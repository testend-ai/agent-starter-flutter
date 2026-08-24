"""Batch execution of conversation cells (tau2 runner layers 2+3).

Layer 2 (build): wires Environment + M2 agent adapter + M1 user simulator +
orchestrator from env/config — the only place components are instantiated.
Layer 3 (batch): iterates tasks x trials x seeds, checkpoints completed
(trial, task_id, seed) cells by scanning artifacts, and writes run-data
artifacts.

Also hosts the CSV -> Task loader (drop ``Ausgeschlossen=Ja``, map
persona/goal/opener) and the text-mode M2 adapter used when the real LiveKit
worker is not in the loop.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from environment import DEFAULT_POLICY, Environment, TelephonyDB
from m1_simulator import LLMConfig, M1UserSimulator, derive_seed, generate_knobs, litellm_llm_fn
from orchestrator import FullDuplexTickOrchestrator, HalfDuplexOrchestrator
from report import build_report, existing_cells, write_report
from schemas import DifficultyTier, Message, Persona, RunMode, SimulationRun, Task, ToolCall

load_dotenv(Path(__file__).with_name(".env"))
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

logger = logging.getLogger("batch-runner")

REQUIRED_COLUMNS = ("intent_name", "description", "scenario", "caller_name", "gender", "anrede")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip().strip('"').strip("'")


def _mask(key: str) -> str:
    return f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"


# ------------------------------------------------------------------ CSV loading
def load_tasks(csv_path: Path, limit: int | None = None) -> list[Task]:
    """IVA_Test.csv -> normalized Task list (drops ``Ausgeschlossen=Ja`` rows).

    The intent label survives only as task_id prefix / metadata for
    traceability — it is not used to steer the conversation.
    """
    tasks: list[Task] = []
    dropped_excluded = 0
    seen_ids: set[str] = set()
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            if (row.get("Ausgeschlossen") or "").strip().lower() == "ja":
                dropped_excluded += 1
                continue
            missing = [c for c in REQUIRED_COLUMNS if not (row.get(c) or "").strip()]
            if missing:
                logger.debug("row skipped, missing %s", missing)
                continue
            contact_id = (row.get("ID des Kontakts") or "").strip() or str(len(tasks) + 1)
            intent = (row.get("intent_name") or "").strip()
            task_id = f"{intent}#{int(contact_id):04d}"
            if task_id in seen_ids:
                continue
            seen_ids.add(task_id)
            tasks.append(
                Task(
                    task_id=task_id,
                    goal=row["description"].strip(),
                    opener=row["scenario"].strip(),
                    persona=Persona(
                        caller_name=row["caller_name"].strip(),
                        gender=row["gender"].strip() or "weiblich",
                        anrede=row["anrede"].strip() or "Sie",
                        pronouns_de=[],
                    ),
                    metadata={
                        "intent": intent,
                        "callernbr": (row.get("callernbr") or "").strip(),
                        "priority": (row.get("Priorität") or "").strip(),
                        "contact_id": contact_id,
                    },
                    filters={"excluded": False},
                )
            )
            if limit and len(tasks) >= limit:
                break
    logger.info("loaded %d tasks (%d excluded rows dropped)", len(tasks), dropped_excluded)
    return tasks


# ------------------------------------------------------------------ Layer 2 build
class M2TextAgent:
    """Text-mode stand-in / harness for the M2 calling model.

    Talks to any OpenAI-compatible endpoint using the *M2_* env config so the
    exact model that serves the LiveKit worker is exercised without a room.
    When ``endpoint`` is empty the adapter echoes a minimal deterministic reply,
    which keeps offline tests honest about plumbing without network calls.

    The adapter implements the same contract as tau2's LLMAgent: respond to the
    history, optionally emit tool calls; never both with text.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        environment: Environment | None = None,
        instructions: str = DEFAULT_POLICY,
        llm_fn=None,
    ) -> None:
        self.config = config or LLMConfig(
            endpoint=_env("M2_MODEL_ENDPOINT"),
            api_key=_env("M2_MODEL_API_KEY"),
            model=_env("M2_MODEL_NAME", "gpt-4o-mini"),
            temperature=float(_env("M2_TEMPERATURE", "0.7") or 0.7),
        )
        self.environment = environment
        self.instructions = instructions
        self.llm_fn = llm_fn or litellm_llm_fn(self.config)
        if self.config.endpoint:
            logger.info("M2TextAgent model=%s endpoint=%s key=%s", self.config.model, self.config.endpoint, _mask(self.config.api_key))

    @property
    def model_id(self) -> str:
        return self.config.model

    def _payload(self, history: list[Message]) -> list[dict]:
        """Chat payload with tool exchanges rendered textually.

        Real ``tools`` schemas are still advertised so capable endpoints emit
        structured tool_calls; the textual rendering of tool results keeps the
        adapter portable across strict and lax OpenAI-compatible providers.
        """
        payload = [{"role": "system", "content": self.instructions}]
        for msg in history:
            if msg.role == "assistant" and msg.content:
                payload.append({"role": "assistant", "content": msg.content})
            elif msg.role == "assistant" and msg.tool_calls:
                calls = ", ".join(f"{c.name}({c.arguments})" for c in msg.tool_calls)
                payload.append({"role": "assistant", "content": f"[tool_calls: {calls}]"})
            elif msg.role == "tool":
                who = msg.requestor or "agent"
                prefix = "error" if msg.error else "result"
                payload.append({"role": "user", "content": f"[{who} tool {prefix}: {msg.content}]"})
            elif msg.role == "user" and msg.content:
                payload.append({"role": "user", "content": msg.content})
        if not payload[1:]:
            payload.append({"role": "user", "content": "(caller connected)"})
        elif payload[-1]["role"] != "user":
            payload.append({"role": "user", "content": "(the caller is waiting)"})
        return payload

    def generate_next_message(self, history: list[Message]) -> Message:
        tools = self.environment.assistant_tools.tool_schemas() if self.environment else None
        raw = self.llm_fn(model=self.config.model, messages=self._payload(history), tools=tools)
        content = raw.strip()
        tool_calls: list[ToolCall] = []
        pending = getattr(self.llm_fn, "last_tool_calls", None)
        if pending:
            for idx, tc in enumerate(pending):
                tool_calls.append(
                    ToolCall(
                        id=tc.id or f"call_{idx}",
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments or "{}"),
                    )
                )
        if tool_calls:
            content = ""
        return Message(role="assistant", content=content, tool_calls=tool_calls)


def build_orchestrator(
    task: Task,
    mode: RunMode = "text",
    tier: DifficultyTier | None = None,
    trial: int = 1,
    base_seed: int = 0,
    max_turns: int = 10,
    timeout_s: float | None = None,
    m1_config: LLMConfig | None = None,
    m1_llm_fn=None,
    m2_llm_fn=None,
) -> tuple[HalfDuplexOrchestrator | FullDuplexTickOrchestrator, Environment]:
    """Instantiate one live conversation cell from config (build layer)."""
    env = Environment(db=TelephonyDB())

    seed = derive_seed(base_seed, task.task_id, trial)
    knobs = generate_knobs(task.task_id, trial, base_seed, tier=tier)
    user = M1UserSimulator(
        task=task,
        knobs=knobs,
        seed=seed,
        config=m1_config or LLMConfig.from_env("M1"),
        llm_fn=m1_llm_fn,
        environment=env,
    )
    agent = M2TextAgent(environment=env, llm_fn=m2_llm_fn)

    if mode == "audio":
        orchestrator = FullDuplexTickOrchestrator(env=env, agent=agent, user=user, seed=seed)
    else:
        kwargs = {"max_turns": max_turns}
        if timeout_s is not None:
            kwargs["timeout_s"] = timeout_s
        orchestrator = HalfDuplexOrchestrator(env=env, agent=agent, user=user, **kwargs)
    return orchestrator, env


# ------------------------------------------------------------------ Layer 3 batch
def run_tasks(
    tasks: list[Task],
    modes: list[RunMode],
    trials: int = 3,
    base_seed: int = 0,
    results_dir: Path = Path("results"),
    tier: DifficultyTier | None = None,
    resume: bool = True,
    max_turns: int = 8,
    timeout_s: float | None = None,
    m1_config: LLMConfig | None = None,
    m1_llm_fn=None,
    m2_llm_fn=None,
) -> dict[str, object]:
    """Run every (task, trial, seed[, mode]) cell; skip already-written cells."""
    done = existing_cells(results_dir) if resume else set()
    written: list[Path] = []
    skipped = 0
    for trial in range(1, trials + 1):
        for task in tasks:
            seed = derive_seed(base_seed, task.task_id, trial)
            for mode in modes:
                if (trial, task.task_id, seed) in done:
                    logger.info("resume: skipping completed cell %s trial=%d seed=%d mode=%s", task.task_id, trial, seed, mode)
                    skipped += 1
                    continue
                orchestrator, _env_ = build_orchestrator(
                    task=task,
                    mode=mode,
                    tier=tier,
                    trial=trial,
                    base_seed=base_seed,
                    max_turns=max_turns,
                    timeout_s=timeout_s,
                    m1_config=m1_config,
                    m1_llm_fn=m1_llm_fn,
                    m2_llm_fn=m2_llm_fn,
                )
                run: SimulationRun = orchestrator.run(task, trial=trial, seed=seed)
                report = build_report(task, run)
                path = write_report(results_dir, report)
                logger.info("%s -> %s [%d turns, stop=%s]", run.run_id, path, report.turns, report.stop_reason)
                written.append(path)
    return {"runs": len(written), "skipped": skipped, "artifacts": [str(p) for p in written]}
