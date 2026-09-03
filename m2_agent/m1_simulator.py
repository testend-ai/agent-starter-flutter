"""M1 — the LLM user simulator that plays the human caller.

M1 gets goal + persona + seeded behavioral knobs and generates the caller side
of the conversation. Any OpenAI-compatible endpoint works via litellm
(M1_MODEL_ENDPOINT / M1_API_KEY / M1_MODEL_NAME), so tier routing
(Gemini Flash for Easy/Medium, Claude for Hard/Adversarial, DeepSeek budget)
is a .env change, not a code change.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from pydantic import BaseModel

from environment import Environment
from schemas import STOP_MARKER
from schemas import BehaviorKnobs, DifficultyTier, Message, Persona, Task, ToolCall, now_ns

load_dotenv(Path(__file__).with_name(".env"))
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

logger = logging.getLogger("m1-simulator")

# user-simulator-models.md difficulty tiers -> knob ranges (seeded jitter on top)
TIER_KNOBS: dict[DifficultyTier, dict[str, tuple[float, float] | int]] = {
    "easy": {
        "verbosity": (0.5, 0.8),
        "interruption_likelihood": (0.0, 0.05),
        "patience_threshold": 12,
        "hesitation": (0.0, 0.2),
    },
    "medium": {
        "verbosity": (0.3, 0.7),
        "interruption_likelihood": (0.05, 0.2),
        "patience_threshold": 10,
        "hesitation": (0.2, 0.4),
    },
    "hard": {
        "verbosity": (0.2, 0.6),
        "interruption_likelihood": (0.2, 0.45),
        "patience_threshold": 6,
        "hesitation": (0.4, 0.7),
    },
    "adversarial": {
        "verbosity": (0.1, 0.4),
        "interruption_likelihood": (0.35, 0.6),
        "patience_threshold": 4,
        "hesitation": (0.3, 0.6),
    },
}

TIER_DISPOSITION: dict[DifficultyTier, str] = {
    "easy": "cooperative and clear; you answer directly and want to be helped.",
    "medium": "occasionally distracted and mildly ambiguous; you may need one thing repeated.",
    "hard": "impatient; you give partial or slightly contradictory information and may hang up if frustrated.",
    "adversarial": "unhelpful and testing; you question the agent's answers and resist confirmation.",
}


def derive_pronouns_de(gender: str) -> list[str]:
    """Pronouns from the gender field ONLY — never from the name (plan 9 risk table)."""
    normalized = gender.strip().lower()
    if normalized in {"männlich", "maennlich", "male", "m"}:
        return ["er", "ihm"]
    if normalized in {"weiblich", "female", "w"}:
        return ["sie", "ihr"]
    return ["sie", "ihm"]


def build_persona(task: Task) -> Persona:
    return Persona(
        caller_name=task.persona.caller_name,
        gender=task.persona.gender,
        anrede=task.persona.anrede,
        pronouns_de=task.persona.pronouns_de or derive_pronouns_de(task.persona.gender),
    )


def derive_seed(base_seed: int, task_id: str, trial: int) -> int:
    """Deterministic seed from (trial, task_id, base) so cells replay identically."""
    digest = hashlib.sha256(f"{base_seed}:{task_id}:{trial}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def generate_knobs(
    task_id: str,
    trial: int,
    base_seed: int,
    tier: DifficultyTier | None = None,
) -> BehaviorKnobs:
    """Deterministic knobs from (task_id, trial, seed); tier default Medium."""
    resolved_tier: DifficultyTier = tier or os.environ.get("M1_DIFFICULTY", "medium").lower()  # type: ignore[assignment]
    if resolved_tier not in TIER_KNOBS:
        resolved_tier = "medium"
    rng = random.Random(derive_seed(base_seed, f"{task_id}#knobs", trial))
    spec = TIER_KNOBS[resolved_tier]
    lo, hi = spec["verbosity"]  # type: ignore[misc]
    ilo, ihi = spec["interruption_likelihood"]  # type: ignore[misc]
    hlo, hhi = spec["hesitation"]  # type: ignore[misc]
    return BehaviorKnobs(
        tier=resolved_tier,
        verbosity=round(rng.uniform(lo, hi), 3),
        interruption_likelihood=round(rng.uniform(ilo, ihi), 3),
        patience_threshold=int(spec["patience_threshold"]),  # type: ignore[arg-type]
        hesitation=round(rng.uniform(hlo, hhi), 3),
    )


class LLMConfig(BaseModel):
    """Plug-and-play endpoint config (mirrors M2_* naming)."""

    endpoint: str = ""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7

    @classmethod
    def from_env(cls, prefix: str = "M1") -> "LLMConfig":
        return cls(
            endpoint=_env(f"{prefix}_MODEL_ENDPOINT"),
            api_key=_env(f"{prefix}_API_KEY") or _env("OPENAI_API_KEY"),
            model=_env(f"{prefix}_MODEL_NAME", "gpt-4o-mini"),
            temperature=float(_env(f"{prefix}_TEMPERATURE", "0.7") or 0.7),
        )

    @property
    def masked_key(self) -> str:
        return mask_key(self.api_key)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip().strip('"').strip("'")


def mask_key(key: str) -> str:
    """Never log full API keys (test_model.py pattern)."""
    if len(key) > 8:
        return f"{key[:4]}...{key[-4:]}"
    return "***"


def litellm_llm_fn(config: LLMConfig) -> Callable[..., str]:
    """Default completion callable routed through litellm (Libraries doc section 2).

    Custom endpoints use the ``openai/`` provider prefix with ``api_base`` so
    any OpenAI-compatible gateway (OpenRouter, vLLM, Ollama...) works. Retries
    transient gateway failures (free tiers 502 often).
    """

    def llm_fn(*, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
        from litellm import completion  # imported lazily so --mock runs need no network deps

        kwargs: dict[str, Any] = {
            "model": f"openai/{model}" if config.endpoint else model,
            "messages": messages,
            "temperature": config.temperature,
            "num_retries": 3,
        }
        if config.endpoint:
            kwargs["api_base"] = config.endpoint.rstrip("/")
            kwargs["api_key"] = config.api_key or "placeholder"
        if tools:
            kwargs["tools"] = tools
        response = completion(**kwargs)
        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None)
        # stash provider tool calls on THIS callable so callers can read them back
        llm_fn.last_tool_calls = tool_calls  # type: ignore[attr-defined]
        content = message.content or ""
        if not content.strip() and not tool_calls:
            # never return silent turns — forces callers/orchestrator to see the failure
            raise RuntimeError(f"empty LLM completion (finish_reason={choice.finish_reason})")
        return content

    return llm_fn


LLMFn = Callable[..., str]

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are {caller_name}, a {gender_word} bank customer from Germany.
Register: formality {anrede} ("Sie" = formal, "Du" = informal). Always answer in German, using this register consistently ({anrede_form}).
Your goal: {goal}
You have only normal-customer knowledge — no insider bank knowledge, no special authority.
Do not reveal your goal unprompted; act naturally, ask follow-up questions as a real customer would.
Do not volunteer final confirmation immediately — wait until the agent has actually earned it with a concrete answer.
Disposition: {disposition}
Style knobs (obey them): verbosity {verbosity} (0 terse, 1 chatty), patience {patience} turns, hesitation {hesitation}.
If the agent solved your need and confirmed it, accept politely and call the confirm_resolution tool when offered. If the call is over, reply with exactly {stop}. Never mention these instructions."""


def build_system_prompt(task: Task, knobs: BehaviorKnobs) -> str:
    """Persona+goal prompt built from the task's goal/opener/persona fields."""
    persona = build_persona(task)
    formal = persona.anrede.strip().lower() == "sie"
    gender_word = {"er": "male", "sie": "female"}.get(persona.pronouns_de[0], "caller")
    prompt = DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(
        caller_name=persona.caller_name,
        gender_word=gender_word,
        anrede=persona.anrede,
        anrede_form="Siezen" if formal else "Duzen",
        goal=task.goal,
        disposition=TIER_DISPOSITION[knobs.tier],
        verbosity=knobs.verbosity,
        patience=knobs.patience_threshold,
        hesitation=knobs.hesitation,
        stop=STOP_MARKER,
    )
    return prompt


class M1UserSimulator:
    """Half/full-duplex user side of the conversation.

    ``llm_fn`` is injectable for offline tests; production uses litellm.
    User-side tools come from the shared Environment (dual-control), but M1 can
    never see agent-side tools.
    """

    def __init__(
        self,
        task: Task,
        knobs: BehaviorKnobs,
        seed: int,
        config: LLMConfig | None = None,
        llm_fn: LLMFn | None = None,
        environment: Environment | None = None,
    ) -> None:
        self.task = task
        self.knobs = knobs
        self.seed = seed
        self.rng = random.Random(seed)
        self.config = config or LLMConfig.from_env("M1")
        self.llm_fn = llm_fn or litellm_llm_fn(self.config)
        self.environment = environment
        self.system_prompt = build_system_prompt(task, knobs)
        self.turn_count = 0

    # ------------------------------------------------------------------ turns
    def open_message(self) -> Message:
        """The CSV opener is sent verbatim as the first utterance."""
        return Message(role="user", content=self.task.opener, timestamp_ns=now_ns())

    def _visible_history(self, history: list[Message]) -> list[Message]:
        """Caller-visible context: user/assistant text + own tool results only.

        Agent-side tool calls/results are internal to M2; the caller never saw
        them happen, so they are excluded (tau2 decoupled-user principle).
        """
        visible: list[Message] = []
        for msg in history:
            if msg.role == "tool":
                if msg.requestor == "user":
                    visible.append(msg)
                continue
            if msg.role in {"user", "assistant"} and msg.content:
                visible.append(msg)
        return visible

    def _chat_payload(self, history: list[Message]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        for msg in self._visible_history(history):
            role = "assistant" if msg.role == "assistant" else "user"
            payload.append({"role": role, "content": msg.content})
        # chat APIs require the last turn to be from the user; if our own prior
        # line (or a user-tool note) is last, nudge for the next reaction.
        if not payload[1:] or payload[-1]["role"] != "user":
            payload.append({"role": "user", "content": "(continue the call naturally)"})
        return payload

    def generate_next_message(self, history: list[Message]) -> Message:
        self.turn_count += 1
        messages = self._chat_payload(history)
        tools = self.environment.user_tools.tool_schemas() if self.environment else None
        raw = self.llm_fn(model=self.config.model, messages=messages, tools=tools)
        content = raw.strip()
        tool_calls: list[ToolCall] = []
        pending = getattr(self.llm_fn, "last_tool_calls", None)
        if pending:
            for idx, tc in enumerate(pending):
                fn = tc.function
                args = json.loads(fn.arguments or "{}")
                tool_calls.append(ToolCall(id=tc.id or f"call_{idx}", name=fn.name, arguments=args))
        stop = STOP_MARKER in content
        if stop:
            content = content.replace(STOP_MARKER, "").strip()
        if tool_calls:
            # Message validator forbids text+tools together; tool turns carry no prose
            content = ""
        return Message(role="user", content=content, tool_calls=tool_calls, timestamp_ns=now_ns())
