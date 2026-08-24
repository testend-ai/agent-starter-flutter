"""Artifact writing + L1/L2-ready grading helpers (iva plan sections 5.3 & 7).

Produces ``results/<task_id>/<run>.json`` with the audit-tagged field set from
the implementation plan, extracts the predicted label (-> ``real_output``),
scores intent match via the exact rules of Workflow-design.md 4.3
(normalise -> SCREAMING_SNAKE -> exact else token-F1), and aggregates trials
with mean / variance / Wilson CI per Workflow-design.md 4.6.

Grading is deliberately kept offline-gradeable: every field a judge needs is in
the artifact; nothing here calls an LLM.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

from schemas import STOP_MARKER, Message, RunReport, SimulationRun, Task

logger = logging.getLogger("report")

SCREAMING_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")
INTENT_TOOL_NAMES = {"detect_intent", "record_intent", "classify_intent", "route_intent"}


# ---------------------------------------------------------------------------- L2 rules
def normalize_label(label: str) -> str:
    """Uppercase, trim, collapse whitespace, CamelCase -> SCREAMING_SNAKE.

    Camel boundaries are expanded on the ORIGINAL casing (before upper-casing),
    so 'RequestProofOfFunds' and 'REQUEST PROOF OF FUNDS' both normalise to
    REQUEST_PROOF_OF_FUNDS per Workflow-design.md 4.3 step A.
    """
    expanded = re.sub(r"(?<=[a-zäöüß0-9])(?=[A-ZÄÖÜ])", "_", label.strip())
    value = expanded.upper()
    return re.sub(r"[\s_]+", "_", value).strip("_")


def token_f1(pred: str, exp: str) -> float:
    """Token-F1 over '_' tokens (Libraries & Dependencies.md section 7)."""
    p = set(normalize_label(pred).split("_")) - {""}
    e = set(normalize_label(exp).split("_")) - {""}
    if not p and not e:
        return 1.0
    if not p or not e:
        return 0.0
    overlap = len(p & e)
    precision = overlap / len(p)
    recall = overlap / len(e)
    return round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0


def extract_predicted_label(run: SimulationRun) -> str:
    """Predicted intent = agent tool-call args first, else final message codes."""
    for msg in run.trajectory:
        if msg.role == "assistant":
            for call in msg.tool_calls:
                if call.name.lower() in INTENT_TOOL_NAMES:
                    arg = str(call.arguments.get("intent_code") or call.arguments.get("intent") or "")
                    if arg.strip():
                        return normalize_label(arg)
    assistant_texts = [m.content for m in run.trajectory if m.role == "assistant" and m.content]
    for text in reversed(assistant_texts):
        matches = SCREAMING_TOKEN_RE.findall(text.replace(STOP_MARKER, ""))
        if matches:
            return normalize_label(matches[-1])
    return ""


def grade_intent(task: Task, predicted: str) -> tuple[bool, float, str]:
    """Returns (exact_match, f1, outcome) per the 4-way classification."""
    expected = task.expected_output or task.intent_name
    if not predicted:
        return False, 0.0, "no_result"
    f1 = token_f1(predicted, expected)
    if normalize_label(predicted) == normalize_label(expected):
        return True, 1.0, "pass"
    return False, f1, "wrong"


# ---------------------------------------------------------------------------- artifacts
def build_report(
    task: Task,
    run: SimulationRun,
    judge_model: str | None = None,
) -> RunReport:
    """Assemble the per-run artifact payload (grading-ready)."""
    predicted = extract_predicted_label(run)
    matched, f1, outcome = grade_intent(task, predicted)
    turns = sum(1 for m in run.trajectory if m.role == "user")
    telemetry_summary: dict[str, object] = {}
    latencies = [t for t in run.telemetry if t.agent_done_ns]
    if latencies:
        e2e = [t.e2e_ms for t in latencies if t.e2e_ms is not None]
        ttft = [t.ttft_ms for t in latencies if t.ttft_ms is not None]
        telemetry_summary = {
            "exchanges": len(latencies),
            "e2e_ms": sorted(e2e) if e2e else [],
            "ttft_ms": sorted(ttft) if ttft else [],
        }
    report = RunReport(
        task_id=task.task_id,
        run_id=run.run_id,
        trial=run.trial,
        seed=run.seed,
        mode=run.mode,
        turns=turns,
        final_transcript=_final_transcript(run.trajectory),
        intent_match=matched,
        intent_f1=f1,
        outcome=outcome,  # type: ignore[arg-type]
        r_intent=round(f1, 4),
        predicted_label=predicted,
        expected_output=task.expected_output,
        stop_reason=run.stop_reason,
        duration_ms=run.duration_ms,
        model_version=run.m1_model,
        temperature=run.m1_temperature,
        judge_model=judge_model,
        m2_model=run.m2_model,
        telemetry_summary=telemetry_summary,
        turn_taking=run.turn_taking,
    )
    _audit_no_gold_leak_in_m1_context(report)
    return report


def _final_transcript(trajectory: list[Message]) -> str:
    lines = []
    for msg in trajectory:
        if msg.role == "user" and msg.content:
            lines.append(f"user: {msg.content}")
        elif msg.role == "assistant" and msg.content:
            lines.append(f"agent: {msg.content}")
        elif msg.role == "tool":
            who = msg.requestor or "agent"
            lines.append(f"tool({who}): {msg.content}")
    return "\n".join(lines)


def _audit_no_gold_leak_in_m1_context(report: RunReport) -> None:
    """Defence-in-depth: artifact must show gold only as comparator fields."""
    logger.debug(
        "artifact %s predicted=%s expected=%s",
        report.run_id,
        report.predicted_label or "-",
        report.expected_output,
    )


def write_report(results_dir: Path, report: RunReport) -> Path:
    """Write results/<task_id>/<mode>-<trial>-<seed>.json atomically enough."""
    target_dir = results_dir / report.task_id
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{report.mode}-trial{report.trial}-seed{report.seed}.json"
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)
    return path


def existing_cells(results_dir: Path) -> set[tuple[int, str, int]]:
    """Scan completed checkpoint cells keyed by (trial, task_id, seed).

    Mirrors tau2 runner/checkpoint.try_resume so batch runs are resumable.
    """
    done: set[tuple[int, str, int]] = set()
    if not results_dir.exists():
        return done
    for path in results_dir.glob("*/*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            done.add((int(data["trial"]), str(data["task_id"]), int(data["seed"])))
        except Exception:  # noqa: BLE001 — corrupt files never block a batch
            logger.warning("skipping unreadable artifact: %s", path)
    return done


def aggregate(reports: list[RunReport]) -> dict[str, object]:
    """Per-task mean + variance + Wilson CI over trials (Workflow-design.md 4.6).

    Dev-phase reward is R_intent (exact match); token-F1 mean is reported
    alongside so partial credit stays visible without ever counting as success.
    """
    by_task: dict[str, list[RunReport]] = {}
    for report in reports:
        by_task.setdefault(report.task_id, []).append(report)
    summary: dict[str, object] = {}
    for task_id, group in sorted(by_task.items()):
        scores = [1.0 if r.intent_match else 0.0 for r in group]
        f1s = [r.intent_f1 for r in group]
        n = len(scores)
        mean = sum(scores) / n if n else 0.0
        variance = sum((s - mean) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
        ci = wilson_interval(successes=sum(1 for s in scores if s >= 1.0), n=n)
        summary[task_id] = {
            "n": n,
            "success_rate": round(mean, 4),
            "variance": round(variance, 4),
            "wilson_ci": ci,
            "mean_token_f1": round(sum(f1s) / n, 4) if n else 0.0,
        }
    return summary


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval; empty list when N < 3 (never publish thin CIs)."""
    if n < 3:
        return []
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]
