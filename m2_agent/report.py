"""Run-data artifact writing (core data-obtaining output).

Produces ``results/<task_id>/<run>.json`` holding the raw conversation
evidence — transcript, turn counts, telemetry latencies and run metadata.
Checkpoint cell scanning keyed by ``(trial, task_id, seed)`` supports resumable
batch runs. No scoring/grading lives here; consumers get the raw data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from schemas import Message, RunReport, SimulationRun, Task, transcript_text

logger = logging.getLogger("report")


def build_report(task: Task, run: SimulationRun) -> RunReport:
    """Assemble the per-run data artifact from a finished conversation."""
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
    return RunReport(
        task_id=task.task_id,
        run_id=run.run_id,
        trial=run.trial,
        seed=run.seed,
        mode=run.mode,
        turns=turns,
        final_transcript=transcript_text(run.trajectory),
        trajectory=run.trajectory,
        ticks=run.ticks,
        stop_reason=run.stop_reason,
        duration_ms=run.duration_ms,
        model_version=run.m1_model,
        temperature=run.m1_temperature,
        m2_model=run.m2_model,
        telemetry_summary=telemetry_summary,
        turn_taking=run.turn_taking,
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


def load_trajectory(path: Path) -> list[Message]:
    """Read back the full ordered trajectory from a saved artifact."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Message(**m) for m in data.get("trajectory", [])]
