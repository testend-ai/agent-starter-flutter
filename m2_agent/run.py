#!/usr/bin/env python3
"""One-command M1<->M2 conversation harness (plug and play).

Point it at the golden dataset and it starts conversation sessions
automatically, saving transcript/telemetry artifacts per run:

    python m2_agent/run.py                          # text mode, first 3 tasks
    python m2_agent/run.py --limit 20 --trials 3
    python m2_agent/run.py --mode room              # full LiveKit sessions
    python m2_agent/run.py --mode room,text         # both, same tasks

The golden CSV is auto-discovered: --csv flag > GOLDEN_DATASET_CSV env >
~/AI-eval-testing/IVA_Test.csv > ./IVA_Test.csv.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from batch_runner import load_tasks, run_tasks  # noqa: E402
from m1_simulator import derive_seed  # noqa: E402
from m1_simulator import LLMConfig  # noqa: E402
from report import existing_cells  # noqa: E402
from room_session import run_room_session  # noqa: E402
from schemas import DifficultyTier  # noqa: E402

logger = logging.getLogger("run")

CSV_CANDIDATES = (
    Path.home() / "AI-eval-testing" / "IVA_Test.csv",
    Path("IVA_Test.csv"),
    Path("../AI-eval-testing/IVA_Test.csv"),
)


def locate_csv(explicit: str | None) -> Path:
    """Resolve the golden dataset: flag > env > auto-discovery.

    Accepts a file or a directory (uses ``IVA_Test.csv`` inside it, else the
    first CSV found there).
    """

    def _resolve(path: Path) -> Path | None:
        path = path.expanduser()
        if path.is_dir():
            named = path / "IVA_Test.csv"
            if named.exists():
                return named
            csvs = sorted(path.glob("*.csv"))
            return csvs[0] if csvs else None
        return path if path.exists() else None

    if explicit:
        resolved = _resolve(Path(explicit))
        if resolved is None:
            raise SystemExit(f"CSV not found: {explicit}")
        return resolved
    env_path = os.environ.get("GOLDEN_DATASET_CSV", "").strip()
    if env_path:
        resolved = _resolve(Path(env_path))
        if resolved is None:
            raise SystemExit(f"GOLDEN_DATASET_CSV points to a missing dataset: {env_path}")
        return resolved
    for candidate in CSV_CANDIDATES:
        resolved = _resolve(candidate)
        if resolved is not None:
            return resolved
    searched = ", ".join(str(c) for c in CSV_CANDIDATES)
    raise SystemExit(f"Golden dataset IVA_Test.csv not found (searched: {searched}). Pass --csv.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Start M1<->M2 conversations from the golden dataset")
    parser.add_argument("--csv", default=None, help="Path to IVA_Test.csv (auto-discovered by default)")
    parser.add_argument("--limit", type=int, default=0, help="Max usable rows (0 = sweep the whole dataset)")
    parser.add_argument("--task", default=None, help="Only run tasks whose task_id contains this substring")
    parser.add_argument("--trials", type=int, default=1, help="Conversation trials per task")
    parser.add_argument("--mode", default="text", help="Comma list: text, room (default: text)")
    parser.add_argument("--tier", choices=["easy", "medium", "hard", "adversarial"], default="medium")
    parser.add_argument("--max-turns", type=int, default=8, help="Text-mode turn budget (room mode caps at 6)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Wall-clock budget per text session (s)")
    parser.add_argument("--room-wait", type=int, default=180, help="Total budget per room session (s)")
    parser.add_argument("--no-worker", action="store_true", help="Room mode: do NOT spawn agent.py; drive your own `lk agent dev`")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-resume", action="store_true", help="Re-run completed cells instead of skipping them")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    csv_path = locate_csv(args.csv)
    results_dir = Path(args.results_dir)

    tasks = load_tasks(csv_path)  # full usable dataset; resume decides what runs
    if args.task:
        tasks = [t for t in tasks if args.task.lower() in t.task_id.lower()]
    if args.limit and args.limit > 0:
        pending = existing_cells(results_dir)
        # apply the limit to rows that still have work to do, not to the file
        remaining: list = []
        for task in tasks:
            has_pending = any(
                (trial, task.task_id, derive_seed(0, task.task_id, trial)) not in pending
                for trial in range(1, args.trials + 1)
            ) if not args.no_resume else True
            if args.no_resume or has_pending:
                remaining.append(task)
            if len(remaining) >= args.limit:
                break
        tasks = remaining
    if not tasks:
        logger.error("no usable tasks after filtering (%s)", csv_path)
        return 1

    modes = [m.strip().lower() for m in args.mode.split(",") if m.strip()]
    unknown = [m for m in modes if m not in {"text", "room"}]
    if unknown:
        parser.error(f"unknown mode(s): {unknown} — use text and/or room")
    tier: DifficultyTier = args.tier  # type: ignore[assignment]

    print("=" * 70)
    print(f"golden dataset : {csv_path}")
    print(f"tasks          : {len(tasks)} ({', '.join(t.task_id for t in tasks[:5])}{' …' if len(tasks) > 5 else ''})")
    print(f"modes          : {modes} · trials={args.trials} · tier={tier} · results={results_dir}/")
    print("=" * 70)

    artifacts: list[str] = []
    skipped = 0

    if "text" in modes:
        result = run_tasks(
            tasks=tasks,
            modes=["text"],
            trials=args.trials,
            results_dir=results_dir,
            tier=tier,
            resume=not args.no_resume,
            max_turns=args.max_turns,
            timeout_s=args.timeout,
        )
        artifacts.extend(result["artifacts"])
        skipped += result["skipped"]
        print(f"[text] {result['runs']} new session artifact(s), {skipped} resumed")

    if "room" in modes:
        m1_cfg = LLMConfig.from_env("M1")
        room_turns = max(1, min(args.max_turns, 6))
        for trial in range(1, args.trials + 1):
            for task in tasks:
                print(f"\n--- room session {task.task_id} trial={trial} ---")
                out = run_room_session(
                    task=task,
                    results_dir=results_dir,
                    session_budget_s=args.room_wait,
                    max_turns=room_turns,
                    spawn_worker=not args.no_worker,
                    trial=trial,
                    m1_cfg=m1_cfg,
                )
                if out:
                    artifacts.append(str(out))

    print("=" * 70)
    print(f"done — {len(artifacts)} artifact(s) in {results_dir}/")
    for artifact in artifacts:
        print(f"  {artifact}")
    return 0 if (artifacts or skipped) else 1


if __name__ == "__main__":
    sys.exit(main())
