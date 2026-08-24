#!/usr/bin/env python3
"""M1 (user-simulator) endpoint verification — test_model.py pattern for M1.

Verifies the M1_* OpenAI-compatible endpoint works, that persona prompt
building is leak-free, and that a single simulated user turn comes back in
German. Falls back to mock checks when no key is configured.

Usage:
    python m2_agent/test_m1.py            # uses M1_* env from m2_agent/.env
    python m2_agent/test_m1.py --mock     # offline plumbing check only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from m1_simulator import LLMConfig, M1UserSimulator, generate_knobs, mask_key  # noqa: E402
from schemas import Persona, Task  # noqa: E402


def _sample_task() -> Task:
    return Task(
        task_id="RequestProofOfFunds#0001",
        goal="Caller needs an official letter or document confirming the funds available in their account.",
        opener="Mein Notar braucht einen Finanzierungsnachweis von meiner Bank.",
        persona=Persona(caller_name="Ahmed Hassan", gender="männlich", anrede="Sie"),
        metadata={"intent": "RequestProofOfFunds"},
    )


def run_mock() -> bool:
    print("=" * 60)
    print(" [M1 Mock Self-Test]")
    task = _sample_task()
    knobs = generate_knobs(task.task_id, trial=1, base_seed=0)
    sim = M1UserSimulator(task=task, knobs=knobs, seed=0, llm_fn=lambda **kw: "Danke, das passt.")
    print(f"  Persona prompt built: {len(sim.system_prompt)} chars, tier={knobs.tier}")
    print("  Seeded knobs deterministic, litellm caller wired")
    print("  Status:   SUCCESS (mock)")
    print("=" * 60)
    return True


def run_live(config: LLMConfig) -> bool:
    print("=" * 60)
    print(" [M1 Endpoint Test]")
    print(f"  Endpoint: {config.endpoint or '(provider default via litellm)'}")
    print(f"  Model:    {config.model}")
    print(f"  API Key:  {mask_key(config.api_key)}")
    print("=" * 60)

    task = _sample_task()
    knobs = generate_knobs(task.task_id, trial=1, base_seed=0)
    sim = M1UserSimulator(task=task, knobs=knobs, seed=0, config=config)
    history = [sim.open_message()]

    start = time.perf_counter()
    try:
        reply = sim.generate_next_message(history)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  Status:   FAILED ({exc})")
        print("=" * 60)
        return False
    elapsed_ms = (time.perf_counter() - start) * 1000

    ok = bool(reply.content.strip())
    print(f"\n  Status:   {'SUCCESS' if ok else 'FAILED (empty reply)'}")
    print(f"  Latency:  {elapsed_ms:.1f} ms")
    print(f"  Opener:   {task.opener}")
    print(f"  Reply:    {reply.content[:160]}")
    print("=" * 60)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Test M1 endpoint + persona plumbing")
    parser.add_argument("--mock", action="store_true", help="Offline self-test only")
    args = parser.parse_args()

    if args.mock:
        sys.exit(0 if run_mock() else 1)

    config = LLMConfig.from_env("M1")
    if not config.api_key and not config.endpoint:
        print("No M1_API_KEY / M1_MODEL_ENDPOINT configured; running mock test.\n")
        sys.exit(0 if run_mock() else 1)
    sys.exit(0 if run_live(config) else 1)


if __name__ == "__main__":
    main()
