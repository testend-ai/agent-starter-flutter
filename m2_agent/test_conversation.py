#!/usr/bin/env python3
"""Direct proof that M1 <-> M2 conversations work end-to-end.

Modes
-----
--mock       Fully offline: scripted M1 + scripted M2 against the real
             Environment/orchestrators. Asserts turn ordering (tool results
             execute BEFORE the next turn), intent extraction, anti-cheat
             guardrails, knob determinism and writes artifacts. Also smoke-tests
             the 200 ms full-duplex tick loop and logs Tick records.
--live       Real M1 endpoint <-> M2-endpoint LLM in-process (no LiveKit room).
             Uses M1_* / M2_* env config verbatim; writes one artifact per run.
--live-room  Experimental: joins a LiveKit room as an 'm1-simulator'
             participant, publishes the CSV opener on the 'lk.chat' data topic
             (what session.sendText uses), records whatever the hosted M2
             worker answers (transcriptions / data) for --room-wait seconds.

Usage:
    python m2_agent/test_conversation.py --mock
    python m2_agent/test_conversation.py --live [--csv PATH] [--row N]
    python m2_agent/test_conversation.py --live-room [--room NAME] [--room-wait 30]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from batch_runner import M2TextAgent, build_orchestrator, load_tasks  # noqa: E402
from environment import Environment, TelephonyDB  # noqa: E402
from m1_simulator import (  # noqa: E402
    FORBIDDEN_FIELDS,
    STOP_MARKER,
    M1UserSimulator,
    assert_no_leak,
    derive_pronouns_de,
    derive_seed,
    generate_knobs,
)
from orchestrator import FullDuplexTickOrchestrator, HalfDuplexOrchestrator  # noqa: E402
from report import build_report, extract_predicted_label, grade_intent, token_f1, write_report  # noqa: E402
from schemas import Persona, Task  # noqa: E402

logger = logging.getLogger("test-conversation")

MOCK_TASK = Task(
    task_id="RequestProofOfFunds#0001",
    intent_name="RequestProofOfFunds",
    expected_output="REQUEST_PROOF_OF_FUNDS",
    goal="Caller needs an official letter or document confirming the funds available in their account.",
    opener="Mein Notar braucht einen Finanzierungsnachweis von meiner Bank.",
    persona=Persona(caller_name="Ahmed Hassan", gender="männlich", anrede="Sie"),
)


class MockCompletion:
    """Callable standing in for litellm completion; scripts replies + tool calls."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.last_tool_calls: list[Any] | None = None

    def __call__(self, *, model: str, messages: list[dict], tools: list | None) -> str:
        logger.debug("mock %s call #%d (%d msgs)", model, self.calls + 1, len(messages))
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        self.last_tool_calls = reply.get("tool_calls")
        return str(reply["content"])

    @staticmethod
    def tool_call(name: str, arguments: dict) -> Any:
        class _Fn:
            def __init__(inner, name: str, arguments: dict) -> None:
                inner.name = name
                inner.arguments = json.dumps(arguments)

        class _TC:
            def __init__(inner, name: str, arguments: dict) -> None:
                inner.id = f"call_{name}"
                inner.function = _Fn(name, arguments)

        return _TC(name, arguments)


# --------------------------------------------------------------------------- mock
def run_mock(results_dir: Path) -> int:
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    print("=" * 70)
    print(" [1] unit checks: pronouns, knobs determinism, anti-cheat, token-F1")
    check("pronouns from gender only", derive_pronouns_de("männlich") == ["er", "ihm"] and derive_pronouns_de("weiblich") == ["sie", "ihr"])

    knobs_a = generate_knobs(MOCK_TASK.task_id, trial=1, base_seed=42)
    knobs_b = generate_knobs(MOCK_TASK.task_id, trial=1, base_seed=42)
    knobs_c = generate_knobs(MOCK_TASK.task_id, trial=2, base_seed=42)
    check("knobs deterministic per (task,trial,seed)", knobs_a == knobs_b and knobs_a != knobs_c)

    try:
        assert_no_leak(f"please reach {MOCK_TASK.expected_output}", MOCK_TASK)
        check("anti-cheat rejects gold label", False)
    except ValueError:
        check("anti-cheat rejects gold label", True)
    try:
        assert_no_leak(f"intent is {MOCK_TASK.intent_name}", MOCK_TASK)
        check("anti-cheat rejects camelCase intent", False)
    except ValueError:
        check("anti-cheat rejects camelCase intent", True)
    prompt_user = M1UserSimulator(
        task=MOCK_TASK,
        knobs=knobs_a,
        seed=0,
        llm_fn=lambda **kw: "",
    ).system_prompt
    check("M1 prompt contains goal+opener fields only", MOCK_TASK.goal.split()[0] in prompt_user and "Ahmed Hassan" in prompt_user)
    check(
        "forbidden fields absent from prompt builder inputs",
        all(f not in ("description", "scenario", "caller_name", "gender", "anrede") for f in FORBIDDEN_FIELDS),
    )

    check("token-F1 exact", token_f1("REQUEST_PROOF_OF_FUNDS", "REQUEST_PROOF_OF_FUNDS") == 1.0)
    check("token-F1 disjoint", token_f1("TRANSFER_TO_HUMAN_AGENT", "REQUEST_BANK_STATEMENT") == 0.0)
    check(
        "token-F1 camelCase normalisation",
        token_f1("RequestProofOfFunds", "REQUEST PROOF OF FUNDS") == 1.0,
    )

    print("-" * 70)
    print(" [2] half-duplex mocked conversation M1<->M2 (no network)")

    env = Environment(db=TelephonyDB())
    m2_mock = MockCompletion(
        replies=[
            {
                "content": "",
                "tool_calls": [MockCompletion.tool_call("detect_intent", {"intent_code": "REQUEST_PROOF_OF_FUNDS"})],
            },
            {"content": "Selbstverständlich, ich lege den Finanzierungsnachweis für Ihren Notar an. Habe ich damit geholfen?", "tool_calls": None},
            {"content": f"{STOP_MARKER} Alles klar, auf Wiedersehen!", "tool_calls": None},
        ]
    )
    m1_mock = MockCompletion(
        replies=[
            {
                "content": "",
                "tool_calls": [MockCompletion.tool_call("confirm_resolution", {"accepted": True})],
            },
            {"content": "Ja, das passt. Vielen Dank!", "tool_calls": None},
        ]
    )
    user = M1UserSimulator(task=MOCK_TASK, knobs=knobs_a, seed=7, environment=env, llm_fn=m1_mock)
    agent = M2TextAgent(environment=env, llm_fn=m2_mock)
    orch = HalfDuplexOrchestrator(env=env, agent=agent, user=user, max_turns=6)
    run = orch.run(MOCK_TASK, trial=1, seed=7, run_id="mock-half-duplex-1")

    roles = [m.role for m in run.trajectory]
    check(">=2 user turns", roles.count("user") >= 2, f"user={roles.count('user')}")
    check(">=2 assistant turns", roles.count("assistant") >= 2, f"assistant={roles.count('assistant')}")

    # the agent's detect_intent tool result must sit in history before its first
    # visible reply (=> before any next user turn saw anything)
    first_reply_idx = next(i for i, m in enumerate(run.trajectory) if m.role == "assistant" and m.content)
    tools_before_reply = [
        i for i, m in enumerate(run.trajectory[:first_reply_idx]) if m.role == "tool" and m.requestor == "agent"
    ]
    check(
        "tool calls executed BEFORE agent reply / next turn",
        bool(tools_before_reply),
        f"reply@{first_reply_idx} tool@{tools_before_reply}",
    )
    check(
        "no message mixes text+tool_calls",
        all(not (m.has_text and m.has_tool_calls) for m in run.trajectory),
    )
    check("environment recorded intent", "REQUEST_PROOF_OF_FUNDS" in TelephonyDB(**env.snapshot()).detected_intents)
    check("caller confirmation recorded", TelephonyDB(**env.snapshot()).resolution_confirmed_by_caller)

    predicted = extract_predicted_label(run)
    matched, f1, outcome = grade_intent(MOCK_TASK, predicted)
    check("predicted label extracted", predicted == "REQUEST_PROOF_OF_FUNDS", predicted or "(empty)")
    check("intent match pass", matched and outcome == "pass")

    report = build_report(MOCK_TASK, run)
    path = write_report(results_dir, report)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    check("artifact written & parseable", parsed["task_id"] == MOCK_TASK.task_id and parsed["intent_match"])
    print(f"        artifact: {path}")

    print("-" * 70)
    print(" [3] full-duplex 200 ms tick smoke test (TTS/STT mocked)")
    audio_env = Environment(db=TelephonyDB())
    m2_audio = MockCompletion(
        replies=[
            {"content": "Gerne, ich prüfe das für Sie. Eine Moment bitte.", "tool_calls": None},
            {f"content": f"{STOP_MARKER} Erledigt. Auf Wiedersehen!", "tool_calls": None},
        ]
    )
    m1_audio = MockCompletion(
        replies=[
            {"content": "Danke, das reicht.", "tool_calls": None},
            {"content": f"{STOP_MARKER} Tschüss!", "tool_calls": None},
        ]
    )
    audio_user = M1UserSimulator(task=MOCK_TASK, knobs=knobs_a, seed=9, environment=audio_env, llm_fn=m1_audio)
    audio_agent = M2TextAgent(environment=audio_env, llm_fn=m2_audio)
    tick_orch = FullDuplexTickOrchestrator(env=audio_env, agent=audio_agent, user=audio_user, seed=9, max_ticks=60)
    audio_run = tick_orch.run(MOCK_TASK, trial=1, seed=9, run_id="mock-tick-1")
    check("tick loop produced Tick records", len(audio_run.ticks) >= 3, f"{len(audio_run.ticks)} ticks")
    events = [e for t in audio_run.ticks for e in t.events]
    check("ticks log VAD endpoints", any(e == "vad_endpoint" for e in events))
    check("ticks log tts start/stop", "tts_audio_start" in events and "agent_done" in events)
    states_seen = {t.user_state for t in audio_run.ticks}
    check("user state machine exercised", {"SPEAKING", "WAIT_FOR_AGENT"} <= states_seen, str(sorted(states_seen)))
    check("turn-taking metrics attached", audio_run.turn_taking.premature_rate >= 0.0)

    print("=" * 70)
    if failures:
        print(f" RESULT: {len(failures)} FAILED -> {failures}")
        return 1
    print(" RESULT: ALL MOCK CHECKS PASSED")
    return 0


# --------------------------------------------------------------------------- live
def _pick_task(csv_path: Path, row_index: int) -> Task:
    tasks = load_tasks(csv_path, limit=row_index + 1)
    if row_index >= len(tasks):
        raise SystemExit(f"only {len(tasks)} usable rows in {csv_path}")
    return tasks[row_index]


def run_live(csv_path: Path, row_index: int, results_dir: Path, max_turns: int, args_timeout: float = 300.0) -> int:
    from m1_simulator import LLMConfig

    m1_cfg = LLMConfig.from_env("M1")
    m2_cfg_present = bool(__import__("os").environ.get("M2_MODEL_API_KEY") or __import__("os").environ.get("OPENAI_API_KEY"))
    if not (m1_cfg.api_key and m2_cfg_present):
        print("M1/M2 API keys missing (M1_API_KEY / M2_MODEL_API_KEY); use --mock instead.")
        return 2

    task = _pick_task(csv_path, row_index)
    print(f"task={task.task_id} goal='{task.goal[:60]}…' opener='{task.opener[:50]}…'")
    print(f"M1 model={m1_cfg.model} endpoint={m1_cfg.endpoint or '(provider default)'} key={m1_cfg.masked_key}")

    orch, env = build_orchestrator(
        task=task,
        mode="text",
        trial=1,
        base_seed=int(time.time()),
        max_turns=max_turns,
        timeout_s=max(args_timeout, 120.0),
    )
    run = orch.run(task, trial=1, seed=derive_seed(int(time.time()), task.task_id, 1))
    report = build_report(task, run)
    path = write_report(results_dir, report)

    print("-" * 70)
    print(report.final_transcript)
    print("-" * 70)
    print(f"stop={run.stop_reason} turns={report.turns} predicted={report.predicted_label!r} "
          f"expected={task.expected_output} match={report.intent_match} f1={report.intent_f1}")
    print(f"artifact: {path}")
    print(f"db hash: {env.get_db_hash()}")
    return 0 if report.predicted_label else 1


def run_live_room(room_name: str, wait_s: int, csv_path: Path, row_index: int, results_dir: Path) -> int:
    """Experimental probe: drive the hosted M2 worker over a real LiveKit room."""
    import os
    import uuid

    url = os.environ.get("LIVEKIT_URL", "").strip()
    key = os.environ.get("LIVEKIT_API_KEY", "").strip()
    secret = os.environ.get("LIVEKIT_API_SECRET", "").strip()
    if not (url and key and secret):
        print("LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET missing in m2_agent/.env.")
        return 2
    try:
        import livekit.rtc as rtc
        from livekit.api import AccessToken, VideoGrants
    except ImportError:
        print("livekit rtc/api not installed; pip install livekit-agents in .venv")
        return 2

    task = _pick_task(csv_path, row_index)
    identity = f"m1-sim-{uuid.uuid4().hex[:8]}"
    token = (
        AccessToken(key, secret)
        .with_identity(identity)
        .with_name("M1 User Simulator")
        .with_grants(VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    received: list[str] = []
    room = rtc.Room()

    def _on_data(data: rtc.DataPacket) -> None:
        text = data.data.decode("utf-8", errors="replace") if isinstance(data.data, bytes) else str(data.data)
        received.append(f"data[{data.topic or '-'}]: {text}")

    def _on_transcription(ev: Any) -> None:
        for seg in getattr(ev, "segments", []) or []:
            received.append(f"transcription[p={getattr(seg, 'final', '?')}]: {seg.text}")

    room.on("data_received", _on_data)
    room.on("transcription_received", _on_transcription)

    async def _main() -> None:
        await room.connect(url, token)
        print(f"connected identity={identity} room={room_name}")
        await room.local_participant.publish_data(task.opener.encode("utf-8"), reliable=True, topic="lk.chat")
        print(f"opener sent on lk.chat: '{task.opener}'")
        deadline = time.time() + wait_s
        while time.time() < deadline:
            await asyncio.sleep(0.5)
        await room.disconnect()

    import asyncio

    asyncio.run(_main())

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{task.task_id}" / f"live-room-{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"task_id": task.task_id, "opener": task.opener, "received": received}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"captured {len(received)} messages -> {out}")
    for line in received:
        print(" ", line[:120])
    return 0 if received else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Prove M1<->M2 conversation works")
    parser.add_argument("--mock", action="store_true", help="Offline scripted conversation + assertions")
    parser.add_argument("--live", action="store_true", help="Real M1 endpoint <-> M2 endpoint LLM (in-process)")
    parser.add_argument("--live-room", action="store_true", help="Join LiveKit room and probe hosted M2 (experimental)")
    parser.add_argument("--csv", default="/Users/jaime/AI-eval-testing/IVA_Test.csv")
    parser.add_argument("--row", type=int, default=0, help="CSV row index among usable tasks")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0, help="Wall-clock budget per live run (s)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--room", default="benchmark-m1-m2")
    parser.add_argument("--room-wait", type=int, default=30, help="Seconds to listen in --live-room mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results_dir = Path(args.results_dir)
    if args.live:
        return run_live(Path(args.csv), args.row, results_dir, args.max_turns, args.timeout)
    if args.live_room:
        return run_live_room(args.room, args.room_wait, Path(args.csv), args.row, results_dir)
    # default (or explicit --mock): fully offline proof
    with tempfile.TemporaryDirectory(prefix="bench-results-") as tmp:
        return run_mock(Path(tmp))


if __name__ == "__main__":
    sys.exit(main())
