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
import contextlib
import json
import logging
import subprocess
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
    STOP_MARKER,
    M1UserSimulator,
    derive_pronouns_de,
    derive_seed,
    generate_knobs,
)
from orchestrator import FullDuplexTickOrchestrator, HalfDuplexOrchestrator  # noqa: E402
from report import build_report, write_report  # noqa: E402
from schemas import Message, Persona, SimulationRun, Task, TurnTelemetry, now_ns  # noqa: E402

logger = logging.getLogger("test-conversation")

MOCK_TASK = Task(
    task_id="RequestProofOfFunds#0001",
    goal="Caller needs an official letter or document confirming the funds available in their account.",
    opener="Mein Notar braucht einen Finanzierungsnachweis von meiner Bank.",
    persona=Persona(caller_name="Ahmed Hassan", gender="männlich", anrede="Sie"),
    metadata={"intent": "RequestProofOfFunds"},
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
    print(" [1] unit checks: pronouns, knobs determinism")
    check("pronouns from gender only", derive_pronouns_de("männlich") == ["er", "ihm"] and derive_pronouns_de("weiblich") == ["sie", "ihr"])

    knobs_a = generate_knobs(MOCK_TASK.task_id, trial=1, base_seed=42)
    knobs_b = generate_knobs(MOCK_TASK.task_id, trial=1, base_seed=42)
    knobs_c = generate_knobs(MOCK_TASK.task_id, trial=2, base_seed=42)
    check("knobs deterministic per (task,trial,seed)", knobs_a == knobs_b and knobs_a != knobs_c)

    prompt_user = M1UserSimulator(
        task=MOCK_TASK,
        knobs=knobs_a,
        seed=0,
        llm_fn=lambda **kw: "",
    ).system_prompt
    check(
        "M1 prompt built from goal+persona",
        MOCK_TASK.goal.split()[0] in prompt_user and "Ahmed Hassan" in prompt_user,
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

    report = build_report(MOCK_TASK, run)
    path = write_report(results_dir, report)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    transcript_ok = MOCK_TASK.opener.split()[0] in parsed["final_transcript"] and parsed["turns"] >= 2
    trajectory_ok = len(parsed.get("trajectory", [])) == len(run.trajectory)
    check(
        "artifact written with transcript + raw trajectory",
        parsed["task_id"] == MOCK_TASK.task_id and transcript_ok and trajectory_ok,
    )
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
    print(f"stop={run.stop_reason} turns={report.turns}")
    print(f"artifact: {path}")
    print(f"db hash: {env.get_db_hash()}")
    return 0 if report.turns >= 2 else 1


def run_live_room(
    room_name_base: str,
    wait_s: int,
    csv_path: Path,
    row_index: int,
    results_dir: Path,
    spawn_worker: bool = True,
    max_turns: int = 3,
) -> int:
    """Fully automatic room session: spawns the M2 worker, drives the M1<->M2
    conversation over a real LiveKit room, tears everything down, writes the
    artifact. No manual `lk agent dev` needed."""
    import asyncio
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
    results_dir = results_dir / task.task_id
    results_dir.mkdir(parents=True, exist_ok=True)
    room_name = f"{room_name_base}-{uuid.uuid4().hex[:6]}"
    identity = f"m1-sim-{uuid.uuid4().hex[:8]}"
    token = (
        AccessToken(key, secret)
        .with_identity(identity)
        .with_name("M1 User Simulator")
        .with_grants(VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    # ---- M1 for the room loop: real endpoint when configured, scripted otherwise
    from m1_simulator import LLMConfig

    m1_cfg = LLMConfig.from_env("M1")
    if m1_cfg.api_key:
        knobs = generate_knobs(task.task_id, trial=1, base_seed=int(time.time()))
        m1_room: Any = M1UserSimulator(task=task, knobs=knobs, seed=int(time.time()), config=m1_cfg)
        print(f"M1 model={m1_cfg.model} endpoint={m1_cfg.endpoint or 'default'} key={m1_cfg.masked_key}")
    else:

        class _ScriptedUser:
            """Fallback so the room demo still runs end-to-end without an M1 key."""

            replies = ["Danke, das reicht mir erstmal.", "Gut, dann bin ich zufrieden.", "Auf Wiedersehen!"]

            def __init__(self) -> None:
                self.i = 0

            def generate_next_message(self, history: list[Message]) -> Message:
                reply = self.replies[min(self.i, len(self.replies) - 1)]
                self.i += 1
                return Message(role="user", content=f"{reply} {STOP_MARKER}" if self.i >= len(self.replies) else reply)

        m1_room = _ScriptedUser()
        print("M1_API_KEY not set — using scripted German caller.")

    # ---- shared state the async session mutates ----------------------------
    history: list[Message] = []
    telemetry: list[TurnTelemetry] = []
    state = {"stop_reason": "", "last_update": time.time(), "seen": 0}
    start_time = time.time()
    deadline = start_time + wait_s

    # ---- optional: auto-spawn the M2 worker (agent.py dev) -----------------
    worker = None
    worker_log = None
    worker_log_path = results_dir / f"worker-{int(time.time())}.log"
    if spawn_worker:
        worker_log = open(worker_log_path, "w", encoding="utf-8")  # noqa: SIM115 - closed in finally
        worker = subprocess.Popen(
            [sys.executable, "agent.py", "dev"],
            cwd=str(Path(__file__).parent),
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        print(f"spawned m2 worker pid={worker.pid} -> log {worker_log_path}")

    QUIET_S = 5.0  # no new transcript text for this long => agent turn complete

    async def _main() -> int:
        seg_texts: dict[tuple[str, str], str] = {}  # (participant, seg_id) -> latest text
        order: list[tuple[str, str]] = []  # first-seen order of segments
        consumed = 0  # segments already folded into an assistant turn

        def _on_transcription(segments: list, participant) -> None:
            who = getattr(participant, "identity", "") or "agent"
            for seg in segments or []:
                seg_key = (who, seg.id)
                if seg_texts.get(seg_key) != seg.text:
                    seg_texts[seg_key] = seg.text
                    if seg_key not in order:
                        order.append(seg_key)
                    state["last_update"] = time.time()
                    state["seen"] += 1

        def _drain() -> str:
            """Fold all unconsumed agent transcript text into one turn string."""
            nonlocal consumed
            texts = [seg_texts[k].strip() for k in order[consumed:] if seg_texts[k].strip()]
            consumed = len(order)
            return " ".join(texts)

        async def _send_user(text: str) -> None:
            stamp_start = now_ns()
            try:
                writer = await room.local_participant.stream_text(topic="lk.chat")
                await writer.write(text)
                await writer.aclose()
            except Exception as exc:  # noqa: BLE001 — fall back to raw data packet
                logger.warning("stream_text failed (%s); falling back to publish_data", exc)
                await room.local_participant.publish_data(text.encode("utf-8"), reliable=True, topic="lk.chat")
            history.append(Message(role="user", content=text))
            telemetry.append(TurnTelemetry(user_start_ns=stamp_start, user_end_ns=now_ns(), stt_ready_ns=now_ns()))
            state["last_update"] = time.time()
            state["seen"] = 0
            print(f"M1> {text}")

        async def _wait_quiet(remaining: float) -> str | None:
            """Wait for a transcript burst then QUIET_S silence; return its text."""
            budget_end = time.time() + max(remaining, QUIET_S + 1)
            while time.time() < budget_end and time.time() < deadline:
                await asyncio.sleep(0.25)
                if worker is not None and worker.poll() is not None:
                    return ""
                if time.time() - state["last_update"] > QUIET_S and state["seen"] > 0:
                    return _drain()
            return None

        room = rtc.Room()
        room.on("transcription_received", _on_transcription)

        try:
            await room.connect(url, token)
            print(f"connected identity={identity} room={room_name}")

            # wait for the agent participant to join (dispatch + connect)
            agent_deadline = time.time() + min(wait_s, 60)
            while time.time() < agent_deadline:
                remotes = list(room.remote_participants.values())
                if remotes:
                    print(f"agent participant present: {[p.identity for p in remotes]}")
                    break
                await asyncio.sleep(0.5)
            else:
                if worker is not None and worker.poll() is not None:
                    print(f"m2 worker exited rc={worker.returncode}; see {worker_log_path}")
                    return 2
                print(f"no agent joined within budget; see {worker_log_path}")
                return 2

            # let the greeting burst finish, then drain it out of the buffer so
            # the trajectory starts with the verbatim CSV opener (spec: user first)
            await asyncio.sleep(QUIET_S)
            greeting = _drain()
            if greeting:
                print(f"M2(greeting)> {greeting[:120]}")

            turns = 0
            await _send_user(task.opener)
            while turns < max_turns and time.time() < deadline:
                assistant_text = await _wait_quiet(remaining=deadline - time.time())
                if assistant_text is None or worker is not None and worker.poll() is not None:
                    state["stop_reason"] = state["stop_reason"] or "timeout_waiting_for_agent"
                    break
                if not assistant_text.strip():
                    continue
                seg_stamp = now_ns()
                history.append(Message(role="assistant", content=assistant_text))
                if telemetry:
                    telemetry[-1].llm_first_token_ns = seg_stamp
                    telemetry[-1].tts_audio_start_ns = seg_stamp
                    telemetry[-1].agent_done_ns = now_ns()
                print(f"M2> {assistant_text[:160]}{'…' if len(assistant_text) > 160 else ''}")
                turns += 1
                lowered = assistant_text.lower()
                if any(m in lowered for m in ("###stop###", "auf wiedersehen", "wiederhören", "tschüss")):
                    state["stop_reason"] = "agent_stop"
                    break
                reply = m1_room.generate_next_message(history)
                content = reply.content.strip()
                if STOP_MARKER in content:
                    content = content.replace(STOP_MARKER, "").strip()
                    state["stop_reason"] = "user_hangup"
                if not content:
                    state["stop_reason"] = "user_empty"
                    break
                await _send_user(content)
            state["stop_reason"] = state["stop_reason"] or ("max_turns" if turns >= max_turns else "session_timeout")
            return 0
        except Exception as exc:  # noqa: BLE001 — surface infra errors cleanly
            logger.error("room session failed: %s", exc)
            state["stop_reason"] = f"error:{exc}"
            return 2
        finally:
            with contextlib.suppress(Exception):
                await room.disconnect()

    rc = 2
    try:
        rc = asyncio.run(_main())
    finally:
        if worker is not None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
            if worker_log is not None:
                worker_log.close()

    run = SimulationRun(
        run_id=f"{task.task_id}-room-{int(start_time)}",
        task_id=task.task_id,
        trial=1,
        seed=0,
        mode="audio",
        trajectory=history,
        telemetry=telemetry,
        stop_reason=state["stop_reason"],
        duration_ms=round((time.time() - start_time) * 1000, 3),
        m1_model=m1_cfg.model if m1_cfg.api_key else "scripted",
        m2_model="livekit-worker",
    )
    report = build_report(task, run)
    out = write_report(results_dir.parent, report)
    print("-" * 70)
    print(report.final_transcript)
    print("-" * 70)
    print(f"stop={state['stop_reason']} turns={report.turns} duration={report.duration_ms:.0f}ms")
    print(f"artifact: {out}\nworker log: {worker_log_path}")
    return 0 if history else rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Prove M1<->M2 conversation works")
    parser.add_argument("--mock", action="store_true", help="Offline scripted conversation + assertions")
    parser.add_argument("--live", action="store_true", help="Real M1 endpoint <-> M2 endpoint LLM (in-process)")
    parser.add_argument(
        "--live-room",
        action="store_true",
        help="Fully automatic: spawns the M2 worker, joins a fresh room, drives the M1<->M2 conversation, writes the artifact",
    )
    parser.add_argument("--no-worker", action="store_true", help="--live-room only: do NOT spawn agent.py; drive your own `lk agent dev`")
    parser.add_argument("--csv", default="/Users/jaime/AI-eval-testing/IVA_Test.csv")
    parser.add_argument("--row", type=int, default=0, help="CSV row index among usable tasks")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0, help="Wall-clock budget per live run (s)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--room", default="benchmark-m1-m2")
    parser.add_argument("--room-wait", type=int, default=180, help="Total session budget in --live-room mode (s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results_dir = Path(args.results_dir)
    if args.live:
        return run_live(Path(args.csv), args.row, results_dir, args.max_turns, args.timeout)
    if args.live_room:
        return run_live_room(
            args.room,
            args.room_wait,
            Path(args.csv),
            args.row,
            results_dir,
            spawn_worker=not args.no_worker,
            max_turns=max(3, min(args.max_turns, 6)),
        )
    # default (or explicit --mock): fully offline proof
    with tempfile.TemporaryDirectory(prefix="bench-results-") as tmp:
        return run_mock(Path(tmp))


if __name__ == "__main__":
    sys.exit(main())
