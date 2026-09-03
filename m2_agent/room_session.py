"""Automated LiveKit room session: M1 caller <-> hosted M2 worker.

One call to :func:`run_room_session` does everything:
  1. spawns the M2 worker (``agent.py dev`` subprocess) unless one is running
  2. creates a fresh uniquely-named LiveKit room
  3. waits for the agent participant to join and captures its greeting
  4. sends the task opener over the same channel as the Flutter client
     (``stream_text``, topic ``lk.chat``)
  5. drives up to ``max_turns`` of real M1<->M2 conversation from live
     transcription events (segment-deduplicated, quiet-period turn detection)
  6. tears down (disconnect + terminate worker) and writes a data artifact
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from m1_simulator import STOP_MARKER, LLMConfig, M1UserSimulator, generate_knobs  # noqa: E402
from report import build_report, write_report  # noqa: E402
from schemas import Message, SimulationRun, Task, TurnTelemetry, now_ns  # noqa: E402

logger = logging.getLogger("room-session")

QUIET_S = 5.0  # no new transcript text for this long => agent turn complete


def resolve_m1_config() -> tuple[LLMConfig | None, str]:
    """Dataset-driven M1 config: dedicated M1_* creds, else reuse M2's.

    Returns (config_or_None, notice). None means no usable endpoint at all.
    """
    cfg = LLMConfig.from_env("M1")
    if cfg.api_key:
        return cfg, ""
    m2 = LLMConfig.from_env("M2")
    if m2.api_key:
        return (
            LLMConfig(
                endpoint=m2.endpoint,
                api_key=m2.api_key,
                model=m2.model,
                temperature=float(os.environ.get("M1_TEMPERATURE", "0.7") or 0.7),
            ),
            f"M1_* not set — reusing M2 endpoint/model '{m2.model}' for the caller",
        )
    return None, "neither M1_API_KEY nor M2_MODEL_API_KEY configured"


def _spawn_worker(log_path: Path) -> tuple[subprocess.Popen, Any]:
    log_handle = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - closed by caller
    proc = subprocess.Popen(
        [sys.executable, "agent.py", "dev"],
        cwd=str(Path(__file__).parent),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    return proc, log_handle


def _teardown_worker(worker: subprocess.Popen | None, log_handle: Any) -> None:
    if worker is not None:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
    if log_handle is not None:
        with contextlib.suppress(Exception):
            log_handle.close()


def run_room_session(
    task: Task,
    results_dir: Path,
    *,
    session_budget_s: int = 180,
    max_turns: int = 3,
    spawn_worker: bool = True,
    room_name_base: str = "benchmark-m1-m2",
    trial: int = 1,
    m1_cfg: LLMConfig | None = None,
) -> Path | None:
    """Run one fully automatic M1<->M2 room conversation; returns artifact path."""
    url = os.environ.get("LIVEKIT_URL", "").strip()
    key = os.environ.get("LIVEKIT_API_KEY", "").strip()
    secret = os.environ.get("LIVEKIT_API_SECRET", "").strip()
    if not (url and key and secret):
        logger.error("LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET missing in m2_agent/.env")
        return None
    try:
        import livekit.rtc as rtc
        from livekit.api import AccessToken, VideoGrants
    except ImportError:
        logger.error("livekit rtc/api not installed; pip install livekit-agents in .venv")
        return None

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

    m1_cfg = m1_cfg or resolve_m1_config()[0]
    if m1_cfg is None:
        logger.error("no M1 endpoint available (set M1_API_KEY or M2_MODEL_API_KEY)")
        return None
    knobs = generate_knobs(task.task_id, trial=trial, base_seed=int(time.time()))
    m1_room: Any = M1UserSimulator(task=task, knobs=knobs, seed=int(time.time()), config=m1_cfg)
    print(f"M1 model={m1_cfg.model} endpoint={m1_cfg.endpoint or 'default'} key={m1_cfg.masked_key}")

    history: list[Message] = []
    telemetry: list[TurnTelemetry] = []
    state = {"stop_reason": "", "last_update": time.time(), "seen": 0}
    start_time = time.time()
    deadline = start_time + session_budget_s

    worker = None
    worker_log = None
    worker_log_path = results_dir / f"worker-{int(start_time)}.log"
    if spawn_worker:
        worker, worker_log = _spawn_worker(worker_log_path)
        print(f"spawned m2 worker pid={worker.pid} -> log {worker_log_path}")

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
            agent_deadline = time.time() + min(session_budget_s, 60)
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
            # the trajectory starts with the verbatim CSV opener (caller first)
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
        _teardown_worker(worker, worker_log)

    run = SimulationRun(
        run_id=f"{task.task_id}-room-{int(start_time)}",
        task_id=task.task_id,
        trial=trial,
        seed=0,
        mode="audio",
        trajectory=history,
        telemetry=telemetry,
        stop_reason=state["stop_reason"],
        duration_ms=round((time.time() - start_time) * 1000, 3),
        m1_model=m1_cfg.model,
        m2_model="livekit-worker",
    )
    report = build_report(task, run)
    out = write_report(results_dir.parent, report)
    print("-" * 70)
    print(report.final_transcript)
    print("-" * 70)
    print(f"stop={state['stop_reason']} turns={report.turns} duration={report.duration_ms:.0f}ms")
    print(f"artifact: {out}")
    if worker_log_path.exists():
        print(f"worker log: {worker_log_path}")
    return out if history else None
