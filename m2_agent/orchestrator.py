"""Conversation orchestrators (tau2 layer: orchestrator/).

HalfDuplexOrchestrator — text-mode turn loop (Workflow-design.md 3.3):
    user opener (verbatim) -> agent turn -> tool calls executed on the
    Environment BEFORE anything else sees them -> user turn -> repeat until
    agent STOP / user hang-up / max_turns.

FullDuplexTickOrchestrator — audio-mode skeleton driven by fixed 200 ms ticks
    (TAU2_BENCH_ARCHITECTURE.md 5.2) with the caller turn-taking state machine
    IDLE -> SPEAKING -> WAIT_FOR_AGENT -> INTERRUPTED, per-tick tool-call
    flushing and Tick records feeding turn-taking metrics.

Rules are enforced HERE, never inside M2/M1: a message carrying text and tool
calls simultaneously is rejected; every loop terminates; no CLI/disk I/O.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from typing import Callable, Protocol

from environment import Environment
from schemas import (
    BehaviorKnobs,
    Message,
    RunMode,
    SimulationRun,
    Tick,
    TurnTakingMetrics,
    TurnTelemetry,
    Task,
    ToolCall,
    now_ns,
)

logger = logging.getLogger("orchestrator")

AGENT_STOP_MARKER = "###STOP###"
USER_STOP_MARKER = "###STOP###"
GOODBYE_PATTERNS = ("auf wiedersehen", "wiederhören", "tschüss", "goodbye")
MAX_TOOL_ROUNDS = 4  # bound consecutive tool-only exchanges so the loop always terminates


class AgentSide(Protocol):
    """Adapter contract every agent under test must satisfy (Workflow-design.md 3.5)."""

    model_id: str

    def generate_next_message(self, history: list[Message]) -> Message: ...


class UserSide(Protocol):
    knobs: BehaviorKnobs

    def open_message(self) -> Message: ...

    def generate_next_message(self, history: list[Message]) -> Message: ...


def _is_goodbye(text: str) -> bool:
    lowered = text.lower()
    return AGENT_STOP_MARKER in lowered or any(p in lowered for p in GOODBYE_PATTERNS)


def _execute_tool_calls(
    env: Environment,
    calls: list[ToolCall],
    requestor: str,
    history: list[Message],
) -> None:
    """Run tool calls against the environment and append ToolMessages.

    Called by the orchestrator immediately after the actor's message and before
    any further turn is generated — 'tools before next turn' rule.
    """
    for call in calls:
        result, error = env.make_tool_call(call, requestor=requestor)  # type: ignore[arg-type]
        history.append(
            Message(
                role="tool",
                content=str(result),
                tool_call_id=call.id,
                requestor=requestor,  # type: ignore[arg-type]
                error=error,
            )
        )
        logger.debug("tool %s(%s) by %s -> %s", call.name, call.arguments, requestor, result)


class HalfDuplexOrchestrator:
    """Deterministic, terminating text-mode conversation runner."""

    def __init__(
        self,
        env: Environment,
        agent: AgentSide,
        user: UserSide,
        max_turns: int = 10,
        timeout_s: float = 120.0,
        mode: RunMode = "text",
    ) -> None:
        self.env = env
        self.agent = agent
        self.user = user
        self.max_turns = max_turns
        self.timeout_s = timeout_s
        self.mode = mode

    def run(self, task: Task, trial: int = 1, seed: int = 0, run_id: str | None = None) -> SimulationRun:
        start_ns = now_ns()
        run_id = run_id or f"{task.task_id}-t{trial}-s{seed}-{int(time.time())}"
        history: list[Message] = []
        telemetry: list[TurnTelemetry] = []
        stop_reason = ""
        protocol_violation = False

        # --- user speaks first: the CSV opener, verbatim -------------------
        opener = self.user.open_message()
        history.append(opener)
        telemetry.append(TurnTelemetry(user_start_ns=opener.timestamp_ns, user_end_ns=now_ns(), stt_ready_ns=now_ns()))

        turns = 0
        while turns < self.max_turns and not stop_reason:
            if (now_ns() - start_ns) / 1e9 > self.timeout_s:
                stop_reason = "timeout"
                break

            # --- agent turn -------------------------------------------------
            agent_msg = self._safe_agent_call(history)
            if agent_msg.has_text and agent_msg.has_tool_calls:
                # enforced here, not by the agent (TAU2 arch rule section 14.5)
                logger.warning("protocol violation: text+tool_calls rejected")
                history.append(Message(role="assistant", content="", timestamp_ns=now_ns()))
                stop_reason = "protocol_violation"
                protocol_violation = True
                break
            first_token_ns = now_ns()
            if agent_msg.has_tool_calls:
                history.append(agent_msg)  # tau2 keeps tool-call turns in the trajectory
                _execute_tool_calls(self.env, agent_msg.tool_calls, "agent", history)
                # give the model the tool results so it can produce its reply
                for _ in range(MAX_TOOL_ROUNDS - 1):
                    follow_up = self._safe_agent_call(history)
                    if follow_up.has_tool_calls:
                        history.append(follow_up)
                        _execute_tool_calls(self.env, follow_up.tool_calls, "agent", history)
                        continue
                    agent_msg = follow_up
                    break
            if agent_msg.has_text:
                history.append(agent_msg)
                done_ns = now_ns()
                telemetry.append(_agent_telemetry(telemetry, first_token_ns, done_ns))
                if _is_goodbye(agent_msg.content):
                    stop_reason = "agent_stop"
                turns += 1
                if stop_reason:
                    break
            elif not agent_msg.has_tool_calls:
                stop_reason = "agent_empty"
                break

            # --- user turn --------------------------------------------------
            try:
                user_msg = self.user.generate_next_message(history)
            except Exception as exc:  # noqa: BLE001 — simulator failures end the run, visibly
                logger.error("user simulator failed: %s", exc)
                history.append(Message(role="user", content=f"(user error: {exc})"))
                stop_reason = "user_error"
                break
            if user_msg.has_tool_calls:
                history.append(user_msg)
                _execute_tool_calls(self.env, user_msg.tool_calls, "user", history)
                user_msg = self.user.generate_next_message(history)
            history.append(user_msg)
            turns += 1
            if USER_STOP_MARKER in user_msg.content or user_msg.content.strip().lower().startswith("auflegen"):
                stop_reason = "user_hangup"

        if not stop_reason:
            stop_reason = "max_turns"

        duration_ms = (now_ns() - start_ns) / 1_000_000
        return SimulationRun(
            run_id=run_id,
            task_id=task.task_id,
            trial=trial,
            seed=seed,
            mode=self.mode,
            trajectory=history,
            telemetry=telemetry,
            stop_reason="protocol_violation" if protocol_violation else stop_reason,
            duration_ms=round(duration_ms, 3),
            m1_model=getattr(self.user, "config", None).model if hasattr(self.user, "config") else "",
            m1_temperature=getattr(self.user, "config", None).temperature if hasattr(self.user, "config") else 0.0,
            m2_model=getattr(self.agent, "model_id", ""),
        )

    def _safe_agent_call(self, history: list[Message]) -> Message:
        try:
            return self.agent.generate_next_message(history)
        except Exception as exc:  # noqa: BLE001 — infra errors must not hang the loop
            logger.error("agent adapter failed: %s", exc)
            return Message(role="assistant", content=f"(agent error: {exc})")


def _agent_telemetry(previous: list[TurnTelemetry], first_token_ns: int, done_ns: int) -> TurnTelemetry:
    """Pair agent response timings with the user turn that triggered them."""
    anchor = previous[-1] if previous else TurnTelemetry(user_start_ns=first_token_ns, user_end_ns=first_token_ns)
    return TurnTelemetry(
        user_start_ns=anchor.user_start_ns,
        user_end_ns=anchor.user_end_ns,
        stt_ready_ns=anchor.stt_ready_ns or anchor.user_end_ns,
        llm_first_token_ns=first_token_ns,
        tts_audio_start_ns=None,
        agent_done_ns=done_ns,
    )


class FullDuplexTickOrchestrator:
    """Audio-mode harness: both sides act every TICK_MS step.

    Speech plumbing is pluggable so Phase 4 can wire real TTS/room capture;
    defaults simulate audio deterministically (text chunked into ~one-tick
    pieces), which keeps the smoke test offline.
    """

    TICK_MS = 200
    MISSED_TURN_THRESHOLD_MS = 2000  # Workflow-design.md 4.7

    def __init__(
        self,
        env: Environment,
        agent: AgentSide,
        user: UserSide,
        seed: int = 0,
        max_ticks: int = 900,  # ~3 minutes
        tts_fn: Callable[[str], list[str]] | None = None,
        stt_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.user = user
        self.seed = seed
        self.max_ticks = max_ticks
        self.tts_fn = tts_fn or self._default_tts
        self.stt_fn = stt_fn or (lambda text: text)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _default_tts(text: str) -> list[str]:
        words = text.split()
        return [" ".join(words[i : i + 4]) for i in range(0, len(words), 4)] or [text]

    # ------------------------------------------------------------------ main loop
    def run(self, task: Task, trial: int = 1, seed: int = 0, run_id: str | None = None) -> SimulationRun:
        start_ns = now_ns()
        run_id = run_id or f"{task.task_id}-t{trial}-s{seed}-audio-{int(time.time())}"
        history: list[Message] = []
        ticks: list[Tick] = []
        rng = random.Random(seed)

        user_state = "SPEAKING"  # caller opens the call with the verbatim opener
        pending_user_chunks: deque[str] = deque(self.tts_fn(task.opener))
        user_text_buffer: list[str] = []
        agent_chunks: deque[str] = deque()
        agent_speaking = False
        agent_last_text = ""
        silent_wait_ticks = 0
        dead_air_ticks = 0
        premature_events = 0
        agent_exchanges = 0
        missed_turns = 0
        barge_in_recovery: list[float] = []
        barge_onset_tick: int | None = None
        stop_reason = ""

        tick_id = 0
        while tick_id < self.max_ticks and not stop_reason:
            events: list[str] = []
            flushed: list[ToolCall] = []

            # ---- user side acts -------------------------------------------
            if user_state == "INTERRUPTED" and barge_onset_tick is not None and not agent_speaking:
                recovery_ms = (tick_id - barge_onset_tick) * self.TICK_MS
                barge_in_recovery.append(float(recovery_ms))
                events.append(f"barge_in_recovery:{recovery_ms}ms")
                barge_onset_tick = None
                user_state = "SPEAKING"

            if user_state == "SPEAKING" and pending_user_chunks:
                chunk = pending_user_chunks.popleft()
                user_text_buffer.append(chunk)
                events.append("user_audio_chunk")

            if not pending_user_chunks and user_text_buffer and user_state == "SPEAKING":
                # VAD endpoint -> hand the transcript to the agent
                user_state = "WAIT_FOR_AGENT"
                events.append("vad_endpoint")
                spoken = self.stt_fn(" ".join(user_text_buffer))
                user_text_buffer = []
                history.append(Message(role="user", content=spoken))
                try:
                    agent_reply = self.agent.generate_next_message(history)
                except Exception as exc:  # noqa: BLE001
                    logger.error("agent adapter failed (tick): %s", exc)
                    agent_reply = Message(role="assistant", content=f"(agent error: {exc})")
                agent_exchanges += 1
                silent_wait_ticks = 0
                if agent_reply.has_tool_calls:
                    _execute_tool_calls(self.env, agent_reply.tool_calls, "agent", history)
                    flushed.extend(agent_reply.tool_calls)
                    events.append("tool_flush")
                if agent_reply.has_text:
                    history.append(agent_reply)
                    agent_last_text = agent_reply.content
                    agent_chunks.extend(self.tts_fn(agent_reply.content))
                    agent_speaking = True
                    events.append("tts_audio_start")
                    if AGENT_STOP_MARKER in agent_reply.content or _is_goodbye(agent_reply.content):
                        stop_reason = "agent_stop"
                else:
                    stop_reason = stop_reason or "agent_empty"

            elif (
                user_state == "WAIT_FOR_AGENT"
                and agent_speaking
                and rng.random() < self.user.knobs.interruption_likelihood * 0.25
                and agent_last_text
                and AGENT_STOP_MARKER not in agent_last_text
            ):
                # caller barges in while the agent is talking
                user_state = "INTERRUPTED"
                barge_onset_tick = tick_id
                agent_chunks.clear()
                agent_speaking = False
                events.append("barge_in")
                pending_user_chunks.extend(
                    self.tts_fn("Entschuldigung, das war nicht meine Frage.")
                    if self.user.knobs.tier == "adversarial"
                    else self.tts_fn("Kurz etwas anderes:")
                )

            # ---- agent side speaks (one chunk per tick) --------------------
            if agent_speaking and agent_chunks:
                agent_chunks.popleft()
                events.append("agent_audio_chunk")
                if user_state == "SPEAKING":
                    premature_events += 1
                    events.append("premature_agent_speech")
                if not agent_chunks:
                    agent_speaking = False
                    events.append("agent_done")
            elif user_state == "WAIT_FOR_AGENT" and not agent_speaking and not stop_reason:
                dead_air_ticks += 1
                silent_wait_ticks += 1
                if silent_wait_ticks * self.TICK_MS > self.MISSED_TURN_THRESHOLD_MS:
                    missed_turns += 1
                    events.append("missed_turn")
                    silent_wait_ticks = 0
                # patience: caller hangs up after too much dead air
                if dead_air_ticks > self.user.knobs.patience_threshold * 10:
                    user_state = "HANGUP"
                    stop_reason = "user_hangup_dead_air"

            ticks.append(
                Tick(
                    tick_id=tick_id,
                    t_ns=now_ns(),
                    user_state=user_state,
                    agent_speaking=agent_speaking,
                    user_audio_emitted="user_audio_chunk" in events,
                    events=events,
                    tool_calls_executed=flushed,
                )
            )
            if user_state == "HANGUP" and not stop_reason:
                stop_reason = "user_hangup"
            tick_id += 1

        if not stop_reason:
            stop_reason = "max_ticks"

        total = len(ticks) or 1
        metrics = TurnTakingMetrics(
            premature_rate=round(premature_events / max(agent_exchanges, 1), 4),
            missed_turn_rate=round(missed_turns / max(agent_exchanges, 1), 4),
            dead_air_ms=float(dead_air_ticks * self.TICK_MS),
            barge_in_recovery_ms=barge_in_recovery,
        )
        return SimulationRun(
            run_id=run_id,
            task_id=task.task_id,
            trial=trial,
            seed=seed,
            mode="audio",
            trajectory=history,
            ticks=ticks,
            turn_taking=metrics,
            stop_reason=stop_reason,
            duration_ms=round((now_ns() - start_ns) / 1_000_000, 3),
            m1_model=getattr(self.user, "config", None).model if hasattr(self.user, "config") else "",
            m1_temperature=getattr(self.user, "config", None).temperature if hasattr(self.user, "config") else 0.0,
            m2_model=getattr(self.agent, "model_id", ""),
        )


def compute_turn_taking(run: SimulationRun) -> TurnTakingMetrics:
    """Public metric hook used by report.py for artifact summaries."""
    return run.turn_taking
