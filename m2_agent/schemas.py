"""Pydantic V2 data contracts for the M1<->M2 benchmark harness.

Flattened equivalent of tau2's ``data_model/`` package (see
TAU2_BENCH_ARCHITECTURE.md section 4.9): messages, tasks, behavioral knobs,
tick records, telemetry and simulation runs. Everything downstream imports
from here so the wire format of artifacts stays in one place.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "user", "assistant", "tool"]
RunMode = Literal["text", "audio"]
DifficultyTier = Literal["easy", "medium", "hard", "adversarial"]

STOP_MARKER = "###STOP###"


def now_ns() -> int:
    """Monotonic wall clock stamp in nanoseconds (Workflow-design.md 4.7)."""
    return time.perf_counter_ns()


class ToolCall(BaseModel):
    """A single tool invocation requested by an agent or the user simulator."""

    id: str = Field(description="Correlation id echoed back on the ToolResult.")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """One ordered entry of a conversation trajectory.

    ``role="tool"`` messages carry the environment's response to a ToolCall and
    must set ``tool_call_id`` / ``requestor`` so replay can distinguish
    assistant-side from user-side mutations.
    """

    role: Role
    content: str = ""
    timestamp_ns: int = Field(default_factory=now_ns)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    requestor: Literal["agent", "user"] | None = None
    error: bool = False

    @property
    def has_text(self) -> bool:
        return bool(self.content.strip())

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @model_validator(mode="after")
    def _reject_text_with_tool_calls(self) -> "Message":
        # Workflow-design.md 3.3: a message that tries to send text AND call a
        # tool at the same time is invalid; the orchestrator rejects it.
        if self.has_tool_calls and self.has_text:
            raise ValueError("message may carry text or tool_calls, never both")
        if self.role == "tool" and self.requestor is None:
            raise ValueError("tool messages must declare their requestor")
        return self


class Persona(BaseModel):
    """CSV persona fields plus German pronouns derived from gender only."""

    model_config = ConfigDict(frozen=True)

    caller_name: str
    gender: str = Field(description="'maennlich' | 'weiblich' — trusted over the name.")
    anrede: str = Field(description="'Sie' (formal) | 'Du' (informal).")
    pronouns_de: list[str] = Field(
        default_factory=list, description="e.g. ['er','ihm'] — derived from gender, never the name."
    )


class Task(BaseModel):
    """Canonical conversation-task schema (iva-golden-dataset-implementation-plan.md 4.2).

    Pure data-collection view: what M1 needs to play the caller. The source
    intent label is kept in ``metadata`` for traceability only.
    """

    task_id: str
    goal: str = Field(description="EN caller need — becomes the M1 goal block.")
    opener: str = Field(description="DE first utterance — sent verbatim as the first user turn.")
    persona: Persona
    metadata: dict[str, Any] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)


class BehaviorKnobs(BaseModel):
    """Seeded per-trial user-simulator knobs (Workflow-design.md 3.4)."""

    tier: DifficultyTier = "medium"
    verbosity: float = Field(ge=0.0, le=1.0, description="0 terse .. 1 chatty")
    interruption_likelihood: float = Field(ge=0.0, le=1.0)
    patience_threshold: int = Field(ge=1, description="Turns of dead-air the caller tolerates.")
    hesitation: float = Field(ge=0.0, le=1.0, description="Fillers/self-correction tendency.")


class TurnTelemetry(BaseModel):
    """Stage timestamps for one utterance exchange (ns since orchestrator start).

    Derived latencies follow Workflow-design.md 4.7:
    stt_latency = stt_ready - user_end, ttft = llm_first_token - stt_ready,
    ttfa = tts_audio_start - llm_first_token, e2e = tts_audio_start - user_end.
    Text-mode leaves stt_ready == user_end and tts_audio_start None.
    """

    user_start_ns: int
    user_end_ns: int
    stt_ready_ns: int | None = None
    llm_first_token_ns: int | None = None
    tts_audio_start_ns: int | None = None
    agent_done_ns: int | None = None

    def _delta_ms(self, start: int | None, end: int | None) -> float | None:
        if start is None or end is None:
            return None
        return round((end - start) / 1_000_000, 3)

    @property
    def stt_latency_ms(self) -> float | None:
        return self._delta_ms(self.user_end_ns, self.stt_ready_ns)

    @property
    def ttft_ms(self) -> float | None:
        return self._delta_ms(self.stt_ready_ns, self.llm_first_token_ns)

    @property
    def ttfa_ms(self) -> float | None:
        return self._delta_ms(self.llm_first_token_ns, self.tts_audio_start_ns)

    @property
    def e2e_ms(self) -> float | None:
        end = self.tts_audio_start_ns if self.tts_audio_start_ns else self.agent_done_ns
        return self._delta_ms(self.user_end_ns, end)


class Tick(BaseModel):
    """One 200 ms full-duplex step (TAU2_BENCH_ARCHITECTURE.md 5.2)."""

    tick_id: int
    t_ns: int
    user_state: str = Field(description="IDLE|SPEAKING|WAIT_FOR_AGENT|INTERRUPTED|HANGUP")
    agent_speaking: bool = False
    user_audio_emitted: bool = False
    events: list[str] = Field(default_factory=list)
    tool_calls_executed: list[ToolCall] = Field(default_factory=list)


class TurnTakingMetrics(BaseModel):
    """Deterministic turn-taking summary computed from ticks/timestamps."""

    premature_rate: float = 0.0
    missed_turn_rate: float = 0.0
    dead_air_ms: float = 0.0
    barge_in_recovery_ms: list[float] = Field(default_factory=list)


class SimulationRun(BaseModel):
    """Raw output of one (trial, task_id, seed) conversation — Layer 1 artifact."""

    run_id: str
    task_id: str
    trial: int
    seed: int
    mode: RunMode
    trajectory: list[Message] = Field(default_factory=list)
    ticks: list[Tick] = Field(default_factory=list)
    telemetry: list[TurnTelemetry] = Field(default_factory=list)
    turn_taking: TurnTakingMetrics = Field(default_factory=TurnTakingMetrics)
    stop_reason: str = ""
    duration_ms: float = 0.0
    m1_model: str = ""
    m1_temperature: float = 0.7
    m2_model: str = ""


class RunReport(BaseModel):
    """Per-run data artifact written to results/<task_id>/<run>.json.

    Core data-obtaining payload only: transcript, trajectory metadata and
    telemetry. No scoring/grading fields.
    """

    schema_version: str = "2.0"
    task_id: str
    run_id: str
    trial: int
    seed: int
    mode: RunMode
    turns: int
    final_transcript: str
    trajectory: list[Message] = Field(default_factory=list)
    ticks: list[Tick] = Field(default_factory=list)
    stop_reason: str
    duration_ms: float
    model_version: str = Field(description="M1 user-simulator model id.")
    temperature: float
    m2_model: str = ""
    telemetry_summary: dict[str, Any] = Field(default_factory=dict)
    turn_taking: TurnTakingMetrics = Field(default_factory=TurnTakingMetrics)


def transcript_text(trajectory: list[Message]) -> str:
    """Human-readable transcript used as final_transcript artifact field."""
    lines: list[str] = []
    for msg in trajectory:
        if msg.role in {"user", "assistant"} and msg.content:
            lines.append(f"{msg.role}: {msg.content}")
    return "\n".join(lines)
