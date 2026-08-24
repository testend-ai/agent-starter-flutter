# M1 User Simulator + M1↔M2 Conversation Infrastructure — Implementation & Testing Guide

> Branch `feat/m1-user-simulator-orchestrator` · All Python modules under `m2_agent/` · No Dart changes
>
> Scope: **data collection only** — M1 (simulated caller), M2 adapter, text + live-room conversation
> infrastructure, transcript/telemetry capture. No scoring or grading.

## 1. What Is Implemented

| Module | File | Responsibility |
|---|---|---|
| **Schemas** | `m2_agent/schemas.py` | Pydantic V2 contracts: `Task` (goal/opener/persona from CSV), `Persona`, `Message`/`ToolCall`, `BehaviorKnobs`, `TurnTelemetry`, `Tick`, `SimulationRun`, `RunReport`. Single source of truth for artifacts. |
| **Environment** | `m2_agent/environment.py` | Stateful mock backend for conversations: `DB.get_hash()` SHA-256 over canonical JSON, `@is_tool` + `ToolKitBase` metaclass, telephony toolkits — agent side (`detect_intent`, `transfer_to_human`, `lookup_policy`) and user side (`confirm_resolution`, `hang_up`). |
| **M1 Simulator** | `m2_agent/m1_simulator.py` | Persona prompt built from CSV `description` (goal) + `scenario` (verbatim opener) + `caller_name`/`gender`/`anrede`; German register `Sie`/`Du`, pronouns from `gender` only; behavioral knobs (`verbosity`, `interruption_likelihood`, `patience_threshold`, `hesitation`) deterministic per `(task_id, trial, seed)` across tiers easy→adversarial; LLM calls via litellm with injectable `llm_fn` for offline tests. |
| **Orchestrator** | `m2_agent/orchestrator.py` | `HalfDuplexOrchestrator`: text turn loop `M1 opener → M2 → tool execution BEFORE next turn → M1 → repeat`; rejects text+tool-simultaneous messages; terminates on STOP/goodbye, hangup, `max_turns`, timeout. `FullDuplexTickOrchestrator`: 200 ms ticks, caller state machine `IDLE→SPEAKING→WAIT_FOR_AGENT→INTERRUPTED`, per-tick tool flush, `Tick` records, turn-taking metrics (premature-rate, missed-turn-rate, dead-air, barge-in recovery). No CLI/disk I/O. |
| **Artifacts** | `m2_agent/report.py` | Writes the core dataset per run: `results/<task_id>/<mode>-trial<n>-seed<s>.json` containing full trajectory, transcript, telemetry latencies and run metadata; checkpoint cell scanner keyed `(trial,task_id,seed)`. |
| **Batch Runner** | `m2_agent/batch_runner.py` | CSV→`Task` loader (BOM-safe, `;` delimiter, drops `Ausgeschlossen=Ja`), component wiring, batch cells with resume, `M2TextAgent` adapter (any OpenAI-compatible endpoint via `M2_*` env). |
| **Tests** | `m2_agent/test_conversation.py` | `--mock` offline proof · `--live` in-process real endpoints · `--live-room` fully automatic room session (spawns worker, drives conversation, writes artifacts) |
|  | `m2_agent/test_m1.py` | M1 endpoint verification (masked keys) |

**Env contract** (`m2_agent/.env.example`):

```
LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET   # room sessions

M2_MODEL_ENDPOINT / M2_MODEL_API_KEY / M2_MODEL_NAME / M2_TEMPERATURE
M2_STT_* / M2_TTS_* / M2_VOICE / M2_WORKFLOW

M1_MODEL_ENDPOINT=https://openrouter.ai/api/v1
M1_API_KEY=sk-or-...
M1_MODEL_NAME=google/gemini-2.5-flash   # Easy/Med tier; Hard/Adv → claude-sonnet-5
M1_TEMPERATURE=0.7
M1_DIFFICULTY=medium                    # easy | medium | hard | adversarial
```

## 2. File Layout

```
m2_agent/
  schemas.py              # contracts
  m1_simulator.py         # persona builder + litellm caller
  environment.py          # DB hash + ToolKitBase toolkits
  orchestrator.py         # half-duplex + 200ms tick orchestrators
  report.py               # artifact writer + checkpoint scan
  batch_runner.py         # CSV loader + build wiring + batch CLI
  test_conversation.py    # --mock/--live/--live-room
  test_m1.py              # M1 endpoint check
  agent.py                # M2 LiveKit worker (unchanged)
results/<task_id>/        # per-run data artifacts + worker logs
docs/M1_IMPLEMENTATION_AND_TESTING.md
```

Flutter app unchanged — remains visualizer/debugger.

## 3. Setup

```bash
source .venv/bin/activate
pip install -r m2_agent/requirements.txt
cp m2_agent/.env.example m2_agent/.env   # then fill LIVEKIT_*, M1_*, M2_* values
# golden CSV default path: /Users/jaime/AI-eval-testing/IVA_Test.csv (override with --csv)
```

## 4. How to Run

### 4.1 Offline proof — no network, no LiveKit

```bash
python m2_agent/test_conversation.py --mock    # 14 checks, all offline
python m2_agent/test_m1.py --mock
python m2_agent/test_model.py --mock           # existing M2 check
```

Asserts: pronouns from `gender` only, knob determinism, prompt built from goal+persona,
half-duplex ≥2 turns/side, tool-before-next-turn ordering, no mixed text+tool messages,
environment mutations recorded, artifact contains transcript + raw trajectory, tick smoke
(≥3 Ticks, VAD/tts events, state machine exercised).

### 4.2 M1 endpoint only

```bash
python m2_agent/test_m1.py   # uses M1_* from .env, key masked as sk-o...b0f4
```

### 4.3 Live text-mode M1↔M2 (in-process, no room)

```bash
set -a; source m2_agent/.env; set +a
export M1_MODEL_ENDPOINT="$M2_MODEL_ENDPOINT" M1_API_KEY="$M2_MODEL_API_KEY" M1_MODEL_NAME="$M2_MODEL_NAME"
python m2_agent/test_conversation.py --live --row 0 --max-turns 6 --timeout 300
# -> results/<task_id>/text-trial1-seed*.json
```

### 4.4 LiveKit room session — FULLY AUTOMATIC

One command does everything: spawns the M2 worker (`agent.py dev` subprocess), creates a
fresh uniquely-named room, waits for the agent participant, captures its greeting, sends
the opener over the same channel as Flutter (`stream_text`, topic `lk.chat`), drives up to
N turns of real M1↔M2 conversation from live transcriptions, then disconnects, terminates
the worker and writes artifacts:

```bash
python m2_agent/test_conversation.py --live-room            # defaults: --room-wait 180 --max-turns 3
python m2_agent/test_conversation.py --live-room --row 42 --room-wait 240 --max-turns 5

# drive your own already-running `lk agent dev` instead of auto-spawning:
python m2_agent/test_conversation.py --live-room --no-worker --room-wait 120
```

Console output is the live transcript (`M2(greeting)>`, `M1>`, `M2>`). Artifacts:

- `results/<task_id>/audio-trial1-seed0.json` — trajectory + transcript + telemetry
- `results/<task_id>/worker-<ts>.log` — full M2 worker log

Without `M1_API_KEY` it falls back to a scripted German caller so the demo still runs.

### 4.5 Batch collection (CSV → many cells)

```bash
python m2_agent/batch_runner.py --limit 5 --trials 3 --modes text
python m2_agent/batch_runner.py --limit 20 --trials 3 --modes text audio --tier hard
```

Checkpointing: re-running with the same `results/` skips completed `(trial,task_id,seed)`
cells (`0 runs` on second invocation proves resume). `--no-resume` re-runs everything.

## 5. How to Test / Verify

```bash
source .venv/bin/activate
python -m py_compile m2_agent/*.py
python m2_agent/test_conversation.py --mock     # ALL MOCK CHECKS PASSED
python m2_agent/test_m1.py --mock

# one live room session with artifacts
python m2_agent/test_conversation.py --live-room --room-wait 240

# Flutter gates (unchanged, must stay green)
export PATH="$HOME/flutter/bin:$PATH"
flutter pub get && dart format --set-exit-if-changed -l 120 . && flutter analyze --no-fatal-infos && flutter test
```

## 6. Artifact Schema (core data only)

`results/<task_id>/<mode>-trial<n>-seed<s>.json`:

```json
{
  "schema_version": "2.0",
  "task_id": "RequestProofOfFunds#0001",
  "run_id": "...",
  "trial": 1,
  "seed": 7,
  "mode": "text",
  "turns": 3,
  "final_transcript": "user: ...\nagent: ...\ntool(agent): ...",
  "trajectory": [ {"role": "user", "content": "...", ...}, ... ],
  "ticks": [],
  "stop_reason": "agent_stop",
  "duration_ms": 4210.5,
  "model_version": "google/gemini-2.5-flash",
  "temperature": 0.7,
  "m2_model": "...",
  "telemetry_summary": { "exchanges": 3, "e2e_ms": [...], "ttft_ms": [...] },
  "turn_taking": { "premature_rate": 0.0, "missed_turn_rate": 0.0, "dead_air_ms": 0.0 }
}
```

Read a trajectory back: `report.load_trajectory(Path("results/.../text-trial1-seed7.json"))`.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `tenacity import failed` | missing `tenacity` for litellm retries | `pip install tenacity` (in requirements.txt) |
| `empty LLM completion` | transient 502 on free OpenRouter tier | auto-retries 3×; raise `--timeout` for live runs |
| `no agent joined within budget` | wrong LIVEKIT creds or worker crashed | inspect `results/<task_id>/worker-*.log` |
| resume reports `0 runs` | all cells completed already | `rm -rf results` or pass `--no-resume` |
