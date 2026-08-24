# M1 User Simulator + M1↔M2 Conversation Harness — Implementation & Testing Guide

> Branch `feat/m1-user-simulator-orchestrator` · All Python modules under `m2_agent/` · No Dart changes

## 1. What Was Implemented

M1 plays the **human caller**; the orchestrator lets M1 and M2 talk in **text mode** (turn loop) and **audio mode** (200 ms tick loop), wired to `IVA_Test.csv` and the benchmark grading contract.

| Module | File | Responsibility |
|---|---|---|
| **Schemas** | `m2_agent/schemas.py` | Pydantic V2 contracts: `Task` (canonical `iva-plan §4.2`), `Persona`, `Message`/`ToolCall`, `BehaviorKnobs`, `TurnTelemetry` (Workflow-design §4.7), `Tick`, `SimulationRun`, `RunReport` (`results/<task_id>/<run>.json` per iva-plan §5.3). Single source of truth for artifacts. |
| **Environment** | `m2_agent/environment.py` | Stateful mock backend. `DB.get_hash()` = SHA-256 over canonical JSON (Workflow-design §4.2/Libraries doc). `ToolKitBase` metaclass + `@is_tool` registration (tau2 pattern). `TelephonyDB` for intent-only tasks + `AssistantToolkit` (`detect_intent`, `transfer_to_human`, `lookup_policy`) and `UserToolkit` (`confirm_resolution`, `hang_up`) — dual-control. Extensible to per-intent `tool_spec` without touching tasks. |
| **M1 Simulator** | `m2_agent/m1_simulator.py` | Persona prompt from CSV `description` (goal) + `scenario` (verbatim opener) + `caller_name`/`gender`/`anrede` only — `expected_output`/`intent_name` hard-blocked via `assert_no_leak()` (Workflow-design §3.4, How-we-benchmark §Four-Actor). Seeded knobs `verbosity`/`interruption_likelihood`/`patience_threshold`/`hesitation` deterministic from `(task_id,trial,seed)` per difficulty tier (`easy`/`medium`/`hard`/`adversarial`). German register `Sie`/`Du` from `anrede`, pronouns from `gender` only. Plug-and-play via `litellm` (`M1_MODEL_ENDPOINT`/`M1_API_KEY`/`M1_MODEL_NAME`/`M1_TEMPERATURE` in `m2_agent/.env`). |
| **Orchestrator** | `m2_agent/orchestrator.py` | `HalfDuplexOrchestrator`: text turn loop `M1 opener → M2 → tool execution BEFORE next turn → M1 → repeat`; rejects text+tool simultaneous (TAU2 §14 rule); terminations: `STOP`/`goodbye`, hangup, `max_turns`, timeout. `FullDuplexTickOrchestrator`: 200 ms ticks, caller state machine `IDLE→SPEAKING→WAIT_FOR_AGENT→INTERRUPTED`, per-tick tool flush, `Tick` records, turn-taking metrics (`premature-rate`, `missed-turn-rate`, `dead-air`, `barge-in recovery <300ms`). No CLI/disk I/O. |
| **Report** | `m2_agent/report.py` | L2 grading: `normalize_label()` (CamelCase→`SCREAMING_SNAKE`) + `token_f1()` per Libraries doc §7; `extract_predicted_label()` (tool args first, then final message regex) → `real_output`; 4-way outcome + `RunReport` artifact `results/<task_id>/<mode>-trial<n>-seed<s>.json`; checkpoint scanner keyed `(trial,task_id,seed)`; Wilson CI / `aggregate()`. |
| **Batch Runner** | `m2_agent/batch_runner.py` | CSV→`Task` loader (BOM-safe, `;` delimiter, drops `Ausgeschlossen=Ja`, 235/151 usable/intents verified), build wiring (Layer 2), batch cells with resume (Layer 3), `M2TextAgent` adapter (OpenAI-compatible tools, text+tool rendered portably). |
| **Tests** | `m2_agent/test_conversation.py` | `--mock` offline proof + `--live` in-process real endpoints + `--live-room` LiveKit probe |
|  | `m2_agent/test_m1.py` | M1 endpoint verification (masked keys, `test_model.py` pattern) |

**Env contract** (`m2_agent/.env.example` section 6, mirrored in root `.env.example` comment):

```
M1_MODEL_ENDPOINT=https://openrouter.ai/api/v1
M1_API_KEY=sk-or-...
M1_MODEL_NAME=google/gemini-2.5-flash  # Easy/Med; Hard/Adv → anthropic/claude-sonnet-5
M1_TEMPERATURE=0.7
M1_DIFFICULTY=medium
```

---

## 2. File Layout

```
m2_agent/
  schemas.py              # contracts
  m1_simulator.py         # persona builder + litellm caller + anti-cheat
  environment.py          # DB hash + ToolKitBase
  orchestrator.py         # half-duplex + 200ms tick orchestrators
  report.py               # grading + artifact writer + checkpoint
  batch_runner.py         # CSV loader + build + batch CLI
  test_conversation.py    # --mock/--live/--live-room
  test_m1.py              # M1 endpoint check
  requirements.txt        # now includes litellm>=1.50, pydantic>=2.7, tenacity>=8.2
  .env.example            # M1_* block documented
results/<task_id>/<run>.json  # artifacts (gitignored in sample)
```

Flutter app unchanged — remains visualizer/debugger.

## 3. Setup

```bash
# from repo root
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -r m2_agent/requirements.txt
cp m2_agent/.env.example m2_agent/.env   # then edit LIVEKIT_* and M1_*/M2_* keys
# CSV is at AI-eval-testing/IVA_Test.csv (1000 rows) or provide your own path via --csv
```

Required env in `m2_agent/.env`:

```env
LIVEKIT_URL=wss://xxx.livekit.cloud
LIVEKIT_API_KEY=xxx
LIVEKIT_API_SECRET=xxx

M2_MODEL_ENDPOINT=https://openrouter.ai/api/v1
M2_MODEL_API_KEY=sk-or-...
M2_MODEL_NAME=nvidia/nemotron-3-ultra-550b-a55b:free
M2_TEMPERATURE=0.7
M2_STT_PROVIDER=deepgram
M2_TTS_PROVIDER=cartesia

M1_MODEL_ENDPOINT=https://openrouter.ai/api/v1
M1_API_KEY=sk-or-...           # can reuse M2 key for smoke tests
M1_MODEL_NAME=google/gemini-2.5-flash
M1_TEMPERATURE=0.7
M1_DIFFICULTY=medium
```

## 4. How to Run

### 4.1 Offline proof — no network, no LiveKit

```bash
source .venv/bin/activate
python m2_agent/test_conversation.py --mock
python m2_agent/test_m1.py --mock
python m2_agent/test_model.py --mock   # existing M2 check
```

`--mock` asserts: pronouns from `gender` only, knob determinism, anti-cheat (gold leaked → ValueError), token-F1, half-duplex ≥2 turns per side + tool-before-next-turn ordering, db mutation, tick smoke (≥3 Ticks, VAD/tts events, state machine exercised). Writes artifacts to a temp dir.

### 4.2 M1 endpoint only

```bash
python m2_agent/test_m1.py               # uses M1_* from .env, masks key as sk-...b0f4
python m2_agent/test_m1.py --endpoint https://api.openai.com/v1 --api-key sk-... --model gpt-4o-mini
```

### 4.3 Live text-mode M1↔M2 (in-process, no room)

```bash
# single task, 6 turns, 5 min wall-clock budget
python m2_agent/test_conversation.py --live --row 0 --max-turns 6 --timeout 300

# choose CSV row among the 235 usable tasks
python m2_agent/test_conversation.py --live --csv /path/to/IVA_Test.csv --row 42

# artifacts at results/<task_id>/text-trial1-seed*.json
cat results/RequestProofOfFunds#0001/*.json | python -m json.tool | head -n 40
```

For free-tier OpenRouter models export M1 creds from M2 in one shell:

```bash
set -a; source m2_agent/.env; set +a
export M1_MODEL_ENDPOINT="$M2_MODEL_ENDPOINT" M1_API_KEY="$M2_MODEL_API_KEY" M1_MODEL_NAME="$M2_MODEL_NAME"
python m2_agent/test_conversation.py --live --row 0
```

### 4.4 LiveKit room session — FULLY AUTOMATIC

One command does everything: **spawns the M2 worker** (`agent.py dev` subprocess), creates a fresh uniquely-named room, waits for the agent participant to join, captures its greeting, sends the CSV opener over the same channel as Flutter (`stream_text` topic `lk.chat`), drives up to N turns of real M1↔M2 conversation from live transcriptions, then disconnects, terminates the worker and writes artifacts:

```bash
python m2_agent/test_conversation.py --live-room            # defaults: --room-wait 180 --max-turns 3
python m2_agent/test_conversation.py --live-room --row 42 --room-wait 240 --max-turns 5

# drive your own already-running `lk agent dev` instead of auto-spawning:
python m2_agent/test_conversation.py --live-room --no-worker --room-wait 120
```

Console output is the live transcript (`M2(greeting)>`, `M1>`, `M2>`). Artifacts per run:

- `results/<task_id>/audio-trial1-seed0.json` — grading-ready RunReport (trajectory, telemetry e2e/ttft estimates, intent_match)
- `results/<task_id>/worker-<ts>.log` — full M2 worker log for debugging dispatch/STT/TTS issues

Requires only `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`; without `M1_API_KEY` it falls back to a scripted German caller so the demo still runs end-to-end.

### 4.5 Batch benchmark (CSV → many cells)

```bash
python m2_agent/batch_runner.py --limit 5 --trials 3 --modes text
python m2_agent/batch_runner.py --limit 20 --trials 3 --modes text audio --tier hard
python m2_agent/batch_runner.py --csv /path/to/IVA_Test.csv --limit 100 --trials 5 --results-dir results
```

`--trials 3-5` per Workflow-design §4.6. Checkpoint: re-running the same `results/` skips completed `(trial,task_id,seed)` cells — `0 runs` on second invocation proves resume.

## 5. How to Test / Verify (CI-equivalent)

```bash
source .venv/bin/activate
python -m py_compile m2_agent/*.py
python m2_agent/test_conversation.py --mock   # must be 23 PASS / 0 FAIL
python m2_agent/test_m1.py --mock

# With keys present — one live text-mode run with artifact
set -a; source m2_agent/.env; set +a
export M1_MODEL_ENDPOINT="$M2_MODEL_ENDPOINT" M1_API_KEY="$M2_MODEL_API_KEY" M1_MODEL_NAME="$M2_MODEL_NAME"
python m2_agent/test_conversation.py --live --row 0 --max-turns 6 --timeout 480
ls results/*/*.json   # artifact exists, contains final_transcript + intent_match + telemetry

# Flutter gates (unchanged, must stay green)
export PATH="$HOME/flutter/bin:$PATH"
flutter pub get
dart format --set-exit-if-changed -l 120 .
flutter analyze --no-fatal-infos
flutter test
```

## 6. Artifacts

Per-run JSON `results/<task_id>/<mode>-trial<n>-seed<s>.json` (iva-plan §5.3 + Workflow-design §4.1):

```json
{
  "task_id": "RequestProofOfFunds#0001",
  "run_id": "...",
  "mode": "text",
  "turns": 4,
  "final_transcript": "user: ...\nagent: ...",
  "predicted_label": "REQUEST_PROOF_OF_FUNDS",
  "expected_output": "REQUEST_PROOF_OF_FUNDS",
  "intent_match": true,
  "intent_f1": 1.0,
  "outcome": "pass",
  "stop_reason": "agent_stop",
  "duration_ms": 4210,
  "model_version": "google/gemini-2.5-flash",
  "temperature": 0.7,
  "m2_model": "nvidia/nemotron-3-ultra-550b-a55b:free",
  "telemetry_summary": { "e2e_ms": [...], "ttft_ms": [...] },
  "turn_taking": { "premature_rate": 0.0, "missed_turn_rate": 0.0, "dead_air_ms": 0.0 }
}
```

### Dual-mode attribution

Run the same `(task,seed)` in `text` and `audio`; `media_cost = R_text − R_audio` per Workflow-design §4.4. Reported separately, never blended.

Feed L1–L4 grading offline per Workflow-design §4.2–4.8 (`R = R_task×R_intent×R_comm×R_tone`, any 0 kills the run).

## 7. Design Decisions for Review

1. **Prompt source:** `goal=description`, `opener=scenario` verbatim, `persona=caller_name/gender/anrede`; `expected_output`/`intent_name` excluded (How-we-benchmark Four-Actor + Infrastructure Plan §4 anti-cheating; `assert_no_leak()` also checks camel-split leakage).
2. **Model tiers:** Gemini Flash-class for Easy/Medium, Claude Sonnet/Haiku for Hard/Adversarial, DeepSeek budget — via `user-simulator-models.md` Decision Matrix; routed through `litellm` so tier switching is config-only.
3. **Orchestrator split:** half-duplex = Workflow-design §3.3; 200 ms tick + caller state machine = tau2 `FullDuplexOrchestrator`; runner layers 1–3 (`orchestrator.py` pure execution / `batch_runner.py` build / batch+checkpoint) mirrors TAU2 §4.9.
4. **Location:** `m2_agent/` flat modules — shared `.venv` and `.env` loading pattern, exact verification commands `python m2_agent/test_conversation.py --mock` from spec §4, and `test_model.py` precedent. Added `schemas.py` (justified deviation: tau2 `data_model/` flattened, used by every module).

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `tenacity import failed` | missing `tenacity` for `num_retries` | `pip install tenacity` (in `requirements.txt`) |
| `empty LLM completion` | transient 502 on free OpenRouter tier | retries 3× inside `litellm_llm_fn`; increase `--timeout` for live runs |
| `anti-cheat violation` in tests | gold string in prompt | prompts never built from `expected_output`/`intent_name`; check custom tasks |
| `agent_empty` in live transcript | tools schema malformed | fixed: `self` excluded from schema (`environment.py:31`); re-run |
| `results` resume does `0 runs` | all cells completed | `rm -rf results` or `--no-resume` |

## 9. References to Quote in PR Description

- M1 prompt derivation + anti-cheating guard → Infrastructure Plan §4 + How-we-benchmark §Four-Actor
- Model tier routing → `user-simulator-models.md` Decision Matrix + §7 dual-mode cost
- Orchestrator text vs 200 ms tick + Layer 1–3 runner → Workflow-design §3.3 + TAU2 §4.9
- Artifacts feeding L1–L4 + multiplicative `R` + Wilson CI → Workflow-design §4.2–4.6 + Libraries §7
