# M1↔M2 Conversation Infrastructure — Plug-and-Play Guide

> Branch `feat/m1-user-simulator-orchestrator` · All Python modules under `m2_agent/` · No Dart changes
>
> Scope: **data collection only** — M1 (simulated caller), M2 adapter, text + live-room
> conversation sessions, transcript/telemetry capture. No scoring or grading.

## 1. One Command

```bash
python m2_agent/run.py
```

That's it. It auto-discovers the golden dataset (`~/AI-eval-testing/IVA_Test.csv`),
starts conversation sessions, and writes transcript artifacts to `results/`.

## 2. What Is Implemented

| Module | File | Responsibility |
|---|---|---|
| **Entry point** | `m2_agent/run.py` | The single CLI. Auto-discovers the golden CSV, loads tasks, starts sessions (text and/or room), prints/writes artifacts. |
| **Schemas** | `m2_agent/schemas.py` | Pydantic V2 contracts: `Task` (goal/opener/persona), `Persona`, `Message`/`ToolCall`, `BehaviorKnobs`, `TurnTelemetry`, `Tick`, `SimulationRun`, `RunReport`. |
| **Environment** | `m2_agent/environment.py` | Stateful mock backend: `DB.get_hash()` SHA-256, `@is_tool` + `ToolKitBase` toolkits — agent side (`detect_intent`, `transfer_to_human`, `lookup_policy`) and user side (`confirm_resolution`, `hang_up`). |
| **M1 Simulator** | `m2_agent/m1_simulator.py` | Persona prompt from CSV `description` (goal) + `scenario` (verbatim opener) + `caller_name`/`gender`/`anrede`; German register `Sie`/`Du`, pronouns from `gender` only; seeded knobs (`verbosity`, `interruption_likelihood`, `patience_threshold`, `hesitation`) deterministic per `(task_id, trial, seed)` across tiers easy→adversarial; litellm caller with injectable `llm_fn`. |
| **Orchestrator** | `m2_agent/orchestrator.py` | `HalfDuplexOrchestrator`: text turn loop `M1 opener → M2 → tools BEFORE next turn → M1 → repeat`; terminates on STOP/goodbye/hangup/max_turns/timeout. `FullDuplexTickOrchestrator`: 200 ms ticks, caller state machine `IDLE→SPEAKING→WAIT_FOR_AGENT→INTERRUPTED`, per-tick tool flush, `Tick` records, turn-taking metrics. |
| **Room session** | `m2_agent/room_session.py` | Fully automatic LiveKit session: spawns the M2 worker (`agent.py dev` subprocess), creates a fresh room, waits for agent join + greeting, sends opener via `stream_text(topic="lk.chat")` (same channel as Flutter `sendText`), drives multi-turn conversation from live transcriptions (segment dedupe + quiet-period turn detection), tears down, writes artifact + worker log. |
| **Artifacts** | `m2_agent/report.py` | Writes `results/<task_id>/<mode>-trial<n>-seed<s>.json`: full trajectory, transcript, telemetry; checkpoint scanner keyed `(trial,task_id,seed)`. |
| **Batch runner** | `m2_agent/batch_runner.py` | CSV→`Task` loader (drops `Ausgeschlossen=Ja`), component wiring, batch cells with resume, `M2TextAgent` adapter (any OpenAI-compatible endpoint). |
| **M2 worker** | `m2_agent/agent.py` | Unchanged LiveKit agent server (STT→LLM→TTS pipeline / realtime). |

**Env contract** (`m2_agent/.env.example`):

```
LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET   # only needed for --mode room

M2_MODEL_ENDPOINT / M2_MODEL_API_KEY / M2_MODEL_NAME / M2_TEMPERATURE
M2_STT_* / M2_TTS_* / M2_VOICE / M2_WORKFLOW

M1_MODEL_ENDPOINT=https://openrouter.ai/api/v1
M1_API_KEY=sk-or-...
M1_MODEL_NAME=google/gemini-2.5-flash   # Easy/Med tier; Hard/Adv → claude-sonnet-5
M1_TEMPERATURE=0.7
M1_DIFFICULTY=medium                    # easy | medium | hard | adversarial
```

Optional env:
- `GOLDEN_DATASET_CSV=/path/to/IVA_Test.csv` to pin the dataset location
- `M2_INSTRUCTIONS` — the harness text-mode M2 uses the same persona as the deployed worker

## 3. File Layout

```
m2_agent/
  run.py                  # ← THE entry point (one command)
  room_session.py         # automated LiveKit room session
  batch_runner.py         # CSV loader + wiring + text-mode batch cells
  orchestrator.py         # half-duplex + 200ms tick orchestrators
  m1_simulator.py         # persona builder + litellm caller
  environment.py          # DB hash + ToolKitBase toolkits
  schemas.py              # pydantic contracts
  report.py               # artifact writer + checkpoint scan
  agent.py                # M2 LiveKit worker (unchanged)
results/<task_id>/        # per-run data artifacts + worker logs
docs/M1_IMPLEMENTATION_AND_TESTING.md
```

## 4. Setup (once)

```bash
source .venv/bin/activate
pip install -r m2_agent/requirements.txt
cp m2_agent/.env.example m2_agent/.env   # fill M1_*/M2_* (+LIVEKIT_* for room mode)
```

## 5. Running Conversations

```bash
# defaults: text mode, the ENTIRE usable dataset (233 rows), medium tier.
# Progress is checkpointed — each re-run continues with the next pending rows.
python m2_agent/run.py

# bound how many pending rows run per invocation
python m2_agent/run.py --limit 20 --trials 3
python m2_agent/run.py --task RequestProofOfFunds
python m2_agent/run.py --tier hard          # easy | medium | hard | adversarial

# full LiveKit sessions (auto-spawns the M2 worker per run)
python m2_agent/run.py --mode room --room-wait 240 --max-turns 3

# both modes back-to-back on the same tasks
python m2_agent/run.py --mode room,text --limit 5

# drive your own already-running `lk agent dev` instead of auto-spawning
lk agent dev   # terminal A (in m2_agent/)
python m2_agent/run.py --mode room --no-worker   # terminal B

# explicit dataset path (file OR directory containing IVA_Test.csv)
python m2_agent/run.py --csv /path/to/IVA_Test.csv
```

Everything on the conversation side comes from the dataset: goal, opener,
persona and tier knobs are read per row; nothing about a row is hardcoded.

**M1 endpoint fallback:** if `M1_API_KEY` is unset, the caller automatically
reuses `M2_MODEL_ENDPOINT`/`M2_API_KEY`/`M2_MODEL_NAME` (notice printed) so a
dataset sweep runs with only M2 credentials configured.

Checkpointing: completed `(trial, task_id, seed)` cells are skipped on re-run
(`--no-resume` forces re-running). A fully-resumed run exits 0.

## 6. Artifacts (core data only)

`results/<task_id>/<mode>-trial<n>-seed<s>.json`:

```json
{
  "schema_version": "2.0",
  "task_id": "RequestProofOfFunds#0001",
  "mode": "text",
  "turns": 3,
  "final_transcript": "user: ...\nagent: ...",
  "trajectory": [ {"role": "user", "content": "...", ...} ],
  "ticks": [],
  "stop_reason": "agent_stop",
  "duration_ms": 4210.5,
  "model_version": "google/gemini-2.5-flash",
  "temperature": 0.7,
  "m2_model": "...",
  "telemetry_summary": { "exchanges": 3, "e2e_ms": [...], "ttft_ms": [...] },
  "turn_taking": { "premature_rate": 0.0, "missed_turn_rate": 0.0 }
}
```

Room mode adds `results/<task_id>/worker-<ts>.log` (full M2 worker log).

Read a trajectory back:
```python
from report import load_trajectory
msgs = load_trajectory(Path("results/.../text-trial1-seed7.json"))
```

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Golden dataset IVA_Test.csv not found` | dataset not in default locations | pass `--csv` or set `GOLDEN_DATASET_CSV` |
| `tenacity import failed` | missing litellm retry dep | `pip install tenacity` |
| `empty LLM completion` | transient free-tier 502 | auto-retries 3×; raise `--timeout` |
| `no agent joined within budget` (room) | wrong LIVEKIT creds / worker crash | inspect `results/<task_id>/worker-*.log` |
