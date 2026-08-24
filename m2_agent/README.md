# M2 Agent Server + M1↔M2 Conversation Infrastructure

This directory hosts the **M2 Agent Server** and the **M1 user simulator** with the
conversation harness that runs golden-dataset sessions automatically.

In our architecture:
- **M1**: The user-simulator model playing the human caller.
- **M2**: The hosted calling AI model under evaluation.
- **Flutter Client (`agent-starter-flutter`)**: LiveKit client for visualizer, audio, transcriptions.

---

## 1. Quick Start (one command)

```bash
source ../.venv/bin/activate          # repo-root venv
pip install -r requirements.txt       # once
cp .env.example .env                  # fill M1_*/M2_* (+LIVEKIT_* for room mode)

python run.py                         # text sessions on the golden dataset
python run.py --mode room             # full automatic LiveKit sessions
```

`run.py` auto-discovers `~/AI-eval-testing/IVA_Test.csv`, sweeps the whole
usable dataset (checkpointed — re-runs continue with pending rows), and writes
artifacts to `results/<task_id>/`. Full guide: `docs/M1_IMPLEMENTATION_AND_TESTING.md`.

Useful flags:

```bash
python run.py --limit 20 --trials 3            # more tasks / repeated trials
python run.py --task RequestProofOfFunds       # filter by task id substring
python run.py --tier hard                      # easy | medium | hard | adversarial
python run.py --mode room,text                 # both modes on the same tasks
python run.py --mode room --no-worker          # drive your own `lk agent dev`
python run.py --csv /path/to/IVA_Test.csv      # explicit dataset path
```

Completed `(trial, task_id, seed)` cells are skipped on re-run; a fully resumed
run exits 0.

---

## 2. Supported M2 Workflows

### A. STT-LLM-TTS Pipeline Workflow (`M2_WORKFLOW=pipeline`)
- **STT**: Deepgram / Whisper / custom OpenAI-compatible endpoint.
- **LLM**: Any OpenAI-compatible model endpoint (`M2_MODEL_ENDPOINT` + `M2_MODEL_API_KEY`).
- **TTS**: Cartesia / ElevenLabs / OpenAI TTS / custom TTS endpoint.
- **Turn Detection**: Silero VAD.

### B. Real-time Model Workflow (`M2_WORKFLOW=realtime`)
- Direct stream to an OpenAI-compatible Realtime API endpoint.

---

## 3. Configuration

Copy `.env.example` → `.env`:

```env
# LiveKit Server Connection (needed for --mode room)
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

M2_WORKFLOW=pipeline
M2_MODEL_ENDPOINT=https://your-model-endpoint.com/v1
M2_MODEL_API_KEY=your_api_key
M2_MODEL_NAME=your-model-name

# M1 user simulator (any OpenAI-compatible endpoint via litellm)
M1_MODEL_ENDPOINT=https://openrouter.ai/api/v1
M1_API_KEY=sk-or-...
M1_MODEL_NAME=google/gemini-2.5-flash   # Easy/Med tier
M1_TEMPERATURE=0.7
M1_DIFFICULTY=medium                    # easy | medium | hard | adversarial
```

---

## 4. Module Map

| File | Responsibility |
|---|---|
| `run.py` | Single entry point: dataset discovery → session start → artifacts |
| `room_session.py` | Automated LiveKit session (spawns worker, drives conversation from live transcriptions) |
| `batch_runner.py` | CSV loader, component wiring, text-mode batch cells with resume |
| `orchestrator.py` | Half-duplex turn loop + full-duplex 200 ms tick loop |
| `m1_simulator.py` | Persona prompt builder, seeded behavioral knobs, litellm caller |
| `environment.py` | Stateful DB (`get_hash()` SHA-256), `@is_tool` / `ToolKitBase` toolkits |
| `schemas.py` | Pydantic contracts: Task / Message / Tick / Telemetry / SimulationRun |
| `report.py` | Artifact writer `results/<task_id>/<run>.json` + checkpoint scanner |
| `agent.py` | M2 LiveKit agent worker |

**Scope note:** this infrastructure collects raw conversation data only — trajectory,
transcript and telemetry per run. No scoring/grading is applied.
