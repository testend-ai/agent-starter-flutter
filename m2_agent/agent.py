"""M2 Agent Server Demo for LiveKit Voice AI Architecture.

Supports:
  1. Pipeline (STT -> custom LLM endpoint -> TTS)
  2. Realtime (Multimodal Realtime model endpoint)

Plug-and-play via M2_MODEL_ENDPOINT + M2_MODEL_API_KEY in .env.
"""

from __future__ import annotations

import logging
import os

from pathlib import Path

from dotenv import load_dotenv
import httpx
from livekit.agents import AgentServer, JobContext, JobProcess, WorkerOptions, cli, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import cartesia, deepgram, openai, silero

# Load m2_agent/.env first (explicit), then fallback to project root .env
# This ensures LIVEKIT_URL / API keys resolve regardless of cwd when running
# `python m2_agent/agent.py` vs `lk agent dev`.
load_dotenv(Path(__file__).with_name(".env"))
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

logger = logging.getLogger("m2-agent")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip().strip('"').strip("'").strip()


def _normalize_base_url(url: str | None) -> str | None:
    if not url:
        return None
    url = url.strip().rstrip("/")
    # livekit plugins expect base_url without /chat/completions suffix
    for suffix in ("/chat/completions", "/v1/chat/completions"):
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    return url or None


class _OpenRouterFluxTTS(tts.TTS):
    def __init__(self, *, api_key: str, model: str, voice: str, base_url: str) -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1)
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._base_url = base_url.rstrip("/")

    def synthesize(self, text: str, *, conn_options=DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        return _FluxChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _FluxChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        tts_inst: _OpenRouterFluxTTS = self._tts  # type: ignore
        url = f"{tts_inst._base_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {tts_inst._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "m2-agent",
        }
        payload = {
            "model": tts_inst._model,
            "input": self._input_text,
            "voice": tts_inst._voice,
            "response_format": "pcm",
        }
        output_emitter.initialize(
            request_id=tts_inst._base_url,
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
        )
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            pcm: bytes = resp.content
            output_emitter.push(pcm)
            output_emitter.flush()


def _build_pipeline_session() -> AgentSession:
    endpoint = _normalize_base_url(_env("M2_MODEL_ENDPOINT"))
    api_key = _env("M2_MODEL_API_KEY") or _env("OPENAI_API_KEY")
    model = _env("M2_MODEL_NAME", "gpt-4o-mini")
    try:
        temp = float(_env("M2_TEMPERATURE", "0.7"))
    except ValueError:
        temp = 0.7

    logger.info("Pipeline LLM: model=%s endpoint=%s", model, endpoint or "default")

    llm = openai.LLM(
        base_url=endpoint,
        api_key=api_key or "placeholder",
        model=model,
        temperature=temp,
    )

    stt_provider = _env("M2_STT_PROVIDER", "deepgram").lower()
    stt_endpoint = _normalize_base_url(_env("M2_STT_ENDPOINT"))
    stt_key = _env("M2_STT_API_KEY") or _env("DEEPGRAM_API_KEY") or api_key
    stt_model = _env("M2_STT_MODEL", "nova-2-general")

    if stt_provider == "openai" or stt_endpoint:
        stt = openai.STT(
            base_url=stt_endpoint,
            api_key=stt_key or "placeholder",
            model=stt_model if stt_model != "nova-2-general" else "whisper-1",
        )
    else:
        stt = deepgram.STT(model=stt_model, api_key=stt_key or None)

    tts_provider = _env("M2_TTS_PROVIDER", "cartesia").lower()
    tts_endpoint = _normalize_base_url(_env("M2_TTS_ENDPOINT"))
    tts_key = _env("M2_TTS_API_KEY") or _env("CARTESIA_API_KEY") or api_key
    tts_model = _env("M2_TTS_MODEL", "sonic-english")
    voice = _env("M2_VOICE", "79a125e8-cd45-4c13-8a67-188112f4dd22")

    is_flux = "flux" in tts_model.lower() and tts_endpoint and "openrouter" in tts_endpoint
    if is_flux:
        tts = _OpenRouterFluxTTS(api_key=tts_key or "placeholder", model=tts_model, voice=voice, base_url=tts_endpoint)
    elif tts_provider == "openai" or tts_endpoint:
        tts = openai.TTS(
            base_url=tts_endpoint,
            api_key=tts_key or "placeholder",
            model=tts_model if tts_model != "sonic-english" else "tts-1",
            voice=voice if len(voice) < 20 else "alloy",
        )
    else:
        tts = cartesia.TTS(model=tts_model, voice=voice, api_key=tts_key or None)

    return AgentSession(stt=stt, llm=llm, tts=tts, vad=silero.VAD.load())


def _build_realtime_session(instructions: str) -> AgentSession:
    endpoint = _normalize_base_url(_env("M2_MODEL_ENDPOINT"))
    api_key = _env("M2_MODEL_API_KEY") or _env("OPENAI_API_KEY")
    model = _env("M2_MODEL_NAME", "gpt-4o-realtime-preview")
    voice = _env("M2_VOICE", "alloy")
    try:
        temp = float(_env("M2_TEMPERATURE", "0.7"))
    except ValueError:
        temp = 0.7

    logger.info("Realtime model: %s endpoint=%s", model, endpoint or "default")

    rt_model = openai.realtime.RealtimeModel(
        base_url=endpoint,
        api_key=api_key or "placeholder",
        model=model,
        instructions=instructions,
        voice=voice if len(voice) < 20 else "alloy",
        temperature=temp,
    )
    return AgentSession(llm=rt_model)


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(setup_fnc=prewarm)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    instructions = _env(
        "M2_INSTRUCTIONS",
        "You are a helpful and professional customer service voice assistant. Keep answers concise and natural.",
    )
    greeting = _env("M2_GREETING", "Hello! How can I help you today?")
    workflow = _env("M2_WORKFLOW", "pipeline").lower()

    logger.info("M2 connecting to %s workflow=%s", ctx.room.name, workflow)
    await ctx.connect()

    session = (
        _build_realtime_session(instructions) if workflow == "realtime" else _build_pipeline_session()
    )
    agent = Agent(instructions=instructions)

    await session.start(room=ctx.room, agent=agent)

    if greeting and workflow != "realtime":
        await session.say(greeting, allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
