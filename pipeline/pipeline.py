"""
pipeline.py  -  Pipecat voice agent
STT (Deepgram) -> LLM (Groq) -> TTS (megakernel) -> audio output
"""
import asyncio, os, sys, time
from typing import AsyncGenerator
import aiohttp, numpy as np
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import AudioRawFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.groq import GroqLLMService
from pipecat.services.tts_service import TTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioParams

DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GROQ_KEY     = os.getenv("GROQ_API_KEY", "")
TTS_SERVER   = os.getenv("TTS_SERVER_URL",     "http://localhost:8000")
SAMPLE_RATE  = 24_000


class MegakernelTTSService(TTSService):
    def __init__(self, server_url=TTS_SERVER, **kwargs):
        super().__init__(sample_rate=SAMPLE_RATE, **kwargs)
        self._url     = f"{server_url}/synthesize"
        self._session = None

    async def start(self, frame):
        await super().start(frame)
        self._session = aiohttp.ClientSession()

    async def stop(self, frame):
        await super().stop(frame)
        if self._session:
            await self._session.close()
            self._session = None

    async def run_tts(self, text):
        if not self._session:
            self._session = aiohttp.ClientSession()
        t0 = time.perf_counter()
        logged = False
        async with self._session.post(
            self._url, json={"text": text},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            buf = b""
            FRAME_BYTES = SAMPLE_RATE // 10 * 2
            async for chunk in resp.content.iter_chunked(8192):
                if not logged:
                    print(f"[pipeline] TTFC: {(time.perf_counter()-t0)*1000:.1f}ms")
                    logged = True
                buf += chunk
                while len(buf) >= FRAME_BYTES:
                    raw, buf = buf[:FRAME_BYTES], buf[FRAME_BYTES:]
                    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    yield TTSAudioRawFrame(audio=pcm.tobytes(),
                                          sample_rate=SAMPLE_RATE, num_channels=1)
            if len(buf) >= 2:
                raw = buf[:len(buf)-len(buf)%2]
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                yield TTSAudioRawFrame(audio=pcm.tobytes(),
                                      sample_rate=SAMPLE_RATE, num_channels=1)


SYSTEM_PROMPT = "You are a helpful, concise voice assistant. Keep responses under 3 sentences."


async def main():
    transport = LocalAudioTransport(LocalAudioParams(
        audio_in_enabled=True, audio_out_enabled=True,
        vad_enabled=True, vad_analyzer=SileroVADAnalyzer(),
        vad_audio_passthrough=True,
        input_sample_rate=16_000, output_sample_rate=SAMPLE_RATE,
    ))
    stt = DeepgramSTTService(api_key=DEEPGRAM_KEY, sample_rate=16_000)
    llm = GroqLLMService(api_key=GROQ_KEY, model="llama-3.3-70b-versatile")
    tts = MegakernelTTSService(server_url=TTS_SERVER)
    ctx = OpenAILLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    ctx_agg = llm.create_context_aggregator(ctx)

    pipeline = Pipeline([
        transport.input(), stt,
        ctx_agg.user(), llm, tts,
        transport.output(), ctx_agg.assistant(),
    ])
    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def _on_conn(transport, client):
        await task.queue_frames([ctx_agg.user().get_context_frame()])

    await PipelineRunner().run(task)


if __name__ == "__main__":
    asyncio.run(main())
