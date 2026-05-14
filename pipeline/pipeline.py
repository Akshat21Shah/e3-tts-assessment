"""
pipeline.py  -  Pipecat 1.1.0 voice agent
STT (Deepgram) -> LLM (Groq) -> TTS (megakernel) -> local audio output

Run:
    export DEEPGRAM_API_KEY=<key>
    export GROQ_API_KEY=<key>
    # TTS server must be running (default: http://localhost:8000)
    python3 pipeline/pipeline.py

Prerequisites (Mac):
    brew install portaudio
    pip install "pipecat-ai[local]" deepgram-sdk groq aiohttp numpy
"""
import asyncio, os, time
import aiohttp

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMRunFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.tts_service import TTSService
from pipecat.services.settings import TTSSettings
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GROQ_KEY     = os.getenv("GROQ_API_KEY", "")
TTS_SERVER   = os.getenv("TTS_SERVER_URL", "http://localhost:8000")
SAMPLE_RATE  = 24_000
SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. "
    "Keep responses short — under 3 sentences. "
    "Do not use bullet points or markdown formatting."
)


class AudioInputGate(FrameProcessor):
    """
    Mutes microphone audio while the bot is speaking.
    Prevents the mic from picking up speaker output (echo / feedback loop)
    which would cause the bot to hear and repeat its own speech.
    A short post-speech delay allows residual speaker audio to dissipate.
    """

    def __init__(self, post_speech_mute_secs: float = 0.4):
        super().__init__()
        self._muted = False
        self._post_speech_mute_secs = post_speech_mute_secs
        self._unmute_handle: asyncio.TimerHandle | None = None

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            # Cancel any pending unmute and mute immediately
            if self._unmute_handle:
                self._unmute_handle.cancel()
                self._unmute_handle = None
            self._muted = True
            await self.push_frame(frame, direction)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Keep muted briefly after bot finishes so lingering speaker
            # audio doesn't leak into the mic stream
            loop = asyncio.get_event_loop()
            self._unmute_handle = loop.call_later(
                self._post_speech_mute_secs, self._unmute
            )
            await self.push_frame(frame, direction)

        elif isinstance(frame, AudioRawFrame) and self._muted:
            pass  # drop mic audio while bot is speaking / cooling down

        else:
            await self.push_frame(frame, direction)

    def _unmute(self):
        self._muted = False
        self._unmute_handle = None


class MegakernelTTSService(TTSService):
    """Pipecat TTS service backed by the local megakernel HTTP server."""

    def __init__(self, server_url=TTS_SERVER, **kwargs):
        super().__init__(
            sample_rate=SAMPLE_RATE,
            settings=TTSSettings(model=None, voice=None, language=None),
            **kwargs,
        )
        self._url     = f"{server_url}/synthesize"
        self._session: aiohttp.ClientSession | None = None

    async def start(self, frame):
        await super().start(frame)
        self._session = aiohttp.ClientSession()

    async def stop(self, frame):
        await super().stop(frame)
        if self._session:
            await self._session.close()
            self._session = None

    async def run_tts(self, text: str, context_id: str):
        """Stream audio chunks from the megakernel TTS server."""
        if not self._session:
            self._session = aiohttp.ClientSession()
        t0 = time.perf_counter()
        ttfc_logged = False
        async with self._session.post(
            self._url,
            json={"text": text},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            buf = b""
            FRAME_BYTES = SAMPLE_RATE // 10 * 2   # 100 ms of 16-bit mono PCM
            async for chunk in resp.content.iter_chunked(8192):
                if not ttfc_logged:
                    print(f"[pipeline] TTFC: {(time.perf_counter()-t0)*1000:.1f} ms")
                    ttfc_logged = True
                buf += chunk
                while len(buf) >= FRAME_BYTES:
                    raw, buf = buf[:FRAME_BYTES], buf[FRAME_BYTES:]
                    yield TTSAudioRawFrame(
                        audio=raw,
                        sample_rate=SAMPLE_RATE,
                        num_channels=1,
                        context_id=context_id,
                    )
            if len(buf) >= 2:
                raw = buf[: len(buf) - len(buf) % 2]
                yield TTSAudioRawFrame(
                    audio=raw,
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                )


async def main():
    if not DEEPGRAM_KEY:
        raise RuntimeError("Set DEEPGRAM_API_KEY environment variable")
    if not GROQ_KEY:
        raise RuntimeError("Set GROQ_API_KEY environment variable")

    # Local microphone + speakers via PyAudio / PortAudio
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=SAMPLE_RATE,
        )
    )

    stt = DeepgramSTTService(api_key=DEEPGRAM_KEY, sample_rate=16_000)

    llm = GroqLLMService(
        api_key=GROQ_KEY,
        settings=GroqLLMService.Settings(
            model="llama-3.3-70b-versatile",
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    tts = MegakernelTTSService(server_url=TTS_SERVER)
    gate = AudioInputGate(post_speech_mute_secs=0.4)

    # Context + aggregators — VAD is wired into the user aggregator
    context = LLMContext()
    pair = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(),
        gate,         # <-- mute mic while bot speaks; prevents echo feedback
        stt,
        pair.user(),
        llm,
        tts,
        transport.output(),
        pair.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Kick off with a greeting as soon as the pipeline starts
    context.add_message({"role": "user", "content": "Hello!"})
    await task.queue_frames([LLMRunFrame()])

    print(f"[pipeline] Voice agent ready  (TTS server: {TTS_SERVER})")
    print("[pipeline] Speak into your microphone. Press Ctrl+C to quit.\n")

    await PipelineRunner().run(task)


if __name__ == "__main__":
    asyncio.run(main())
