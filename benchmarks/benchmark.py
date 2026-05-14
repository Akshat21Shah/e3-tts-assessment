"""
benchmark.py  -  TTFC / RTF / tok/s benchmarks for the TTS server
Usage: python benchmark.py   (ensure tts_server.py is running first)
"""
import asyncio, time
import aiohttp

TTS_URL     = "http://localhost:8000/synthesize"
SAMPLE_RATE = 24_000
TEXTS = [
    "Hello.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we interact with technology.",
    ("In the rapidly evolving landscape of AI, large language models have emerged "
     "as powerful tools for natural language understanding and generation."),
]
WARMUP, RUNS = 1, 3


async def _measure(session, text):
    t0, ttfc, total = time.perf_counter(), None, 0
    async with session.post(TTS_URL, json={"text": text},
                            timeout=aiohttp.ClientTimeout(total=300)) as resp:
        resp.raise_for_status()
        async for chunk in resp.content.iter_chunked(4096):
            if ttfc is None:
                ttfc = (time.perf_counter()-t0)*1000
            total += len(chunk)
    wall  = time.perf_counter()-t0
    audio = (total//2) / SAMPLE_RATE
    rtf   = wall / max(audio, 1e-6)
    tps   = (audio * 12 * 16) / max(wall, 1e-6)
    return {"ttfc_ms": ttfc, "rtf": rtf, "toks_per_s": tps}


async def run():
    async with aiohttp.ClientSession() as s:
        async with s.get("http://localhost:8000/health") as r:
            print("Server:", await r.json())
        print(f"{'Text':<42} {'TTFC':>9} {'RTF':>7} {'tok/s':>8}")
        print("-"*72)
        for text in TEXTS:
            ab = text[:40]+"..." if len(text)>42 else text
            for _ in range(WARMUP): await _measure(s, text)
            rs = [await _measure(s, text) for _ in range(RUNS)]
            tt = sum(r["ttfc_ms"] for r in rs)/RUNS
            rv = sum(r["rtf"] for r in rs)/RUNS
            tp = sum(r["toks_per_s"] for r in rs)/RUNS
            print(f"{ab:<42} {tt:>7.1f}ms {'✓' if tt<60 else '✗'} "
                  f"{rv:>5.3f}{'✓' if rv<0.15 else '✗'} {tp:>7.0f}")
        print("Targets: TTFC<60ms  RTF<0.15")

asyncio.run(run())
