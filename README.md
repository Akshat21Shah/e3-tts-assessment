# Megakernel-Accelerated Qwen3-TTS Voice Agent

Real-time voice agent: mic → Deepgram STT → Groq LLaMA-3.3-70B → Qwen3-TTS on a rented GPU via a hand-rolled CUDA megakernel → speaker.

**TTFC ≈ 36 ms · RTF ≈ 0.13** (RTX 5090, Blackwell)

---

## Architecture decisions

### System overview

```
Laptop (Mac)                         GPU server (vast.ai)
────────────────────────────         ──────────────────────────────
Mic → Deepgram STT (cloud)
     ↓
  Groq LLaMA-3.3-70B (cloud)
     ↓
  pipeline.py ──HTTP POST──────────► tts_server.py (FastAPI :8000)
  (via SSH tunnel)                        ↓
                                   PyTorch prefill (Qwen3-TTS talker)
                                        ↓
                                   CUDA megakernel decode (all 28 layers)
                                        ↓
                                   Code predictor (CUDA graph)
                                        ↓
                                   Vocoder → int16 PCM stream
◄────────────────────────────────────────
     ↓
  PyAudio → speakers
```

### Why a CUDA megakernel

The Qwen3-TTS talker is a 28-layer Transformer that autoregressively generates codec tokens at 12 Hz. Standard HuggingFace `generate()` costs ~20 ms per token (Python/CUDA kernel launch overhead per layer). At 12 tokens/second that gives RTF ≈ 0.24 — too slow for real-time.

The [AlpinDale megakernel](https://github.com/AlpinDale/qwen_megakernel) fuses all 28 layers into **one CUDA kernel launch** (128 blocks × 512 threads). No Python overhead between layers → **0.86 ms/step** → talker RTF ≈ 0.01.

### AudioInputGate (echo prevention)

The laptop mic picks up speaker output, creating a feedback loop (bot hears itself and responds again). `AudioInputGate` is a Pipecat `FrameProcessor` that drops all `AudioRawFrame` events while the bot is speaking and for 400 ms after it finishes, preventing the loop.

### Additional server optimisations

| Optimisation | Effect |
|---|---|
| `torch.compile(code_predictor)` | 18 ms → 5 ms per frame |
| CUDA graph for code-predictor loop | Eliminates Python overhead |
| Vocoder batching (`CHUNK_FRAMES=4`) | First frame sent immediately (TTFC); rest batched at 3 ms/frame |

---

## Kernel modifications

The [AlpinDale megakernel](https://github.com/AlpinDale/qwen_megakernel) was written for LLMs with a 151,936-token vocabulary. Three changes were needed for TTS:

### 1. LDG_VOCAB_SIZE patch

`kernel.cu` had `constexpr int LDG_VOCAB_SIZE = 151936` which a `-D` compiler flag cannot override. `scripts/patch_kernel.py` rewrites it to:

```c
#ifndef LDG_VOCAB_SIZE
#define LDG_VOCAB_SIZE 151936
#endif
```

`server/tts_build.py` then compiles with `-DLDG_VOCAB_SIZE=3072` (the TTS codec vocabulary size).

### 2. Slot-0 embedding trick

The kernel reads `embed_weight[token_id × HIDDEN]` expecting a single integer token. In TTS, each decode step's input is a sum of 16 codec embeddings + a text-guidance vector (no single integer). Fix: pre-compute the combined embedding on the Python side, write it into row 0 of a 1-row fake embedding table, and always pass `token_id=0` to the kernel.

### 3. Zero-copy KV cache bridge

Prefill runs in standard PyTorch (qwen-tts). Rather than converting KV tensors, they are `copy_()` directly into the megakernel's pre-allocated CUDA buffers. This works because Qwen3's m-RoPE collapses to 1D RoPE for text tokens, and `rope_theta=1_000_000` matches — values are bit-identical.

---

## How to run the Pipecat demo

### Prerequisites

- **GPU server** running `server/tts_server.py` (see below)
- **Mac laptop** with a microphone
- Deepgram API key ([console.deepgram.com](https://console.deepgram.com))
- Groq API key ([console.groq.com](https://console.groq.com))

### 1. Start the GPU server

```bash
# On the GPU server (e.g. vast.ai instance)
git clone https://github.com/Akshat21Shah/e3-tts-assessment.git
cd e3-tts-assessment
bash scripts/setup.sh          # clone megakernel, patch, download model, compile .so (~3 min)
python3 server/tts_server.py   # wait for "Application startup complete"
```

### 2. Open SSH tunnel (on your laptop)

```bash
ssh -N -f -L 8000:localhost:8000 -p <PORT> root@<IP>

# Verify:
curl http://localhost:8000/health
# → {"status":"ok","sample_rate":24000}
```

### 3. Install local dependencies (Mac)

```bash
brew install portaudio
pip install "pipecat-ai[local,silero]" deepgram-sdk groq aiohttp numpy
```

### 4. Run the agent

```bash
export DEEPGRAM_API_KEY=<your_key>
export GROQ_API_KEY=<your_key>

python3 pipeline/pipeline.py
```

Speak naturally. The agent greets you, transcribes your speech, generates a reply via Groq, and synthesises audio on the GPU. Press `Ctrl+C` to quit.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `DEEPGRAM_API_KEY` | *(required)* | Deepgram STT key |
| `GROQ_API_KEY` | *(required)* | Groq LLM key |
| `TTS_SERVER_URL` | `http://localhost:8000` | TTS server URL |
| `MODEL_PATH` | `<repo>/model/tts_base` | Qwen3-TTS weights path |

---

## Benchmark results

*RTX 5090 · sm_120a · WARMUP=1, RUNS=3*

| Text | TTFC | RTF |
|---|---|---|
| "Hello." | 35.7 ms | 0.149 |
| "The quick brown fox jumps over the lazy dog." | 35.9 ms | 0.128 |
| "Artificial intelligence is transforming…" | 38.0 ms | 0.133 |
| "In the rapidly evolving landscape of AI…" | 35.9 ms | 0.122 |

**TTFC < 60 ms ✓ · RTF < 0.15 ✓**
