# e3 Group Technical Assessment – Megakernel-Accelerated Qwen3-TTS Voice Agent

## Overview

This solution wires the [AlpinDale CUDA megakernel](https://github.com/AlpinDale/qwen_megakernel) (designed for Qwen3-0.6B) to power **Qwen3-TTS-12Hz-0.6B-Base** inference inside a [Pipecat](https://github.com/pipecat-ai/pipecat) voice agent.

Full pipeline:

```
Microphone
  → Deepgram STT  (WebSocket, real-time)
  → Groq LLaMA-3.3-70B  (streaming LLM)
  → MegakernelTTSService  (→ tts_server.py)
      • Prefill: standard PyTorch (qwen-tts)
      • Decode:  RTX 5090 CUDA megakernel (128 blocks × 512 threads, sm_120a)
  → Audio output
```

**Hardware:** NVIDIA RTX 5090 (Blackwell, sm_120a) · CUDA 13.1 · PyTorch 2.10 NGC

**Performance targets:** TTFC < 60 ms · RTF < 0.15

---

## Architecture & Design Decisions

### 1 – Megakernel Integration Strategy

The Qwen3-TTS talker backbone (28 layers, hidden=1024) is architecturally **identical** to Qwen3-0.6B except for:

| Property | Qwen3-0.6B | Qwen3-TTS talker |
|---|---|---|
| `rope_theta` | 10 000 | 1 000 000 |
| LM head vocab | 151 936 (text) | 3 072 (codec) |
| Embed table | 151 936 × 1024 | 3 072 × 1024 |
| Weight prefix | `model.` | `talker.model.` |

The megakernel decode loop runs unchanged. Only two adaptations are needed:

**a) Pre-computed embeddings via slot-0 trick**  
The megakernel takes `embed_weight` (the full embedding table) and a `token_id` integer, and reads `embed_weight[token_id * HIDDEN_SIZE]`. For TTS generation, each step's input embedding is a *sum* of up to 16 codec embeddings plus a text-guidance vector — not a simple table lookup. The fix: maintain a 1-row `embed_weight` tensor, write the pre-computed combined vector into row 0, and always call `decode(token_id=0)`. This adds one ~2 KB GPU write per step, negligible overhead.

**b) LDG_VOCAB_SIZE compile-time override**  
`kernel.cu` had `constexpr int LDG_VOCAB_SIZE = 151936` at line ~74. A `constexpr` cannot be overridden by a `-D` compile flag. We patch this to:
```cpp
#ifndef LDG_VOCAB_SIZE
#define LDG_VOCAB_SIZE 151936
#endif
```
Then `tts_build.py` compiles a separate `.so` with `-DLDG_VOCAB_SIZE=3072`, pointed at a different cache directory so both kernels coexist.

### 2 – m-RoPE / 1D-RoPE Equivalence

Qwen3's multimodal-RoPE degenerates to standard 1D RoPE for text-only sequences: all three position dimensions (temporal, height, width) receive the same scalar position index. With `rope_theta=1_000_000` matching the TTS config, the KV values computed by qwen-tts (PyTorch prefill) and by the megakernel (autoregressive decode) are bit-identical. This enables **zero-copy KV cache transfer**: the KV tensors from `talker.model.forward(..., use_cache=True)` are `copy_()` into the megakernel's pre-allocated `_k_cache` / `_v_cache` buffers.

### 3 – Code Predictor Residency

After each `decode()` call, the megakernel leaves the final normalized hidden state (1024 × float32) in the `_norm_out` scratch buffer. This is exactly the input `past_hidden` required by the code predictor to generate codebooks c1..c15. No kernel modification is needed — we simply call `get_last_hidden()` after each step.

### 4 – Non-Streaming Mode Simplification

qwen-tts supports two modes:
- **Streaming**: `trailing_text_hidden[step]` = text projection of the t-th text token, providing token-level guidance during codec generation.  
- **Non-streaming**: `trailing_text_hidden` = constant `tts_pad_embed` for all steps.

We use **non-streaming mode** for simplicity. The text is fully baked into the prefill, and `tts_pad_embed` is a single pre-computed constant vector that guides all codec decode steps uniformly.

---

## File Structure

```
e3-tts-assessment/
├── server/
│   ├── tts_server.py        # FastAPI streaming TTS server (CUDA graphs + vocoder batching)
│   ├── tts_talker.py        # TTSTalkerDecoder – megakernel adapter + codec decoder
│   └── tts_build.py         # JIT-compiles megakernel with VOCAB=3072
├── pipeline/
│   └── pipeline.py          # Pipecat voice agent: Deepgram STT → Groq LLM → MegakernelTTS
├── benchmarks/
│   └── benchmark.py         # TTFC / RTF benchmarks (WARMUP=1, RUNS=3)
├── scripts/
│   ├── setup.sh             # One-shot setup: clone → patch → download model → compile
│   └── patch_kernel.py      # Applies #ifndef LDG_VOCAB_SIZE patch to kernel.cu
├── qwen_megakernel/         # AlpinDale megakernel – cloned + patched by scripts/setup.sh
│   └── csrc/kernel.cu       # Patched: LDG_VOCAB_SIZE → #ifndef macro
├── model/tts_base/          # Qwen3-TTS-12Hz-0.6B-Base weights – downloaded by scripts/setup.sh
├── requirements.txt
└── README.md
```

All paths are computed relative to each file's location (no hardcoded system paths).
`MODEL_PATH` in `server/tts_server.py` can be overridden via the `MODEL_PATH` environment variable.

---

## Setup & Running

### Prerequisites

- NVIDIA GPU (tested: RTX 5090, sm_120a / CUDA 13.1). sm_86+ should work.
- Python 3.10+, CUDA toolkit matching your driver
- `pip install torch --index-url https://download.pytorch.org/whl/cu121` (or cu118/cu124 as needed)

### One-shot setup

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo

# Clone megakernel, patch kernel.cu, download model, install deps, compile
bash scripts/setup.sh
```

### Step 1 – Patch kernel.cu (if running manually)

```bash
python3 scripts/patch_kernel.py
```

### Step 2 – Compile the TTS kernel

```bash
python3 server/tts_build.py   # ~3 min first time; cached after
```

### Step 3 – Start the TTS server

```bash
python3 server/tts_server.py
# Alternatively: MODEL_PATH=/path/to/model python3 server/tts_server.py
# Server starts on http://0.0.0.0:8000
```

### Step 4 – Quick synthesis test

```bash
curl -s -X POST http://localhost:8000/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text":"Hello, this is Qwen three T T S powered by the megakernel."}' \
  | aplay -r 24000 -f S16_LE -c 1
```

### Step 5 – Run benchmarks

```bash
python3 benchmarks/benchmark.py
```

### Step 6 – Start the Pipecat voice agent

```bash
# In a second terminal (server must be running in first)
export DEEPGRAM_API_KEY=your_key
export GROQ_API_KEY=your_key
python3 pipeline/pipeline.py
# Speak into the microphone; responses are synthesized via megakernel
```

---

## Benchmark Results

*(Measured on RTX 5090, driver 595.58.03, sm_120a · WARMUP=1, RUNS=3)*

| Text | TTFC | RTF | Codec tok/s |
|---|---|---|---|
| "Hello." | 35.7 ms | 0.149 | 1 291 |
| "The quick brown fox jumps over the lazy dog." | 35.9 ms | 0.128 | 1 501 |
| "Artificial intelligence is transforming…" | 38.0 ms | 0.133 | 1 443 |
| "In the rapidly evolving landscape of AI…" | 35.9 ms | 0.122 | 1 574 |

All results within targets: **TTFC < 60 ms ✓ · RTF < 0.15 ✓**

### Optimisations applied

1. **CUDA megakernel** (`tts_talker.py`) – replaces HuggingFace autoregressive decode for the talker backbone (28-layer Transformer). Decode step: 0.86 ms.
2. **`torch.compile(mode="default")`** on `code_predictor.model` – inductor/triton kernel fusion for the 5-layer code predictor.
3. **CUDA graph** (`_build_cp_graph` in `tts_server.py`) – captures the full per-frame code predictor loop (prefill + 15 decode steps + top-k sampling + embedding accumulation) as a single replayable graph. Per-frame latency: ~5 ms (vs ~18 ms with HF generate).
4. **Vocoder batching** – first codec frame emitted immediately (preserves TTFC); subsequent frames sent to the speech tokenizer in batches of 4 (reduces vocoder overhead from ~11 ms/frame to ~3 ms/frame).

Combined per-frame wall time: ~5 (code pred) + ~0.9 (megakernel) + ~3 (vocoder) ≈ **8.9 ms** vs 80 ms of audio per frame → **RTF ≈ 0.11**.

---

## Simplifications / Known Limitations

1. **No voice cloning / ICL prompt**: voice clone and instruct modes are not implemented. The server uses the default speaker with no conditioning.  
2. **Non-streaming text guidance**: `trailing_text_hidden` is always `tts_pad_embed`. This sacrifices some prosody alignment for implementation simplicity.  
3. **Batch size = 1**: the megakernel is a single-sequence decode kernel.  
4. **Temperature / sampling**: the megakernel uses argmax (greedy decoding) for c0; the code predictor uses top-k/top-p sampling for c1..c15.
