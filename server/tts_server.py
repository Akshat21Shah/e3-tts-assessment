# tts_server.py  – Qwen3-TTS streaming server with CUDA-graph code predictor
# Optimisations applied:
#   1. torch.compile(code_predictor.model, mode="default")
#   2. CUDA graph for full code_predictor generate (prefill + 14 decode + sampling)
#   3. Vocoder batching: first codec frame emitted immediately (TTFC),
#      subsequent frames batched in groups of CHUNK_FRAMES (lower RTF)

import sys, types, importlib.util

def _make_stub(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    m.__loader__ = None; m.__package__ = name; m.__path__ = []
    return m

_ta = _make_stub("torchaudio")
_co = _make_stub("torchaudio.compliance")
_ka = _make_stub("torchaudio.compliance.kaldi")
_ta.compliance = _co; _co.kaldi = _ka
sys.modules.update({"torchaudio": _ta, "torchaudio.compliance": _co,
                    "torchaudio.compliance.kaldi": _ka})
# ─────────────────────────────────────────────────────────────────────────────

import asyncio, os, time
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import StaticCache

_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)                                        # find tts_talker
sys.path.insert(0, os.path.join(_REPO_ROOT, "qwen_megakernel"))  # find qwen_tts
from tts_talker import (
    TTSTalkerDecoder,
    CODEC_EOS_ID, CODEC_BOS_ID, CODEC_PAD_ID,
    CODEC_NOTHINK_ID, CODEC_THINK_BOS, CODEC_THINK_EOS,
    TTS_BOS_ID, TTS_EOS_ID, TTS_PAD_ID,
    NUM_CODE_GROUPS, HIDDEN_SIZE, NUM_LAYERS,
)
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models import Qwen3TTSProcessor

MODEL_PATH   = os.getenv("MODEL_PATH", os.path.join(_REPO_ROOT, "model", "tts_base"))
SAMPLE_RATE  = 24_000
CHUNK_FRAMES = 4       # batch vocoder calls after the first frame
TEMPERATURE  = 0.9
TOP_K        = 50

# ─── code-predictor CUDA-graph constants ─────────────────────────────────────
_CP_CONTEXT_LEN = 2          # [past_hidden, c0_emb]
_CP_MAX_NEW     = NUM_CODE_GROUPS - 1   # 15  (c1..c15)
_CP_MAX_LEN     = _CP_CONTEXT_LEN + _CP_MAX_NEW + 1   # 18 slots in static KV cache


def _build_cp_graph(cp, codec_embed):
    """
    Set up torch.compile + CUDA-graph for code_predictor.
    Returns a namespace with the graph and all its I/O buffers.
    """
    import types
    ns = types.SimpleNamespace()

    cfg    = cp.model.config
    HIDDEN = cfg.hidden_size   # 1024

    # ── 1. torch.compile the inner Transformer ──────────────────────────────
    cp.model = torch.compile(cp.model, mode="default", fullgraph=False)
    print("[server] torch.compile applied to code_predictor.model", flush=True)

    # ── 2. Pre-fixed constant tensors (never change) ─────────────────────────
    ns.ctx_pos = torch.arange(_CP_CONTEXT_LEN, device="cuda")

    # Prefill causal mask [1, 1, 2, MAX_LEN]
    ns.prefill_mask = torch.zeros(
        1, 1, _CP_CONTEXT_LEN, _CP_MAX_LEN, dtype=torch.bfloat16, device="cuda")
    ns.prefill_mask[:, :, :, _CP_CONTEXT_LEN:] = float("-inf")
    ns.prefill_mask[:, :, 0, 1] = float("-inf")   # pos 0 cannot attend pos 1

    # Per-decode-step causal masks [MAX_NEW × (1,1,1,MAX_LEN)]
    ns.dec_pos   = [torch.tensor([_CP_CONTEXT_LEN + g], device="cuda")
                    for g in range(_CP_MAX_NEW)]
    ns.dec_masks = []
    for g in range(_CP_MAX_NEW):
        pos = _CP_CONTEXT_LEN + g
        m = torch.full((1, 1, 1, _CP_MAX_LEN), float("-inf"),
                       dtype=torch.bfloat16, device="cuda")
        m[:, :, :, :pos + 1] = 0.0
        ns.dec_masks.append(m)

    # ── 3. Mutable I/O buffers ────────────────────────────────────────────────
    ns.ctx_emb   = torch.zeros(1, _CP_CONTEXT_LEN, HIDDEN,
                               dtype=torch.bfloat16, device="cuda")   # input
    ns.out_codes = torch.zeros(_CP_MAX_NEW, dtype=torch.long, device="cuda")  # sampled c1..c15
    ns.codec_sum = torch.zeros(1, 1, HIDDEN,
                               dtype=torch.bfloat16, device="cuda")   # sum(c0..c15) embeds

    # ── 4. Static KV cache (never reset – every position is overwritten) ──────
    ns.sc = StaticCache(config=cfg, batch_size=1, max_cache_len=_CP_MAX_LEN,
                        device="cuda", dtype=torch.bfloat16)

    # ── 5. Top-k sampling helper (captured inside CUDA graph) ─────────────────
    def _topk_sample(logits):           # logits: [1, vocab]
        v, i = logits.topk(TOP_K, dim=-1)
        f = torch.full_like(logits, float("-inf"))
        f.scatter_(-1, i, v)
        return torch.multinomial((f / TEMPERATURE).softmax(-1), 1)   # [1, 1]

    # ── 6. Warmup (triggers torch.compile) ────────────────────────────────────
    print("[server] Warming up code_predictor (triggers torch.compile)…",
          flush=True)
    for _ in range(5):
        out = cp.model(inputs_embeds=ns.ctx_emb, past_key_values=ns.sc,
                       use_cache=True, cache_position=ns.ctx_pos,
                       attention_mask=ns.prefill_mask)
        h = out.last_hidden_state[:, -1, :]
        for g in range(_CP_MAX_NEW):
            lgts = cp.lm_head[g](h)
            c    = _topk_sample(lgts)
            ns.out_codes[g:g+1].copy_(c.view(-1))
            if g < _CP_MAX_NEW - 1:
                emb = cp.get_input_embeddings()[g](c)
                out = cp.model(inputs_embeds=emb, past_key_values=ns.sc,
                               use_cache=True, cache_position=ns.dec_pos[g],
                               attention_mask=ns.dec_masks[g])
                h = out.last_hidden_state[:, -1, :]
    torch.cuda.synchronize()
    print("[server] Warmup done.", flush=True)

    # ── 7. Capture CUDA graph ─────────────────────────────────────────────────
    print("[server] Capturing CUDA graph…", flush=True)
    ns.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ns.graph):
        # Prefill [1, 2, 1024]
        _g_out = cp.model(inputs_embeds=ns.ctx_emb, past_key_values=ns.sc,
                          use_cache=True, cache_position=ns.ctx_pos,
                          attention_mask=ns.prefill_mask)
        _g_h = _g_out.last_hidden_state[:, -1, :]

        # Accumulate sum starting from c0_emb (from ns.ctx_emb[:,1:2,:])
        ns.codec_sum.copy_(ns.ctx_emb[:, 1:2, :])

        for g_idx in range(_CP_MAX_NEW):
            _g_lgts = cp.lm_head[g_idx](_g_h)
            _g_c    = _topk_sample(_g_lgts)
            ns.out_codes[g_idx:g_idx+1].copy_(_g_c.view(-1))

            _g_emb = cp.get_input_embeddings()[g_idx](_g_c)   # [1, 1, 1024]
            ns.codec_sum.add_(_g_emb)                          # accumulate in-place

            if g_idx < _CP_MAX_NEW - 1:
                _g_fwd = cp.model(inputs_embeds=_g_emb, past_key_values=ns.sc,
                                  use_cache=True,
                                  cache_position=ns.dec_pos[g_idx],
                                  attention_mask=ns.dec_masks[g_idx])
                _g_h = _g_fwd.last_hidden_state[:, -1, :]

    torch.cuda.synchronize()
    print("[server] CUDA graph captured.", flush=True)
    return ns


# ─── model container ─────────────────────────────────────────────────────────

class _Models:
    def __init__(self):
        print("[server] Loading processor…", flush=True)
        self.processor = Qwen3TTSProcessor.from_pretrained(MODEL_PATH)

        print("[server] Loading qwen-tts model…", flush=True)
        self.qwen_tts = Qwen3TTSForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map="cuda")
        self.qwen_tts.eval()

        print("[server] Loading megakernel decoder…", flush=True)
        self.talker_dec = TTSTalkerDecoder(MODEL_PATH, verbose=True)

        talker = self.qwen_tts.talker

        # Embedding + projection modules
        self.text_embed   = talker.get_text_embeddings()
        self.text_proj    = talker.text_projection
        self.codec_embed  = talker.get_input_embeddings()   # Embedding(3072, 1024)
        self.codec_head   = talker.codec_head
        self.code_pred    = talker.code_predictor
        self.talker_model = talker
        self.speech_tok   = self.qwen_tts.speech_tokenizer

        # Pre-compute BOS/EOS/PAD embeds
        with torch.no_grad():
            sp_ids = torch.tensor([[TTS_BOS_ID, TTS_EOS_ID, TTS_PAD_ID]],
                                   dtype=torch.long, device="cuda")
            bos_e, eos_e, pad_e = self.text_proj(
                self.text_embed(sp_ids)).chunk(3, dim=1)
        self.tts_bos_embed = bos_e
        self.tts_eos_embed = eos_e
        self.tts_pad_embed = pad_e

        # CUDA graph for code_predictor (includes torch.compile + warmup)
        self.cp_ns = _build_cp_graph(self.code_pred, self.codec_embed)

        print("[server] Ready.\n", flush=True)


_models: "_Models | None" = None


def get_models():
    global _models
    if _models is None:
        _models = _Models()
    return _models


# ─── prefill ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _build_prefill_embed(m, text):
    """Build the talker prefill embed sequence (streaming mode, English)."""
    tc  = m.qwen_tts.config.talker_config
    full_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    proc_out  = m.processor(text=[full_text], return_tensors="pt")
    input_ids = proc_out["input_ids"].to("cuda")   # [1, L]

    # English language id
    lang_id = tc.codec_language_id["english"]  # 2050

    # codec_input_emebdding_0: [think_id, think_bos_id, lang_id, think_eos_id]
    codec_e0 = m.codec_embed(torch.tensor(
        [[tc.codec_think_id, tc.codec_think_bos_id, lang_id, tc.codec_think_eos_id]],
        dtype=torch.long, device="cuda"))  # [1, 4, 1024]

    # codec_input_emebdding_1: [pad_id, bos_id]
    codec_e1 = m.codec_embed(torch.tensor(
        [[tc.codec_pad_id, tc.codec_bos_id]],
        dtype=torch.long, device="cuda"))  # [1, 2, 1024]

    # codec_input_emebdding: 6 tokens
    codec_emb = torch.cat([codec_e0, codec_e1], dim=1)  # [1, 6, 1024]

    # Role embed: input_ids[:, :3] = [<|im_start|>, assistant, \n]
    role_embed = m.text_proj(m.text_embed(input_ids[:, :3]))  # [1, 3, 1024]

    # talker_mid: (tts_pad x 4 + tts_bos) + codec_emb[:, :5]  = 5 tokens
    n_codec = codec_emb.shape[1]  # 6
    pad_blk    = m.tts_pad_embed.expand(-1, n_codec - 2, -1)   # [1, 4, 1024]
    talker_mid = torch.cat([pad_blk, m.tts_bos_embed], dim=1) + codec_emb[:, :-1]  # [1, 5, 1024]

    # first content token + codec_bos embed (streaming: only first token in prefill)
    first_e = (m.text_proj(m.text_embed(input_ids[:, 3:4]))
               + codec_emb[:, -1:])  # [1, 1, 1024]

    prefill_embed = torch.cat([role_embed, talker_mid, first_e], dim=1)  # [1, 9, 1024]

    # trailing_text_hidden: remaining content tokens (4:-5) + tts_eos
    remaining_ids = input_ids[:, 4:-5]   # tokens after first content, before suffix
    if remaining_ids.shape[1] > 0:
        trailing_text_hidden = torch.cat(
            [m.text_proj(m.text_embed(remaining_ids)), m.tts_eos_embed], dim=1
        )  # [1, N_remaining+1, 1024]
    else:
        trailing_text_hidden = m.tts_eos_embed  # [1, 1, 1024]

    seq_len = prefill_embed.shape[1]
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device="cuda")
    return prefill_embed.to(torch.bfloat16), attention_mask, trailing_text_hidden


@torch.no_grad()
def _run_prefill(m, prefill_embed, attn_mask):
    """Run talker transformer on prefill, return past_kv + initial c0."""
    out = m.talker_model.model(
        inputs_embeds=prefill_embed,
        attention_mask=attn_mask,
        use_cache=True,
        output_hidden_states=False,
    )
    past_kv     = out.past_key_values
    past_hidden = out.last_hidden_state[:, -1:, :].to(torch.bfloat16)  # [1,1,1024]

    # Sample c0_0 from codec_head logits at last prefill position
    logits_c0 = m.codec_head(past_hidden)       # [1, 1, 3072]
    probs = torch.softmax(logits_c0[0, 0] / TEMPERATURE, dim=-1)
    c0_0  = torch.multinomial(probs, 1).item()

    return past_kv, past_hidden, c0_0


# ─── per-frame decode (CUDA-graph accelerated) ───────────────────────────────

@torch.no_grad()
def _decode_frame(m, c0_val, past_hidden, step_embed):
    """
    Given c0 for the current frame and the talker's last hidden state,
    generate c1..c15 via CUDA-graphed code_predictor, then run one
    megakernel step to produce the next c0.

    Returns:
      frame       – list of 16 ints [c0_val, c1, …, c15]
      next_c0_val – int
      next_hidden – [1, 1, 1024] bfloat16
    """
    ns = m.cp_ns

    # Build context embedding [past_hidden | c0_emb] into mutable buffer
    c0_ids = torch.tensor([[c0_val]], dtype=torch.long, device="cuda")
    ns.ctx_emb[:, 0:1, :].copy_(past_hidden)
    ns.ctx_emb[:, 1:2, :].copy_(m.codec_embed(c0_ids))

    # Replay CUDA graph → fills ns.out_codes[0..14] and ns.codec_sum
    ns.graph.replay()

    # ns.codec_sum holds  sum(c0_emb + c1_emb + … + c15_emb)
    next_in_emb = ns.codec_sum + step_embed             # [1, 1, 1024]

    # Megakernel step → next c0
    embed_vec   = next_in_emb[0, 0].to(torch.bfloat16)  # [1024]
    next_c0_val = m.talker_dec.step_with_embed(embed_vec)
    next_hidden = m.talker_dec.get_last_hidden().to(torch.bfloat16)  # [1,1,1024]

    frame = [c0_val] + ns.out_codes.tolist()
    return frame, next_c0_val, next_hidden


# ─── audio decode ─────────────────────────────────────────────────────────────

def _decode_audio(frames, m):
    """Convert list of codec frames → int16 PCM bytes."""
    codes = torch.tensor(frames, dtype=torch.long)   # (N, 16)
    wavs, _sr = m.speech_tok.decode({"audio_codes": codes})
    wav = wavs[0]
    wav = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
    return wav.tobytes()


# ─── main synthesis loop ──────────────────────────────────────────────────────

@torch.no_grad()
async def _synthesize_streaming(text, m):
    t0 = time.perf_counter()

    prefill_embed, attn_mask, trailing_text_hidden = _build_prefill_embed(m, text)
    seq_len = prefill_embed.shape[1]

    past_kv, past_hidden, c0_val = _run_prefill(m, prefill_embed, attn_mask)

    m.talker_dec.reset()
    m.talker_dec.load_kv_cache(past_kv, seq_len)
    del past_kv

    pending   = []
    ttfc_done = False

    for step in range(4096):
        if c0_val == CODEC_EOS_ID:
            break

        n_trailing = trailing_text_hidden.shape[1]
        step_embed = (trailing_text_hidden[:, step:step+1]
                      if step < n_trailing else m.tts_pad_embed)
        frame, c0_val, past_hidden = _decode_frame(
            m, c0_val, past_hidden, step_embed=step_embed)
        pending.append(frame)

        if not ttfc_done:
            ttfc_ms = (time.perf_counter() - t0) * 1000
            print(f"[server] TTFC: {ttfc_ms:.1f} ms", flush=True)
            ttfc_done = True
            # Emit first frame immediately for lowest TTFC
            yield _decode_audio(pending, m)
            pending = []
            await asyncio.sleep(0)
        elif len(pending) >= CHUNK_FRAMES:
            yield _decode_audio(pending, m)
            pending = []
            await asyncio.sleep(0)

    if pending:
        yield _decode_audio(pending, m)

    wall = time.perf_counter() - t0
    print(f"[server] Done in {wall:.3f}s  steps={step}", flush=True)


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Qwen3-TTS Megakernel Server")


class SynthReq(BaseModel):
    text: str


@app.on_event("startup")
async def _startup():
    get_models()


@app.post("/synthesize")
async def synthesize(req: SynthReq):
    m = get_models()

    async def _gen():
        async for chunk in _synthesize_streaming(req.text, m):
            yield chunk

    return StreamingResponse(
        _gen(), media_type="audio/pcm",
        headers={"X-Sample-Rate": "24000", "X-Channels": "1", "X-Encoding": "int16-le"})


@app.get("/health")
async def health():
    return {"status": "ok", "sample_rate": SAMPLE_RATE}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
