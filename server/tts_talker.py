"""
tts_talker.py  -  Megakernel-accelerated Qwen3-TTS talker decoder

Integration strategy
--------------------
1. Prefill (text context): use qwen-tts PyTorch forward to prime KV cache
2. Transfer KV cache into megakernel buffers
3. Autoregressive codec-token generation (megakernel + code_predictor)
4. Audio synthesis via 12Hz codec decoder
"""

import math, os, struct, sys, time
from typing import Optional
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SERVER_DIR)
sys.path.insert(0, _SERVER_DIR)  # for tts_build
sys.path.insert(0, os.path.join(_REPO_ROOT, "qwen_megakernel"))
from tts_build import get_tts_extension

# Architecture constants (from talker config.json)
NUM_LAYERS       = 28
NUM_KV_HEADS     = 8
HEAD_DIM         = 128
HIDDEN_SIZE      = 1024
INTERMEDIATE     = 3072
Q_SIZE           = 2048
KV_SIZE          = 1024
CODEC_VOCAB_SIZE = 3072
NUM_CODE_GROUPS  = 16
ROPE_THETA       = 1_000_000.0
MAX_SEQ_LEN      = 8192

# Special codec token IDs
CODEC_EOS_ID     = 2150
CODEC_BOS_ID     = 2149
CODEC_PAD_ID     = 2148
CODEC_NOTHINK_ID = 2155
CODEC_THINK_ID   = 2154
CODEC_THINK_BOS  = 2156
CODEC_THINK_EOS  = 2157
TTS_BOS_ID       = 151672
TTS_EOS_ID       = 151673
TTS_PAD_ID       = 151671


def _build_rope_tables(max_seq_len=MAX_SEQ_LEN, theta=ROPE_THETA):
    half_dim = HEAD_DIM // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
    t     = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    cos_half = torch.cos(angles).to(torch.bfloat16).cuda()
    sin_half = torch.sin(angles).to(torch.bfloat16).cuda()
    # Kernel expects HEAD_DIM entries per row (= cat(half, half))
    cos = torch.cat([cos_half, cos_half], dim=-1).contiguous()
    sin = torch.cat([sin_half, sin_half], dim=-1).contiguous()
    return cos, sin


def _pack_layer_weights(layer_weights):
    ptr_size, n_ptrs = 8, 11
    buf = bytearray(NUM_LAYERS * n_ptrs * ptr_size)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size,
                             layer_weights[i * n_ptrs + j].data_ptr())
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class TTSTalkerDecoder:
    """Megakernel-powered Qwen3-TTS talker backbone."""

    def __init__(self, model_path=None, verbose=True):
        if model_path is None:
            model_path = os.path.join(_REPO_ROOT, "model", "tts_base")
        get_tts_extension()
        self._decode_op = torch.ops.qwen_tts_megakernel_C.decode

        if verbose:
            print("[TTSTalkerDecoder] Loading weights...", flush=True)
        state = load_file(f"{model_path}/model.safetensors")

        layer_weights = []
        for i in range(NUM_LAYERS):
            p = f"talker.model.layers.{i}."
            layer_weights.extend([
                state[p+"input_layernorm.weight"         ].to(torch.bfloat16).cuda().contiguous(),
                state[p+"self_attn.q_proj.weight"        ].to(torch.bfloat16).cuda().contiguous(),
                state[p+"self_attn.k_proj.weight"        ].to(torch.bfloat16).cuda().contiguous(),
                state[p+"self_attn.v_proj.weight"        ].to(torch.bfloat16).cuda().contiguous(),
                state[p+"self_attn.q_norm.weight"        ].to(torch.bfloat16).cuda().contiguous(),
                state[p+"self_attn.k_norm.weight"        ].to(torch.bfloat16).cuda().contiguous(),
                state[p+"self_attn.o_proj.weight"        ].to(torch.bfloat16).cuda().contiguous(),
                state[p+"post_attention_layernorm.weight"].to(torch.bfloat16).cuda().contiguous(),
                state[p+"mlp.gate_proj.weight"           ].to(torch.bfloat16).cuda().contiguous(),
                state[p+"mlp.up_proj.weight"             ].to(torch.bfloat16).cuda().contiguous(),
                state[p+"mlp.down_proj.weight"           ].to(torch.bfloat16).cuda().contiguous(),
            ])
        self._layer_weights_ref    = layer_weights
        self._layer_weights_packed = _pack_layer_weights(layer_weights)

        self._lm_head_weight = state["talker.codec_head.weight"].to(torch.bfloat16).cuda().contiguous()
        self._final_norm     = state["talker.model.norm.weight"].to(torch.bfloat16).cuda().contiguous()

        self._cos_table, self._sin_table = _build_rope_tables()

        # 1-row embed for pre-computed embeddings (token_id=0 always)
        self._embed_weight = torch.zeros(1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda").contiguous()

        self._k_cache = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
                                    dtype=torch.bfloat16, device="cuda")
        self._v_cache = torch.zeros_like(self._k_cache)
        self._position = 0

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32  = dict(dtype=torch.float32,  device="cuda")
        self._hidden    = torch.empty(HIDDEN_SIZE,  **bf16)
        self._act       = torch.empty(HIDDEN_SIZE,  **f32)
        self._res       = torch.empty(HIDDEN_SIZE,  **f32)
        self._q         = torch.empty(Q_SIZE,       **f32)
        self._k         = torch.empty(KV_SIZE,      **f32)
        self._v         = torch.empty(KV_SIZE,      **f32)
        self._attn_out  = torch.empty(Q_SIZE,       **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE, **f32)
        self._norm_out  = torch.empty(HIDDEN_SIZE,  **f32)
        self._bmax_vals = torch.empty(4096,         **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1,    dtype=torch.int32, device="cuda")
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        del state
        torch.cuda.empty_cache()
        if verbose:
            print("[TTSTalkerDecoder] Ready.", flush=True)

    def load_kv_cache(self, past_key_values, seq_len):
        assert len(past_key_values) == NUM_LAYERS
        for idx, (key, val) in enumerate(past_key_values):
            self._k_cache[idx, :, :seq_len, :] = key[0].to(torch.bfloat16)
            self._v_cache[idx, :, :seq_len, :] = val[0].to(torch.bfloat16)
        self._position = seq_len

    @torch.no_grad()
    def step_with_embed(self, embed_vec):
        self._embed_weight[0].copy_(embed_vec)
        self._decode_op(
            self._out_token, 0,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm, self._lm_head_weight,
            self._cos_table, self._sin_table,
            self._k_cache, self._v_cache,
            self._hidden, self._act, self._res,
            self._q, self._k, self._v, self._attn_out,
            self._mlp_inter, self._norm_out,
            self._bmax_vals, self._bmax_idxs,
            NUM_LAYERS, self._position, MAX_SEQ_LEN, self._attn_scale,
        )
        self._position += 1
        return self._out_token.item()

    def get_last_hidden(self):
        return self._norm_out.clone().to(torch.float32).unsqueeze(0).unsqueeze(0)

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self):
        return self._position
