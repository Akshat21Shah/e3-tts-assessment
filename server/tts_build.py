"""
tts_build.py  -  JIT-compile the CUDA megakernel for Qwen3-TTS (vocab=3072)
"""
import os
import torch
from torch.utils.cpp_extension import load

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE  = os.path.join(_REPO_ROOT, "qwen_megakernel")
CACHE = os.path.expanduser("~/.cache/torch_extensions/qwen_tts_megakernel_C")

_module = None


def get_tts_extension():
    global _module
    if _module is not None:
        return _module
    os.makedirs(CACHE, exist_ok=True)
    print("[tts_build] Compiling TTS megakernel (LDG_VOCAB_SIZE=3072) ...", flush=True)
    _module = load(
        name="qwen_tts_megakernel_C",
        sources=[
            f"{BASE}/csrc/kernel.cu",
            f"{BASE}/csrc/torch_bindings.cpp",
        ],
        extra_cuda_cflags=[
            "-arch=sm_120a",
            "-O3",
            "--use_fast_math",
            "-DLDG_NUM_BLOCKS=128",
            "-DLDG_BLOCK_SIZE=512",
            "-DLDG_LM_NUM_BLOCKS=1184",
            "-DLDG_LM_BLOCK_SIZE=256",
            "-DLDG_VOCAB_SIZE=3072",
        ],
        build_directory=CACHE,
        verbose=True,
    )
    print("[tts_build] TTS megakernel ready.", flush=True)
    return _module


if __name__ == "__main__":
    get_tts_extension()
    print("Build SUCCESS")
