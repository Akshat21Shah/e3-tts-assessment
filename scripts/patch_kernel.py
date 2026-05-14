#!/usr/bin/env python3
"""
patch_kernel.py  –  Apply the LDG_VOCAB_SIZE portability fix to kernel.cu

The upstream qwen_megakernel uses:
    constexpr int LDG_VOCAB_SIZE = 151936;

A constexpr cannot be overridden by a compile-time -D flag.
We change it to an #ifndef macro so that tts_build.py can inject
-DLDG_VOCAB_SIZE=3072 without modifying the kernel source at build time.

Run once after cloning qwen_megakernel:
    python3 patch_kernel.py
"""
import pathlib, sys

kernel_path = pathlib.Path(__file__).parent.parent / "qwen_megakernel" / "csrc" / "kernel.cu"

if not kernel_path.exists():
    print(f"ERROR: {kernel_path} not found. Did you clone qwen_megakernel first?")
    print("  git clone https://github.com/AlpinDale/qwen_megakernel")
    sys.exit(1)

src = kernel_path.read_text()

OLD = "constexpr int LDG_VOCAB_SIZE = 151936;"
NEW = "#ifndef LDG_VOCAB_SIZE\n#define LDG_VOCAB_SIZE 151936\n#endif"

if NEW in src:
    print("kernel.cu already patched – nothing to do.")
    sys.exit(0)

if OLD not in src:
    print(f"ERROR: expected pattern not found in {kernel_path}")
    print(f"  Looking for: {OLD!r}")
    print("  The upstream kernel may have changed. Check manually.")
    sys.exit(1)

kernel_path.write_text(src.replace(OLD, NEW))
print(f"Patched: {kernel_path}")
print("You can now run:  python3 tts_build.py")
