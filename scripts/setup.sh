#!/usr/bin/env bash
# setup.sh  –  One-shot environment setup for the Qwen3-TTS megakernel project
# Tested on: Ubuntu 22.04, RTX 5090, CUDA 13.1, Python 3.12
set -e

echo "=== Step 1: Clone qwen_megakernel ==="
if [ ! -d "qwen_megakernel" ]; then
    git clone https://github.com/AlpinDale/qwen_megakernel
else
    echo "qwen_megakernel/ already present – skipping clone"
fi

echo ""
echo "=== Step 2: Patch kernel.cu (LDG_VOCAB_SIZE #ifndef guard) ==="
python3 scripts/patch_kernel.py

echo ""
echo "=== Step 3: Download Qwen3-TTS model ==="
if [ ! -d "model/tts_base" ]; then
    pip install -q huggingface_hub
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --local-dir model/tts_base \
        --local-dir-use-symlinks False
else
    echo "model/tts_base/ already present – skipping download"
fi

echo ""
echo "=== Step 4: Install Python dependencies ==="
pip install -r requirements.txt

echo ""
echo "=== Step 5: Compile TTS CUDA megakernel ==="
python3 server/tts_build.py

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start the TTS server:"
echo "  python3 server/tts_server.py"
echo ""
echo "To run benchmarks (server must be running):"
echo "  python3 benchmark.py"
echo ""
echo "To run the Pipecat voice agent (server must be running):"
echo "  export DEEPGRAM_API_KEY=your_key"
echo "  export GROQ_API_KEY=your_key"
echo "  python3 pipeline.py"
