#!/usr/bin/env bash
# setup_env.sh — Fast environment bootstrap with uv
# Usage: bash setup_env.sh
set -euo pipefail

# ---- System deps ----
sudo apt-get update -y
sudo apt-get install -y git git-lfs
git lfs install

# ---- Core wheels ----
# If PyTorch is not present, install a CUDA 12 build (adjust if your base image differs).
# Comment this line if your image already includes torch.
# uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 "torch>=2.3,<2.6" torchvision torchaudio
uv venv .venv
source .venv/bin/activate

# ---- Training stack ----
uv pip install --upgrade pip
uv pip install "unsloth[zoo]" trl peft datasets accelerate bitsandbytes

# ---- Optional: FlashAttention (big speedups on RTX 50xx/Ampere+)
# If this fails on your image, just skip it; everything else will still work.
uv pip install --no-build-isolation flash-attn || echo "[warn] flash-attn install failed — continuing without it."

echo
echo "✅ Environment ready."
echo "Activate with: source .venv/bin/activate"
python -c "import torch;print('PyTorch:', torch.__version__);print('CUDA available:', torch.cuda.is_available())"