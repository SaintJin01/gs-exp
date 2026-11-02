#!/usr/bin/env bash
# Run in WSL Ubuntu: bash scripts/setup_ppisp_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA_EXE="${CONDA_EXE:-$HOME/anaconda3/bin/conda}"
eval "$("$CONDA_EXE" shell.bash hook)"
conda create -n 3dgs-ppisp python=3.10 pip -y
conda activate 3dgs-ppisp
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8 -y
# Ubuntu 22.04 GCC 11 is compatible with CUDA 11.8.
command -v g++ >/dev/null || { echo "Install build-essential first."; exit 1; }
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=4
python -m pip install "pip<26" "setuptools>=77,<81" wheel ninja
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4 plyfile==1.1.2 tqdm pillow opencv-python==4.10.0.84 tensorboard matplotlib
git submodule update --init --recursive submodules/simple-knn submodules/diff-gaussian-rasterization submodules/fused-ssim
python -m pip install --no-build-isolation ./submodules/simple-knn ./submodules/diff-gaussian-rasterization ./submodules/fused-ssim
python -m pip install --no-build-isolation "ppisp @ git+https://github.com/nv-tlabs/ppisp.git@a247368047651db2e03e6f97a36e4f5bebbe5c5f"
python scripts/test_ppisp_integration.py
echo "Ready. Run: conda activate 3dgs-ppisp"
