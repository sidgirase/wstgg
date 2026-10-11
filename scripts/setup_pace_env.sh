#!/bin/bash
# ==============================================================================
# One-click Automated Environment Setup for Georgia Tech PACE-ICE Cluster
# Installs directly into ~/scratch to avoid PACE /home [Errno 122] disk quota limits.
# ==============================================================================

set -e

echo "===================================================================="
echo " 1. Loading PACE Modules (Anaconda & CUDA 11.8)..."
echo "===================================================================="
module load anaconda3/2023.03
module load cuda/11.8

# PACE /home has a strict ~10GB quota. All large environments and caches MUST live in ~/scratch
SCRATCH_DIR="$HOME/scratch"
ENV_DIR="$SCRATCH_DIR/vlm_placement_env"

mkdir -p "$SCRATCH_DIR/.cache/pip"
export PIP_CACHE_DIR="$SCRATCH_DIR/.cache/pip"

echo ""
echo "===================================================================="
echo " 2. Setting up Conda Environment in scratch: $ENV_DIR..."
echo "===================================================================="
if [ -d "$ENV_DIR" ]; then
    echo "Environment at '$ENV_DIR' already exists. Reusing existing environment."
else
    echo "Creating clean Conda environment in scratch (Python 3.10)..."
    conda create -y -p "$ENV_DIR" python=3.10
fi

echo ""
echo "===================================================================="
echo " 3. Activating Environment..."
echo "===================================================================="
source activate "$ENV_DIR" || conda activate "$ENV_DIR"

echo ""
echo "===================================================================="
echo " 4. Installing PyTorch with CUDA 11.8 Support (direct to scratch)..."
echo "===================================================================="
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "===================================================================="
echo " 5. Installing Project Requirements..."
echo "===================================================================="
pip install --no-cache-dir -r requirements.txt

echo ""
echo "===================================================================="
echo " [SUCCESS] Environment is fully installed in scratch and ready!"
echo " Activate it anytime on PACE using:"
echo "     module load anaconda3/2023.03"
echo "     module load cuda/11.8"
echo "     source activate $ENV_DIR"
echo "===================================================================="
