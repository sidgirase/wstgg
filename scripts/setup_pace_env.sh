#!/bin/bash
# ==============================================================================
# One-click Automated Environment Setup for Georgia Tech PACE-ICE Cluster
# ==============================================================================

set -e

echo "===================================================================="
echo " 1. Loading PACE Modules (Anaconda & CUDA 11.8)..."
echo "===================================================================="
module load anaconda3/2023.03
module load cuda/11.8

ENV_NAME="vlm_placement_env"

echo ""
echo "===================================================================="
echo " 2. Setting up Conda Environment: $ENV_NAME (Python 3.10)..."
echo "===================================================================="
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Reusing existing environment."
else
    echo "Creating clean Conda environment '$ENV_NAME'..."
    conda create -y -n "$ENV_NAME" python=3.10
fi

echo ""
echo "===================================================================="
echo " 3. Activating Environment..."
echo "===================================================================="
source activate "$ENV_NAME" || conda activate "$ENV_NAME"

echo ""
echo "===================================================================="
echo " 4. Installing PyTorch with CUDA 11.8 Support..."
echo "===================================================================="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "===================================================================="
echo " 5. Installing Project Requirements..."
echo "===================================================================="
pip install -r requirements.txt

echo ""
echo "===================================================================="
echo " [SUCCESS] Environment '$ENV_NAME' is fully installed and ready!"
echo " Activate it anytime on PACE using:"
echo "     module load anaconda3/2023.03"
echo "     module load cuda/11.8"
echo "     source activate $ENV_NAME"
echo "===================================================================="
