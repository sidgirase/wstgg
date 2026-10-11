#!/bin/bash
#SBATCH --job-name=vlm_profile
#SBATCH --account=pace-ice
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/profile_%j.out
#SBATCH --error=logs/profile_%j.err

# ==============================================================================
# VRAM Profiling on PACE GPU: compares peak memory and step time between LM and Vision
# ==============================================================================

set -e

mkdir -p logs
mkdir -p ./runs/profiling

echo "Loading PACE Modules..."
module load anaconda3/2023.03
module load cuda/11.8

echo "Activating Conda Environment..."
if [ -d "$HOME/scratch/vlm_placement_env" ]; then
    source activate "$HOME/scratch/vlm_placement_env" || conda activate "$HOME/scratch/vlm_placement_env"
else
    source activate vlm_placement_env || conda activate vlm_placement_env
fi

echo "Profiling Model Ranks..."
python scripts/match_ranks.py --model all

echo "Profiling InternVL3-2B Placements..."
python scripts/run_smoke.py --model internvl3_2b

echo "Profiling SmolVLM2-2.2B Placements..."
python scripts/run_smoke.py --model smolvlm2_2b

echo "Profiling Qwen3-VL-2B Placements..."
python scripts/run_smoke.py --model qwen3_vl_2b

echo "Profiling finished! Inspect ./runs/smoke_* for peak VRAM and step timings."
