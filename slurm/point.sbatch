#!/bin/bash
#SBATCH --job-name=vlm_point
#SBATCH --account=pace-ice
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err

# ==============================================================================
# Georgia Tech PACE-ICE Cluster Submission Script for Single VLM Placement Run
# ==============================================================================

set -e

# Default arguments or override via environment variables
MODEL=${MODEL:-"internvl3_2b"}
TASK=${TASK:-"infovqa"}
PLACEMENT=${PLACEMENT:-"lm"}
SEED=${SEED:-42}
STEPS=${STEPS:-500}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-2e-4}
RUN_ID=${RUN_ID:-"run_${MODEL}_${TASK}_${PLACEMENT}_s${SEED}"}
OUTPUT_DIR=${OUTPUT_DIR:-"./runs/${RUN_ID}"}

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "===================================================================="
echo " Starting VLM Placement Run on PACE-ICE"
echo " Host: $(hostname)"
echo " GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'N/A')"
echo " Model:     $MODEL"
echo " Task:      $TASK"
echo " Placement: $PLACEMENT"
echo " Seed:      $SEED"
echo " Run ID:    $RUN_ID"
echo "===================================================================="

echo "Loading PACE Modules..."
module load anaconda3/2023.03
module load cuda/11.8

echo "Activating Conda Environment..."
if [ -d "$HOME/scratch/vlm_placement_env" ]; then
    source activate "$HOME/scratch/vlm_placement_env" || conda activate "$HOME/scratch/vlm_placement_env"
else
    source activate vlm_placement_env || conda activate vlm_placement_env
fi

echo "Executing Training..."
python scripts/run_train.py train \
    --model "$MODEL" \
    --task "$TASK" \
    --placement "$PLACEMENT" \
    --seed "$SEED" \
    --max_steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --lr "$LR" \
    --mixed_precision "bf16" \
    --run_id "$RUN_ID" \
    --output_dir "$OUTPUT_DIR" \
    --manifest_path "./runs/manifest.json"

echo "===================================================================="
echo " Run $RUN_ID completed successfully!"
echo " Results and metrics committed to $OUTPUT_DIR"
echo "===================================================================="
