#!/bin/bash
# ==============================================================================
# Resumable PACE-ICE SLURM Grid Sweep Launcher
# Submits jobs for all pending/failed runs in the experiment manifest.
# Skips already completed runs.
# ==============================================================================

set -e

mkdir -p logs
mkdir -p runs

MANIFEST_PATH="./runs/manifest.json"
FILTER_ASSIGNEE=${1:-""}
FILTER_MODEL=${2:-""}

echo "Initializing / loading manifest at $MANIFEST_PATH..."
python scripts/manifest_tool.py init --manifest_path "$MANIFEST_PATH"

echo "Current status of experiment grid:"
python scripts/manifest_tool.py status --manifest_path "$MANIFEST_PATH"

echo ""
echo "Submitting jobs to SLURM queue on PACE-ICE..."

# Read pending runs and submit each as an independent SLURM job
SUBMIT_ARGS=()
if [ -n "$FILTER_ASSIGNEE" ]; then
    SUBMIT_ARGS+=(--assigned_to "$FILTER_ASSIGNEE")
fi
if [ -n "$FILTER_MODEL" ]; then
    SUBMIT_ARGS+=(--model "$FILTER_MODEL")
fi

python scripts/manifest_tool.py submit \
    --manifest_path "$MANIFEST_PATH" \
    --sbatch_script "slurm/point.sbatch" \
    "${SUBMIT_ARGS[@]}"

echo "Done! Check queue with: squeue -u \$USER"
