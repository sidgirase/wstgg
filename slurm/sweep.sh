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
python -c "
import json, subprocess, os, sys

with open('$MANIFEST_PATH', 'r') as f:
    runs = json.load(f)

filter_assignee = '$FILTER_ASSIGNEE'
filter_model = '$FILTER_MODEL'

submitted = 0
for r in runs:
    if r['status'] == 'completed':
        continue
    if filter_assignee and r['assigned_to'] != filter_assignee:
        continue
    if filter_model and r['model'] != filter_model:
        continue

    run_id = r['run_id']
    model = r['model']
    task = r['task']
    placement = r['placement']
    seed = str(r['seed'])

    env = dict(os.environ)
    env['MODEL'] = model
    env['TASK'] = task
    env['PLACEMENT'] = placement
    env['SEED'] = seed
    env['RUN_ID'] = run_id

    export_vars = f"ALL,MODEL={model},TASK={task},PLACEMENT={placement},SEED={seed},RUN_ID={run_id}"
    cmd = [
        'sbatch',
        f'--job-name={run_id}',
        f'--export={export_vars}',
    ]
    if os.environ.get('PACE_ACCOUNT'):
        cmd.append(f"--account={os.environ['PACE_ACCOUNT']}")
    if os.environ.get('PACE_QOS'):
        cmd.append(f"--qos={os.environ['PACE_QOS']}")
    cmd.append('slurm/point.sbatch')

    print(f'Submitting: {run_id} (Assigned to: {r[\"assigned_to\"]})...')
    try:
        res = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        print(f'  -> {res.stdout.strip()}')
        submitted += 1
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.strip() if e.stderr else (e.stdout.strip() if e.stdout else str(e))
        print(f'  -> Failed to submit: {err_msg}')
    except Exception as e:
        print(f'  -> Failed to submit: {e}')

print(f'\nTotal jobs submitted: {submitted}')
"

echo "Done! Check queue with: squeue -u \$USER"
