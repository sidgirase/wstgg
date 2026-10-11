# wstgg (Where Should the Gradient Go?)
### Aligning LoRA Placement with Pre-Training Perception/Reasoning Profiles in Small VLMs

**CS 8803-VLM Fall 2026 · Final Project**  
**Lane 2 / Lane B: Stage B Placement, Parameter Matching, and Training Framework**

---

## 1. Executive Summary & Research Goal

When adapting a small Vision-Language Model (VLM, ~2B parameters) to downstream reasoning tasks under a strict computation and parameter budget, practitioners face a binary structural decision:
1. **Language-Side Adaptation (`lm`)**: Freeze the entire vision pathway (encoder + connector) and apply LoRA exclusively to the language decoder.
2. **Vision-Side Adaptation (`vision`)**: Freeze the language decoder completely and apply LoRA exclusively to the vision encoder and connector.

Our project investigates whether an inference-only **oracle-text rescue diagnostic** measured *before* training can reliably predict which parameter locus yields higher accuracy under a strictly matched parameter budget.

### Separation from Concurrent Work (§0a Rules)
To guarantee strict independence from concurrent distillation research:
* **Excluded Benchmarks**: Under no circumstances do we train on or evaluate on **ChartQA** or **Geometry3K**.
* **Target Benchmarks**: **InfographicVQA**, **TabMWP**, and **AI2D** (with **DocVQA** held out for out-of-distribution evaluation).
* **Target Models**: **InternVL3-2B** (`OpenGVLab/InternVL2_5-2B` / `InternVL3-2B`), **SmolVLM2-2.2B** (`HuggingFaceTB/SmolVLM2-2.2B-Instruct`), and **Qwen3-VL-2B** (`Qwen/Qwen2-VL-2B-Instruct`).
* **Supervision**: Standard answer-only SFT; **no teacher models**, no distilled supervision.

---

## 2. Two-Person Collaboration Architecture (Lane 2 Division)

To allow 2 teammates to collaborate efficiently without stepping on each other's runs or duplicating compute on the Georgia Tech PACE cluster, the 36-run experimental grid (6 pairs × 2 placements × 3 seeds) is partitioned into clear ownership tracks with cross-training capability.

### Work Partition

| Teammate | Primary Model Ownership | Benchmark Tasks | Seeds | Placements | Total Runs |
|---|---|---|---|---|---|
| **Team Member 1** | **InternVL3-2B** | InfographicVQA, TabMWP, AI2D | 42, 43, 44 | `lm`, `vision` | **18 runs** |
| **Team Member 2** | **SmolVLM2-2.2B** | InfographicVQA, TabMWP, AI2D | 42, 43, 44 | `lm`, `vision` | **18 runs** |
| **Shared / Expansion** | **Qwen3-VL-2B** | InfographicVQA, TabMWP, AI2D | 42, 43, 44 | `lm`, `vision` | As compute permits |

### Collaboration Protocol
1. **Isolated Output Paths**: Each run has a deterministic output folder: `./runs/run_<index>_<model>_<task>_<placement>_s<seed>/`.
2. **Shared State Manifest**: All runs are tracked in `runs/manifest.json`. You can check progress or list pending runs anytime via:
   ```bash
   python scripts/manifest_tool.py status
   ```
   *(Note: `manifest_tool.py` is pure Python and has zero external dependencies, so it runs directly on login nodes without requiring PyTorch).*
3. **Cross-Training Assistance**: If Team Member 2 finishes their runs early or Team Member 1 needs assistance running remaining seeds on PACE, either teammate can run:
   ```bash
   # Filter and submit only pending runs assigned to a specific teammate or model:
   bash slurm/sweep.sh "Person_1" "internvl3_2b"
   bash slurm/sweep.sh "Person_2" "smolvlm2_2b"
   ```
4. **Resumable Execution**: If a PACE job hits walltime or fails, re-running the sweep script will automatically resume from the latest checkpoint without re-running completed seeds.

---

## 3. Repository Structure

```
wstgg/
├── .gitignore                      # Git ignore patterns (ignores checkpoints, runs, data)
├── README.md                       # Comprehensive project & navigation guide
├── requirements.txt                # Python dependencies
├── environment_pace.yml            # Conda environment definition for PACE-ICE
├── docs/                           # Proposals, slides, and team todos
│   ├── VLM Project Proposal1.pdf   # Official presentation slides
│   ├── proposal_draft.pdf          # Full proposal document
│   └── team_todo.md                # Four-lane work breakdown and milestones
├── vlm_placement/                  # Core Python package
│   ├── __init__.py
│   ├── cli.py                      # Unified CLI entry point
│   ├── modeling/
│   │   ├── adapter.py              # LoRALinear injection & check_trainable invariants
│   │   ├── builder.py              # Model loader with synthetic test architectures
│   │   ├── matcher.py              # Parameter-count rank matcher (<= 10% budget)
│   │   └── module_maps.py          # Layer naming definitions for all 3 models
│   ├── data/
│   │   ├── collator.py             # Dynamic resolution, 768px cap, answer-only masking
│   │   └── datasets.py             # InfographicVQA, TabMWP, AI2D & smoke generator
│   ├── training/
│   │   ├── trainer.py              # Unified trainer with gradient checkpointing
│   │   └── metrics.py              # Telemetry tracker (peak VRAM GB, step latency)
│   └── utils/
│       └── manifest.py             # 36-run grid generator and progress tracker
├── scripts/                        # Standalone utility scripts
│   ├── setup_pace_env.sh           # One-click environment installer for PACE-ICE
│   ├── run_smoke.py                # 100-sample smoke test (Week 6 milestone)
│   ├── run_train.py                # Direct training launcher
│   ├── match_ranks.py              # Rank matching solver tool
│   └── manifest_tool.py            # Manifest query & status tool (zero dependencies)
├── slurm/                          # Georgia Tech PACE-ICE SLURM submission scripts
│   ├── point.sbatch                # Single run SLURM job
│   ├── sweep.sh                    # Master resumable sweep runner
│   ├── sweep_person1.sh            # Dedicated sweep for Team Member 1 (InternVL3-2B)
│   ├── sweep_person2.sh            # Dedicated sweep for Team Member 2 (SmolVLM2-2.2B)
│   └── profile_vram.sbatch         # VRAM and step latency profiling job
└── tests/                          # Automated unit test suite
    ├── test_modeling.py            # Invariant tests and rank-matching tests
    ├── test_data.py                # Image cap, token bounds, and label masking tests
    └── test_manifest.py            # 2-person grid partition tests
```

---

## 4. Setup Guide

### Local Development Setup
```bash
# 1. Clone repository and navigate to root
git clone https://github.com/<username>/wstgg.git
cd wstgg

# 2. Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

### Georgia Tech PACE-ICE Cluster Setup

On PACE-ICE, conda solver conflicts can occur if attempting multi-channel resolutions across read-only system modules. We provide a **one-click automated setup script** that installs everything cleanly:

```bash
# 1. SSH into PACE login node
ssh <username>@login-ice.pace.gatech.edu

# 2. Navigate to your scratch directory and clone
cd ~/scratch
git clone https://github.com/<username>/wstgg.git
cd wstgg

# 3. Run the one-click PACE environment installer
bash scripts/setup_pace_env.sh
```

#### Manual PACE Setup (Alternative):
If you prefer running commands manually step-by-step:
```bash
module load anaconda3/2023.03
module load cuda/11.8

# Create environment
conda create -y -n vlm_placement_env python=3.10
source activate vlm_placement_env

# Install PyTorch with CUDA 11.8 and requirements
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## 5. Quick Start & Verification

### 1. Check Experiment Status (Zero-Dependency)
On any machine or directly on the PACE login node:
```bash
python scripts/manifest_tool.py status
```

### 2. Verify Parameter Matching (10% Budget Constraint)
Before training, verify that the LoRA rank matcher finds balanced trainable parameter counts within 10% for both placements:
```bash
python scripts/match_ranks.py --model all --lm_rank 16 --tolerance 0.10
```
*Expected Output:* Shows that for each model family, the vision rank is adjusted such that $|N_{\text{vision}} - N_{\text{lm}}| / N_{\text{lm}} \le 10\%$.

### 3. Run the Week 6 Smoke Test
Test the full pipeline end-to-end on 100 InfographicVQA samples (1 seed) for both `lm` and `vision` placements:
```bash
python scripts/run_smoke.py --model internvl3_2b
```

### 4. Run the Unit Test Suite
```bash
pytest tests/ -v
```

---

## 6. Training Execution Guide

### Direct Training Commands (Single Run)

#### Train Language-Side LoRA (`lm`):
```bash
python scripts/run_train.py train \
    --model internvl3_2b \
    --task infovqa \
    --placement lm \
    --seed 42 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr 2e-4 \
    --max_steps 500
```

#### Train Vision-Side LoRA (`vision`):
```bash
python scripts/run_train.py train \
    --model internvl3_2b \
    --task infovqa \
    --placement vision \
    --seed 42 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr 2e-4 \
    --max_steps 500
```

*Note:* When `--rank` is omitted, the trainer automatically queries the parameter matcher to select the rank that balances trainable parameters within 10%!

---

## 7. Running on Georgia Tech PACE-ICE

### 1. Submit a Single Job via SLURM
Submit an individual run with custom environment variables:
```bash
sbatch --export=ALL,MODEL=internvl3_2b,TASK=infovqa,PLACEMENT=vision,SEED=42 slurm/point.sbatch
```

### 2. Launch 2-Person Sweep Queues
Each teammate can submit their assigned 18 runs with a single command:
```bash
# Team Member 1 (InternVL3-2B runs):
bash slurm/sweep_person1.sh

# Team Member 2 (SmolVLM2-2.2B runs):
bash slurm/sweep_person2.sh
```

### 3. Monitor PACE Jobs and Progress
```bash
# Check running SLURM jobs
squeue -u $USER

# Check project experiment grid progress
python scripts/manifest_tool.py status
```

---

## 8. PACE Troubleshooting Guide

1. **`EnvironmentNameNotFound: Could not find conda environment: vlm_placement_env`**:
   * *Cause*: Conda environment was not created yet or `conda env create` failed on solver conflicts.
   * *Fix*: Run `bash scripts/setup_pace_env.sh`. It cleanly creates the environment and installs PyTorch CUDA wheels without solver crashes.
2. **`ModuleNotFoundError: No module named 'torch'` on Login Node**:
   * *Cause*: Trying to run training scripts or importing packages that load PyTorch before activating the conda environment on the login node.
   * *Fix*:
     * Use `python scripts/manifest_tool.py status` for monitoring (which has zero external dependencies).
     * Activate the environment before running training commands: `source activate vlm_placement_env`.
3. **CUDA Out of Memory (OOM) on Vision-Side LoRA**:
   * Vision-side LoRA has higher activation memory under dynamic image resolutions.
   * *Fix 1*: Gradient checkpointing is enabled by default. Ensure it is not disabled.
   * *Fix 2*: Reduce batch size and increase gradient accumulation steps (e.g. `--batch_size 2 --gradient_accumulation_steps 8`).
   * *Fix 3*: Use the fallback flag `--last_k_vision_blocks 4` to adapt only the connector and top visual blocks.
4. **SLURM Account or Partition Errors**:
   * If `#SBATCH --account=pace-ice` fails on your PACE allocation, edit `slurm/point.sbatch` and replace with your class account (e.g. `--account=gts-zkira3`).

---

## 9. Deliverables & Handoff to Lane C

Upon completion of each training run, the output directory `./runs/<run_id>/` contains:
1. `run_config.json`: Committed run configuration, realized trainable parameter counts, matched ranks.
2. `run_metrics.json`: Final telemetry summary, including peak VRAM (GB) and average step latency (ms).
3. `step_trace.csv`: Per-step loss, time, and memory trace.
4. `checkpoint_step_*.pt`: Saved adapter weights for evaluation by Lane C.

These artifacts are directly consumed by Lane C's scorer (`evaluate.py`) to generate the midterm results table and confidence intervals.
