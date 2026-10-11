"""
Unified Command Line Interface for VLM Placement Experiments.

Commands:
- train: Train a single (model, task, placement, seed) configuration
- match-ranks: Compute matched LoRA ranks within 10% budget constraint
- smoke: Run 100-sample end-to-end smoke test for both placements
- manifest: Manage 36-run experiment grid and 2-person work assignments
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional

from .modeling.module_maps import MODEL_SPECS
from .utils.manifest import ManifestManager, DEFAULT_MODELS, TASKS, PLACEMENTS, SEEDS


def cmd_match_ranks(args: argparse.Namespace) -> None:
    """Pre-calculates matched vision ranks for a model or all models."""
    from .modeling.builder import load_vlm_model
    from .modeling.matcher import match_lora_ranks

    models_to_check = list(MODEL_SPECS.keys()) if args.model == "all" else [args.model]

    print("=" * 80)
    print(f" LoRA Parameter-Count Matching (Target LM Rank = {args.lm_rank}, Max Tolerance = {args.tolerance*100}%)")
    print("=" * 80)

    for m in models_to_check:
        model = load_vlm_model(m, use_mock=True)
        res = match_lora_ranks(model, m, lm_rank=args.lm_rank, tolerance=args.tolerance)

        print(f"\nModel: {m}")
        print(f"  LM Placement:     Rank {res['lm_rank']:2d} -> {res['lm_trainable_params']:,} trainable parameters")
        print(f"  Vision Placement: Rank {res['vision_rank']:2d} -> {res['vision_trainable_params']:,} trainable parameters")
        print(f"  Difference:       {res['percentage_difference']}% ({'PASSED <= 10%' if res['matched_within_tolerance'] else 'FAILED > 10%'})")
        print(f"  Result:           {res['message']}")

    print("\n" + "=" * 80)


def cmd_smoke(args: argparse.Namespace) -> None:
    """Runs a 100-sample smoke run for both placements (Week 6 deliverable)."""
    from torch.utils.data import DataLoader
    from .modeling.builder import load_vlm_model
    from .modeling.adapter import apply_lora_to_model, get_trainable_param_stats
    from .modeling.matcher import match_lora_ranks
    from .data.datasets import get_smoke_dataset
    from .data.collator import VLMDataCollator
    from .training.trainer import VLMTrainer, set_seed

    print("=" * 80)
    print(" Running Week 6 Milestone Smoke Test (100 InfographicVQA samples, 1 seed)")
    print("=" * 80)

    model_name = args.model
    seed = 42
    max_steps = 10
    task = "infovqa"

    for placement in ["lm", "vision"]:
        print(f"\n--- Testing placement='{placement}' on {model_name} ---")
        set_seed(seed)
        model = load_vlm_model(model_name, use_mock=True)

        # Match rank
        match_info = match_lora_ranks(model, model_name, lm_rank=16, tolerance=0.10)
        rank = match_info["lm_rank"] if placement == "lm" else match_info["vision_rank"]

        # Apply LoRA
        apply_lora_to_model(model, placement=placement, model_name=model_name, rank=rank)
        stats = get_trainable_param_stats(model, model_name)
        print(f"  Trainable params: {stats['trainable_params']:,} ({stats['trainable_percent']:.2f}% of total)")

        # Prepare 100-sample smoke dataset
        dataset = get_smoke_dataset(num_samples=100, task=task)
        collator = VLMDataCollator()
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator)

        output_dir = f"./runs/smoke_{model_name}_{placement}"
        trainer = VLMTrainer(
            model=model,
            train_dataloader=dataloader,
            placement=placement,
            model_name=model_name,
            output_dir=output_dir,
            run_config={
                "is_smoke_run": True,
                "task": task,
                "seed": seed,
                "rank": rank,
                "match_info": match_info,
            },
        )
        res = trainer.train(max_steps=max_steps, checkpoint_every_steps=max_steps)
        print(f"  [SUCCESS] Smoke run for '{placement}' passed. Peak RAM: {res['telemetry_summary']['peak_memory_mb']} MB")

    print("\n" + "=" * 80)
    print(" [PASSED] Both LM and Vision smoke runs completed successfully!")
    print("=" * 80)


def cmd_train(args: argparse.Namespace) -> None:
    """Executes a single training run from the grid."""
    from torch.utils.data import DataLoader
    from .modeling.builder import load_vlm_model
    from .modeling.adapter import apply_lora_to_model, get_trainable_param_stats
    from .modeling.matcher import match_lora_ranks
    from .data.datasets import load_dataset_for_task
    from .data.collator import VLMDataCollator
    from .training.trainer import VLMTrainer, set_seed

    print("=" * 80)
    print(f" Launching Run: Model={args.model} | Task={args.task} | Placement={args.placement} | Seed={args.seed}")
    print("=" * 80)

    set_seed(args.seed)

    # 1. Load base architecture
    model = load_vlm_model(args.model, use_mock=args.use_mock)

    # 2. Determine rank (matched or custom)
    match_info = match_lora_ranks(model, args.model, lm_rank=args.lm_rank, tolerance=0.10)
    if args.rank is not None:
        rank = args.rank
    else:
        rank = match_info["lm_rank"] if args.placement == "lm" else match_info["vision_rank"]

    # 3. Apply LoRA and verify invariants
    apply_lora_to_model(
        model=model,
        placement=args.placement,
        model_name=args.model,
        rank=rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        last_k_vision_blocks=args.last_k_vision_blocks,
    )
    param_stats = get_trainable_param_stats(model, args.model)
    print(f"[ModelConfig] Realized trainable parameters: {param_stats['trainable_params']:,}")

    # 4. Load dataset
    dataset = load_dataset_for_task(
        task=args.task,
        split="train",
        max_samples=args.max_samples,
        seed=args.seed,
        use_synthetic_fallback=True,
    )
    collator = VLMDataCollator()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    # 5. Output directory & run ID
    run_id = args.run_id or f"run_{args.model}_{args.task}_{args.placement}_s{args.seed}"
    output_dir = args.output_dir or f"./runs/{run_id}"

    # Manifest update to 'running'
    manifest_mgr = ManifestManager(args.manifest_path)
    if os.path.exists(args.manifest_path):
        manifest_mgr.load()
        manifest_mgr.update_status(run_id, "running")

    # 6. Initialize trainer and run
    run_config = {
        "run_id": run_id,
        "model": args.model,
        "task": args.task,
        "placement": args.placement,
        "seed": args.seed,
        "rank": rank,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "match_info": match_info,
    }

    try:
        trainer = VLMTrainer(
            model=model,
            train_dataloader=dataloader,
            placement=args.placement,
            model_name=args.model,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            enable_gradient_checkpointing=not args.disable_gradient_checkpointing,
            mixed_precision=args.mixed_precision,
            output_dir=output_dir,
            run_config=run_config,
        )

        res = trainer.train(
            max_steps=args.max_steps,
            checkpoint_every_steps=args.checkpoint_every_steps,
            resume_from_checkpoint=args.resume_checkpoint,
        )

        peak_vram = res["telemetry_summary"]["peak_memory_gb"]
        if os.path.exists(args.manifest_path):
            manifest_mgr.update_status(run_id, "completed", peak_vram_gb=peak_vram)

        print(f"[Run Success] {run_id} completed successfully. Peak VRAM: {peak_vram} GB")

    except Exception as e:
        if os.path.exists(args.manifest_path):
            manifest_mgr.update_status(run_id, "failed", error_message=str(e))
        print(f"[Run Failed] Error executing {run_id}: {e}")
        raise e


def cmd_manifest(args: argparse.Namespace) -> None:
    """Manages the experiment manifest for 2-person collaboration."""
    mgr = ManifestManager(args.manifest_path)

    if args.action == "init":
        mgr.initialize_if_missing(
            person1_name=args.person1,
            person2_name=args.person2,
            include_expansion_models=args.include_expansion,
        )
        print(f"Manifest initialized at {args.manifest_path} with {len(mgr.runs)} runs.")

    elif args.action == "status":
        if not os.path.exists(args.manifest_path):
            mgr.initialize_if_missing()
        else:
            mgr.load()
        print(mgr.render_markdown_summary())

    elif args.action == "pending":
        if not os.path.exists(args.manifest_path):
            mgr.initialize_if_missing()
        else:
            mgr.load()
        pending = mgr.get_pending_runs(assigned_to=args.assigned_to, model=args.model)
        print(f"Pending runs ({len(pending)}):")
        for r in pending:
            print(f"  - [{r['run_id']}] {r['model']} | {r['task']} | {r['placement']} | seed={r['seed']} (Assigned: {r['assigned_to']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM LoRA Placement & Training Framework (Stage B)")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. match-ranks
    p_match = subparsers.add_parser("match-ranks", help="Calculate matched LoRA ranks within 10% budget")
    p_match.add_argument("--model", type=str, default="all", choices=list(MODEL_SPECS.keys()) + ["all"])
    p_match.add_argument("--lm_rank", type=int, default=16, help="Reference LM LoRA rank")
    p_match.add_argument("--tolerance", type=float, default=0.10, help="Max budget tolerance (default: 0.10)")

    # 2. smoke
    p_smoke = subparsers.add_parser("smoke", help="Run 100-sample smoke test for both placements")
    p_smoke.add_argument("--model", type=str, default="internvl3_2b", choices=list(MODEL_SPECS.keys()))

    # 3. train
    p_train = subparsers.add_parser("train", help="Run a single training configuration")
    p_train.add_argument("--model", type=str, required=True, choices=list(MODEL_SPECS.keys()))
    p_train.add_argument("--task", type=str, required=True, choices=["infovqa", "tabmwp", "ai2d", "docvqa"])
    p_train.add_argument("--placement", type=str, required=True, choices=["lm", "vision"])
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--rank", type=int, default=None, help="Custom LoRA rank (auto-matches if not set)")
    p_train.add_argument("--lm_rank", type=int, default=16, help="Reference LM rank for auto-matching")
    p_train.add_argument("--lora_alpha", type=float, default=None)
    p_train.add_argument("--lora_dropout", type=float, default=0.05)
    p_train.add_argument("--last_k_vision_blocks", type=int, default=None, help="Fallback: adapt only last k blocks")
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--weight_decay", type=float, default=0.01)
    p_train.add_argument("--batch_size", type=int, default=4)
    p_train.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p_train.add_argument("--max_steps", type=int, default=200)
    p_train.add_argument("--checkpoint_every_steps", type=int, default=50)
    p_train.add_argument("--max_samples", type=int, default=None)
    p_train.add_argument("--disable_gradient_checkpointing", action="store_true")
    p_train.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p_train.add_argument("--use_mock", action="store_true", help="Use lightweight architecture mock for testing")
    p_train.add_argument("--run_id", type=str, default=None)
    p_train.add_argument("--output_dir", type=str, default=None)
    p_train.add_argument("--resume_checkpoint", type=str, default=None)
    p_train.add_argument("--manifest_path", type=str, default="./runs/manifest.json")

    # 4. manifest
    p_manifest = subparsers.add_parser("manifest", help="Manage 2-person experiment tracking manifest")
    p_manifest.add_argument("action", choices=["init", "status", "pending"])
    p_manifest.add_argument("--manifest_path", type=str, default="./runs/manifest.json")
    p_manifest.add_argument("--person1", type=str, default="Person_1")
    p_manifest.add_argument("--person2", type=str, default="Person_2")
    p_manifest.add_argument("--include_expansion", action="store_true")
    p_manifest.add_argument("--assigned_to", type=str, default=None)
    p_manifest.add_argument("--model", type=str, default=None)

    args = parser.parse_args()

    if args.subcommand == "match-ranks":
        cmd_match_ranks(args)
    elif args.subcommand == "smoke":
        cmd_smoke(args)
    elif args.subcommand == "train":
        cmd_train(args)
    elif args.subcommand == "manifest":
        cmd_manifest(args)


if __name__ == "__main__":
    main()
