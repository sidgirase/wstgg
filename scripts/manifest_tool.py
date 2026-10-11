#!/usr/bin/env python3
"""
CLI helper for viewing and updating the 2-person experiment manifest.
Pure standard-library implementation (runs anywhere without torch/transformers installed).
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vlm_placement.utils.manifest import ManifestManager


import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment Manifest Manager (Zero External Dependencies)")
    parser.add_argument("action", choices=["init", "status", "pending", "submit"], default="status", nargs="?")
    parser.add_argument("--manifest_path", type=str, default="./runs/manifest.json")
    parser.add_argument("--person1", type=str, default="Person_1")
    parser.add_argument("--person2", type=str, default="Person_2")
    parser.add_argument("--include_expansion", action="store_true")
    parser.add_argument("--assigned_to", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--sbatch_script", type=str, default="slurm/point.sbatch")
    args = parser.parse_args()

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
            mgr.initialize_if_missing(person1_name=args.person1, person2_name=args.person2)
        else:
            mgr.load()
        print(mgr.render_markdown_summary())

    elif args.action == "pending":
        if not os.path.exists(args.manifest_path):
            mgr.initialize_if_missing(person1_name=args.person1, person2_name=args.person2)
        else:
            mgr.load()
        pending = mgr.get_pending_runs(assigned_to=args.assigned_to, model=args.model)
        print(f"Pending runs ({len(pending)}):")
        for r in pending:
            print(f"  - [{r['run_id']}] {r['model']} | {r['task']} | {r['placement']} | seed={r['seed']} (Assigned: {r['assigned_to']})")

    elif args.action == "submit":
        if not os.path.exists(args.manifest_path):
            mgr.initialize_if_missing(person1_name=args.person1, person2_name=args.person2)
        else:
            mgr.load()
        pending = mgr.get_pending_runs(assigned_to=args.assigned_to, model=args.model)
        print(f"Submitting {len(pending)} pending jobs to SLURM...")
        submitted = 0
        for r in pending:
            run_id = r["run_id"]
            model = r["model"]
            task = r["task"]
            placement = r["placement"]
            seed = str(r["seed"])

            env = dict(os.environ)
            env["MODEL"] = model
            env["TASK"] = task
            env["PLACEMENT"] = placement
            env["SEED"] = seed
            env["RUN_ID"] = run_id

            export_vars = f"ALL,MODEL={model},TASK={task},PLACEMENT={placement},SEED={seed},RUN_ID={run_id}"
            cmd = [
                "sbatch",
                f"--job-name={run_id}",
                f"--export={export_vars}",
            ]
            if os.environ.get("PACE_ACCOUNT"):
                cmd.append(f"--account={os.environ['PACE_ACCOUNT']}")
            if os.environ.get("PACE_QOS"):
                cmd.append(f"--qos={os.environ['PACE_QOS']}")
            cmd.append(args.sbatch_script)

            print(f"Submitting: {run_id} (Assigned to: {r['assigned_to']})...")
            try:
                res = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
                print(f"  -> {res.stdout.strip()}")
                submitted += 1
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr.strip() if e.stderr else (e.stdout.strip() if e.stdout else str(e))
                print(f"  -> Failed to submit: {err_msg}")
            except Exception as e:
                print(f"  -> Failed to submit: {e}")
        print(f"\nTotal jobs submitted: {submitted}")


if __name__ == "__main__":
    main()
