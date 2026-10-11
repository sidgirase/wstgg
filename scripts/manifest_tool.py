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


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment Manifest Manager (Zero External Dependencies)")
    parser.add_argument("action", choices=["init", "status", "pending"], default="status", nargs="?")
    parser.add_argument("--manifest_path", type=str, default="./runs/manifest.json")
    parser.add_argument("--person1", type=str, default="Person_1")
    parser.add_argument("--person2", type=str, default="Person_2")
    parser.add_argument("--include_expansion", action="store_true")
    parser.add_argument("--assigned_to", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
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


if __name__ == "__main__":
    main()
