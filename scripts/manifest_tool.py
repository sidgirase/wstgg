#!/usr/bin/env python3
"""
CLI helper for viewing and updating the 2-person experiment manifest.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vlm_placement.cli import cmd_manifest
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Manifest Manager")
    parser.add_argument("action", choices=["init", "status", "pending"], default="status", nargs="?")
    parser.add_argument("--manifest_path", type=str, default="./runs/manifest.json")
    parser.add_argument("--person1", type=str, default="Person_1")
    parser.add_argument("--person2", type=str, default="Person_2")
    parser.add_argument("--include_expansion", action="store_true")
    parser.add_argument("--assigned_to", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    cmd_manifest(args)
