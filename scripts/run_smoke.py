#!/usr/bin/env python3
"""
Executes the Week 6 milestone smoke test:
Runs 100 InfographicVQA samples on InternVL3-2B for both 'lm' and 'vision' placements (1 seed)
to prove the pipeline end-to-end.
"""

import sys
import os

# Ensure package is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vlm_placement.cli import cmd_smoke
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 100-sample smoke test")
    parser.add_argument("--model", type=str, default="internvl3_2b", choices=["internvl3_2b", "smolvlm2_2b", "qwen3_vl_2b"])
    args = parser.parse_args()
    cmd_smoke(args)
