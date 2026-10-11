#!/usr/bin/env python3
"""
Solves and reports matched LoRA ranks across all candidate architectures within 10% budget.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vlm_placement.cli import cmd_match_ranks
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match LoRA ranks within 10% tolerance")
    parser.add_argument("--model", type=str, default="all", choices=["internvl3_2b", "smolvlm2_2b", "qwen3_vl_2b", "all"])
    parser.add_argument("--lm_rank", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=0.10)
    args = parser.parse_args()
    cmd_match_ranks(args)
