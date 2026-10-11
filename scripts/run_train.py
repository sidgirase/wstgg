#!/usr/bin/env python3
"""
Direct training launcher for VLM placement experiments.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vlm_placement.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    main()
