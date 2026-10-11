import sys
import os

# Add repo root to sys.path so tests can import vlm_placement from any working directory
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
