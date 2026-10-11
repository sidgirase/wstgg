"""
Unit tests for the 2-person work allocation and manifest management.
"""

import os
import tempfile
import pytest

from vlm_placement.utils.manifest import (
    generate_experiment_grid,
    ManifestManager,
)


def test_grid_cardinality_and_pairs():
    """
    Validates that the generated grid matches the project specification:
    - 2 primary models: InternVL3-2B, SmolVLM2-2.2B
    - 3 tasks: InfographicVQA, TabMWP, AI2D
    - 2 placements: 'lm', 'vision'
    - 3 seeds: 42, 43, 44
    Total = 2 x 3 x 2 x 3 = 36 runs.
    """
    grid = generate_experiment_grid(person1_name="Siddhesh", person2_name="Collaborator")
    assert len(grid) == 36

    # Verify 18 runs assigned to Person 1, 18 runs to Person 2
    p1_runs = [r for r in grid if r["assigned_to"] == "Siddhesh"]
    p2_runs = [r for r in grid if r["assigned_to"] == "Collaborator"]
    assert len(p1_runs) == 18
    assert len(p2_runs) == 18

    # Ensure all InternVL3 runs belong to Person 1
    assert all("internvl3" in r["model"] for r in p1_runs)
    # Ensure all SmolVLM runs belong to Person 2
    assert all("smolvlm" in r["model"] for r in p2_runs)


def test_manifest_manager_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_manifest.json")
        mgr = ManifestManager(path)
        mgr.initialize_if_missing(person1_name="Siddhesh", person2_name="Teammate")

        assert len(mgr.runs) == 36
        first_id = mgr.runs[0]["run_id"]

        # Initial status is pending
        assert mgr.get_run(first_id)["status"] == "pending"

        # Update to running
        mgr.update_status(first_id, "running")
        assert mgr.get_run(first_id)["status"] == "running"

        # Update to completed with peak VRAM
        mgr.update_status(first_id, "completed", peak_vram_gb=14.2)
        assert mgr.get_run(first_id)["status"] == "completed"
        assert mgr.get_run(first_id)["peak_vram_gb"] == 14.2

        # Check pending filtering
        pending = mgr.get_pending_runs()
        assert len(pending) == 35

        # Render summary markdown
        summary_md = mgr.render_markdown_summary()
        assert "Completed | 1" in summary_md
        assert "Pending | 35" in summary_md
