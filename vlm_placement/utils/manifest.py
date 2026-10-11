"""
Run manifest and work distribution system for 2-person collaboration.
Manages the 36-run experimental grid (6 pairs x 2 placements x 3 seeds),
tracks execution state, and allows seamless handoff and cross-training between teammates.
"""

import json
import os
from typing import Dict, List, Optional, Any


DEFAULT_MODELS = ["internvl3_2b", "smolvlm2_2b"]
EXPANSION_MODELS = ["qwen3_vl_2b"]
TASKS = ["infovqa", "tabmwp", "ai2d"]
PLACEMENTS = ["lm", "vision"]
SEEDS = [42, 43, 44]  # 3 distinct seeds


def generate_experiment_grid(
    models: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    placements: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    person1_name: str = "Person_1",
    person2_name: str = "Person_2",
) -> List[Dict[str, Any]]:
    """
    Generates the complete 36-run experiment grid partitioned between two teammates.
    By default:
    - Person 1 owns InternVL3-2B (18 runs: 3 tasks x 2 placements x 3 seeds)
    - Person 2 owns SmolVLM2-2.2B (18 runs: 3 tasks x 2 placements x 3 seeds)
    """
    models = models or DEFAULT_MODELS
    tasks = tasks or TASKS
    placements = placements or PLACEMENTS
    seeds = seeds or SEEDS

    grid = []
    idx = 1

    for model in models:
        for task in tasks:
            for placement in placements:
                for seed in seeds:
                    run_id = f"run_{idx:03d}_{model}_{task}_{placement}_s{seed}"

                    # Assign between person 1 and person 2
                    if "internvl3" in model:
                        assignee = person1_name
                    elif "smolvlm" in model:
                        assignee = person2_name
                    else:
                        assignee = person1_name if idx % 2 == 1 else person2_name

                    grid.append({
                        "run_id": run_id,
                        "index": idx,
                        "model": model,
                        "task": task,
                        "placement": placement,
                        "seed": seed,
                        "assigned_to": assignee,
                        "status": "pending",  # pending, running, completed, failed
                        "output_dir": f"./runs/{run_id}",
                        "peak_vram_gb": None,
                        "completed_at": None,
                        "error_message": None,
                    })
                    idx += 1

    return grid


class ManifestManager:
    """Manages reading, updating, and querying the experiment manifest file."""
    def __init__(self, manifest_path: str = "./runs/manifest.json"):
        self.manifest_path = manifest_path
        self.runs: List[Dict[str, Any]] = []

    def initialize_if_missing(
        self,
        person1_name: str = "Person_1",
        person2_name: str = "Person_2",
        include_expansion_models: bool = False,
    ) -> None:
        if not os.path.exists(self.manifest_path):
            models = DEFAULT_MODELS + (EXPANSION_MODELS if include_expansion_models else [])
            self.runs = generate_experiment_grid(
                models=models,
                person1_name=person1_name,
                person2_name=person2_name,
            )
            self.save()
        else:
            self.load()

    def load(self) -> None:
        with open(self.manifest_path, "r") as f:
            self.runs = json.load(f)

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.manifest_path)), exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.runs, f, indent=2)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        for r in self.runs:
            if r["run_id"] == run_id:
                return r
        return None

    def sync_from_disk(self, runs_dir: str = "./runs") -> None:
        """
        Scans runs_dir for completed run_metrics.json files and synchronizes
        the manifest state. This ensures robust progress tracking across
        parallel SLURM jobs without race condition clobbering.
        """
        if not os.path.exists(runs_dir):
            return
        if os.path.exists(self.manifest_path):
            try:
                self.load()
            except Exception:
                pass

        updated = False
        for r in self.runs:
            run_id = r["run_id"]
            metrics_file = os.path.join(runs_dir, run_id, "run_metrics.json")
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, "r") as f:
                        data = json.load(f)
                    peak_vram = data.get("telemetry_summary", {}).get("peak_memory_gb", None)
                    if r["status"] != "completed" or r.get("peak_vram_gb") != peak_vram:
                        r["status"] = "completed"
                        if peak_vram is not None:
                            r["peak_vram_gb"] = peak_vram
                        updated = True
                except Exception:
                    pass

        if updated:
            self.save()

    def update_status(
        self,
        run_id: str,
        status: str,
        peak_vram_gb: Optional[float] = None,
        error_message: Optional[str] = None,
        completed_at: Optional[str] = None,
    ) -> None:
        if os.path.exists(self.manifest_path):
            try:
                self.load()
            except Exception:
                pass
        run = self.get_run(run_id)
        if run:
            run["status"] = status
            if peak_vram_gb is not None:
                run["peak_vram_gb"] = peak_vram_gb
            if error_message is not None:
                run["error_message"] = error_message
            if completed_at is not None:
                run["completed_at"] = completed_at
            self.save()

    def get_pending_runs(
        self,
        assigned_to: Optional[str] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        placement: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filters runs that are still pending or failed (resumable)."""
        pending = []
        for r in self.runs:
            if r["status"] in ["pending", "failed"]:
                if assigned_to and r["assigned_to"] != assigned_to:
                    continue
                if model and r["model"] != model:
                    continue
                if task and r["task"] != task:
                    continue
                if placement and r["placement"] != placement:
                    continue
                pending.append(r)
        return pending

    def render_markdown_summary(self) -> str:
        """Returns a Markdown table of experiment progress for reports and syncs."""
        self.sync_from_disk()
        total = len(self.runs)
        completed = sum(1 for r in self.runs if r["status"] == "completed")
        running = sum(1 for r in self.runs if r["status"] == "running")
        failed = sum(1 for r in self.runs if r["status"] == "failed")
        pending = sum(1 for r in self.runs if r["status"] == "pending")

        md = [
            f"### Experiment Grid Progress ({completed}/{total} Completed)\n",
            f"| Status | Count |",
            f"|---|---|",
            f"| Completed | {completed} |",
            f"| Running | {running} |",
            f"| Pending | {pending} |",
            f"| Failed | {failed} |\n",
            "| ID | Model | Task | Placement | Seed | Assigned To | Status | Peak VRAM |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in self.runs:
            vram_str = f"{r['peak_vram_gb']} GB" if r["peak_vram_gb"] else "-"
            md.append(
                f"| `{r['run_id']}` | {r['model']} | {r['task']} | `{r['placement']}` | "
                f"{r['seed']} | {r['assigned_to']} | **{r['status']}** | {vram_str} |"
            )
        return "\n".join(md)
