"""
Hardware telemetry and experiment metric logging.
Tracks peak VRAM (GB), per-step execution time (ms), loss curves, and parameter stats.
"""

import csv
import json
import os
import time
from typing import Dict, List, Optional, Any
import psutil
import torch


class TelemetryTracker:
    """
    Monitors peak VRAM usage (on GPU) and system RAM (on CPU),
    along with step latency.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.step_times: List[float] = []
        self.losses: List[float] = []
        self.step_vram_mb: List[float] = []
        self._step_start_time: Optional[float] = None

        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_step_start(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        self._step_start_time = time.perf_counter()

    def on_step_end(self, loss_val: float) -> Dict[str, float]:
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter() - self._step_start_time) * 1000.0 if self._step_start_time else 0.0
        self.step_times.append(elapsed)
        self.losses.append(loss_val)

        if self.is_cuda:
            peak_bytes = torch.cuda.max_memory_allocated(self.device)
            peak_mb = peak_bytes / (1024.0 * 1024.0)
        else:
            process = psutil.Process(os.getpid())
            peak_mb = process.memory_info().rss / (1024.0 * 1024.0)

        self.step_vram_mb.append(peak_mb)
        return {
            "step_time_ms": elapsed,
            "peak_memory_mb": peak_mb,
            "peak_memory_gb": peak_mb / 1024.0,
            "loss": loss_val,
        }

    def get_summary(self) -> Dict[str, Any]:
        avg_step_ms = sum(self.step_times) / len(self.step_times) if self.step_times else 0.0
        max_step_ms = max(self.step_times) if self.step_times else 0.0
        peak_mb = max(self.step_vram_mb) if self.step_vram_mb else 0.0

        return {
            "total_steps": len(self.step_times),
            "avg_step_time_ms": round(avg_step_ms, 2),
            "max_step_time_ms": round(max_step_ms, 2),
            "peak_memory_mb": round(peak_mb, 2),
            "peak_memory_gb": round(peak_mb / 1024.0, 3),
            "final_loss": round(self.losses[-1], 4) if self.losses else None,
            "min_loss": round(min(self.losses), 4) if self.losses else None,
        }

    def save_log(self, output_dir: str, run_config: Dict[str, Any]) -> str:
        os.makedirs(output_dir, exist_ok=True)
        summary = self.get_summary()
        result_payload = {
            "run_config": run_config,
            "telemetry_summary": summary,
            "losses": [round(l, 5) for l in self.losses],
            "step_times_ms": [round(t, 2) for t in self.step_times],
        }

        json_path = os.path.join(output_dir, "run_metrics.json")
        with open(json_path, "w") as f:
            json.dump(result_payload, f, indent=2)

        # Write CSV step trace
        csv_path = os.path.join(output_dir, "step_trace.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "step_time_ms", "peak_memory_mb"])
            for i, (l, t, m) in enumerate(zip(self.losses, self.step_times, self.step_vram_mb)):
                writer.writerow([i + 1, round(l, 5), round(t, 2), round(m, 2)])

        return json_path
