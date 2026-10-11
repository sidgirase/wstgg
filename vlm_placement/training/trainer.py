"""
Core training engine for small VLM placement experiments.
Features:
- Gradient checkpointing
- Answer-only SFT loss calculation
- Resumable checkpointing
- Automatic telemetry logging (peak VRAM, step latency)
- Clean integration with PACE-ICE and local environments
"""

import os
import random
import time
from typing import Dict, List, Optional, Any, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..modeling.adapter import get_trainable_param_stats, check_trainable
from .metrics import TelemetryTracker


def set_seed(seed: int) -> None:
    """Enforces strict reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class VLMTrainer:
    """
    Standardized trainer for matched-budget LoRA experiments.
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        placement: str,
        model_name: str,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        enable_gradient_checkpointing: bool = True,
        mixed_precision: str = "bf16",  # 'no', 'fp16', 'bf16'
        device: Optional[torch.device] = None,
        output_dir: str = "./runs/run_01",
        run_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.placement = placement
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        self.output_dir = output_dir
        self.run_config = run_config or {}

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

        # Invariant check
        check_trainable(self.model, self.placement, self.model_name)

        # Gradient checkpointing
        if enable_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        # Optimizer: only optimize parameters with requires_grad=True
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Hardware tracker
        self.telemetry = TelemetryTracker(self.device)

        # Precision scaler
        self.use_amp = self.device.type == "cuda" and mixed_precision in ["fp16", "bf16"]
        if self.mixed_precision == "bf16":
            self.amp_dtype = torch.bfloat16
        elif self.mixed_precision == "fp16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=(self.mixed_precision == "fp16"))
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.mixed_precision == "fp16"))

    def compute_loss(self, outputs: Any, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes standard cross-entropy loss over unmasked answer tokens (labels != -100).
        """
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        # Shift logits and labels for autoregressive next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss

    def train(
        self,
        max_steps: int = 100,
        checkpoint_every_steps: int = 50,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs the training loop up to max_steps with telemetry tracking.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.train()

        start_step = 0
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            ckpt = torch.load(resume_from_checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt.get("step", 0)
            print(f"[VLMTrainer] Resumed training from step {start_step} ({resume_from_checkpoint})")

        global_step = start_step
        data_iter = iter(self.train_dataloader)
        self.optimizer.zero_grad()

        print(f"[VLMTrainer] Starting training: {max_steps} steps, placement='{self.placement}', model='{self.model_name}' on {self.device}")

        while global_step < max_steps:
            self.telemetry.on_step_start()

            accumulated_loss = 0.0
            for _ in range(self.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device)

                # Context-aware mixed precision
                if self.use_amp:
                    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                        amp_ctx = torch.amp.autocast("cuda", dtype=self.amp_dtype)
                    else:
                        amp_ctx = torch.cuda.amp.autocast(dtype=self.amp_dtype)
                    with amp_ctx:
                        # Forward pass
                        if hasattr(self.model, "forward_multimodal"):
                            outputs = self.model.forward_multimodal(input_ids, pixel_values, labels=labels)
                        else:
                            outputs = self.model(input_ids=input_ids, labels=labels)
                        loss = self.compute_loss(outputs, labels) / self.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    if hasattr(self.model, "forward_multimodal"):
                        outputs = self.model.forward_multimodal(input_ids, pixel_values, labels=labels)
                    else:
                        outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = self.compute_loss(outputs, labels) / self.gradient_accumulation_steps
                    loss.backward()

                accumulated_loss += loss.item() * self.gradient_accumulation_steps

            # Gradient clipping & step
            if self.use_amp and self.mixed_precision == "fp16":
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            self.optimizer.zero_grad()
            global_step += 1

            step_stats = self.telemetry.on_step_end(accumulated_loss)

            if global_step % 10 == 0 or global_step == max_steps:
                print(
                    f"Step {global_step}/{max_steps} | Loss: {accumulated_loss:.4f} | "
                    f"Peak RAM: {step_stats['peak_memory_mb']:.1f} MB | "
                    f"Step Time: {step_stats['step_time_ms']:.1f} ms"
                )

            # Periodic checkpointing
            if global_step % checkpoint_every_steps == 0 or global_step == max_steps:
                ckpt_path = os.path.join(self.output_dir, f"checkpoint_step_{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "model_state_dict": {k: v for k, v in self.model.state_dict().items() if "lora_" in k},
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "run_config": self.run_config,
                }, ckpt_path)

        # Save final metrics and configuration
        param_stats = get_trainable_param_stats(self.model, self.model_name)
        full_run_config = {
            **self.run_config,
            "placement": self.placement,
            "model_name": self.model_name,
            "max_steps": max_steps,
            "learning_rate": self.learning_rate,
            "device": str(self.device),
            "trainable_param_stats": param_stats,
        }
        metrics_file = self.telemetry.save_log(self.output_dir, full_run_config)
        print(f"[VLMTrainer] Training complete. Run metrics committed to: {metrics_file}")

        return {
            "metrics_path": metrics_file,
            "telemetry_summary": self.telemetry.get_summary(),
            "trainable_param_stats": param_stats,
        }
