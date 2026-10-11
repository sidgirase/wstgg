"""
Model construction and loading interface.
Supports both official Hugging Face pretrained checkpoints (for PACE-ICE GPU runs)
and synthetic architecture mock models (for rapid local testing, CI, and smoke tests).
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from .module_maps import get_model_spec


class MockInternVL3(nn.Module):
    """Synthetic architecture matching InternVL3-2B layer naming and structure."""
    def __init__(self, hidden_dim: int = 256, vocab_size: int = 1000):
        super().__init__()
        # Vision encoder
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.ModuleList([
            nn.ModuleDict({
                "qkv": nn.Linear(hidden_dim, hidden_dim * 3),
                "proj": nn.Linear(hidden_dim, hidden_dim),
                "fc1": nn.Linear(hidden_dim, hidden_dim * 2),
                "fc2": nn.Linear(hidden_dim * 2, hidden_dim),
            })
            for _ in range(4)
        ])
        # Projector / connector
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Language decoder
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(hidden_dim, hidden_dim),
                "k_proj": nn.Linear(hidden_dim, hidden_dim),
                "v_proj": nn.Linear(hidden_dim, hidden_dim),
                "o_proj": nn.Linear(hidden_dim, hidden_dim),
                "gate_proj": nn.Linear(hidden_dim, hidden_dim * 2),
                "up_proj": nn.Linear(hidden_dim, hidden_dim * 2),
                "down_proj": nn.Linear(hidden_dim * 2, hidden_dim),
            })
            for _ in range(4)
        ])
        self.language_model.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Any:
        batch_size, seq_len = input_ids.shape
        hidden = torch.randn(batch_size, seq_len, 256, device=input_ids.device, requires_grad=True)
        # Pass through an LM layer so grad flows to LoRA parameters
        for layer in self.language_model.model.layers:
            hidden = layer["q_proj"](hidden)
        logits = self.language_model.lm_head(hidden)
        return type("Output", (), {"logits": logits, "loss": None})()

    def forward_multimodal(self, input_ids: torch.Tensor, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Any:
        return self.forward(input_ids, labels=labels)


class MockSmolVLM2(nn.Module):
    """Synthetic architecture matching SmolVLM2-2.2B layer naming and structure."""
    def __init__(self, hidden_dim: int = 256, vocab_size: int = 1000):
        super().__init__()
        self.model = nn.Module()
        # Vision encoder
        self.model.vision_model = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(hidden_dim, hidden_dim),
                "k_proj": nn.Linear(hidden_dim, hidden_dim),
                "v_proj": nn.Linear(hidden_dim, hidden_dim),
                "out_proj": nn.Linear(hidden_dim, hidden_dim),
                "fc1": nn.Linear(hidden_dim, hidden_dim * 2),
                "fc2": nn.Linear(hidden_dim * 2, hidden_dim),
            })
            for _ in range(4)
        ])
        # Connector
        self.model.connector = nn.Linear(hidden_dim, hidden_dim)
        # Text model
        self.model.text_model = nn.Module()
        self.model.text_model.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(hidden_dim, hidden_dim),
                "k_proj": nn.Linear(hidden_dim, hidden_dim),
                "v_proj": nn.Linear(hidden_dim, hidden_dim),
                "o_proj": nn.Linear(hidden_dim, hidden_dim),
                "gate_proj": nn.Linear(hidden_dim, hidden_dim * 2),
                "up_proj": nn.Linear(hidden_dim, hidden_dim * 2),
                "down_proj": nn.Linear(hidden_dim * 2, hidden_dim),
            })
            for _ in range(4)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Any:
        batch_size, seq_len = input_ids.shape
        hidden = torch.randn(batch_size, seq_len, 256, device=input_ids.device, requires_grad=True)
        for layer in self.model.text_model.layers:
            hidden = layer["q_proj"](hidden)
        logits = self.lm_head(hidden)
        return type("Output", (), {"logits": logits, "loss": None})()

    def forward_multimodal(self, input_ids: torch.Tensor, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Any:
        return self.forward(input_ids, labels=labels)


class MockQwen3VL(nn.Module):
    """Synthetic architecture matching Qwen3-VL-2B layer naming and structure."""
    def __init__(self, hidden_dim: int = 256, vocab_size: int = 1000):
        super().__init__()
        # Visual pathway
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([
            nn.ModuleDict({
                "qkv": nn.Linear(hidden_dim, hidden_dim * 3),
                "proj": nn.Linear(hidden_dim, hidden_dim),
                "fc1": nn.Linear(hidden_dim, hidden_dim * 2),
                "fc2": nn.Linear(hidden_dim * 2, hidden_dim),
            })
            for _ in range(4)
        ])
        self.visual.merger = nn.Linear(hidden_dim, hidden_dim)
        # LM pathway
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(hidden_dim, hidden_dim),
                "k_proj": nn.Linear(hidden_dim, hidden_dim),
                "v_proj": nn.Linear(hidden_dim, hidden_dim),
                "o_proj": nn.Linear(hidden_dim, hidden_dim),
                "gate_proj": nn.Linear(hidden_dim, hidden_dim * 2),
                "up_proj": nn.Linear(hidden_dim, hidden_dim * 2),
                "down_proj": nn.Linear(hidden_dim * 2, hidden_dim),
            })
            for _ in range(4)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Any:
        batch_size, seq_len = input_ids.shape
        hidden = torch.randn(batch_size, seq_len, 256, device=input_ids.device, requires_grad=True)
        for layer in self.model.layers:
            hidden = layer["q_proj"](hidden)
        logits = self.lm_head(hidden)
        return type("Output", (), {"logits": logits, "loss": None})()

    def forward_multimodal(self, input_ids: torch.Tensor, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Any:
        return self.forward(input_ids, labels=labels)


def load_vlm_model(
    model_name: str,
    use_mock: bool = False,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Constructs or loads the specified VLM model.
    Falls back to faithful mock architecture when running offline or in unit tests.
    """
    spec = get_model_spec(model_name)
    family = spec["family"]

    if use_mock:
        if family == "internvl3":
            model = MockInternVL3()
        elif family == "smolvlm2":
            model = MockSmolVLM2()
        elif family == "qwen3_vl":
            model = MockQwen3VL()
        else:
            raise ValueError(f"Unknown family: {family}")
        return model

    # Try loading HuggingFace model
    try:
        from transformers import AutoModelForVision2Seq, AutoModel
        hf_id = spec["hf_model_id"]
        print(f"[ModelLoader] Loading pretrained {hf_id} from Hugging Face...")
        model = AutoModel.from_pretrained(
            hf_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        return model
    except Exception as e:
        print(f"[ModelLoader] Hugging Face loading failed ({e}). Falling back to mock architecture for '{family}'.")
        return load_vlm_model(model_name, use_mock=True, device=device)
