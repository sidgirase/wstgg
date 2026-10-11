"""
Modeling utilities for VLM parameter locus adaptation.
"""

from .module_maps import get_model_spec, is_vision_module, is_lm_module, MODEL_SPECS
from .adapter import (
    LoRALinear,
    apply_lora_to_model,
    check_trainable,
    count_parameters,
    get_trainable_param_stats,
)
from .matcher import match_lora_ranks, compute_layer_capacities
from .builder import load_vlm_model, MockInternVL3, MockSmolVLM2, MockQwen3VL

__all__ = [
    "get_model_spec",
    "is_vision_module",
    "is_lm_module",
    "MODEL_SPECS",
    "LoRALinear",
    "apply_lora_to_model",
    "check_trainable",
    "count_parameters",
    "get_trainable_param_stats",
    "match_lora_ranks",
    "compute_layer_capacities",
    "load_vlm_model",
    "MockInternVL3",
    "MockSmolVLM2",
    "MockQwen3VL",
]
