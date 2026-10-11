"""
Parameter-count matcher for balanced LoRA budget allocation.

Finds the optimal LoRA rank for the 'vision' placement such that the number
of trainable parameters matches the 'lm' placement within a specified tolerance
(default 10%, per course specification and proposal).
"""

import copy
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn

from .module_maps import get_model_spec, is_vision_module, is_lm_module
from .adapter import LoRALinear


def compute_layer_capacities(
    model: nn.Module,
    model_name: str,
) -> Tuple[int, int, List[str], List[str]]:
    """
    Computes sum of (in_features + out_features) for all eligible linear layers
    in the LM pathway vs the Vision pathway.

    Returns:
        (c_lm, c_vision, lm_layer_names, vision_layer_names)
    """
    spec = get_model_spec(model_name)
    lm_targets = spec["lm_target_modules"]
    vision_targets = spec["vision_target_modules"]

    c_lm = 0
    c_vision = 0
    lm_layers = []
    vision_layers = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, LoRALinear)):
            in_f = module.in_features if isinstance(module, LoRALinear) else module.in_features
            out_f = module.out_features if isinstance(module, LoRALinear) else module.out_features
            cap = in_f + out_f

            if is_lm_module(name, spec) and any(t in name for t in lm_targets):
                c_lm += cap
                lm_layers.append(name)
            elif is_vision_module(name, spec) and any(t in name for t in vision_targets):
                c_vision += cap
                vision_layers.append(name)

    return c_lm, c_vision, lm_layers, vision_layers


def match_lora_ranks(
    model: nn.Module,
    model_name: str,
    lm_rank: int = 16,
    max_search_rank: int = 128,
    tolerance: float = 0.10,
) -> Dict[str, Any]:
    """
    Finds the optimal vision LoRA rank that matches the LM LoRA parameter count
    within tolerance (default 10%).

    Args:
        model: PyTorch model.
        model_name: Model identifier (e.g. 'internvl3_2b', 'smolvlm2_2b', 'qwen3_vl_2b').
        lm_rank: Reference rank for the language decoder LoRA (default: 16).
        max_search_rank: Upper bound for searching vision rank.
        tolerance: Maximum allowed relative difference (default: 0.10 = 10%).

    Returns:
        Dictionary with:
            - lm_rank: Reference LM rank
            - vision_rank: Matched vision rank
            - lm_trainable_params: Exact trainable count for LM placement
            - vision_trainable_params: Exact trainable count for Vision placement
            - relative_difference: |N_vision - N_lm| / N_lm
            - matched_within_tolerance: bool
            - message: Descriptive summary
    """
    c_lm, c_vision, lm_layers, vision_layers = compute_layer_capacities(model, model_name)

    if c_lm == 0:
        raise ValueError(f"No eligible LM linear layers found for model '{model_name}'.")
    if c_vision == 0:
        raise ValueError(f"No eligible Vision linear layers found for model '{model_name}'.")

    target_lm_params = lm_rank * c_lm

    # Find best integer rank for vision
    ideal_vision_rank = target_lm_params / c_vision
    candidate_ranks = sorted(list(set([
        max(1, int(ideal_vision_rank)),
        max(1, int(ideal_vision_rank) + 1),
        max(1, round(ideal_vision_rank)),
    ])))

    best_rank = 1
    best_diff = float("inf")
    best_vision_params = 0

    # Search through candidates or full range up to max_search_rank
    search_space = candidate_ranks if candidate_ranks else range(1, max_search_rank + 1)
    for r in search_space:
        params = r * c_vision
        diff = abs(params - target_lm_params) / target_lm_params
        if diff < best_diff:
            best_diff = diff
            best_rank = r
            best_vision_params = params

    within_tolerance = best_diff <= tolerance

    result = {
        "model_name": model_name,
        "lm_rank": lm_rank,
        "vision_rank": best_rank,
        "lm_trainable_params": target_lm_params,
        "vision_trainable_params": best_vision_params,
        "relative_difference": round(best_diff, 4),
        "percentage_difference": round(best_diff * 100.0, 2),
        "matched_within_tolerance": within_tolerance,
        "tolerance": tolerance,
        "c_lm_factor": c_lm,
        "c_vision_factor": c_vision,
        "num_lm_layers": len(lm_layers),
        "num_vision_layers": len(vision_layers),
    }

    if within_tolerance:
        result["message"] = (
            f"Successfully matched budget within {result['percentage_difference']}% "
            f"(<= {tolerance*100}%): LM rank {lm_rank} ({target_lm_params:,} params) vs "
            f"Vision rank {best_rank} ({best_vision_params:,} params)."
        )
    else:
        result["message"] = (
            f"WARNING: Closest match is {result['percentage_difference']}% difference: "
            f"LM rank {lm_rank} ({target_lm_params:,} params) vs "
            f"Vision rank {best_rank} ({best_vision_params:,} params)."
        )

    return result
