"""
Unit tests for VLM placement modeling and parameter-budget matching.
Guards the invariant assertions specified in Week 5 milestone.
"""

import pytest
import torch

from vlm_placement.modeling.builder import load_vlm_model
from vlm_placement.modeling.adapter import (
    apply_lora_to_model,
    check_trainable,
    get_trainable_param_stats,
)
from vlm_placement.modeling.matcher import match_lora_ranks


@pytest.mark.parametrize("model_name", ["internvl3_2b", "smolvlm2_2b", "qwen3_vl_2b"])
def test_lm_placement_trainable_invariants(model_name: str):
    """
    Placement 'lm' MUST have:
    - 0 trainable parameters in the vision encoder or connector
    - > 0 trainable parameters in the language decoder
    """
    model = load_vlm_model(model_name, use_mock=True)
    apply_lora_to_model(model, placement="lm", model_name=model_name, rank=16)

    # Invariant assertion should pass cleanly
    check_trainable(model, placement="lm", model_name=model_name)

    stats = get_trainable_param_stats(model, model_name)
    assert stats["vision_trainable_params"] == 0, f"Expected 0 vision params, found {stats['vision_trainable_params']}"
    assert stats["lm_trainable_params"] > 0, "Expected > 0 LM params"
    assert stats["trainable_params"] == stats["lm_trainable_params"]


@pytest.mark.parametrize("model_name", ["internvl3_2b", "smolvlm2_2b", "qwen3_vl_2b"])
def test_vision_placement_trainable_invariants(model_name: str):
    """
    Placement 'vision' MUST have:
    - 0 trainable parameters in the language decoder
    - > 0 trainable parameters in the vision encoder / connector
    """
    model = load_vlm_model(model_name, use_mock=True)
    apply_lora_to_model(model, placement="vision", model_name=model_name, rank=16)

    # Invariant assertion should pass cleanly
    check_trainable(model, placement="vision", model_name=model_name)

    stats = get_trainable_param_stats(model, model_name)
    assert stats["lm_trainable_params"] == 0, f"Expected 0 LM params, found {stats['lm_trainable_params']}"
    assert stats["vision_trainable_params"] > 0, "Expected > 0 vision params"
    assert stats["trainable_params"] == stats["vision_trainable_params"]


def test_violation_triggers_assertion_error():
    """
    Tests that check_trainable raises AssertionError if a parameter from the
    wrong pathway is unfreezed.
    """
    model = load_vlm_model("internvl3_2b", use_mock=True)
    apply_lora_to_model(model, placement="lm", model_name="internvl3_2b", rank=16)

    # Artificially unfreeze a vision parameter
    for p in model.vision_model.parameters():
        p.requires_grad = True
        break

    with pytest.raises(AssertionError, match="Placement 'lm' invariant violated"):
        check_trainable(model, placement="lm", model_name="internvl3_2b")


@pytest.mark.parametrize("model_name", ["internvl3_2b", "smolvlm2_2b", "qwen3_vl_2b"])
def test_parameter_count_matcher_within_tolerance(model_name: str):
    """
    Tests that match_lora_ranks selects a vision rank whose trainable count
    matches the LM budget within the 10% tolerance threshold.
    """
    model = load_vlm_model(model_name, use_mock=True)
    res = match_lora_ranks(model, model_name, lm_rank=16, tolerance=0.10)

    assert res["matched_within_tolerance"] is True, (
        f"Model {model_name} failed 10% tolerance: {res['percentage_difference']}%"
    )
    assert res["relative_difference"] <= 0.10
    assert res["vision_rank"] >= 1
    assert res["lm_trainable_params"] > 0
    assert res["vision_trainable_params"] > 0
