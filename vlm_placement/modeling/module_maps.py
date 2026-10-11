"""
Module maps and target layer specifications for small VLMs.
Maps architecture components to 'lm' vs 'vision' placement categories
for InternVL3-2B, SmolVLM2-2.2B, and Qwen3-VL-2B.
"""

from typing import Dict, List, Any


MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "internvl3_2b": {
        "hf_model_id": "OpenGVLab/InternVL2_5-2B",  # or OpenGVLab/InternVL3-2B
        "family": "internvl3",
        "auto_model_class": "AutoModel",
        "vision_module_prefixes": [
            "vision_model",
            "mlp1",
        ],
        "vision_connector_prefixes": [
            "mlp1",
        ],
        "lm_module_prefixes": [
            "language_model",
        ],
        # Target linear modules for LoRA
        "vision_target_modules": [
            "qkv",
            "proj",
            "fc1",
            "fc2",
            "mlp1.0",
            "mlp1.2",
            "linear_fc1",
            "linear_fc2",
        ],
        "lm_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "smolvlm2_2b": {
        "hf_model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "family": "smolvlm2",
        "auto_model_class": "AutoModelForImageTextToText",
        "vision_module_prefixes": [
            "vision_model",
            "connector",
            "modality_projection",
            "multi_modal_projector",
        ],
        "vision_connector_prefixes": [
            "connector",
            "modality_projection",
            "multi_modal_projector",
        ],
        "lm_module_prefixes": [
            "text_model",
        ],
        "vision_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "fc1",
            "fc2",
            "connector",
            "modality_projection",
            "multi_modal_projector",
        ],
        "lm_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "qwen3_vl_2b": {
        "hf_model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "family": "qwen3_vl",
        "vision_module_prefixes": [
            "visual",
        ],
        "vision_connector_prefixes": [
            "visual.merger",
        ],
        "lm_module_prefixes": [
            "model.layers",
            "lm_head",
        ],
        "vision_target_modules": [
            "qkv",
            "proj",
            "fc1",
            "fc2",
            "linear_fc1",
            "linear_fc2",
            "merger.linear",
        ],
        "lm_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
}


def get_model_spec(model_name: str) -> Dict[str, Any]:
    """Retrieve the module specification for a given model family."""
    name_clean = model_name.lower().replace("-", "_").replace(".", "_")
    for key, spec in MODEL_SPECS.items():
        if key in name_clean or name_clean in key:
            return spec
    raise ValueError(f"Unknown model name: '{model_name}'. Supported: {list(MODEL_SPECS.keys())}")


def is_vision_module(param_or_module_name: str, spec: Dict[str, Any]) -> bool:
    """Check if a parameter/module name belongs to the vision pathway (encoder or connector)."""
    return any(p in param_or_module_name for p in spec["vision_module_prefixes"])


def is_lm_module(param_or_module_name: str, spec: Dict[str, Any]) -> bool:
    """Check if a parameter/module name belongs to the language decoder pathway."""
    return any(p in param_or_module_name for p in spec["lm_module_prefixes"])
