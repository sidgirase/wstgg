"""
Unit tests for data collator, 768px image cap, token bounding, and answer-only label masking.
"""

import pytest
from PIL import Image
import torch

from vlm_placement.data.collator import (
    resize_with_max_side,
    calculate_image_token_bounds,
    VLMDataCollator,
)
from vlm_placement.data.datasets import get_smoke_dataset


def test_resize_with_max_side():
    # Image larger than 768 on width
    img_wide = Image.new("RGB", (1200, 600))
    resized_wide = resize_with_max_side(img_wide, max_side=768)
    assert resized_wide.size[0] == 768
    assert resized_wide.size[1] == 384

    # Image larger than 768 on height
    img_tall = Image.new("RGB", (500, 1000))
    resized_tall = resize_with_max_side(img_tall, max_side=768)
    assert resized_tall.size[1] == 768
    assert resized_tall.size[0] == 384

    # Image smaller than 768 remains unchanged
    img_small = Image.new("RGB", (400, 300))
    resized_small = resize_with_max_side(img_small, max_side=768)
    assert resized_small.size == (400, 300)


def test_calculate_image_token_bounds():
    # Should stay within [300, 600]
    for (w, h) in [(768, 768), (1024, 768), (600, 800), (300, 300)]:
        gw, gh, tokens = calculate_image_token_bounds(w, h, min_tokens=300, max_tokens=600)
        assert 300 <= tokens <= 600, f"Dimensions ({w}, {h}) produced {tokens} tokens, outside [300, 600]"


def test_answer_only_sft_label_masking():
    dataset = get_smoke_dataset(num_samples=4)
    collator = VLMDataCollator()
    batch = [dataset[i] for i in range(len(dataset))]

    collated = collator(batch)
    labels = collated["labels"]
    input_ids = collated["input_ids"]

    assert labels.shape == input_ids.shape
    # Ensure there are masked prompt tokens (-100)
    assert (labels == -100).sum() > 0, "Labels must contain masked prompt tokens (-100)"
    # Ensure there are unmasked target tokens (valid target answer)
    assert (labels != -100).sum() > 0, "Labels must contain unmasked answer tokens"

    # Verify first token of every sequence is masked (-100) because it belongs to the prompt
    assert (labels[:, 0] == -100).all(), "Prompt tokens must have label -100"


def test_smoke_dataset_cardinality():
    ds = get_smoke_dataset(num_samples=100)
    assert len(ds) == 100
    sample = ds[0]
    assert "image" in sample
    assert "question" in sample
    assert "answer" in sample
    assert "task" in sample
    assert sample["task"] == "infovqa"
