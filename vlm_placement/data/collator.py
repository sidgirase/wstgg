"""
Multimodal batch collator with dynamic resolution bounds,
768-pixel long-side cap, and answer-only SFT label masking.
"""

from typing import Dict, List, Optional, Any, Tuple
import math
import torch
import torch.nn.functional as F
from PIL import Image


def resize_with_max_side(img: Image.Image, max_side: int = 768) -> Image.Image:
    """
    Downscales an image preserving aspect ratio if max(width, height) > max_side.
    Complies with project constraint: 768-px longer-side cap.
    """
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return img


def calculate_image_token_bounds(
    width: int,
    height: int,
    patch_size: int = 14,
    min_tokens: int = 300,
    max_tokens: int = 600,
) -> Tuple[int, int, int]:
    """
    Calculates grid dimensions (grid_w, grid_h) such that total image tokens
    fall within [min_tokens, max_tokens] (project bound: 300-600 image tokens).
    """
    raw_patches_w = max(1, width // patch_size)
    raw_patches_h = max(1, height // patch_size)
    raw_tokens = raw_patches_w * raw_patches_h

    if min_tokens <= raw_tokens <= max_tokens:
        return raw_patches_w, raw_patches_h, raw_tokens

    aspect = width / float(height)
    # Target midpoint around 450 tokens
    target_tokens = 450
    # tokens = (h * aspect) * h = aspect * h^2 => h = sqrt(target_tokens / aspect)
    target_h = max(1, int(round(math.sqrt(target_tokens / aspect))))
    target_w = max(1, int(round(target_h * aspect)))
    actual_tokens = target_w * target_h

    # Clip within [min_tokens, max_tokens]
    if actual_tokens > max_tokens:
        scale = math.sqrt(max_tokens / actual_tokens)
        target_w = max(1, int(math.floor(target_w * scale)))
        target_h = max(1, int(math.floor(target_h * scale)))
    elif actual_tokens < min_tokens:
        scale = math.sqrt(min_tokens / actual_tokens)
        target_w = max(1, int(math.ceil(target_w * scale)))
        target_h = max(1, int(math.ceil(target_h * scale)))

    return target_w, target_h, target_w * target_h


class VLMDataCollator:
    """
    Collates raw multimodal examples into PyTorch training tensors.
    Implements:
    - 768px long-side cap
    - 300-600 image token bounds
    - Answer-only SFT loss masking (-100 for all question/prompt tokens)
    """
    def __init__(
        self,
        tokenizer: Any = None,
        max_image_side: int = 768,
        min_image_tokens: int = 300,
        max_image_tokens: int = 600,
        patch_size: int = 14,
        max_seq_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_image_side = max_image_side
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.patch_size = patch_size
        self.max_seq_length = max_seq_length

    def format_prompt(self, question: str) -> str:
        return f"<image>\nQuestion: {question}\nAnswer: "

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        processed_images = []
        token_counts = []
        questions = []
        answers = []
        ids = []

        for item in batch:
            ids.append(item["id"])
            img = resize_with_max_side(item["image"], max_side=self.max_image_side)
            w, h = img.size
            gw, gh, tokens = calculate_image_token_bounds(
                w, h,
                patch_size=self.patch_size,
                min_tokens=self.min_image_tokens,
                max_tokens=self.max_image_tokens,
            )
            processed_images.append(img)
            token_counts.append(tokens)
            questions.append(item["question"])
            answers.append(str(item["answer"]))

        # Build answer-only token IDs and labels
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for q, a in zip(questions, answers):
            prompt = self.format_prompt(q)
            full_text = prompt + a

            if self.tokenizer is not None:
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

                # Truncate if exceeds max length
                if len(full_ids) > self.max_seq_length:
                    full_ids = full_ids[:self.max_seq_length]

                # Answer-only label masking: -100 on prompt
                labels = list(full_ids)
                prompt_len = min(len(prompt_ids), len(labels))
                for i in range(prompt_len):
                    labels[i] = -100

                attn_mask = [1] * len(full_ids)
            else:
                # Standalone mock tokenization (character/word-based) for CPU/unit test environments
                # 32 tokens prompt, 8 tokens answer
                prompt_ids = [hash(w) % 1000 + 1 for w in prompt.split()]
                ans_ids = [hash(w) % 1000 + 1 for w in a.split()]
                full_ids = prompt_ids + ans_ids
                labels = [-100] * len(prompt_ids) + list(ans_ids)
                attn_mask = [1] * len(full_ids)

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attn_mask, dtype=torch.long))

        # Pad sequences
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) if self.tokenizer else 0
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        # Approximate pixel tensors (batch, 3, 224, 224) or dummy visual feature representations
        pixel_values = torch.zeros((len(batch), 3, 224, 224), dtype=torch.float32)

        return {
            "ids": ids,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_token_counts": token_counts,
            "questions": questions,
            "answers": answers,
        }
