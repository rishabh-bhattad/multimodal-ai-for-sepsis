"""Unit tests for Step 5 model configuration, QLoRA initialization, and multimodal data collator."""

import pytest
import torch
import torch.nn as nn
from PIL import Image
from peft import PeftModel, TaskType
from transformers.modeling_outputs import CausalLMOutputWithPast

import config
from src.step5_training import (
    get_qlora_config,
    get_quantization_config,
    count_trainable_parameters,
    apply_qlora_to_model,
    MultimodalDataCollator,
    predict_sepsis,
)


class DummyAttentionLM(nn.Module):
    """Minimal model mimicking Transformer linear projections for LoRA adapter attachment."""

    def __init__(self, hidden_dim: int = 32, vocab_size: int = 100):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}

    def forward(self, x):
        h = self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
        return self.lm_head(self.o_proj(h))


def test_get_qlora_config_defaults():
    cfg = get_qlora_config()
    assert cfg.r == config.STEP5_LORA_R  # 16
    assert cfg.lora_alpha == config.STEP5_LORA_ALPHA  # 32
    assert cfg.lora_dropout == config.STEP5_LORA_DROPOUT  # 0.05
    assert set(cfg.target_modules) == set(config.STEP5_LORA_TARGET_MODULES)
    assert cfg.bias == config.STEP5_LORA_BIAS  # "none"
    assert cfg.task_type == TaskType.CAUSAL_LM


def test_get_qlora_config_custom_overrides():
    custom_modules = ["q_proj", "v_proj"]
    cfg = get_qlora_config(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=custom_modules,
        bias="none",
    )
    assert cfg.r == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.1
    assert cfg.target_modules == set(custom_modules)


def test_get_quantization_config():
    bnb_cfg = get_quantization_config()
    assert bnb_cfg.load_in_4bit is True
    assert bnb_cfg.bnb_4bit_quant_type == "nf4"
    assert bnb_cfg.bnb_4bit_compute_dtype == torch.bfloat16
    assert bnb_cfg.bnb_4bit_use_double_quant is True


def test_count_trainable_parameters():
    model = nn.Sequential(
        nn.Linear(10, 5, bias=True),   # 10*5 + 5 = 55
        nn.Linear(5, 2, bias=True),    # 5*2 + 2 = 12
    )  # Total 67
    trainable, total, pct = count_trainable_parameters(model)
    assert total == 67
    assert trainable == 67
    assert pytest.approx(pct, rel=1e-3) == 100.0

    # Freeze layer 0
    for p in model[0].parameters():
        p.requires_grad = False

    trainable_frozen, total_frozen, pct_frozen = count_trainable_parameters(model)
    assert total_frozen == 67
    assert trainable_frozen == 12
    assert pytest.approx(pct_frozen, rel=1e-3) == (12 / 67) * 100.0


def test_apply_qlora_to_model():
    base_model = DummyAttentionLM(hidden_dim=32, vocab_size=100)
    peft_model = apply_qlora_to_model(base_model)

    assert isinstance(peft_model, PeftModel)
    trainable, total, pct = count_trainable_parameters(peft_model)

    # Base projection weights must be frozen
    assert not base_model.q_proj.weight.requires_grad
    assert not base_model.lm_head.weight.requires_grad

    # Only LoRA weights should be trainable
    assert trainable > 0
    assert trainable < total
    assert pct < 50.0  # LoRA parameters are a tiny fraction of total model weights


def test_multimodal_data_collator_padding():
    collator = MultimodalDataCollator(pad_token_id=0, label_pad_token_id=-100, padding_side="right")
    features = [
        {
            "input_ids": [10, 20, 30],
            "attention_mask": [1, 1, 1],
            "labels": [-100, -100, 1001],
        },
        {
            "input_ids": [10, 20, 30, 40, 50],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [-100, -100, -100, 1002, 1003],
        },
    ]

    batch = collator(features)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    assert input_ids.shape == (2, 5)
    assert attention_mask.shape == (2, 5)
    assert labels.shape == (2, 5)

    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.long
    assert labels.dtype == torch.long

    # First row check (padded from len 3 to len 5)
    assert input_ids[0].tolist() == [10, 20, 30, 0, 0]
    assert attention_mask[0].tolist() == [1, 1, 1, 0, 0]
    assert labels[0].tolist() == [-100, -100, 1001, -100, -100]

    # Second row check (unpadded length 5)
    assert input_ids[1].tolist() == [10, 20, 30, 40, 50]
    assert attention_mask[1].tolist() == [1, 1, 1, 1, 1]
    assert labels[1].tolist() == [-100, -100, -100, 1002, 1003]


def test_multimodal_data_collator_left_padding():
    collator = MultimodalDataCollator(pad_token_id=0, label_pad_token_id=-100, padding_side="left")
    features = [
        {
            "input_ids": [10, 20],
            "attention_mask": [1, 1],
            "labels": [-100, 50],
        },
        {
            "input_ids": [1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1],
            "labels": [-100, -100, -100, 99],
        },
    ]

    batch = collator(features)
    assert batch["input_ids"].shape == (2, 4)
    # Row 0 padded on the left
    assert batch["input_ids"][0].tolist() == [0, 0, 10, 20]
    assert batch["attention_mask"][0].tolist() == [0, 0, 1, 1]
    assert batch["labels"][0].tolist() == [-100, -100, -100, 50]


def test_multimodal_data_collator_auxiliary_fields():
    collator = MultimodalDataCollator()
    features = [
        {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "labels": [-100, 10],
            "stay_id": 10001,
        },
        {
            "input_ids": [3, 4],
            "attention_mask": [1, 1],
            "labels": [-100, 20],
            "stay_id": 10002,
        },
    ]

    batch = collator(features)
    assert batch["stay_id"] == [10001, 10002]


class SyntheticProcessor:
    """Exercise the processor interface without weights, a tokenizer download, or real images."""

    tokenizer = None
    pad_token_id = 0

    def __init__(self):
        self.tokenizer = self

    def encode(self, text, add_special_tokens=False):
        return {"Sepsis": [2, 3], "Non-Sepsis": [2, 4]}[text]

    def decode(self, ids, skip_special_tokens=True):
        assert ids.tolist() == [7]
        return "Observed findings warrant review."

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "".join(part.get("text", "<vision>") for part in messages[0]["content"])

    def __call__(self, *, text, images, return_tensors, add_special_tokens, truncation):
        assert text[0].count("<vision>") == len(images or [])
        assert return_tensors == "pt" and not truncation
        ids = torch.tensor([[1, 1] + [6] * len(images or [])])
        result = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if images:
            assert all(image.mode == "RGB" for image in images)
            result["pixel_values"] = torch.tensor([image.getpixel((0, 0)) for image in images])
            result["image_grid_thw"] = torch.ones(len(images), 3, dtype=torch.long)
            result["mm_token_type_ids"] = (ids == 6).long()
        return result


def test_processor_packs_ordered_images_and_masks_prompt(tmp_path):
    images = []
    for index, color in enumerate(("red", "blue")):
        path = tmp_path / f"synthetic_{index}.png"
        Image.new("RGB", (8, 8), color=color).save(path)
        images.append(str(path))
    records = [
        {"prompt_text": "First <image> then <image>", "image_paths": images,
         "target_completion": "Sepsis", "subject_id": 1},
        {"prompt_text": "No image", "image_paths": [], "target_completion": "Non-Sepsis"},
    ]
    batch = MultimodalDataCollator(processor=SyntheticProcessor())(records)
    assert batch["pixel_values"].tolist() == [[255, 0, 0], [0, 0, 255]]
    assert batch["image_grid_thw"].shape == (2, 3)
    assert batch["mm_token_type_ids"].tolist() == [[0, 0, 1, 1, 0, 0], [0] * 6]
    assert batch["labels"].tolist() == [[-100] * 4 + [2, 3], [-100] * 2 + [2, 4, -100, -100]]
    assert "image_paths" not in batch and "subject_id" not in batch
    with pytest.raises(ValueError, match="processor is required"):
        MultimodalDataCollator()(records)


def test_prediction_uses_the_token_after_a_shared_label_prefix(monkeypatch):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(8, 2)

        def get_input_embeddings(self):
            return self.embeddings

        def forward(self, input_ids, attention_mask):
            logits = torch.zeros((*input_ids.shape, 8))
            logits[..., 3] = (input_ids == 2) * 5
            return CausalLMOutputWithPast(logits=logits)

        def generate(self, input_ids, attention_mask, **kwargs):
            return torch.cat((input_ids, input_ids.new_tensor([[7]])), dim=1)

    monkeypatch.setattr(config, "STEP5_RATIONALE_MAX_NEW_TOKENS", 10)
    output = predict_sepsis(Model(), SyntheticProcessor(), {
        "prompt_text": f"Observations\n\n{config.STEP4_TARGET_TASK}", "image_paths": [],
    }, max_seq_length=32)
    assert output["sepsis_probability"] == pytest.approx(0.9933, abs=1e-4)
    assert output["reasoning_chain"] == "Observed findings warrant review."


def test_multimodal_data_collator_validations():
    collator = MultimodalDataCollator()

    # Empty features
    with pytest.raises(ValueError, match="Cannot collate an empty list"):
        collator([])

    # Missing required key
    with pytest.raises(KeyError, match="missing required key"):
        collator([{"input_ids": [1, 2], "attention_mask": [1, 1]}])

    # Mismatched tensor lengths within a feature
    with pytest.raises(ValueError, match="mismatched tensor lengths"):
        collator([{"input_ids": [1, 2, 3], "attention_mask": [1, 1], "labels": [1, 2, 3]}])

    # Invalid padding side
    with pytest.raises(ValueError, match="padding_side must be 'right' or 'left'"):
        MultimodalDataCollator(padding_side="center")
