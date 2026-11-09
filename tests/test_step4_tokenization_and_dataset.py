"""Unit tests for Step 4 tokenization, target-only causal loss masking, and DatasetDict creation."""

import sys
from pathlib import Path

import pytest
from datasets import DatasetDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.step4_serialization import (
    build_huggingface_dataset,
    build_stay_example,
    tokenize_and_mask_labels,
)


class MockTokenizer:
    """Simple deterministic whitespace/word tokenizer for testing tokenization and masking."""

    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1, "<image>": 2, "Sepsis": 3, "Non-Sepsis": 4}

    def _get_id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def encode(self, text: str) -> list[int]:
        tokens = text.split()
        return [self._get_id(t) for t in tokens]

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        ids = self.encode(text)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def make_synthetic_windows():
    """Helper to generate 4 minimal synthetic windows."""
    return [
        {"window_idx": i, "heart_rate_mean": 80.0, "map_latest": 70.0, "cxr_path": None}
        for i in range(4)
    ]


def test_tokenize_and_mask_labels_target_only_supervision():
    """Verify that prompt tokens are assigned -100 and completion tokens are active."""
    tokenizer = MockTokenizer()
    prompt = "This is a clinical trajectory prompt for patient"
    completion = "Sepsis"

    tokenized = tokenize_and_mask_labels(prompt, completion, tokenizer=tokenizer)

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = tokenized["labels"]

    assert len(input_ids) == len(attention_mask) == len(labels)
    # The last token corresponds to "Sepsis"
    prompt_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt_ids)

    # Prompt tokens must be strictly -100
    for idx in range(prompt_len):
        assert labels[idx] == -100, f"Token at {idx} was not masked with -100"

    # Completion tokens must be active and match input_ids
    for idx in range(prompt_len, len(input_ids)):
        assert labels[idx] == input_ids[idx]
        assert labels[idx] != -100


def test_tokenize_and_mask_labels_truncation_preserves_target():
    """Verify that truncation keeps the target completion tokens intact."""
    tokenizer = MockTokenizer()
    prompt = "word " * 100
    completion = "Non-Sepsis"

    max_len = 10
    tokenized = tokenize_and_mask_labels(prompt, completion, tokenizer=tokenizer, max_seq_length=max_len)

    assert len(tokenized["input_ids"]) == max_len
    assert len(tokenized["labels"]) == max_len

    # The final token must be active (the target completion token)
    assert tokenized["labels"][-1] != -100
    # Earlier tokens should be masked
    assert tokenized["labels"][0] == -100


def test_build_stay_example_contract():
    """Verify single stay dictionary matches the technical contract."""
    tokenizer = MockTokenizer()
    windows = make_synthetic_windows()

    example = build_stay_example(
        stay_id="30001",
        subject_id="10001",
        split="val",
        stay_windows=windows,
        label=1,
        tokenizer=tokenizer,
    )

    assert example["stay_id"] == 30001
    assert example["subject_id"] == 10001
    assert example["split"] == "validation"  # normalized
    assert example["target_completion"] == "Sepsis"
    assert example["label"] == 1
    assert isinstance(example["prompt_text"], str)
    assert isinstance(example["image_paths"], list)
    assert "input_ids" in example
    assert "attention_mask" in example
    assert "labels" in example


def test_build_huggingface_dataset_splits_and_isolation():
    """Verify DatasetDict assembly and strict subject split isolation enforcement."""
    tokenizer = MockTokenizer()
    windows = make_synthetic_windows()

    examples = [
        build_stay_example("1", "101", "train", windows, 1, tokenizer=tokenizer),
        build_stay_example("2", "102", "train", windows, 0, tokenizer=tokenizer),
        build_stay_example("3", "103", "val", windows, 1, tokenizer=tokenizer),
        build_stay_example("4", "104", "test", windows, 0, tokenizer=tokenizer),
    ]

    dataset_dict = build_huggingface_dataset(examples)
    assert isinstance(dataset_dict, DatasetDict)
    assert set(dataset_dict.keys()) == {"train", "validation", "test"}
    assert len(dataset_dict["train"]) == 2
    assert len(dataset_dict["validation"]) == 1
    assert len(dataset_dict["test"]) == 1

    # Verify column schema
    for split_name in ("train", "validation", "test"):
        features = dataset_dict[split_name].column_names
        for required_col in ("stay_id", "subject_id", "prompt_text", "image_paths", "target_completion", "label", "input_ids", "labels"):
            assert required_col in features


def test_build_huggingface_dataset_rejects_subject_leakage():
    """Verify that any subject overlap across splits immediately raises ValueError."""
    windows = make_synthetic_windows()
    examples_with_leakage = [
        build_stay_example("1", "101", "train", windows, 1),
        build_stay_example("2", "101", "test", windows, 0),  # Subject 101 leaked into test!
    ]

    with pytest.raises(ValueError, match="Subject split isolation violated"):
        build_huggingface_dataset(examples_with_leakage)

