"""Unit tests for Step 4 tokenization, target-only causal loss masking, and DatasetDict creation."""

import sys
from pathlib import Path

import pytest

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


def test_overlong_observations_fail_instead_of_dropping_windows():
    tokenizer = MockTokenizer()
    prompt = "word " * 100
    completion = "Non-Sepsis"

    max_len = 10
    with pytest.raises(ValueError, match="Four windows do not fit"):
        tokenize_and_mask_labels(prompt, completion, tokenizer=tokenizer, max_seq_length=max_len)


def test_budget_removes_graph_edges_and_keeps_each_observation():
    tokenizer = MockTokenizer()
    observations = [f"Window_{index} Vitals_{index} Labs_{index}" for index in range(4)]
    prompt = "\n\n".join(
        f"{line}\n- Retrieved KG Prior:\n" + "\n".join(
            f"  [Concept_{i}] -> relation -> [Other_{i}]" for i in range(32)
        )
        for line in observations
    )
    bare_length = sum(len(line.split()) for line in observations)
    empty_graph_length = len(f"- Retrieved KG Prior: {config.STEP4_BASELINE_KG_TEXT}".split())
    result = tokenize_and_mask_labels(
        prompt, "Sepsis", tokenizer, max_seq_length=bare_length + 4 * empty_graph_length + 1,
    )
    assert len(result["input_ids"]) == bare_length + 4 * empty_graph_length + 1
    assert not set(tokenizer.encode("[Concept_0]")) & set(result["input_ids"])
    positions = [result["input_ids"].index(tokenizer.encode(f"Window_{i}")[0]) for i in range(4)]
    assert positions == sorted(positions)
    for line in observations:
        assert set(tokenizer.encode(line)).issubset(result["input_ids"])
    assert result["labels"][:-1] == [-100] * (len(result["labels"]) - 1)
    assert result["labels"][-1:] == tokenizer.encode("Sepsis")


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
    from datasets import DatasetDict

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


import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.step4_serialization import (
    compute_sepsis_probability,
    get_target_completion,
)


def test_config_step4_settings_exist_and_valid():
    """Verify that all Step 4 configuration variables are correctly defined."""
    assert hasattr(config, "TOKENIZED_DATASET_DIR")
    assert hasattr(config, "STEP4_VALIDATION_JSON")
    assert hasattr(config, "STEP4_SAMPLE_SIZE")
    assert hasattr(config, "STEP4_MAX_SEQ_LENGTH")
    assert hasattr(config, "STEP4_TARGET_COMPLETION_POSITIVE")
    assert hasattr(config, "STEP4_TARGET_COMPLETION_NEGATIVE")
    assert hasattr(config, "STEP4_CLASS_LABELS")
    assert hasattr(config, "STEP4_SYSTEM_CONTEXT")
    assert hasattr(config, "STEP4_TARGET_TASK")
    assert hasattr(config, "STEP4_IMAGE_PLACEHOLDER")
    assert hasattr(config, "STEP4_NO_IMAGE_PLACEHOLDER")
    assert hasattr(config, "STEP4_BASELINE_KG_TEXT")

    assert config.STEP4_TARGET_COMPLETION_POSITIVE == "Sepsis"
    assert config.STEP4_TARGET_COMPLETION_NEGATIVE == "Non-Sepsis"
    assert config.STEP4_CLASS_LABELS == {0: "Non-Sepsis", 1: "Sepsis"}
    assert config.STEP4_SAMPLE_SIZE == 10
    assert config.STEP4_MAX_SEQ_LENGTH > 0
    assert "<image>" in config.STEP4_IMAGE_PLACEHOLDER
    assert all(label in config.STEP4_TARGET_TASK for label in config.STEP4_CLASS_LABELS.values())
    assert "JSON" not in config.STEP4_TARGET_TASK


def test_get_target_completion():
    """Verify mapping of binary labels to canonical target completion strings."""
    assert get_target_completion(1) == "Sepsis"
    assert get_target_completion(0) == "Non-Sepsis"

    for invalid in (-1, 2, 99, None, "1", 0.0):
        with pytest.raises(ValueError):
            get_target_completion(invalid)


def test_compute_sepsis_probability_equal_logits():
    """Equal logits must yield exactly 0.5 probability."""
    assert compute_sepsis_probability(0.0, 0.0) == 0.5
    assert compute_sepsis_probability(5.0, 5.0) == 0.5
    assert compute_sepsis_probability(-10.0, -10.0) == 0.5


def test_compute_sepsis_probability_extreme_and_typical():
    """Verify numerical stability across large and typical logit ranges."""
    # Positive logit much higher than negative -> probability close to 1.0
    p_high = compute_sepsis_probability(10.0, -10.0)
    assert 0.9999 <= p_high <= 1.0

    # Negative logit much higher than positive -> probability close to 0.0
    p_low = compute_sepsis_probability(-10.0, 10.0)
    assert 0.0 <= p_low <= 0.0001

    # Hand-computed reference: diff = ln(3) -> P = 3 / (3 + 1) = 0.75
    diff = math.log(3.0)
    assert abs(compute_sepsis_probability(diff, 0.0) - 0.75) < 1e-6

    # Massive logits that would overflow naive exp without max subtraction
    p_massive = compute_sepsis_probability(1000.0, 990.0)
    assert 0.9999 <= p_massive <= 1.0


def test_compute_sepsis_probability_inf_and_nan():
    """Verify edge case handling for infinite and NaN values."""
    with pytest.raises(ValueError):
        compute_sepsis_probability(float("nan"), 1.0)
    with pytest.raises(ValueError):
        compute_sepsis_probability(1.0, float("nan"))
    with pytest.raises(ValueError):
        compute_sepsis_probability(float("inf"), float("inf"))

    assert compute_sepsis_probability(float("inf"), 1.0) == 1.0
    assert compute_sepsis_probability(1.0, float("inf")) == 0.0
    assert compute_sepsis_probability(float("-inf"), 0.0) == 0.0
    assert compute_sepsis_probability(0.0, float("-inf")) == 1.0
