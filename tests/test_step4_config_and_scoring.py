"""Unit tests for Step 4 configuration parameters, target completions, and scoring logic."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.step4_serialization import (
    compute_sepsis_probability,
    format_inference_output,
    get_target_completion,
    validate_inference_output,
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
    assert "sepsis_probability" in config.STEP4_TARGET_TASK
    assert "reasoning_chain" in config.STEP4_TARGET_TASK


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


def test_format_and_validate_inference_output():
    """Verify two-field inference formatting and schema validation."""
    formatted = format_inference_output(0.85239, "Tachycardia and high lactate present.")
    assert formatted["sepsis_probability"] == 0.8524
    assert formatted["reasoning_chain"] == "Tachycardia and high lactate present."

    # Validate dict input
    validated = validate_inference_output(formatted)
    assert validated["sepsis_probability"] == 0.8524
    assert validated["reasoning_chain"] == "Tachycardia and high lactate present."

    # Validate JSON string input
    json_str = json.dumps(formatted)
    validated_json = validate_inference_output(json_str)
    assert validated_json == validated


def test_format_inference_output_invalid_inputs():
    """Verify formatting rejects out-of-range probabilities or invalid rationale types."""
    with pytest.raises(ValueError):
        format_inference_output(-0.01, "Rationale")
    with pytest.raises(ValueError):
        format_inference_output(1.01, "Rationale")
    with pytest.raises(TypeError):
        format_inference_output(0.5, 123)


def test_validate_inference_output_rejects_malformed():
    """Verify validation catches missing keys, non-JSON strings, or empty rationale."""
    with pytest.raises(ValueError):
        validate_inference_output('{"broken": json}')
    with pytest.raises(ValueError):
        validate_inference_output({"sepsis_probability": 0.5})  # missing reasoning_chain
    with pytest.raises(ValueError):
        validate_inference_output({"reasoning_chain": "Note"})  # missing sepsis_probability
    with pytest.raises(ValueError):
        validate_inference_output({"sepsis_probability": 1.5, "reasoning_chain": "Valid text"})
    with pytest.raises(ValueError):
        validate_inference_output({"sepsis_probability": 0.5, "reasoning_chain": "   "})  # empty rationale

