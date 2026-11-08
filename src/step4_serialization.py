"""Step 4: Chronological Prompt Serialization & Hugging Face Dataset Building.

Builds chronological multimodal prompts combining ICU vitals, laboratory tests,
aligned chest radiographs, and isolated PrimeKG biomedical subgraphs across four
6-hour observation windows.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


def get_target_completion(label: int) -> str:
    """Return the canonical binary target completion string for a binary cohort label.

    Parameters
    ----------
    label : int
        Binary Sepsis-3 label (0 or 1).

    Returns
    -------
    str
        "Sepsis" for label 1, "Non-Sepsis" for label 0.

    Raises
    ------
    ValueError
        If label is not 0 or 1.
    """
    if type(label) is not int or label not in (0, 1):
        raise ValueError(f"Label must be integer 0 or 1, got: {label} (type {type(label).__name__})")
    return config.STEP4_CLASS_LABELS[label]


def compute_sepsis_probability(
    logit_pos: float,
    logit_neg: float,
) -> float:
    """Compute normalized sepsis probability from positive and negative class logits.

    Uses numerically stable two-class softmax:
        P(sepsis) = exp(z_pos - max_z) / (exp(z_pos - max_z) + exp(z_neg - max_z))

    Parameters
    ----------
    logit_pos : float
        Conditional logit or log-probability for the positive target completion ("Sepsis").
    logit_neg : float
        Conditional logit or log-probability for the negative target completion ("Non-Sepsis").

    Returns
    -------
    float
        Probability value strictly in [0.0, 1.0].

    Raises
    ------
    ValueError
        If logits are NaN or both are infinite with the same sign.
    """
    if math.isnan(logit_pos) or math.isnan(logit_neg):
        raise ValueError("Logits must not be NaN")
    if math.isinf(logit_pos) and math.isinf(logit_neg):
        if logit_pos == logit_neg:
            raise ValueError("Ambiguous probability: both logits are infinite with the same sign")
        return 1.0 if logit_pos > logit_neg else 0.0
    if math.isinf(logit_pos):
        return 1.0 if logit_pos > 0 else 0.0
    if math.isinf(logit_neg):
        return 0.0 if logit_neg > 0 else 1.0

    max_logit = max(logit_pos, logit_neg)
    exp_pos = math.exp(logit_pos - max_logit)
    exp_neg = math.exp(logit_neg - max_logit)
    prob = exp_pos / (exp_pos + exp_neg)
    return max(0.0, min(1.0, prob))


def format_inference_output(
    probability: float,
    reasoning_chain: str,
) -> Dict[str, Any]:
    """Format the two-field inference output dictionary.

    Parameters
    ----------
    probability : float
        Estimated sepsis probability between 0.0 and 1.0.
    reasoning_chain : str
        Evidence-based clinical rationale.

    Returns
    -------
    dict
        Structured output matching the inference contract:
        {"sepsis_probability": float, "reasoning_chain": str}
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"Sepsis probability must be in [0.0, 1.0], got {probability}")
    if not isinstance(reasoning_chain, str):
        raise TypeError(f"reasoning_chain must be a string, got {type(reasoning_chain).__name__}")

    return {
        "sepsis_probability": round(float(probability), 4),
        "reasoning_chain": reasoning_chain.strip(),
    }


def validate_inference_output(output_payload: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate that an inference output payload strictly conforms to the two-field schema.

    Parameters
    ----------
    output_payload : str or dict
        Raw JSON string or parsed dictionary.

    Returns
    -------
    dict
        Validated inference output dictionary.

    Raises
    ------
    ValueError
        If schema is invalid, required fields are missing, or probability is out of bounds.
    """
    if isinstance(output_payload, str):
        try:
            data = json.loads(output_payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Inference output is not valid JSON: {exc}") from exc
    elif isinstance(output_payload, dict):
        data = output_payload
    else:
        raise TypeError(f"Expected str or dict, got {type(output_payload).__name__}")

    required_keys = {"sepsis_probability", "reasoning_chain"}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - set(data.keys())
        raise ValueError(f"Inference output missing required fields: {missing}")

    prob = data["sepsis_probability"]
    if not isinstance(prob, (int, float)) or math.isnan(prob) or not (0.0 <= prob <= 1.0):
        raise ValueError(f"sepsis_probability must be a float between 0.0 and 1.0, got: {prob}")

    rationale = data["reasoning_chain"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("reasoning_chain must be a non-empty string")

    return {
        "sepsis_probability": float(prob),
        "reasoning_chain": rationale.strip(),
    }
