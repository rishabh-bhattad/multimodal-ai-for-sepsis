"""Step 4: Chronological Prompt Serialization & Hugging Face Dataset Building.

Builds chronological multimodal prompts combining ICU vitals, laboratory tests,
aligned chest radiographs, and isolated PrimeKG biomedical subgraphs across four
6-hour observation windows.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import networkx as nx

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


def format_window_header(window_idx: int) -> str:
    """Format the chronological window interval header.

    Parameters
    ----------
    window_idx : int
        Window index (0, 1, 2, or 3).

    Returns
    -------
    str
        Header string like '[TIMELINE: T - 24 hours to T - 18 hours]'.
    """
    if type(window_idx) is not int or window_idx not in range(config.NUM_WINDOWS):
        raise ValueError(f"window_idx must be integer 0..{config.NUM_WINDOWS - 1}, got: {window_idx}")
    start_h = config.OBSERVATION_HOURS - window_idx * config.WINDOW_SIZE_HOURS
    end_h = config.OBSERVATION_HOURS - (window_idx + 1) * config.WINDOW_SIZE_HOURS
    return f"[TIMELINE: T - {start_h} hours to T - {end_h} hours]"


def _is_valid_measurement(val: Any) -> bool:
    """Check if a measurement value is present, non-null, and non-NaN."""
    if val is None:
        return False
    try:
        fval = float(val)
        return not (math.isnan(fval) or math.isinf(fval))
    except (TypeError, ValueError):
        return False


def format_vitals_line(window: Dict[str, Any]) -> str:
    """Format structured vitals for a 6-hour window into a concise text summary.

    Parameters
    ----------
    window : dict
        Window dictionary with vital sign measurements.

    Returns
    -------
    str
        Formatted vitals line, or explicit absence statement if none recorded.
    """
    parts: List[str] = []

    hr = window.get("heart_rate_mean")
    if _is_valid_measurement(hr):
        parts.append(f"Heart Rate {round(float(hr))} bpm")

    map_val = window.get("map_latest")
    if _is_valid_measurement(map_val):
        parts.append(f"MAP {round(float(map_val))} mmHg")

    spo2 = window.get("spo2_mean")
    if _is_valid_measurement(spo2):
        parts.append(f"SpO2 {round(float(spo2))}%")

    temp = window.get("temp_latest")
    if not _is_valid_measurement(temp):
        temp = window.get("temp_mean")
    if _is_valid_measurement(temp):
        parts.append(f"Temp {round(float(temp), 1)} C")

    rr = window.get("resp_rate_mean")
    if _is_valid_measurement(rr):
        parts.append(f"Resp Rate {round(float(rr))} bpm")

    if parts:
        return f"- Vitals: {', '.join(parts)}."
    return "- Vitals: No vital signs recorded in this window."


def format_labs_line(window: Dict[str, Any]) -> str:
    """Format structured laboratory measurements for a 6-hour window.

    Parameters
    ----------
    window : dict
        Window dictionary with lab measurements.

    Returns
    -------
    str
        Formatted labs line, or explicit absence statement if none recorded.
    """
    parts: List[str] = []

    lac = window.get("lactate_latest")
    if _is_valid_measurement(lac):
        parts.append(f"Lactate {round(float(lac), 1)} mmol/L")

    plt = window.get("platelets_latest")
    if _is_valid_measurement(plt):
        parts.append(f"Platelets {round(float(plt))} k/uL")

    cr = window.get("creatinine_latest")
    if _is_valid_measurement(cr):
        parts.append(f"Creatinine {round(float(cr), 1)} mg/dL")

    wbc = window.get("wbc_latest")
    if _is_valid_measurement(wbc):
        parts.append(f"WBC {round(float(wbc), 1)} k/uL")

    bili = window.get("bilirubin_latest")
    if _is_valid_measurement(bili):
        parts.append(f"Bilirubin {round(float(bili), 1)} mg/dL")

    if parts:
        return f"- Labs: {', '.join(parts)}."
    return "- Labs: No laboratory measurements recorded in this window."


def serialize_subgraph_edges(
    subgraph: Optional[nx.MultiDiGraph],
    max_edges: int = config.STEP3_MAX_EDGES_PER_WINDOW,
) -> str:
    """Serialize the edges of a window PrimeKG MultiDiGraph into natural language statements.

    Follows the technical contract:
        "[Entity A] -> [relation] -> [Entity B]"
    Edges are deduplicated, sorted deterministically, and capped at max_edges.
    If the subgraph has 0 edges or is None, returns the baseline physiological context string.

    Parameters
    ----------
    subgraph : nx.MultiDiGraph or None
        Isolated window subgraph from Step 3.
    max_edges : int
        Maximum number of edge statements to serialize.

    Returns
    -------
    str
        Serialized multi-line KG text.
    """
    if subgraph is None or subgraph.number_of_edges() == 0:
        return f"- Retrieved KG Prior: {config.STEP4_BASELINE_KG_TEXT}"

    triples = set()
    for u, v, key, data in subgraph.edges(keys=True, data=True):
        u_node = subgraph.nodes.get(u, {})
        v_node = subgraph.nodes.get(v, {})
        source_name = u_node.get("node_name") or (u[1] if isinstance(u, tuple) else str(u))
        target_name = v_node.get("node_name") or (v[1] if isinstance(v, tuple) else str(v))
        relation = data.get("relation", key)
        triples.add((str(source_name).strip(), str(relation).strip(), str(target_name).strip()))

    sorted_triples = sorted(triples, key=lambda t: (t[0].lower(), t[1].lower(), t[2].lower()))
    retained_triples = sorted_triples[:max_edges]

    lines = ["- Retrieved KG Prior:"]
    for src, rel, tgt in retained_triples:
        lines.append(f"  [{src}] -> {rel} -> [{tgt}]")
    return "\n".join(lines)


def serialize_window_block(
    window_idx: int,
    window_data: Dict[str, Any],
    subgraph: Optional[nx.MultiDiGraph] = None,
    image_line: Optional[str] = None,
    max_edges: int = config.STEP3_MAX_EDGES_PER_WINDOW,
) -> str:
    """Serialize a single 6-hour window into its chronological text block.

    Parameters
    ----------
    window_idx : int
        Window index (0, 1, 2, 3).
    window_data : dict
        Vitals, labs, and metadata for the window.
    subgraph : nx.MultiDiGraph or None
        Isolated PrimeKG subgraph retrieved for this window.
    image_line : str or None
        Formatted image line (e.g. "- Image: <image> Frontal Chest Radiograph acquired."
        or "- Image: No chest radiograph acquired in this window.").
    max_edges : int
        Max edges to serialize for the knowledge graph.

    Returns
    -------
    str
        Complete serialized text block for this observation window.
    """
    header = format_window_header(window_idx)
    vitals = format_vitals_line(window_data)
    labs = format_labs_line(window_data)
    img = image_line if image_line is not None else f"- Image: {config.STEP4_NO_IMAGE_PLACEHOLDER}"
    kg_text = serialize_subgraph_edges(subgraph, max_edges=max_edges)
    return f"{header}\n{vitals}\n{labs}\n{img}\n{kg_text}"

