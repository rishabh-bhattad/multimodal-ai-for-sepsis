"""Step 4: Chronological Prompt Serialization & Hugging Face Dataset Building.

Builds chronological multimodal prompts combining ICU vitals, laboratory tests,
aligned chest radiographs, and isolated PrimeKG biomedical subgraphs across four
6-hour observation windows.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

logger = logging.getLogger("Step4-Serialization")


def get_target_completion(label: int) -> str:
    """Return the canonical binary target completion string for a binary cohort label."""
    if type(label) is not int or label not in (0, 1):
        raise ValueError(f"Label must be integer 0 or 1, got: {label} (type {type(label).__name__})")
    return config.STEP4_CLASS_LABELS[label]


def compute_sepsis_probability(
    logit_pos: float,
    logit_neg: float,
) -> float:
    """Normalize the two class scores with a stable softmax."""
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
    """Format the two-field inference output dictionary."""
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"Sepsis probability must be in [0.0, 1.0], got {probability}")
    if not isinstance(reasoning_chain, str):
        raise TypeError(f"reasoning_chain must be a string, got {type(reasoning_chain).__name__}")

    return {
        "sepsis_probability": round(float(probability), 4),
        "reasoning_chain": reasoning_chain.strip(),
    }


def validate_inference_output(output_payload: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate that an inference output payload strictly conforms to the two-field schema."""
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
    """Format the chronological window interval header."""
    if type(window_idx) is not int or window_idx not in range(config.NUM_WINDOWS):
        raise ValueError(f"window_idx must be integer 0..{config.NUM_WINDOWS - 1}, got: {window_idx}")
    start_h = config.OBSERVATION_HOURS - window_idx * config.WINDOW_SIZE_HOURS
    end_h = config.OBSERVATION_HOURS - (window_idx + 1) * config.WINDOW_SIZE_HOURS
    return f"[TIMELINE: T - {start_h} hours to T - {end_h} hours]"


def _is_valid_measurement(val: Any) -> bool:
    """Check whether a measurement is present and finite enough to format."""
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
    """Format structured laboratory measurements for a 6-hour window."""
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
    """Serialize a bounded, deterministic list of native PrimeKG triples for the window."""
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
    """Serialize a single 6-hour window into its chronological text block."""
    header = format_window_header(window_idx)
    vitals = format_vitals_line(window_data)
    labs = format_labs_line(window_data)
    img = image_line if image_line is not None else f"- Image: {config.STEP4_NO_IMAGE_PLACEHOLDER}"
    kg_text = serialize_subgraph_edges(subgraph, max_edges=max_edges)
    return f"{header}\n{vitals}\n{labs}\n{img}\n{kg_text}"


def resolve_image_path(
    cxr_path: Optional[Any],
    cxr_dir: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Resolve an image path to a normalized string path."""
    if cxr_path is None:
        return None
    try:
        if isinstance(cxr_path, float) and math.isnan(cxr_path):
            return None
    except (TypeError, ValueError):
        pass

    path_str = str(cxr_path).strip()
    if not path_str or path_str.lower() in ("nan", "none"):
        return None

    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return str(path_obj)

    base = Path(config.MIMIC_CXR_DIR if cxr_dir is None else cxr_dir)
    return str(base / path_obj)


def assemble_multimodal_prompt(
    stay_windows: List[Dict[str, Any]],
    subgraphs: Optional[Dict[str, nx.MultiDiGraph]] = None,
    cxr_dir: Optional[Union[str, Path]] = None,
    max_edges_per_window: int = config.STEP3_MAX_EDGES_PER_WINDOW,
) -> Tuple[str, List[str]]:
    """Build the 24-hour prompt and return image paths in placeholder order."""
    if len(stay_windows) != config.NUM_WINDOWS:
        raise ValueError(
            f"Expected exactly {config.NUM_WINDOWS} windows per stay, got: {len(stay_windows)}"
        )

    sorted_windows = sorted(stay_windows, key=lambda w: w.get("window_idx", -1))
    for expected_idx, win in enumerate(sorted_windows):
        win_idx = win.get("window_idx")
        if win_idx != expected_idx:
            raise ValueError(
                f"Windows must have sequential indices 0..{config.NUM_WINDOWS - 1}. "
                f"Expected index {expected_idx}, got {win_idx}"
            )

    window_blocks: List[str] = []
    image_paths: List[str] = []

    for idx, win in enumerate(sorted_windows):
        cxr_path = win.get("cxr_path")
        resolved_img = resolve_image_path(cxr_path, cxr_dir=cxr_dir)
        if resolved_img is not None:
            image_line = f"- Image: {config.STEP4_IMAGE_PLACEHOLDER}"
            image_paths.append(resolved_img)
        else:
            image_line = f"- Image: {config.STEP4_NO_IMAGE_PLACEHOLDER}"

        sg = None
        if subgraphs is not None:
            sg = subgraphs.get(f"window_{idx}")

        block = serialize_window_block(
            window_idx=idx,
            window_data=win,
            subgraph=sg,
            image_line=image_line,
            max_edges=max_edges_per_window,
        )
        window_blocks.append(block)

    prompt_sections = [
        config.STEP4_SYSTEM_CONTEXT,
        *window_blocks,
        config.STEP4_TARGET_TASK,
    ]
    prompt_text = "\n\n".join(prompt_sections)

    placeholder_count = prompt_text.count("<image>")
    if placeholder_count != len(image_paths):
        raise ValueError(
            f"Mismatch between <image> placeholders ({placeholder_count}) and "
            f"resolved image paths ({len(image_paths)})"
        )

    return prompt_text, image_paths


class ContextBudgetError(ValueError):
    """The required observations exceed the available model context."""


def fit_graph_context(prompt_text, encode, max_prompt_tokens):
    """Reduce complete KG edge lists evenly while keeping all observation blocks and images."""
    blocks = prompt_text.split("\n\n")
    edge_header = "- Retrieved KG Prior:"
    cap = max((sum(line.startswith("  [") for line in block.splitlines())
               for block in blocks if edge_header in block), default=0)
    while True:
        encoded = encode(prompt_text)
        ids = encoded["input_ids"]
        length = ids.shape[-1] if hasattr(ids, "shape") else len(ids)
        if length <= max_prompt_tokens:
            return prompt_text, encoded
        if not cap:
            raise ContextBudgetError("Four windows do not fit after removing retrieved graph context")
        cap //= 2
        trimmed = []
        for block in blocks:
            observations, marker, graph = block.partition("\n" + edge_header + "\n")
            if not marker:
                trimmed.append(block)
                continue
            edges = graph.splitlines()[:cap]
            kg_text = "\n".join([edge_header, *edges]) if edges else (
                f"{edge_header} {config.STEP4_BASELINE_KG_TEXT}"
            )
            trimmed.append(f"{observations}\n{kg_text}")
        prompt_text = "\n\n".join(trimmed)


def tokenize_and_mask_labels(
    prompt_text: str,
    target_completion: str,
    tokenizer: Any,
    max_seq_length: int = config.STEP4_MAX_SEQ_LENGTH,
) -> Dict[str, List[int]]:
    """Check the text budget and supervise only the explicitly appended binary completion."""
    def encode(text, *, special):
        if callable(tokenizer):
            return list(tokenizer(text, add_special_tokens=special)["input_ids"])
        return list(tokenizer.encode(text))

    completion_ids = encode(target_completion, special=False)
    if not completion_ids:
        raise ValueError("Binary completion tokenizes to an empty sequence")
    completion_budget = max(
        len(encode(label, special=False)) for label in config.STEP4_CLASS_LABELS.values()
    )
    _, prompt = fit_graph_context(
        prompt_text.rstrip(), lambda text: {"input_ids": encode(text, special=True)},
        max_seq_length - completion_budget,
    )
    prompt_ids = prompt["input_ids"]
    return {
        "input_ids": prompt_ids + completion_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(completion_ids)),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def build_stay_example(
    stay_id: Union[int, str],
    subject_id: Union[int, str],
    split: str,
    stay_windows: List[Dict[str, Any]],
    label: int,
    subgraphs: Optional[Dict[str, nx.MultiDiGraph]] = None,
    tokenizer: Optional[Any] = None,
    cxr_dir: Optional[Union[str, Path]] = None,
    max_seq_length: int = config.STEP4_MAX_SEQ_LENGTH,
    max_edges_per_window: int = config.STEP3_MAX_EDGES_PER_WINDOW,
) -> Dict[str, Any]:
    """Build a stay record with ordered image paths and optional text-only token checks."""
    target_completion = get_target_completion(label)
    prompt_text, image_paths = assemble_multimodal_prompt(
        stay_windows=stay_windows,
        subgraphs=subgraphs,
        cxr_dir=cxr_dir,
        max_edges_per_window=max_edges_per_window,
    )

    norm_split = "validation" if split in ("val", "validation") else split

    record: Dict[str, Any] = {
        "stay_id": int(stay_id),
        "subject_id": int(subject_id),
        "split": norm_split,
        "prompt_text": prompt_text,
        "image_paths": image_paths,
        "target_completion": target_completion,
        "label": int(label),
    }

    if tokenizer is not None:
        tokenized = tokenize_and_mask_labels(
            prompt_text=prompt_text,
            target_completion=target_completion,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        record.update(tokenized)

    return record


def build_huggingface_dataset(
    examples: List[Dict[str, Any]],
) -> Any:
    """Build a Hugging Face DatasetDict partitioned by train, validation, and test splits."""
    from datasets import Dataset, DatasetDict

    splits_data: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    subjects_by_split: Dict[str, set] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    for ex in examples:
        raw_split = ex.get("split", "train")
        split_name = "validation" if raw_split in ("val", "validation") else raw_split
        if split_name not in splits_data:
            raise ValueError(f"Unknown split name: {raw_split}")

        row = {k: v for k, v in ex.items() if k != "split"}
        splits_data[split_name].append(row)
        subjects_by_split[split_name].add(ex["subject_id"])

    for s1, s2 in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = subjects_by_split[s1].intersection(subjects_by_split[s2])
        if overlap:
            raise ValueError(
                f"Subject split isolation violated: {len(overlap)} subjects overlap between {s1} and {s2}"
            )

    return DatasetDict({
        name: Dataset.from_list(rows if rows else [])
        for name, rows in splits_data.items()
    })


def compute_file_sha256(path: Path) -> str:
    """Compute sha256 checksum of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_trajectories_pickle(path: Path) -> Tuple[Dict[str, Any], str]:
    """Load Step 3 subgraph trajectories pickle and compute its checksum."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Trajectories pickle not found: {path}")
    digest = compute_file_sha256(p)
    with p.open("rb") as f:
        trajectories = pickle.load(f)
    if not isinstance(trajectories, dict):
        raise ValueError("Trajectories pickle must be a dict keyed by stay_id")
    return trajectories, digest


def load_patient_windows_parquet(path: Path) -> Tuple[pd.DataFrame, str]:
    """Load Step 1 patient windows parquet and compute its checksum."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Patient windows parquet not found: {path}")
    digest = compute_file_sha256(p)
    df = pd.read_parquet(p)
    required = {"stay_id", "subject_id", "window_idx", "split", "label"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Patient windows parquet missing required columns: {missing}")
    return df, digest


def select_stays_for_dataset(
    trajectories: Dict[str, Any],
    windows_df: pd.DataFrame,
    sample_size: Optional[int] = config.STEP4_SAMPLE_SIZE,
) -> List[int]:
    """Select eligible stay IDs deterministically, preserving patient splits and image coverage.

    If sample_size is None, returns all eligible stay IDs.
    """
    traj_stay_ids = {int(k) for k in trajectories.keys()}
    win_stay_ids = set(windows_df["stay_id"].unique())
    common_stay_ids = sorted(traj_stay_ids.intersection(win_stay_ids))

    if not common_stay_ids:
        raise ValueError("No common stay IDs between trajectories and patient windows")

    if sample_size is None or len(common_stay_ids) <= sample_size:
        return common_stay_ids

    sub_df = windows_df[windows_df["stay_id"].isin(common_stay_ids)].drop_duplicates("stay_id")
    sub_df = sub_df.assign(has_image=sub_df["cxr_path"].notna() if "cxr_path" in sub_df.columns else False)

    representatives: Dict[int, int] = {}
    for row in sub_df.sort_values(["has_image", "stay_id"], ascending=[False, True]).to_dict("records"):
        representatives.setdefault(row["subject_id"], row["stay_id"])

    candidates = sorted(representatives.items())
    rng = random.Random(config.RANDOM_SEED)
    rng.shuffle(candidates)

    coverage_cases: Dict[bool, Tuple[int, int]] = {}
    stay_to_img = dict(zip(sub_df["stay_id"], sub_df["has_image"]))
    for sub, sid in candidates:
        coverage_cases.setdefault(stay_to_img.get(sid, False), (sub, sid))

    selected = [coverage_cases[k] for k in (True, False) if k in coverage_cases][:sample_size]
    selected_subs = {s for s, _ in selected}
    selected.extend((s, sid) for s, sid in candidates if s not in selected_subs)

    final_stays = [sid for _, sid in sorted(selected[:sample_size], key=lambda x: x[1])]
    return final_stays


def summarize_dataset(
    dataset_dict: Any,
    mode: str,
    windows_sha256: str,
    trajectories_sha256: str,
) -> Dict[str, Any]:
    """Return aggregate-only execution diagnostics and dataset provenance without PHI."""
    all_rows = [row for split in dataset_dict.values() for row in split]
    unique_patients = len({row["subject_id"] for row in all_rows})
    total_stays = len(all_rows)
    total_windows = total_stays * config.NUM_WINDOWS

    stays_per_label = dict(sorted(Counter(row["label"] for row in all_rows).items()))
    stays_per_split = {name: len(ds) for name, ds in dataset_dict.items()}
    patients_per_split = {
        name: len({row["subject_id"] for row in ds})
        for name, ds in dataset_dict.items()
    }

    stays_with_images = sum(len(row["image_paths"]) > 0 for row in all_rows)
    total_images = sum(len(row["image_paths"]) for row in all_rows)

    has_tokens = ("input_ids" in all_rows[0] and bool(all_rows[0]["input_ids"])) if all_rows else False
    token_summary = None
    active_tokens = 0
    masked_tokens = 0

    if has_tokens:
        lengths = sorted(len(row["input_ids"]) for row in all_rows)
        n = len(lengths)
        token_summary = {
            "min": lengths[0],
            "p50": lengths[n // 2],
            "p90": lengths[int(n * 0.9)],
            "p99": lengths[min(int(n * 0.99), n - 1)],
            "max": lengths[-1],
        }
        for row in all_rows:
            for lbl in row.get("labels", []):
                if lbl == -100:
                    masked_tokens += 1
                else:
                    active_tokens += 1

    summary = {
        "validation": "passed",
        "mode": mode,
        "unique_patients": unique_patients,
        "stays": total_stays,
        "windows": total_windows,
        "stays_per_label": {str(k): v for k, v in stays_per_label.items()},
        "patients_per_split": patients_per_split,
        "stays_per_split": stays_per_split,
        "stays_with_images": stays_with_images,
        "total_images_referenced": total_images,
        "tokenized": has_tokens,
        "token_length_summary": token_summary,
        "active_supervision_tokens": active_tokens,
        "masked_tokens": masked_tokens,
        "input_sha256": {
            "windows": windows_sha256,
            "trajectories": trajectories_sha256,
        },
    }
    return summary


def run_step4_pipeline(
    windows_path: Path,
    trajectories_path: Path,
    output_dir: Path,
    sample_size: Optional[int] = config.STEP4_SAMPLE_SIZE,
    full: bool = False,
    tokenizer_name: Optional[str] = None,
    cxr_dir: Optional[Path] = None,
    max_seq_length: int = config.STEP4_MAX_SEQ_LENGTH,
    max_edges_per_window: int = config.STEP3_MAX_EDGES_PER_WINDOW,
) -> Dict[str, Any]:
    """Execute the complete Step 4 prompt serialization and dataset assembly pipeline."""
    windows_p = Path(windows_path)
    trajectories_p = Path(trajectories_path)
    output_p = Path(output_dir)

    trajectories, traj_digest = load_trajectories_pickle(trajectories_p)
    windows_df, win_digest = load_patient_windows_parquet(windows_p)

    selected_stays = select_stays_for_dataset(
        trajectories=trajectories,
        windows_df=windows_df,
        sample_size=None if full else sample_size,
    )

    tokenizer = None
    if tokenizer_name:
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        except Exception as exc:
            logger.warning("Could not load tokenizer %s (%s); proceeding without tokenization",
                           tokenizer_name, type(exc).__name__)

    selected_df = windows_df[windows_df["stay_id"].isin(selected_stays)]
    stays_grouped = selected_df.groupby("stay_id")

    examples = []
    for sid in sorted(selected_stays):
        stay_rows = stays_grouped.get_group(sid).sort_values("window_idx").to_dict("records")
        if len(stay_rows) != config.NUM_WINDOWS:
            raise ValueError(f"Stay {sid} does not retain exactly {config.NUM_WINDOWS} windows")

        subject_id = stay_rows[0]["subject_id"]
        split = stay_rows[0]["split"]
        label = stay_rows[0]["label"]

        traj_entry = trajectories.get(str(sid), trajectories.get(sid, {}))
        subgraphs = traj_entry.get("windows") if isinstance(traj_entry, dict) else None

        ex = build_stay_example(
            stay_id=sid,
            subject_id=subject_id,
            split=split,
            stay_windows=stay_rows,
            label=label,
            subgraphs=subgraphs,
            tokenizer=tokenizer,
            cxr_dir=cxr_dir,
            max_seq_length=max_seq_length,
            max_edges_per_window=max_edges_per_window,
        )
        examples.append(ex)

    dataset_dict = build_huggingface_dataset(examples)

    output_p.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_p))

    mode_str = "full" if full else "sample"
    summary = summarize_dataset(
        dataset_dict=dataset_dict,
        mode=mode_str,
        windows_sha256=win_digest,
        trajectories_sha256=traj_digest,
    )

    report_path = output_p.parent / "step4_validation.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    """CLI entrypoint for Step 4 prompt serialization and dataset building."""
    parser = argparse.ArgumentParser(
        description="Step 4: Build chronological multimodal prompts and Hugging Face DatasetDict."
    )
    parser.add_argument("--windows", type=Path, default=config.PATIENT_WINDOWS_PARQUET,
                        help="Path to patient_windows.parquet")
    parser.add_argument("--trajectories", type=Path, default=config.SUBGRAPH_TRAJECTORIES_PKL,
                        help="Path to subgraph_trajectories.pkl")
    parser.add_argument("--output", type=Path, default=config.TOKENIZED_DATASET_DIR,
                        help="Output directory for Hugging Face DatasetDict")
    parser.add_argument("--cxr_dir", type=Path, default=config.MIMIC_CXR_DIR,
                        help="Directory containing CXR images")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Hugging Face tokenizer name or path")
    parser.add_argument("--max_seq_length", type=int, default=config.STEP4_MAX_SEQ_LENGTH,
                        help="Maximum sequence length")
    parser.add_argument("--max_edges", type=int, default=config.STEP3_MAX_EDGES_PER_WINDOW,
                        help="Maximum PrimeKG edges per window")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sample_size", type=int, default=None,
                        help="Number of unique subjects for smoke run")
    mode.add_argument("--full", action="store_true",
                        help="Process the full eligible cohort")

    args = parser.parse_args(argv)

    if args.sample_size is None and not args.full:
        args.sample_size = config.STEP4_SAMPLE_SIZE
    if not args.full and args.sample_size < 1:
        parser.error("--sample_size must be positive")

    try:
        summary = run_step4_pipeline(
            windows_path=args.windows,
            trajectories_path=args.trajectories,
            output_dir=args.output,
            sample_size=args.sample_size,
            full=args.full,
            tokenizer_name=args.tokenizer,
            cxr_dir=args.cxr_dir,
            max_seq_length=args.max_seq_length,
            max_edges_per_window=args.max_edges,
        )
        logger.info("Step 4 complete: %s", json.dumps(summary, sort_keys=True))
    except ContextBudgetError as exc:
        logger.error("Step 4 context budget exceeded: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Step 4 failed (%s); inspect locally using synthetic inputs", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    sys.exit(main())

