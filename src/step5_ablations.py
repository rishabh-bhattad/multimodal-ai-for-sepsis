"""Step 5: Static-RAG and No-KG Prompt Baseline Transforms.

Implements the data transformation pipelines for the two RAG ablation baselines:
1. Static-RAG: All 4 windows' subgraphs collapsed into a single flat union graph
   S_static = union(S_0, S_1, S_2, S_3) and serialized across windows.
2. No-KG Prompt: Completely removes all PrimeKG graph triples while preserving
   identical chronological vitals, labs, and CXR image alignments.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
import pandas as pd
from datasets import Dataset, DatasetDict

import config
from src.step4_serialization import (
    format_window_header,
    format_vitals_line,
    format_labs_line,
    resolve_image_path,
    serialize_subgraph_edges,
    get_target_completion,
    tokenize_and_mask_labels,
)

logger = logging.getLogger("Step5-Ablations")


def merge_subgraphs(subgraphs: Optional[Dict[str, nx.MultiDiGraph]]) -> nx.MultiDiGraph:
    """Collapses multiple window subgraphs into a single merged graph.

    Computes S_static = union_{i=0}^3 S_i. Preserves all node attributes and
    unique multi-edges across all windows.

    Args:
        subgraphs: Dict mapping window keys (e.g. 'window_0'..'window_3') to subgraphs.

    Returns:
        Consolidated nx.MultiDiGraph containing all nodes and edges.
    """
    merged = nx.MultiDiGraph()
    if not subgraphs:
        return merged

    for win_key, sg in subgraphs.items():
        if sg is None or not isinstance(sg, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            continue

        # Merge nodes with attributes
        for node, attrs in sg.nodes(data=True):
            if node not in merged:
                merged.add_node(node, **attrs)
            else:
                merged.nodes[node].update(attrs)

        # Merge edges with attributes
        for u, v, data in sg.edges(data=True):
            merged.add_edge(u, v, **data)

    return merged


def build_static_rag_prompt(
    stay_windows: List[Dict[str, Any]],
    subgraphs: Optional[Dict[str, nx.MultiDiGraph]] = None,
    cxr_dir: Optional[Union[str, Path]] = None,
    max_edges_per_window: int = config.STEP3_MAX_EDGES_PER_WINDOW,
) -> Tuple[str, List[str]]:
    """Assembles a Static-RAG prompt where all 4 windows share the flat union graph.

    Collapses window subgraphs into S_static = union(S_0, S_1, S_2, S_3) and serializes
    the identical merged context across all four 6-hour windows.

    Args:
        stay_windows: List of four window records for a single stay, ordered 0 to 3.
        subgraphs: Dict mapping 'window_0'..'window_3' to isolated subgraphs.
        cxr_dir: Base directory for radiographs (defaults to config.MIMIC_CXR_DIR).
        max_edges_per_window: Maximum edges serialized per window block.

    Returns:
        Tuple of (prompt_text, image_paths).
    """
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

    # Compute flat union graph
    union_graph = merge_subgraphs(subgraphs)

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

        header = format_window_header(idx)
        vitals = format_vitals_line(win)
        labs = format_labs_line(win)
        kg_text = serialize_subgraph_edges(union_graph, max_edges=max_edges_per_window)
        block = f"{header}\n{vitals}\n{labs}\n{image_line}\n{kg_text}"
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


def build_no_kg_prompt(
    stay_windows: List[Dict[str, Any]],
    cxr_dir: Optional[Union[str, Path]] = None,
) -> Tuple[str, List[str]]:
    """Assembles a No-KG prompt with all PrimeKG biomedical triples omitted.

    Retains strictly chronological vitals, laboratory values, and aligned chest
    radiographs to test the empirical value of external biomedical knowledge.

    Args:
        stay_windows: List of four window records for a single stay, ordered 0 to 3.
        cxr_dir: Base directory for radiographs (defaults to config.MIMIC_CXR_DIR).

    Returns:
        Tuple of (prompt_text, image_paths).
    """
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

        header = format_window_header(idx)
        vitals = format_vitals_line(win)
        labs = format_labs_line(win)
        # Note: Zero KG triples or headers emitted in No-KG prompt
        block = f"{header}\n{vitals}\n{labs}\n{image_line}"
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


def build_ablation_dataset_dict(
    stays_by_split: Dict[str, List[Dict[str, Any]]],
    variant: str = "static_rag",
    tokenizer: Optional[Any] = None,
    max_seq_length: int = config.STEP4_MAX_SEQ_LENGTH,
    cxr_dir: Optional[Union[str, Path]] = None,
    max_edges_per_window: int = config.STEP3_MAX_EDGES_PER_WINDOW,
) -> DatasetDict:
    """Builds a Hugging Face DatasetDict for the requested ablation variant.

    Args:
        stays_by_split: Dict mapping split name ('train', 'validation', 'test')
            to lists of stay records. Each record must contain 'stay_id', 'subject_id',
            'label', 'windows' (list of 4 window dicts), and optionally 'subgraphs'.
        variant: Either 'static_rag' or 'no_kg'.
        tokenizer: Optional tokenizer for tokenization and causal loss masking.
        max_seq_length: Maximum allowed sequence length.
        cxr_dir: Radiograph directory root.
        max_edges_per_window: Maximum edges per window for Static-RAG.

    Returns:
        Configured DatasetDict instance.
    """
    if variant not in ("static_rag", "no_kg"):
        raise ValueError(f"Unknown variant '{variant}'. Expected 'static_rag' or 'no_kg'.")

    splits_data: Dict[str, Dataset] = {}

    for split_name, stay_records in stays_by_split.items():
        processed_records: List[Dict[str, Any]] = []

        for record in stay_records:
            stay_id = int(record["stay_id"])
            subject_id = int(record["subject_id"])
            label = int(record["label"])
            windows = record["windows"]
            subgraphs = record.get("subgraphs", None)

            if variant == "static_rag":
                prompt_text, image_paths = build_static_rag_prompt(
                    stay_windows=windows,
                    subgraphs=subgraphs,
                    cxr_dir=cxr_dir,
                    max_edges_per_window=max_edges_per_window,
                )
            else:  # no_kg
                prompt_text, image_paths = build_no_kg_prompt(
                    stay_windows=windows,
                    cxr_dir=cxr_dir,
                )

            target_completion = get_target_completion(label)

            item: Dict[str, Any] = {
                "stay_id": stay_id,
                "subject_id": subject_id,
                "prompt_text": prompt_text,
                "image_paths": image_paths,
                "target_completion": target_completion,
                "label": label,
            }

            if tokenizer is not None:
                tokenized = tokenize_and_mask_labels(
                    prompt_text=prompt_text,
                    target_completion=target_completion,
                    tokenizer=tokenizer,
                    max_seq_length=max_seq_length,
                )
                item["input_ids"] = tokenized["input_ids"]
                item["attention_mask"] = tokenized["attention_mask"]
                item["labels"] = tokenized["labels"]

            processed_records.append(item)

        splits_data[split_name] = Dataset.from_list(processed_records)

    return DatasetDict(splits_data)
