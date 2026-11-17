"""Build static RAG and no-KG comparisons from the same observations and images."""

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
    """Collapses multiple window subgraphs into a single merged graph."""
    merged = nx.MultiDiGraph()
    if not subgraphs:
        return merged

    for win_key, sg in subgraphs.items():
        if sg is None or not isinstance(sg, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            continue

        for node, attrs in sg.nodes(data=True):
            if node not in merged:
                merged.add_node(node, **attrs)
            else:
                merged.nodes[node].update(attrs)

        for u, v, data in sg.edges(data=True):
            merged.add_edge(u, v, **data)

    return merged


def build_static_rag_prompt(
    stay_windows: List[Dict[str, Any]],
    subgraphs: Optional[Dict[str, nx.MultiDiGraph]] = None,
    cxr_dir: Optional[Union[str, Path]] = None,
    max_edges_per_window: int = config.STEP3_MAX_EDGES_PER_WINDOW,
) -> Tuple[str, List[str]]:
    """Render the same merged graph in each window for the static RAG comparison."""
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
    """Assembles a No-KG prompt with all PrimeKG biomedical triples omitted."""
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
    """Build the comparison datasets from the same window records and binary target."""
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
