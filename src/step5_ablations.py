"""Build static RAG and no-KG comparisons from the same observations and images."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
from datasets import Dataset, DatasetDict

import config
from src.step4_serialization import (
    assemble_multimodal_prompt,
    get_target_completion,
    tokenize_and_mask_labels,
)

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
    union_graph = merge_subgraphs(subgraphs)
    return assemble_multimodal_prompt(
        stay_windows,
        subgraphs={f"window_{index}": union_graph for index in range(config.NUM_WINDOWS)},
        cxr_dir=cxr_dir,
        max_edges_per_window=max_edges_per_window,
    )


def build_no_kg_prompt(
    stay_windows: List[Dict[str, Any]],
    cxr_dir: Optional[Union[str, Path]] = None,
) -> Tuple[str, List[str]]:
    """Assembles a No-KG prompt with all PrimeKG biomedical triples omitted."""
    return assemble_multimodal_prompt(stay_windows, cxr_dir=cxr_dir, include_graph=False)


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
