"""Unit tests for Step 5 Static-RAG and No-KG ablation baseline transforms."""

from pathlib import Path
import pytest
import networkx as nx

import config
from src.step4_serialization import assemble_multimodal_prompt, format_window_header
from src.step5_ablations import (
    merge_subgraphs,
    build_static_rag_prompt,
    build_no_kg_prompt,
    build_ablation_dataset_dict,
)


def create_sample_subgraph(edges):
    """Helper to create a MultiDiGraph with given (source, relation, target) tuples."""
    g = nx.MultiDiGraph()
    for src, rel, tgt in edges:
        g.add_node(src, name=src, node_type="phenotype")
        g.add_node(tgt, name=tgt, node_type="disease")
        g.add_edge(src, tgt, relation=rel, display_relation=rel)
    return g


@pytest.fixture
def sample_stay_windows():
    """Provides a synthetic 4-window stay with vitals, labs, and 1 CXR."""
    return [
        {
            "window_idx": 0,
            "heart_rate_mean": 92.0,
            "map_latest": 78.0,
            "spo2_mean": 97.0,
            "lactate_latest": 1.4,
            "wbc_latest": 9.2,
            "creatinine_latest": 0.9,
            "platelets_latest": 220.0,
            "cxr_path": None,
        },
        {
            "window_idx": 1,
            "heart_rate_mean": 105.0,
            "map_latest": 68.0,
            "spo2_mean": 94.0,
            "lactate_latest": 2.2,
            "wbc_latest": 12.8,
            "creatinine_latest": 1.1,
            "platelets_latest": 190.0,
            "cxr_path": "files/p10/p10001/s5001/img01.jpg",
        },
        {
            "window_idx": 2,
            "heart_rate_mean": 118.0,
            "map_latest": 58.0,
            "spo2_mean": 92.0,
            "lactate_latest": 3.6,
            "wbc_latest": 16.4,
            "creatinine_latest": 1.5,
            "platelets_latest": 140.0,
            "cxr_path": None,
        },
        {
            "window_idx": 3,
            "heart_rate_mean": 128.0,
            "map_latest": 50.0,
            "spo2_mean": 89.0,
            "lactate_latest": 4.8,
            "wbc_latest": 19.5,
            "creatinine_latest": 1.9,
            "platelets_latest": 85.0,
            "cxr_path": None,
        },
    ]


@pytest.fixture
def sample_temporal_subgraphs():
    """Provides distinct temporal subgraphs across windows."""
    return {
        "window_0": create_sample_subgraph([("Tachycardia", "manifests", "Sepsis")]),
        "window_1": create_sample_subgraph([("Pneumonia", "triggers", "Sepsis")]),
        "window_2": create_sample_subgraph([("Hypoperfusion", "induces", "Lactic Acidosis")]),
        "window_3": create_sample_subgraph([("Septic Shock", "causes", "Organ Failure")]),
    }


def test_merge_subgraphs_empty():
    assert len(merge_subgraphs(None).nodes) == 0
    assert len(merge_subgraphs({}).nodes) == 0
    assert len(merge_subgraphs({"window_0": None}).nodes) == 0


def test_merge_subgraphs_union(sample_temporal_subgraphs):
    merged = merge_subgraphs(sample_temporal_subgraphs)
    # Total distinct edges: 4
    assert merged.number_of_edges() == 4
    # Check all nodes exist in union
    expected_nodes = {
        "Tachycardia", "Sepsis", "Pneumonia", "Hypoperfusion",
        "Lactic Acidosis", "Septic Shock", "Organ Failure",
    }
    assert expected_nodes.issubset(set(merged.nodes))


def test_build_static_rag_prompt(sample_stay_windows, sample_temporal_subgraphs):
    prompt_text, image_paths = build_static_rag_prompt(
        stay_windows=sample_stay_windows,
        subgraphs=sample_temporal_subgraphs,
    )
    assert isinstance(prompt_text, str)
    assert len(image_paths) == 1
    assert prompt_text.count("<image>") == 1

    # Verify all 4 window headers exist
    for i in range(4):
        header = format_window_header(i)
        assert header in prompt_text

    # In Static-RAG, the union triples should appear in each window's serialized block
    # Check that triple from window 3 appears across the prompt
    assert "[Septic Shock] -> causes -> [Organ Failure]" in prompt_text
    assert "[Pneumonia] -> triggers -> [Sepsis]" in prompt_text


def test_build_static_rag_prompt_validations(sample_stay_windows):
    # Less than 4 windows
    with pytest.raises(ValueError, match="Expected exactly 4 windows"):
        build_static_rag_prompt(sample_stay_windows[:3])

    # Non-sequential / duplicated window indices
    bad_order = [dict(w) for w in sample_stay_windows]
    bad_order[3]["window_idx"] = 1
    with pytest.raises(ValueError, match="sequential indices"):
        build_static_rag_prompt(bad_order)


def test_build_no_kg_prompt(sample_stay_windows):
    prompt_text, image_paths = build_no_kg_prompt(
        stay_windows=sample_stay_windows,
    )
    assert isinstance(prompt_text, str)
    assert len(image_paths) == 1
    assert prompt_text.count("<image>") == 1

    # No-KG MUST NOT contain any PrimeKG section or triple arrows
    assert "- Retrieved KG Prior:" not in prompt_text
    assert " -> " not in prompt_text

    # But it MUST contain all chronological clinical vitals and labs
    assert "Heart Rate 92 bpm" in prompt_text
    assert "Lactate 4.8 mmol/L" in prompt_text
    assert "- Image: <image> Frontal Chest Radiograph acquired." in prompt_text


def test_alignment_consistency_across_variants(sample_stay_windows, sample_temporal_subgraphs):
    """Verifies that T-RAG, Static-RAG, and No-KG preserve identical clinical timelines and CXR alignments."""
    trag_prompt, trag_imgs = assemble_multimodal_prompt(
        stay_windows=sample_stay_windows,
        subgraphs=sample_temporal_subgraphs,
    )
    static_prompt, static_imgs = build_static_rag_prompt(
        stay_windows=sample_stay_windows,
        subgraphs=sample_temporal_subgraphs,
    )
    nokg_prompt, nokg_imgs = build_no_kg_prompt(
        stay_windows=sample_stay_windows,
    )

    # 1. Exact image path consistency
    assert trag_imgs == static_imgs == nokg_imgs

    # 2. Exact image placeholder count
    assert trag_prompt.count("<image>") == static_prompt.count("<image>") == nokg_prompt.count("<image>") == 1

    # 3. System context and Target Task must be identical
    assert config.STEP4_SYSTEM_CONTEXT in trag_prompt
    assert config.STEP4_SYSTEM_CONTEXT in static_prompt
    assert config.STEP4_SYSTEM_CONTEXT in nokg_prompt

    assert config.STEP4_TARGET_TASK in trag_prompt
    assert config.STEP4_TARGET_TASK in static_prompt
    assert config.STEP4_TARGET_TASK in nokg_prompt


class MockTokenizer:
    """Mock tokenizer returning token IDs and attention mask."""

    def __call__(self, text, truncation=True, max_length=4096, add_special_tokens=True):
        tokens = [hash(w) % 1000 for w in text.split()]
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
        }

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 1000 for w in text.split()]


def test_build_ablation_dataset_dict(sample_stay_windows, sample_temporal_subgraphs):
    stays_by_split = {
        "train": [
            {
                "stay_id": 1001,
                "subject_id": 501,
                "label": 1,
                "windows": sample_stay_windows,
                "subgraphs": sample_temporal_subgraphs,
            }
        ],
        "test": [
            {
                "stay_id": 1002,
                "subject_id": 502,
                "label": 0,
                "windows": sample_stay_windows,
                "subgraphs": sample_temporal_subgraphs,
            }
        ],
    }

    tokenizer = MockTokenizer()

    # Static-RAG dataset
    static_dd = build_ablation_dataset_dict(
        stays_by_split=stays_by_split,
        variant="static_rag",
        tokenizer=tokenizer,
    )
    assert "train" in static_dd
    assert "test" in static_dd
    train_item = static_dd["train"][0]
    assert train_item["stay_id"] == 1001
    assert train_item["target_completion"] == "Sepsis"
    assert "input_ids" in train_item
    assert "labels" in train_item
    # Check causal loss masking (-100 present for prompt tokens)
    assert -100 in train_item["labels"]

    # No-KG dataset
    nokg_dd = build_ablation_dataset_dict(
        stays_by_split=stays_by_split,
        variant="no_kg",
        tokenizer=tokenizer,
    )
    test_item = nokg_dd["test"][0]
    assert test_item["stay_id"] == 1002
    assert test_item["target_completion"] == "Non-Sepsis"
    assert "- Retrieved KG Prior:" not in test_item["prompt_text"]

    # Invalid variant error
    with pytest.raises(ValueError, match="Unknown variant"):
        build_ablation_dataset_dict(stays_by_split, variant="invalid_variant")
