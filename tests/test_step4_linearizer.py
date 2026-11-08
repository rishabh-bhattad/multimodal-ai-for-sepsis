"""Unit tests for Step 4 timeline linearizer and PrimeKG knowledge graph serializer."""

import math
import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.step4_serialization import (
    format_labs_line,
    format_vitals_line,
    format_window_header,
    serialize_subgraph_edges,
    serialize_window_block,
)


def test_format_window_header_all_windows():
    """Verify headers for all 4 6-hour windows in the 24-hour observation period."""
    assert format_window_header(0) == "[TIMELINE: T - 24 hours to T - 18 hours]"
    assert format_window_header(1) == "[TIMELINE: T - 18 hours to T - 12 hours]"
    assert format_window_header(2) == "[TIMELINE: T - 12 hours to T - 6 hours]"
    assert format_window_header(3) == "[TIMELINE: T - 6 hours to T - 0 hours]"

    for invalid in (-1, 4, 10, "0", None, 1.0):
        with pytest.raises(ValueError):
            format_window_header(invalid)


def test_format_vitals_line_full_and_partial():
    """Verify vitals formatting with full, partial, and fallback measurements."""
    full_window = {
        "heart_rate_mean": 78.4,
        "map_latest": 82.1,
        "spo2_mean": 98.0,
        "temp_latest": 37.14,
        "resp_rate_mean": 18.2,
    }
    line = format_vitals_line(full_window)
    assert line == "- Vitals: Heart Rate 78 bpm, MAP 82 mmHg, SpO2 98%, Temp 37.1 C, Resp Rate 18 bpm."

    # Partial vitals
    partial_window = {
        "heart_rate_mean": 105.0,
        "map_latest": 65.0,
    }
    assert format_vitals_line(partial_window) == "- Vitals: Heart Rate 105 bpm, MAP 65 mmHg."

    # Fallback to temp_mean if temp_latest is missing
    fallback_window = {
        "temp_latest": None,
        "temp_mean": 38.56,
    }
    assert format_vitals_line(fallback_window) == "- Vitals: Temp 38.6 C."


def test_format_vitals_line_empty_and_nan():
    """Verify explicit absence statement when vitals are missing or NaN."""
    assert format_vitals_line({}) == "- Vitals: No vital signs recorded in this window."
    assert format_vitals_line({"heart_rate_mean": float("nan"), "map_latest": None}) == (
        "- Vitals: No vital signs recorded in this window."
    )


def test_format_labs_line_full_and_partial():
    """Verify labs formatting with full and partial observations."""
    full_labs = {
        "lactate_latest": 1.12,
        "platelets_latest": 240.0,
        "creatinine_latest": 0.89,
        "wbc_latest": 13.5,
        "bilirubin_latest": 0.7,
    }
    line = format_labs_line(full_labs)
    assert line == "- Labs: Lactate 1.1 mmol/L, Platelets 240 k/uL, Creatinine 0.9 mg/dL, WBC 13.5 k/uL, Bilirubin 0.7 mg/dL."

    partial_labs = {
        "lactate_latest": 4.6,
        "platelets_latest": 88.0,
    }
    assert format_labs_line(partial_labs) == "- Labs: Lactate 4.6 mmol/L, Platelets 88 k/uL."


def test_format_labs_line_empty_and_nan():
    """Verify explicit absence statement when labs are missing or NaN."""
    assert format_labs_line({}) == "- Labs: No laboratory measurements recorded in this window."
    assert format_labs_line({"lactate_latest": float("nan"), "platelets_latest": None}) == (
        "- Labs: No laboratory measurements recorded in this window."
    )


def test_serialize_subgraph_edges_empty_and_none():
    """Empty or None subgraphs must serialize to baseline physiological context."""
    assert serialize_subgraph_edges(None) == f"- Retrieved KG Prior: {config.STEP4_BASELINE_KG_TEXT}"

    g_empty = nx.MultiDiGraph()
    assert serialize_subgraph_edges(g_empty) == f"- Retrieved KG Prior: {config.STEP4_BASELINE_KG_TEXT}"

    g_nodes_no_edges = nx.MultiDiGraph()
    g_nodes_no_edges.add_node(("disease", "123"), node_name="Isolated Condition")
    assert serialize_subgraph_edges(g_nodes_no_edges) == f"- Retrieved KG Prior: {config.STEP4_BASELINE_KG_TEXT}"


def test_serialize_subgraph_edges_deterministic_ordering_and_syntax():
    """Verify determinism, sorting, and edge syntax for PrimeKG subgraphs."""
    g = nx.MultiDiGraph()
    g.add_node(("disease", "d1"), node_name="Pneumonia")
    g.add_node(("disease", "d2"), node_name="Bacterial Sepsis")
    g.add_node(("phenotype", "p1"), node_name="Tachycardia")

    # Add edges in non-alphabetical insertion order
    g.add_edge(("disease", "d1"), ("disease", "d2"), key="disease_disease", relation="disease_disease")
    g.add_edge(("disease", "d2"), ("phenotype", "p1"), key="disease_phenotype_positive", relation="disease_phenotype_positive")

    text = serialize_subgraph_edges(g)
    expected = (
        "- Retrieved KG Prior:\n"
        "  [Bacterial Sepsis] -> disease_phenotype_positive -> [Tachycardia]\n"
        "  [Pneumonia] -> disease_disease -> [Bacterial Sepsis]"
    )
    assert text == expected


def test_serialize_subgraph_edges_capping():
    """Verify that edges are budgeted strictly up to max_edges."""
    g = nx.MultiDiGraph()
    for i in range(10):
        g.add_node(f"n{i}", node_name=f"Concept {chr(65 + i)}")
    for i in range(9):
        g.add_edge(f"n{i}", f"n{i+1}", key="rel", relation="rel")

    capped_text = serialize_subgraph_edges(g, max_edges=3)
    lines = capped_text.strip().split("\n")
    assert lines[0] == "- Retrieved KG Prior:"
    # Exactly 3 triple lines
    assert len(lines) == 4


def test_serialize_window_block():
    """Verify end-to-end assembly of a single 6-hour window block."""
    g = nx.MultiDiGraph()
    g.add_node("a", node_name="Consolidation")
    g.add_node("b", node_name="Pneumonia")
    g.add_edge("a", "b", key="disease_disease", relation="disease_disease")

    window_data = {
        "heart_rate_mean": 110.0,
        "map_latest": 62.0,
        "lactate_latest": 3.1,
    }
    block = serialize_window_block(
        window_idx=2,
        window_data=window_data,
        subgraph=g,
        image_line=f"- Image: {config.STEP4_IMAGE_PLACEHOLDER}",
    )

    lines = block.split("\n")
    assert lines[0] == "[TIMELINE: T - 12 hours to T - 6 hours]"
    assert lines[1] == "- Vitals: Heart Rate 110 bpm, MAP 62 mmHg."
    assert lines[2] == "- Labs: Lactate 3.1 mmol/L."
    assert lines[3] == f"- Image: {config.STEP4_IMAGE_PLACEHOLDER}"
    assert lines[4] == "- Retrieved KG Prior:"
    assert lines[5] == "  [Consolidation] -> disease_disease -> [Pneumonia]"

