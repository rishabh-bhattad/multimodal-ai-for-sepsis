"""Exercise the unchanged Step 2 assembler through Step 3's real CLI."""

import csv
import json
import os
import pickle
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src import step2_extraction as step2
from src import step3_graph_rag as step3


@pytest.fixture
def assembled_input(tmp_path):
    """Use real threshold/extraction code with synthetic, prelinked vocabulary."""
    rows, radiographic = [], {}
    for subject in range(1, 11):
        for index in range(4):
            start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=subject, hours=index * 6)
            values = {f"{name}_latest": float("nan") for name in config.LAB_SEVERITY_THRESHOLDS}
            if index == 0:
                values.update(map_latest=60.0, wbc_latest=14.0)
            elif index == 2:
                values["creatinine_latest"] = 2.0
            image = None
            if subject % 2 == 0 and index == 3:
                study = 70000000 + subject
                image = f"synthetic/p{subject}/s{study}/image.jpg"
                radiographic[study] = {
                    "coverage": {"chexpert": "positive", "negbio": "covered_no_positive"},
                    "findings": {"Pneumonia": {"chexpert"}},
                }
            rows.append({
                "subject_id": 80000000 + subject, "stay_id": 90000000 + subject,
                "window_idx": index, "window_start": start,
                "window_end": start + pd.Timedelta(hours=6), "cxr_path": image,
                "split": "train" if subject <= 7 else "val" if subject == 8 else "test",
                **values,
            })
    identities = {
        "Systemic Hypotension": ("effect/phenotype", "001"),
        "Leukocytosis": ("effect/phenotype", "002"),
        "Pneumonia": ("disease", "3"),
        "Elevated Creatinine": None,
    }
    mappings = {
        name: {
            "node": None if identity is None else {
                "node_type": identity[0], "node_id": identity[1], "node_name": name,
            },
            "similarity": 0.5 if identity is None else 0.9,
            "matched_query": name, "mapping_status": "unresolved" if identity is None else "matched",
        }
        for name, identity in identities.items()
    }
    windows = pd.DataFrame(rows)
    original = windows.copy(deep=True)
    mapping_before = deepcopy(mappings)
    trajectories = step2.assemble_trajectories(windows, radiographic, mappings)
    pd.testing.assert_frame_equal(windows, original)
    assert mappings == mapping_before
    source = tmp_path / "temporal_quadruples.json"
    source.write_text(json.dumps(trajectories, allow_nan=False), encoding="utf-8")
    graph_path = tmp_path / "kg.csv"
    with graph_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["relation", "x_type", "x_id", "x_name", "y_type", "y_id", "y_name"])
        writer.writerows([
            ["disease_phenotype_positive", "disease", "4", "Synthetic disease context",
             "effect/phenotype", "001", "Systemic Hypotension"],
            ["disease_phenotype_positive", "disease", "4", "Synthetic disease context",
             "effect/phenotype", "002", "Leukocytosis"],
            ["disease_phenotype_positive", "disease", "3", "Pneumonia",
             "effect/phenotype", "001", "Systemic Hypotension"],
            ["disease_phenotype_negative", "disease", "5", "Synthetic negative context",
             "effect/phenotype", "001", "Systemic Hypotension"],
            ["disease_phenotype_negative", "disease", "3", "Pneumonia",
             "effect/phenotype", "003", "Synthetic phenotype"],
        ])
    return trajectories, source, graph_path


def test_step2_to_step3_cli_preserves_observations_and_is_deterministic_across_processes(tmp_path, assembled_input):
    source_records, source, graph_path = assembled_input
    source_bytes, graph_bytes = source.read_bytes(), graph_path.read_bytes()
    runs, reports = [], []
    for hash_seed in (1, 987):
        output = tmp_path / f"run_{hash_seed}" / "subgraph_trajectories.pkl"
        completed = subprocess.run(
            [sys.executable, str(Path(step3.__file__)), "--input", str(source),
             "--primekg", str(graph_path), "--sample_size", "10", "--output", str(output)],
            env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
            capture_output=True, text=True, timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        with output.open("rb") as stored:
            runs.append(pickle.load(stored))
        reports.append(json.loads((output.parent / "step3_validation.json").read_text()))
    for report in reports:
        assert report["unique_patients"] == report["stays"] == 10
        assert report["windows"] == 40
        assert report["patients_per_split"] == {"train": 7, "val": 1, "test": 2}
        assert report["windows_with_image"] == 5
        assert report["unresolved_findings"] == 10
        assert report["missing_graph_findings"] == 0
        assert report["max_edges_in_window"] == 4
        assert report["path_edges_retained"] == 20
        assert report["negative_association_edges"] == 15
    assert {key: value for key, value in reports[0].items() if key != "pickle_sha256"} == {
        key: value for key, value in reports[1].items() if key != "pickle_sha256"
    }
    first, second = runs
    assert list(first) == list(second)
    graphs = []
    for stay_id, stay in first.items():
        assert stay["subject_id"] == source_records[stay_id]["subject_id"]
        assert stay["split"] == source_records[stay_id]["split"]
        for key, graph in stay["windows"].items():
            graphs.append(graph)
            other = second[stay_id]["windows"][key]
            assert nx.utils.graphs_equal(graph, other)
            assert list(graph) == list(other)
            assert list(graph.edges(keys=True)) == list(other.edges(keys=True))
            source_window = source_records[stay_id]["windows"][key]
            for field in ("start", "end", "findings", "cxr_coverage"):
                assert graph.graph[field] == source_window[field]
        baseline = stay["windows"]["window_0"]
        assert baseline.graph["retrieval"]["active_seed_nodes"] == (
            ("effect/phenotype", "001"), ("effect/phenotype", "002"),
        )
        # A retrieved disease is context, even when it is observed in a later window.
        assert not baseline.nodes[("disease", "3")]["is_active_seed"]
        assert not stay["windows"]["window_1"]
        unresolved = stay["windows"]["window_2"]
        assert not unresolved
        assert unresolved.graph["findings"][0]["name"] == "Elevated Creatinine"
        cxr_graph = stay["windows"]["window_3"]
        if cxr_graph:
            assert cxr_graph.nodes[("disease", "3")]["is_active_seed"]
            assert not cxr_graph.nodes[("effect/phenotype", "001")]["is_active_seed"]
            assert cxr_graph.graph["findings"][0]["label_sources"] == ["chexpert"]
            assert cxr_graph.graph["findings"][0]["value"] is None
    assert len({id(graph) for graph in graphs}) == 40
    assert source.read_bytes() == source_bytes
    assert graph_path.read_bytes() == graph_bytes
