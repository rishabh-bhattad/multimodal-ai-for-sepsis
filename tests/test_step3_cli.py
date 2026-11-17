"""Synthetic CLI, sampling, aggregate-report, and pickle output checks."""

import csv
import hashlib
import json
import logging
import pickle
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import step3_graph_rag as step3


def synthetic_finding(node_id="a", *, unresolved=False):
    return {
        "name": "PRIVATE_FINDING_SENTINEL", "predicate": "exhibits", "source": "map_latest",
        "value": 60.0, "matched_query": "PRIVATE_QUERY_SENTINEL",
        "similarity": 0.5 if unresolved else 0.9,
        "mapping_status": "unresolved" if unresolved else "matched",
        "node": None if unresolved else {
            "node_type": "disease", "node_id": node_id, "node_name": f"Node {node_id}",
        },
    }


@pytest.fixture
def records():
    result = {}
    start = datetime(2020, 1, 1)
    for subject in range(1, 13):
        for variant in range(2):
            windows = {}
            for index in range(4):
                windows[f"window_{index}"] = {
                    "start": (start + timedelta(hours=6 * index)).isoformat(),
                    "end": (start + timedelta(hours=6 * (index + 1))).isoformat(),
                    "cxr_coverage": {"chexpert": "no_image", "negbio": "no_image"},
                    "findings": [],
                }
            windows["window_0"]["findings"] = [synthetic_finding("a"), synthetic_finding("c")]
            windows["window_1"]["findings"] = [synthetic_finding(unresolved=True)]
            windows["window_2"]["findings"] = [synthetic_finding("absent")]
            if subject % 2 == 0 and variant == 1:
                cxr = synthetic_finding()
                cxr.update(source="cxr_labels", predicate="shows_radiographic", value=None,
                           label_sources=["chexpert", "negbio"])
                windows["window_0"]["findings"].append(cxr)
                windows["window_0"]["cxr_coverage"] = {"chexpert": "positive", "negbio": "positive"}
            result[str(90000000 + subject * 10 + variant)] = {
                "subject_id": 80000000 + subject,
                "split": "train" if subject <= 8 else "val" if subject <= 10 else "test",
                "windows": windows,
            }
    return result


@pytest.fixture
def files(tmp_path, records):
    source = tmp_path / "PRIVATE_PATH_SENTINEL.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    graph_path = tmp_path / "kg.csv"
    with graph_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["relation", "x_type", "x_id", "x_name", "y_type", "y_id", "y_name"])
        for first, second in (("a", "b"), ("c", "b")):
            writer.writerow(["disease_disease", "disease", first, f"Node {first}",
                             "disease", second, f"Node {second}"])
        writer.writerow(["disease_phenotype_negative", "disease", "a", "Node a",
                         "effect/phenotype", "p", "Synthetic phenotype"])
    return source, graph_path


def has_image(stay):
    return any(window["cxr_coverage"]["chexpert"] != "no_image" for window in stay["windows"].values())


def test_smoke_sampling_is_unique_reproducible_and_covers_images(records):
    original = deepcopy(records)
    selected = step3.select_trajectories(records)
    assert len(selected) == len({stay["subject_id"] for stay in selected.values()}) == 10
    assert {has_image(stay) for stay in selected.values()} == {True, False}
    assert all(len(stay["windows"]) == 4 for stay in selected.values())
    for key, stay in selected.items():
        subject_stays = [other for other in records.values() if other["subject_id"] == stay["subject_id"]]
        assert has_image(stay) == any(has_image(other) for other in subject_stays)
        assert stay["split"] == records[key]["split"]
    shuffled = dict(reversed(list(records.items())))
    assert list(selected) == list(step3.select_trajectories(shuffled))
    assert records == original


def test_full_mode_keeps_every_stay_and_validates_splits_before_sampling(records):
    assert step3.select_trajectories(records, None) == records
    assert len(step3.select_trajectories(records, None)) == 24
    records[next(reversed(records))]["split"] = "train"
    with pytest.raises(ValueError, match="split isolation"):
        step3.select_trajectories(records, 1)


@pytest.mark.parametrize("size", [0, -1, True, 1.5, 13])
def test_invalid_or_unavailable_sample_fails_without_full_fallback(records, size):
    with pytest.raises(ValueError):
        step3.select_trajectories(records, size)


def test_single_subject_sample_and_all_no_image_inputs(records):
    selected = step3.select_trajectories(records, 1)
    assert len(selected) == 1 and has_image(next(iter(selected.values())))
    no_images = {key: stay for key, stay in records.items() if not has_image(stay)}
    assert len(step3.select_trajectories(no_images, 10)) == 10


@pytest.mark.parametrize("payload", [
    '{"1": {}, "1": {}}', '{"1": {"windows": {}, "windows": {}}}',
    '{"value": NaN}', '{"value": Infinity}',
])
def test_loader_rejects_duplicate_keys_and_nonfinite_constants(tmp_path, payload):
    path = tmp_path / "input.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        step3.load_temporal_quadruples(path)


@pytest.mark.parametrize("full", [False, True])
def test_real_cli_writes_matching_pickle_report_and_private_data_only_locally(tmp_path, records, files, full):
    source, graph_path = files
    output = tmp_path / ("full" if full else "smoke") / "subgraph_trajectories.pkl"
    before = source.read_bytes(), graph_path.read_bytes()
    command = [sys.executable, str(Path(step3.__file__)), "--input", str(source),
               "--primekg", str(graph_path), "--output", str(output)]
    if full:
        command.append("--full")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    with output.open("rb") as stored:
        result = pickle.load(stored)
    report_text = (output.parent / "step3_validation.json").read_text(encoding="utf-8")
    report = json.loads(report_text)
    expected_stays, expected_patients = (24, 12) if full else (10, 10)
    assert report["validation"] == "passed"
    assert report["mode"] == ("full" if full else "sample")
    assert report["stays"] == len(result) == expected_stays
    assert report["unique_patients"] == expected_patients
    assert report["windows"] == expected_stays * 4
    assert report["empty_finding_windows"] == expected_stays
    assert report["empty_graph_windows"] == report["edgeless_windows"] == expected_stays * 3
    assert report["matched_findings"] + report["unresolved_findings"] == report["findings"]
    assert report["missing_graph_findings"] == report["unresolved_findings"] == expected_stays
    assert report["path_edges_retained"] == expected_stays * 2
    assert report["negative_association_edges"] == expected_stays
    assert report["retrieved_edges"] == expected_stays * 3
    assert report["windows_with_image"] >= 1
    assert report["max_edges_in_window"] <= report["limits"]["max_edges_per_window"] == 200
    assert report["pickle_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["graph_provenance"]["source_sha256"] == hashlib.sha256(graph_path.read_bytes()).hexdigest()
    for stay_id, stay in result.items():
        assert stay["subject_id"] == records[stay_id]["subject_id"]
        assert stay["split"] == records[stay_id]["split"]
        for key, graph in stay["windows"].items():
            assert isinstance(graph, nx.MultiDiGraph)
            assert graph.graph["findings"] == records[stay_id]["windows"][key]["findings"]
    published = completed.stdout + completed.stderr + report_text
    for marker in ("PRIVATE_FINDING_SENTINEL", "PRIVATE_QUERY_SENTINEL", "PRIVATE_PATH_SENTINEL", *records):
        assert marker not in published
    assert "Step 3 summary:" in completed.stderr
    assert (source.read_bytes(), graph_path.read_bytes()) == before
    assert not list(output.parent.glob(".step3-*.tmp"))
    if full:
        assert report["patients_per_split"] == {"train": 8, "val": 2, "test": 2}
        assert report["stays_per_split"] == {"train": 16, "val": 4, "test": 4}
        assert report["windows_with_image"] == 6


def test_cli_defaults_use_configuration(tmp_path, files, monkeypatch):
    import config

    source, graph_path = files
    output = tmp_path / "configured" / "subgraph_trajectories.pkl"
    monkeypatch.setattr(config, "TEMPORAL_QUADRUPLES_JSON", source)
    monkeypatch.setattr(config, "PRIMEKG_CSV", graph_path)
    monkeypatch.setattr(config, "SUBGRAPH_TRAJECTORIES_PKL", output)
    assert step3.main([]) == 0
    report = json.loads((output.parent / "step3_validation.json").read_text())
    assert report["unique_patients"] == 10


@pytest.mark.parametrize("arguments", [
    ["--full", "--sample_size", "10"], ["--sample_size", "0"], ["--output", "wrong.json"],
])
def test_cli_rejects_invalid_modes_and_output_extension(arguments):
    with pytest.raises(SystemExit) as caught:
        step3.main(arguments)
    assert caught.value.code == 2


def test_cli_failure_does_not_publish_artifacts_or_sensitive_exceptions(tmp_path, files, caplog):
    source, graph_path = files
    source.write_text('{"PRIVATE_RECORD_SENTINEL": broken}', encoding="utf-8")
    output = tmp_path / "failed" / "subgraph_trajectories.pkl"
    with caplog.at_level(logging.ERROR, logger=step3.logger.name):
        assert step3.main(["--input", str(source), "--primekg", str(graph_path), "--output", str(output)]) == 1
    assert "Step 3 failed (JSONDecodeError)" in caplog.text
    assert "PRIVATE_RECORD_SENTINEL" not in caplog.text
    assert "PRIVATE_PATH_SENTINEL" not in caplog.text
    assert not output.exists()
    assert not (output.parent / "step3_validation.json").exists()


def test_output_serialization_failure_preserves_existing_files_and_cleans_temporary_files(tmp_path):
    output = tmp_path / "subgraph_trajectories.pkl"
    report_path = tmp_path / "step3_validation.json"
    output.write_bytes(b"previous pickle")
    report_path.write_bytes(b"previous report")
    with pytest.raises(TypeError):
        step3.write_trajectory_outputs({}, {"not_json_serializable": {1, 2}}, output)
    assert output.read_bytes() == b"previous pickle"
    assert report_path.read_bytes() == b"previous report"
    assert not list(tmp_path.glob(".step3-*.tmp"))


def test_cli_refuses_to_overwrite_an_input_artifact(tmp_path, files):
    source, graph_path = files
    protected = tmp_path / "input.pkl"
    protected.write_bytes(source.read_bytes())
    before = protected.read_bytes()
    assert step3.main(["--input", str(protected), "--primekg", str(graph_path), "--output", str(protected)]) == 1
    assert protected.read_bytes() == before


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
