"""Synthetic four-window contracts and temporal/stay isolation checks."""

import csv
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.step3_graph_rag import (
    assemble_subgraph_trajectories,
    load_primekg_graph,
    validate_temporal_quadruples,
)


def finding(node_id="a", *, unresolved=False):
    return {
        "name": "Synthetic observation", "predicate": "exhibits", "source": "map_latest",
        "value": 60.0, "matched_query": "Synthetic observation",
        "similarity": 0.5 if unresolved else 0.9,
        "mapping_status": "unresolved" if unresolved else "matched",
        "node": None if unresolved else {
            "node_type": "disease", "node_id": node_id, "node_name": f"Node {node_id}",
        },
    }


@pytest.fixture
def graph(tmp_path):
    path = tmp_path / "kg.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["relation", "x_type", "x_id", "x_name", "y_type", "y_id", "y_name"])
        for first, second in (("a", "b"), ("b", "c"), ("x", "y")):
            writer.writerow(["disease_disease", "disease", first, f"Node {first}",
                             "disease", second, f"Node {second}"])
    result = load_primekg_graph(path)
    result.add_node(("disease", "isolated"), node_type="disease",
                    node_id="isolated", node_name="Isolated synthetic node")
    return result


@pytest.fixture
def trajectories():
    result = {}
    start = datetime(2020, 1, 1)
    for stay_id, subject, split in (("10", 1, "train"), ("11", 1, "train"), ("20", 2, "val")):
        windows = {}
        for index in range(4):
            windows[f"window_{index}"] = {
                "start": (start + timedelta(hours=6 * index)).isoformat(),
                "end": (start + timedelta(hours=6 * (index + 1))).isoformat(),
                "cxr_coverage": {"chexpert": "no_image", "negbio": "no_image"},
                "findings": [],
            }
        windows["window_0"]["findings"] = [finding()]
        windows["window_1"]["findings"] = [finding(unresolved=True)]
        windows["window_2"]["findings"] = [finding("absent")]
        windows["window_2"]["cxr_coverage"] = {
            "chexpert": "covered_no_positive", "negbio": "missing_study",
        }
        result[stay_id] = {"subject_id": subject, "split": split, "windows": windows}
    return result


def test_four_graphs_keep_all_stays_splits_empty_windows_and_provenance(graph, trajectories):
    before_input, before_graph = deepcopy(trajectories), deepcopy(graph)
    result = assemble_subgraph_trajectories(graph, trajectories)
    objects = []
    assert set(result) == set(trajectories)
    for stay_id, stay in result.items():
        source = trajectories[stay_id]
        assert stay["subject_id"] == source["subject_id"]
        assert stay["split"] == source["split"]
        assert list(stay["windows"]) == [f"window_{index}" for index in range(4)]
        for index, (key, window_graph) in enumerate(stay["windows"].items()):
            objects.append(window_graph)
            assert isinstance(window_graph, nx.MultiDiGraph)
            assert window_graph.graph["window_idx"] == index
            assert window_graph.graph["provenance"] == graph.graph["provenance"]
            for field in ("start", "end", "cxr_coverage", "findings"):
                assert window_graph.graph[field] == source["windows"][key][field]
        empty = stay["windows"]["window_3"]
        assert not empty and empty.number_of_edges() == 0
        assert empty.graph["findings"] == []
        assert empty.graph["cxr_coverage"] == {"chexpert": "no_image", "negbio": "no_image"}
    assert len({id(item) for item in objects}) == 12
    assert trajectories == before_input
    assert nx.utils.graphs_equal(graph, before_graph)


def test_unresolved_missing_and_disconnected_concepts_are_separate(graph, trajectories):
    trajectories["10"]["windows"]["window_3"]["findings"] = [finding("isolated"), finding("x")]
    windows = assemble_subgraph_trajectories(graph, trajectories)["10"]["windows"]
    unresolved = windows["window_1"]
    assert not unresolved
    assert unresolved.graph["findings"][0]["mapping_status"] == "unresolved"
    assert unresolved.graph["retrieval"]["unresolved_findings"] == 1
    assert unresolved.graph["retrieval"]["missing_graph_findings"] == 0
    missing = windows["window_2"]
    assert not missing
    assert missing.graph["findings"][0]["mapping_status"] == "matched"
    assert missing.graph["retrieval"]["missing_seed_nodes"] == (("disease", "absent"),)
    assert missing.graph["retrieval"]["missing_graph_findings"] == 1
    assert missing.graph["retrieval"]["unresolved_findings"] == 0
    disconnected = windows["window_3"]
    assert disconnected.graph["retrieval"]["seeds_without_connecting_paths"] == (
        ("disease", "isolated"), ("disease", "x"),
    )
    assert disconnected.graph["retrieval"]["isolated_seed_nodes"] == (("disease", "isolated"),)
    assert set(disconnected) == {("disease", name) for name in ("isolated", "x", "y")}


def test_a_new_finding_changes_only_its_own_window(graph, trajectories):
    original = assemble_subgraph_trajectories(graph, trajectories)
    changed = deepcopy(trajectories)
    changed["10"]["windows"]["window_3"]["findings"].append(finding("c"))
    revised = assemble_subgraph_trajectories(graph, changed)
    for stay_id, stay in original.items():
        for key, window_graph in stay["windows"].items():
            if (stay_id, key) != ("10", "window_3"):
                assert nx.utils.graphs_equal(window_graph, revised[stay_id]["windows"][key])
    assert not original["10"]["windows"]["window_3"]
    assert set(revised["10"]["windows"]["window_3"]) == {("disease", "b"), ("disease", "c")}


def test_mutating_output_cannot_change_other_stays_windows_or_inputs(graph, trajectories):
    trajectories["10"]["windows"]["window_3"]["findings"] = [finding()]
    original_input, original_graph = deepcopy(trajectories), deepcopy(graph)
    result = assemble_subgraph_trajectories(graph, trajectories)
    source = result["10"]["windows"]["window_0"]
    same_window_other_stay = result["11"]["windows"]["window_0"]
    same_stay_other_window = result["10"]["windows"]["window_3"]
    source.nodes[("disease", "a")]["node_name"] = "Changed"
    source.edges[("disease", "a"), ("disease", "b"), "disease_disease"]["relation"] = "changed"
    source.graph["findings"][0]["node"]["node_name"] = "Changed"
    source.graph["cxr_coverage"]["chexpert"] = "positive"
    source.graph["provenance"]["allowed_node_types"].append("changed")
    for untouched in (same_window_other_stay, same_stay_other_window):
        assert untouched.nodes[("disease", "a")]["node_name"] == "Node a"
        assert untouched.edges[("disease", "a"), ("disease", "b"), "disease_disease"]["relation"] == "disease_disease"
        assert untouched.graph["findings"][0]["node"]["node_name"] == "Node a"
        assert untouched.graph["cxr_coverage"]["chexpert"] == "no_image"
        assert "changed" not in untouched.graph["provenance"]["allowed_node_types"]
    assert trajectories == original_input
    assert nx.utils.graphs_equal(graph, original_graph)


def test_input_order_does_not_change_window_order_or_retrieval(graph, trajectories):
    original = assemble_subgraph_trajectories(graph, trajectories, max_edges=1)
    shuffled = {
        key: {**stay, "windows": dict(reversed(list(stay["windows"].items())))}
        for key, stay in reversed(list(trajectories.items()))
    }
    revised = assemble_subgraph_trajectories(graph, shuffled, max_edges=1)
    assert list(original) == list(revised)
    for stay_id in original:
        assert list(original[stay_id]["windows"]) == list(revised[stay_id]["windows"])
        for key, window_graph in original[stay_id]["windows"].items():
            assert nx.utils.graphs_equal(window_graph, revised[stay_id]["windows"][key])
            assert window_graph.number_of_edges() <= 1


def test_cohort_labels_and_anchors_do_not_seed_graphs(graph, trajectories):
    original = assemble_subgraph_trajectories(graph, trajectories)
    for stay in trajectories.values():
        stay.update(label=1, t0="2020-01-02T00:00:00", cohort_type="synthetic_control")
    revised = assemble_subgraph_trajectories(graph, trajectories)
    for stay_id, stay in original.items():
        assert set(revised[stay_id]) == {"subject_id", "split", "windows"}
        for key, window_graph in stay["windows"].items():
            assert nx.utils.graphs_equal(window_graph, revised[stay_id]["windows"][key])


def test_duplicate_seeds_keep_distinct_observations_and_cxr_sources(graph, trajectories):
    window = trajectories["10"]["windows"]["window_0"]
    cxr = finding()
    cxr.update(predicate="shows_radiographic", source="cxr_labels", value=None,
               label_sources=["chexpert", "negbio"])
    window["findings"].append(cxr)
    window["cxr_coverage"] = {"chexpert": "positive", "negbio": "positive"}
    subgraph = assemble_subgraph_trajectories(graph, trajectories)["10"]["windows"]["window_0"]
    assert subgraph.graph["findings"] == window["findings"]
    assert subgraph.graph["cxr_coverage"] == window["cxr_coverage"]
    assert subgraph.graph["retrieval"]["matched_findings"] == 2
    assert subgraph.graph["retrieval"]["active_seed_nodes"] == (("disease", "a"),)


@pytest.mark.parametrize("case", [
    "missing_window", "extra_window", "noncontiguous", "wrong_duration", "bad_timestamp",
    "missing_coverage", "bad_coverage", "missing_findings", "subject_split", "bad_split",
    "bad_subject", "bad_mapping", "unresolved_seed", "numeric_node_id", "nonclinical_node",
])
def test_invalid_inputs_fail_without_echoing_identifiers(graph, trajectories, case):
    stay = trajectories["10"]
    window = stay["windows"]["window_0"]
    item = window["findings"][0]
    if case == "missing_window":
        del stay["windows"]["window_3"]
    elif case == "extra_window":
        stay["windows"]["window_4"] = deepcopy(window)
    elif case == "noncontiguous":
        stay["windows"]["window_1"] = deepcopy(window)
    elif case == "wrong_duration":
        window["end"] = window["start"]
    elif case == "bad_timestamp":
        window["start"] = "SENSITIVE_SYNTHETIC_SENTINEL"
    elif case == "missing_coverage":
        del window["cxr_coverage"]["negbio"]
    elif case == "bad_coverage":
        window["cxr_coverage"]["negbio"] = "invented"
    elif case == "missing_findings":
        del window["findings"]
    elif case == "subject_split":
        trajectories["11"]["split"] = "test"
    elif case == "bad_split":
        stay["split"] = "unknown"
    elif case == "bad_subject":
        stay["subject_id"] = True
    elif case == "bad_mapping":
        item["mapping_status"] = "invented"
    elif case == "unresolved_seed":
        item["mapping_status"] = "unresolved"
    elif case == "numeric_node_id":
        item["node"]["node_id"] = 1
    elif case == "nonclinical_node":
        item["node"]["node_type"] = "drug"
    with pytest.raises(ValueError) as caught:
        assemble_subgraph_trajectories(graph, trajectories)
    assert "SENSITIVE_SYNTHETIC_SENTINEL" not in str(caught.value)


@pytest.mark.parametrize("value", [{}, [], {"not-a-native-id": {}}])
def test_empty_or_malformed_roots_fail(value):
    with pytest.raises(ValueError):
        validate_temporal_quadruples(value)
