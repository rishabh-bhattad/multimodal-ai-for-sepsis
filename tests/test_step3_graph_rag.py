"""Synthetic public-graph contracts; no patient data or model downloads."""

import csv
import hashlib
import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.step3_graph_rag import (
    build_path_connectivity,
    load_primekg_graph,
    retrieve_connecting_path_edges,
)


FIELDS = ["relation", "x_type", "x_id", "x_name", "y_type", "y_id", "y_name"]


def write_graph(tmp_path, rows):
    path = tmp_path / "kg.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    return path


def test_native_identity_direction_parallel_relations_and_negative_semantics(tmp_path):
    positive = ["disease_phenotype_positive", "disease", "001", "Disease, synthetic",
                "effect/phenotype", "001", "Synthetic finding"]
    negative = ["disease_phenotype_negative", *positive[1:]]
    path = write_graph(tmp_path, [positive, negative, positive])
    graph = load_primekg_graph(path)
    disease, finding = ("disease", "001"), ("effect/phenotype", "001")
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == graph.number_of_edges() == 2
    assert graph.nodes[disease]["node_name"] == "Disease, synthetic"
    assert not graph.has_edge(finding, disease)
    assert set(graph[disease][finding]) == {
        "disease_phenotype_positive", "disease_phenotype_negative"
    }
    edge = graph[disease][finding]["disease_phenotype_negative"]
    assert edge["relation"] == "disease_phenotype_negative"
    assert edge["association_polarity"] == "negative"
    assert edge["evidence_role"] == "negative_association"
    assert edge["eligible_for_connecting_paths"] is False
    assert graph[disease][finding]["disease_phenotype_positive"]["eligible_for_connecting_paths"]
    provenance = graph.graph["provenance"]
    assert provenance["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["source_bytes"] == path.stat().st_size
    assert provenance["source_release"] == "unverified_local_file"
    assert provenance["source_rows"] == provenance["retained_rows"] == 3
    assert provenance["duplicate_rows"] == 1


def test_filters_both_endpoints_and_relations(tmp_path):
    rows = [
        ["disease_disease", "disease", "1", "A", "disease", "2", "B"],
        ["indication", "drug", "3", "Drug", "disease", "1", "A"],
        ["contraindication", "disease", "1", "A", "drug", "3", "Drug"],
        ["invented_relation", "disease", "1", "A", "disease", "4", "C"],
    ]
    graph = load_primekg_graph(write_graph(tmp_path, rows))
    assert set(graph) == {("disease", "1"), ("disease", "2")}
    assert graph.number_of_edges() == 1
    assert graph.graph["provenance"]["source_rows"] == 4
    assert graph.graph["provenance"]["retained_rows"] == 1


def test_row_order_does_not_change_graph_or_canonical_names(tmp_path):
    rows = [
        ["disease_disease", "disease", "1", "Z alias", "disease", "2", "B"],
        ["disease_disease", "disease", "2", "B", "disease", "1", "A alias"],
    ]
    first = load_primekg_graph(write_graph(tmp_path, rows))
    second = load_primekg_graph(write_graph(tmp_path, rows[::-1]))
    assert dict(first.nodes(data=True)) == dict(second.nodes(data=True))
    assert dict(((a, b, key), data) for a, b, key, data in first.edges(keys=True, data=True)) == dict(
        ((a, b, key), data) for a, b, key, data in second.edges(keys=True, data=True)
    )
    assert first.nodes[("disease", "1")]["node_name"] == "A alias"


@pytest.mark.parametrize("rows,message", [
    ([], "No eligible"),
    ([["disease_disease", "disease", "", "A", "disease", "2", "B"]], "lacks identity"),
    ([["disease_disease", "disease", "1", "A", "disease", "2", " "]], "lacks identity"),
    ([["disease_disease", "disease", "1"]], "Malformed"),
])
def test_invalid_graph_fails_clearly(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        load_primekg_graph(write_graph(tmp_path, rows))


def test_missing_columns_and_missing_file(tmp_path):
    path = tmp_path / "kg.csv"
    with pytest.raises(FileNotFoundError):
        load_primekg_graph(path)
    path.write_text("x_name,y_name\nA,B\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required"):
        load_primekg_graph(path)


def disease_edge(source, target, relation="disease_disease"):
    return [relation, "disease", source, f"Node {source}", "disease", target, f"Node {target}"]


def test_all_shortest_two_hop_paths_preserve_source_direction(tmp_path):
    graph = load_primekg_graph(write_graph(tmp_path, [
        disease_edge("a", "b"), disease_edge("c", "b"),
        disease_edge("a", "d"), disease_edge("d", "c"),
        disease_edge("a", "e"), disease_edge("e", "f"), disease_edge("f", "c"),
    ]))
    a, c = ("disease", "a"), ("disease", "c")
    edges = retrieve_connecting_path_edges(graph, [a, c])
    assert {(source[1], target[1]) for source, target, _ in edges} == {
        ("a", "b"), ("c", "b"), ("a", "d"), ("d", "c")
    }
    assert edges == tuple(sorted(edges))
    assert all(graph.has_edge(source, target, relation) for source, target, relation in edges)
    assert retrieve_connecting_path_edges(graph, [a, c], max_path_length=1) == ()


def test_direct_connection_excludes_longer_alternative_paths(tmp_path):
    graph = load_primekg_graph(write_graph(tmp_path, [
        disease_edge("a", "c"), disease_edge("a", "b"), disease_edge("b", "c"),
    ]))
    a, c = ("disease", "a"), ("disease", "c")
    assert retrieve_connecting_path_edges(graph, [a, c]) == ((a, c, "disease_disease"),)


def test_negative_associations_neither_bridge_nor_become_path_evidence(tmp_path):
    graph = load_primekg_graph(write_graph(tmp_path, [
        disease_edge("a", "b", "disease_phenotype_negative"),
        disease_edge("b", "c"), disease_edge("a", "d"), disease_edge("d", "c"),
        disease_edge("a", "d", "disease_phenotype_positive"),
        disease_edge("a", "d", "disease_phenotype_negative"),
        disease_edge("d", "a"),
    ]))
    a, b, c, d = [("disease", name) for name in "abcd"]
    connectivity = build_path_connectivity(graph)
    assert not connectivity.has_edge(a, b)
    assert connectivity.has_edge(a, d)
    edges = retrieve_connecting_path_edges(graph, [a, c], connectivity=connectivity)
    assert set(edges) == {
        (a, d, "disease_disease"), (d, a, "disease_disease"),
        (a, d, "disease_phenotype_positive"), (d, c, "disease_disease"),
    }
    assert all(relation != "disease_phenotype_negative" for _, _, relation in edges)
    assert graph.has_edge(a, d, "disease_phenotype_negative")


def test_empty_missing_singleton_and_disconnected_seeds_have_no_paths(tmp_path):
    graph = load_primekg_graph(write_graph(tmp_path, [
        disease_edge("a", "b"), disease_edge("c", "d"),
        disease_edge("e", "f", "disease_phenotype_negative"),
    ]))
    a, c, e, f = [("disease", name) for name in "acef"]
    for seeds in ([], [a], [a, a], [("disease", "absent")], [a, c], [e, f],
                  [("effect/phenotype", "a"), c]):
        assert retrieve_connecting_path_edges(graph, seeds) == ()


def test_path_selection_is_deterministic_and_does_not_mutate_graph(tmp_path):
    rows = [disease_edge("a", "b"), disease_edge("c", "b"), disease_edge("b", "d")]
    first = load_primekg_graph(write_graph(tmp_path, rows))
    second = load_primekg_graph(write_graph(tmp_path, rows[::-1]))
    seeds = [("disease", name) for name in "acd"]
    before = first.copy()
    expected = retrieve_connecting_path_edges(first, seeds)
    assert expected == retrieve_connecting_path_edges(second, seeds[::-1] + seeds)
    assert expected == retrieve_connecting_path_edges(first, seeds)
    assert nx.utils.graphs_equal(first, before)


@pytest.mark.parametrize("limit", [0, -1, 3, True, 2.0, "2"])
def test_invalid_path_limits_fail(limit):
    with pytest.raises(ValueError, match="one or two"):
        retrieve_connecting_path_edges(nx.MultiDiGraph(), [], max_path_length=limit)
