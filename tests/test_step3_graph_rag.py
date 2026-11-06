"""Synthetic public-graph contracts; no patient data or model downloads."""

import csv
import hashlib
import sys
from copy import deepcopy
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.step3_graph_rag import (
    build_path_connectivity,
    load_primekg_graph,
    retrieve_connecting_path_edges,
    retrieve_window_subgraph,
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


def test_neighbors_keep_native_incoming_outgoing_and_negative_context(tmp_path):
    positive = ["disease_phenotype_positive", "disease", "1", "Synthetic disease",
                "effect/phenotype", "2", "Synthetic finding"]
    negative = ["disease_phenotype_negative", *positive[1:]]
    reverse = ["disease_phenotype_positive", *positive[4:], *positive[1:4]]
    graph = load_primekg_graph(write_graph(tmp_path, [positive, negative, reverse]))
    disease, finding = ("disease", "1"), ("effect/phenotype", "2")
    result = retrieve_window_subgraph(graph, [finding])
    assert set(result.edges(keys=True)) == set(graph.edges(keys=True))
    assert result.nodes[finding]["is_active_seed"]
    assert not result.nodes[disease]["is_active_seed"]
    for source, target, relation, data in result.edges(keys=True, data=True):
        assert data["relation"] == graph.edges[source, target, relation]["relation"]
        assert data["retrieval_role"] == "one_hop_neighbor"
    edge = result.edges[disease, finding, "disease_phenotype_negative"]
    assert edge["association_polarity"] == "negative"
    assert edge["evidence_role"] == "negative_association"
    assert edge["eligible_for_connecting_paths"] is False
    assert result.graph["retrieval"]["negative_association_edges"] == 1


def test_neighbor_expansion_is_not_recursive(tmp_path):
    graph = load_primekg_graph(write_graph(tmp_path, [
        disease_edge("a", "b"), disease_edge("b", "c"), disease_edge("d", "a"),
    ]))
    a, b, c, d = [("disease", name) for name in "abcd"]
    result = retrieve_window_subgraph(graph, [a])
    assert set(result) == {a, b, d}
    assert c not in result
    assert result.graph["retrieval"]["neighbor_edges_available"] == 2
    assert not result.graph["retrieval"]["truncated"]


def test_path_edges_take_priority_over_earlier_sorted_neighbors(tmp_path):
    graph = load_primekg_graph(write_graph(tmp_path, [
        disease_edge("a", "b"), disease_edge("b", "z"),
        disease_edge("a", "aa"), disease_edge("z", "zz"),
    ]))
    a, b, z = [("disease", name) for name in ("a", "b", "z")]
    result = retrieve_window_subgraph(graph, [a, z], max_edges=2)
    assert set(result.edges(keys=True)) == {(a, b, "disease_disease"), (b, z, "disease_disease")}
    assert all(data["retrieval_role"] == "connecting_path" for _, _, data in result.edges(data=True))
    summary = result.graph["retrieval"]
    assert summary["path_edges_available"] == summary["path_edges_retained"] == 2
    assert summary["neighbor_edges_pruned"] == 2
    assert summary["neighbor_edges_retained"] == 0
    assert summary["truncated"] and not summary["path_edges_truncated"]


def test_default_200_edge_cap_when_paths_alone_overflow(tmp_path):
    rows = [edge for index in range(120) for edge in (
        disease_edge("a", f"m{index:03}"), disease_edge("z", f"m{index:03}")
    )]
    graph = load_primekg_graph(write_graph(tmp_path, rows + [disease_edge("a", "extra")]))
    seeds = [("disease", name) for name in ("a", "z")]
    paths = retrieve_connecting_path_edges(graph, seeds)
    result = retrieve_window_subgraph(graph, seeds)
    assert len(paths) == 240
    assert result.number_of_edges() == 200
    assert tuple(result.edges(keys=True)) == paths[:200]
    assert all(seed in result for seed in seeds)
    summary = result.graph["retrieval"]
    assert summary["path_edges_retained"] == 200
    assert summary["path_edges_pruned"] == 40
    assert summary["neighbor_edges_available"] == summary["neighbor_edges_pruned"] == 1
    assert summary["neighbor_edges_retained"] == 0
    assert summary["path_edges_truncated"]


def test_default_200_edge_cap_for_neighbors(tmp_path):
    graph = load_primekg_graph(write_graph(tmp_path, [
        disease_edge(f"n{index:03}", "seed") for index in range(220)
    ]))
    result = retrieve_window_subgraph(graph, [("disease", "seed")])
    assert result.number_of_edges() == 200
    assert tuple(result.edges(keys=True)) == tuple(sorted(graph.edges(keys=True)))[:200]
    summary = result.graph["retrieval"]
    assert summary["path_edges_retained"] == 0
    assert summary["neighbor_edges_retained"] == 200
    assert summary["neighbor_edges_pruned"] == 20
    assert summary["truncated"]


def test_empty_missing_and_isolated_seeds_remain_explicit(tmp_path):
    graph = load_primekg_graph(write_graph(tmp_path, [disease_edge("a", "b")]))
    isolated, missing = ("disease", "isolated"), ("disease", "absent")
    graph.add_node(isolated, node_type="disease", node_id="isolated", node_name="Isolated")
    result = retrieve_window_subgraph(graph, [isolated, missing, isolated])
    assert set(result) == {isolated}
    assert result.number_of_edges() == 0
    assert result.graph["retrieval"]["active_seed_nodes"] == (isolated,)
    assert result.graph["retrieval"]["missing_seed_nodes"] == (missing,)
    assert result.graph["retrieval"]["isolated_seed_nodes"] == (isolated,)
    empty = retrieve_window_subgraph(graph, [])
    assert not empty and empty.number_of_edges() == 0
    assert empty.graph["retrieval"]["active_seed_nodes"] == ()


def test_zero_edge_budget_retains_active_seeds(tmp_path):
    graph = load_primekg_graph(write_graph(tmp_path, [disease_edge("a", "b")]))
    seeds = [("disease", "a"), ("disease", "b")]
    result = retrieve_window_subgraph(graph, seeds, max_edges=0)
    assert set(result) == set(seeds)
    assert result.number_of_edges() == 0
    assert result.graph["retrieval"]["isolated_seed_nodes"] == tuple(seeds)
    assert result.graph["retrieval"]["path_edges_pruned"] == 1


def test_neighbor_filter_restricts_both_endpoints_and_native_relations(tmp_path):
    graph = load_primekg_graph(write_graph(tmp_path, [disease_edge("a", "b")]))
    seed, drug = ("disease", "a"), ("drug", "x")
    graph.add_node(drug, node_type="drug", node_id="x", node_name="Excluded drug")
    graph.add_edge(seed, drug, key="indication", relation="indication")
    graph.add_edge(drug, seed, key="contraindication", relation="contraindication")
    graph.add_edge(seed, ("disease", "b"), key="invented", relation="invented")
    result = retrieve_window_subgraph(graph, [seed])
    assert drug not in result
    assert set(result.edges(keys=True)) == {(seed, ("disease", "b"), "disease_disease")}


def test_window_graph_order_is_deterministic_and_attributes_are_independent(tmp_path):
    rows = [disease_edge("a", "b"), disease_edge("b", "z"), disease_edge("a", "c")]
    first = load_primekg_graph(write_graph(tmp_path, rows))
    second = load_primekg_graph(write_graph(tmp_path, rows[::-1]))
    # Same source identity when comparing retrieval rather than CSV order.
    second.graph["provenance"] = deepcopy(first.graph["provenance"])
    seeds = [("disease", "a"), ("disease", "z")]
    first.nodes[seeds[0]]["nested"] = {"values": [1]}
    second.nodes[seeds[0]]["nested"] = {"values": [1]}
    first.edges[seeds[0], ("disease", "b"), "disease_disease"]["nested"] = {"values": [2]}
    second.edges[seeds[0], ("disease", "b"), "disease_disease"]["nested"] = {"values": [2]}
    before = deepcopy(first)
    connectivity = build_path_connectivity(first)
    result = retrieve_window_subgraph(first, seeds, connectivity=connectivity, max_edges=2)
    other = retrieve_window_subgraph(second, seeds[::-1] + seeds, max_edges=2)
    assert nx.utils.graphs_equal(result, other)
    assert list(result) == list(other)
    assert list(result.edges(keys=True)) == list(other.edges(keys=True))
    result.nodes[seeds[0]]["nested"]["values"].append(3)
    result.edges[seeds[0], ("disease", "b"), "disease_disease"]["nested"]["values"].append(4)
    result.graph["provenance"]["allowed_node_types"].append("changed")
    assert nx.utils.graphs_equal(first, before)
    assert other.nodes[seeds[0]]["nested"]["values"] == [1]
    assert other.edges[seeds[0], ("disease", "b"), "disease_disease"]["nested"]["values"] == [2]


@pytest.mark.parametrize("budget", [-1, 201, True, 2.0, "2"])
def test_invalid_window_edge_budgets_fail(budget):
    with pytest.raises(ValueError, match="within the configured cap"):
        retrieve_window_subgraph(nx.MultiDiGraph(), [], max_edges=budget)


def test_window_budget_uses_configured_cap(tmp_path, monkeypatch):
    import config

    graph = load_primekg_graph(write_graph(tmp_path, [disease_edge("a", "b"), disease_edge("a", "c")]))
    monkeypatch.setattr(config, "STEP3_MAX_EDGES_PER_WINDOW", 1)
    result = retrieve_window_subgraph(graph, [("disease", "a")])
    assert result.number_of_edges() == result.graph["retrieval"]["max_edges"] == 1
    with pytest.raises(ValueError, match="within the configured cap"):
        retrieve_window_subgraph(graph, [("disease", "a")], max_edges=2)
