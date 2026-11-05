"""Synthetic public-graph contracts; no patient data or model downloads."""

import csv
import hashlib
import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.step3_graph_rag import load_primekg_graph


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
