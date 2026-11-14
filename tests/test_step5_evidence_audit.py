"""Synthetic tests for reviewer-assisted evidence checks and aggregate privacy."""

from copy import deepcopy
import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src import step5_evidence_audit as audit


@pytest.fixture
def evidence():
    windows = [{"window_idx": i, "stay_id": 10, "subject_id": 1, "split": "test",
                "lactate_latest": [1., 4., None, 2.][i], "map_slope": 0.} for i in range(4)]
    graphs = {f"window_{i}": nx.MultiDiGraph() for i in range(4)}
    for graph in graphs.values():
        graph.graph["findings"] = []
    graphs["window_1"].graph["findings"] = [
        {"name": "Pulmonary edema", "source": "cxr_chexpert"},
        {"name": "Elevated Lactate", "source": "lactate_latest"},
    ]
    graphs["window_1"].add_edge(("disease", "d"), ("effect/phenotype", "p"),
                                key="disease_phenotype_negative", relation="disease_phenotype_negative")
    return windows, graphs


def claim(text, kind, window=1, **fields):
    return {"start": 0, "end": len(text), "quote": text, "kind": kind, "window_idx": window, **fields}


def single(evidence, text, kind, window=1, **fields):
    context = audit.build_audit_context(*evidence)
    return audit.audit_rationale(text, [claim(text, kind, window, **fields)], context)


def test_context_retains_temporal_isolation_and_does_not_mutate_inputs(evidence):
    windows, graphs = evidence
    original = deepcopy(evidence)
    context = audit.build_audit_context(windows, graphs)
    assert context[0]["findings"] == []
    assert context[1]["findings"][0] == {"name": "Pulmonary edema", "source": "cxr_chexpert"}
    assert context[1]["measurements"]["lactate_latest"] == {"value": 4., "unit": "mmol/L"}
    assert "lactate_latest" not in context[2]["measurements"]
    assert context[0]["measurements"]["map_slope"]["value"] == 0
    assert windows == original[0]
    assert all(nx.utils.graphs_equal(graphs[k], original[1][k]) for k in graphs)


@pytest.mark.parametrize("problem", ["count", "duplicate", "foreign_stay", "graph_keys", "undirected", "infinity", "invalid_findings", "native_identity"])
def test_invalid_evidence_context_is_rejected(evidence, problem):
    windows, graphs = evidence
    if problem == "count":
        windows = windows[:3]
    elif problem == "duplicate":
        windows[3]["window_idx"] = 0
    elif problem == "foreign_stay":
        windows[3]["stay_id"] = 20
    elif problem == "graph_keys":
        del graphs["window_3"]
    elif problem == "undirected":
        graphs["window_1"] = nx.Graph()
    elif problem == "infinity":
        windows[0]["map_latest"] = np.inf
    elif problem == "invalid_findings":
        windows[0]["findings"] = ["normal"]
    else:
        graphs["window_0"].add_edge("d", "p", relation="disease_disease")
    with pytest.raises(ValueError):
        audit.build_audit_context(windows, graphs)


@pytest.mark.parametrize("window,finding,assertion,status", [
    (1, "Pulmonary edema", "present", "matched_evidence"),
    (0, "Pulmonary edema", "present", "not_in_evidence"),
    (1, "Pneumonia", "present", "not_in_evidence"),
    (0, "Pneumonia", "absent", "requires_review"),
])
def test_observed_findings_require_positive_same_window_evidence(evidence, window, finding, assertion, status):
    result = single(evidence, "An annotated radiographic claim.", "observed_finding", window,
                    finding=finding, assertion=assertion)
    assert result["claims"][0]["status"] == status
    assert result["review_complete"] is False


@pytest.mark.parametrize("window,value,unit,status", [
    (1, 4., "mmol/L", "matched_evidence"), (1, 9., "mmol/L", "not_in_evidence"),
    (2, 4., "mmol/L", "not_in_evidence"), (1, 4., "mg/dL", "not_in_evidence"),
])
def test_measurements_match_values_units_and_window(evidence, window, value, unit, status):
    result = single(evidence, "Lactate measurement claim.", "measurement", window,
                    field="lactate_latest", value=value, unit=unit)
    assert result["claims"][0]["status"] == status


@pytest.mark.parametrize("problem,status", [
    ("exact", "matched_evidence"), ("reverse", "not_in_evidence"),
    ("positive", "not_in_evidence"), ("wrong_window", "not_in_evidence"),
    ("diagnosis", "not_in_evidence"),
])
def test_graph_checks_preserve_direction_negative_relations_and_context_role(evidence, problem, status):
    fields = {"source": ["disease", "d"], "target": ["effect/phenotype", "p"],
              "relation": "disease_phenotype_negative", "use": "biomedical_context"}
    window = 1
    if problem == "reverse":
        fields["source"], fields["target"] = fields["target"], fields["source"]
    elif problem == "positive":
        fields["relation"] = "disease_phenotype_positive"
    elif problem == "wrong_window":
        window = 0
    elif problem == "diagnosis":
        fields["use"] = "patient_diagnosis"
    result = single(evidence, "An annotated graph-context claim.", "retrieved_relation", window, **fields)
    assert result["claims"][0]["status"] == status


def test_diagnoses_and_unavailable_evidence_are_not_automatic_support(evidence):
    result = single(evidence, "The patient has pneumonia.", "diagnosis")
    assert result["claims"][0]["status"] == "not_in_evidence"
    context = audit.build_audit_context(evidence[0])
    finding = claim("A finding.", "observed_finding", finding="Pulmonary edema", assertion="present")
    result = audit.audit_rationale("A finding.", [finding], context)
    assert result["claims"][0]["status"] == "requires_review"


@pytest.mark.parametrize("problem", ["quote", "span", "overlap", "kind", "window", "field", "value", "assertion"])
def test_invalid_annotations_rejected(evidence, problem):
    text = "Lactate is 4 mmol/L."
    annotation = claim(text, "measurement", field="lactate_latest", value=4., unit="mmol/L")
    annotations = [annotation]
    if problem == "quote":
        annotation["quote"] = "Different text."
    elif problem == "span":
        annotation["start"] = -1
    elif problem == "overlap":
        annotations.append(dict(annotation))
    elif problem == "kind":
        annotation["kind"] = "unknown"
    elif problem == "window":
        annotation["window_idx"] = True
    elif problem == "field":
        annotation["field"] = "label"
    elif problem == "value":
        annotation["value"] = np.nan
    else:
        annotation.update(kind="observed_finding", finding="Edema", assertion="normal")
    with pytest.raises(ValueError):
        audit.audit_rationale(text, annotations, audit.build_audit_context(*evidence))


def test_human_decisions_are_separate_from_automatic_annotation_matches(evidence):
    # The metadata matches evidence but semantics ("not") need human review.
    text = "Lactate is not 4 mmol/L."
    annotation = claim(text, "measurement", field="lactate_latest", value=4., unit="mmol/L")
    review = {"reviewer_role": "SYNTHETIC_PRIVATE_REVIEWER", "claims_complete": True, "decisions": ["unsupported"]}
    result = audit.audit_rationale(text, [annotation], audit.build_audit_context(*evidence), review)
    assert result["claims"][0]["status"] == "matched_evidence"
    assert result["claims"][0]["review_decision"] == "unsupported"
    assert result["review_complete"] is True
    summary = audit.aggregate_audits([result])
    assert summary["automatic_match_fraction"] == 1
    assert summary["reviewer_unsupported_fraction"] == 1
    assert summary["clinical_correctness_validated"] is False
    assert "SYNTHETIC_PRIVATE_REVIEWER" not in json.dumps(summary)


def test_unannotated_rationale_cannot_be_marked_reviewed(evidence):
    result = audit.audit_rationale("Unannotated rationale.", [], audit.build_audit_context(*evidence),
                                  {"reviewer_role": "researcher", "claims_complete": True, "decisions": []})
    summary = audit.aggregate_audits([result])
    assert summary["pending_review_cases"] == 1
    assert summary["automatic_match_fraction"] is None
    assert summary["reviewer_unsupported_fraction"] is None


@pytest.mark.parametrize("review", [
    {"reviewer_role": "", "claims_complete": True, "decisions": ["supported"]},
    {"reviewer_role": "researcher", "claims_complete": "yes", "decisions": ["supported"]},
    {"reviewer_role": "researcher", "claims_complete": True, "decisions": []},
    {"reviewer_role": "researcher", "claims_complete": True, "decisions": ["correct"]},
])
def test_incomplete_or_invalid_review_data_fails(evidence, review):
    text = "Lactate is 4 mmol/L."
    with pytest.raises(ValueError):
        audit.audit_rationale(text, [claim(text, "measurement", field="lactate_latest", value=4., unit="mmol/L")],
                              audit.build_audit_context(*evidence), review)


def sampling_frame():
    return pd.DataFrame([
        {"stay_id": 10 * subject + stay, "subject_id": subject,
         "split": "test" if subject < 35 else "train", "label": subject % 2,
         "sepsis_probability": .1, "reasoning_chain": "SYNTHETIC_PRIVATE_TEXT"}
        for subject in range(40) for stay in range(2)
    ])


def test_sampling_is_outcome_independent_reproducible_and_one_stay_per_subject():
    frame = sampling_frame()
    sample = audit.select_audit_sample(frame)
    frame["label"] = 1 - frame.label
    frame["sepsis_probability"] = .9
    other = audit.select_audit_sample(frame.sample(frac=1, random_state=5))
    pd.testing.assert_frame_equal(sample, other)
    assert len(sample) == sample.subject_id.nunique() == 25
    assert sample.split.eq("test").all()
    assert set(sample.columns) == {"stay_id", "subject_id", "split"}


@pytest.mark.parametrize("problem", ["small", "large", "shortfall", "duplicate", "leakage", "missing"])
def test_invalid_audit_sampling_fails(problem):
    frame, size = sampling_frame(), 25
    if problem == "small":
        size = 19
    elif problem == "large":
        size = 31
    elif problem == "shortfall":
        frame = frame[frame.subject_id < 10]
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "leakage":
        frame.loc[0, "split"] = "train"
    else:
        frame.loc[0, "subject_id"] = np.nan
    with pytest.raises(ValueError):
        audit.select_audit_sample(frame, size=size)


def test_aggregate_persistence_never_exports_case_text_or_identifiers(evidence, tmp_path):
    result = single(evidence, "SYNTHETIC_PRIVATE_TEXT", "diagnosis")
    result["stay_id"] = "SYNTHETIC_PRIVATE_ID"
    path = audit.save_audit_summary([result], tmp_path / "audit")
    saved = json.loads(path.read_text())
    assert saved["cases_with_missing_evidence_flags"] == 1
    assert saved["automatic_status_counts"]["not_in_evidence"] == 1
    assert saved["sample_size_in_planned_range"] is False
    assert "SYNTHETIC_PRIVATE" not in path.read_text()
    assert list(path.parent.iterdir()) == [path]


def test_empty_and_forged_completed_audits_fail(evidence):
    with pytest.raises(ValueError):
        audit.aggregate_audits([])
    result = single(evidence, "A diagnosis claim.", "diagnosis")
    result["review_complete"] = True
    with pytest.raises(ValueError):
        audit.aggregate_audits([result])
