"""Local, reviewer-assisted evidence-grounding audit for held-out rationales.

Annotations quote exact character spans and state the claim being checked.
Automatic results check annotation/evidence consistency, not free-text semantics
or clinical truth. Empty findings never establish absence, and retrieved disease
nodes never establish patient diagnoses. Only aggregate summaries are exported.
"""

from __future__ import annotations

import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd


RUBRIC_VERSION = 1
KINDS = ("observed_finding", "measurement", "retrieved_relation", "diagnosis")
STATUSES = ("matched_evidence", "not_in_evidence", "requires_review")
DECISIONS = ("supported", "unsupported", "uncertain")
MEASUREMENT_UNITS = {
    **{f"hr_{suffix}": "bpm" for suffix in ("mean", "min", "max")},
    "hr_slope": "bpm/hour",
    **{f"map_{suffix}": "mmHg" for suffix in ("mean", "min", "max", "latest")},
    "map_slope": "mmHg/hour", "temp_latest": "C", "temp_mean": "C",
    "spo2_mean": "%", "resp_rate_mean": "breaths/min",
    "lactate_latest": "mmol/L", "platelets_latest": "10^9/L",
    "creatinine_latest": "mg/dL", "bilirubin_latest": "mg/dL", "wbc_latest": "10^9/L",
}


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and bool(np.isfinite(value))


def _window_index(value: Any) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value not in range(4):
        raise ValueError("Window indices must be integers zero through three")
    return int(value)


def _node_identity(value: Any) -> tuple[str, str]:
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError("Graph references require typed native string identities")
    return tuple(value)


def select_audit_sample(predictions: pd.DataFrame, size: int = 25, seed: int = 42) -> pd.DataFrame:
    """Preselect 20–30 test subjects uniformly, then one stay per subject.

    Outcome, probability, and rationale columns are ignored and not returned.
    Fails on a shortfall rather than silently changing the predefined sample.
    Persist the returned manifest locally before inspecting explanations.
    """
    required = ["stay_id", "subject_id", "split"]
    if not isinstance(size, Integral) or isinstance(size, bool) or not 20 <= size <= 30:
        raise ValueError("Audit size must be an integer between twenty and thirty")
    if set(required) - set(predictions.columns) or predictions.empty:
        raise ValueError("Audit sampling requires stay, subject, and split metadata")
    data = predictions[required].copy()
    if data.isna().any().any() or data.stay_id.duplicated().any():
        raise ValueError("Audit metadata must be nonmissing with unique stays")
    if not data.split.isin(["train", "val", "test"]).all() or (data.groupby("subject_id").split.nunique() > 1).any():
        raise ValueError("Audit sampling requires isolated known splits")
    test = data.loc[data.split == "test"].sort_values(["subject_id", "stay_id"])
    subjects = test.subject_id.drop_duplicates().to_numpy()
    if len(subjects) < size:
        raise ValueError("Insufficient unique held-out subjects for the predefined audit size")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(subjects), size=size, replace=False)
    rows = []
    for index in chosen:
        stays = test.loc[test.subject_id == subjects[index]]
        rows.append(stays.iloc[int(rng.integers(len(stays)))])
    return pd.DataFrame(rows).reset_index(drop=True)


def build_audit_context(
    stay_windows: Sequence[Mapping[str, Any]],
    subgraphs: Mapping[str, nx.MultiDiGraph] | None = None,
) -> dict[int, dict[str, Any]]:
    """Adapt existing numerical windows and Step 3 graphs without changing them.

    Finding names/sources come from Step 2 extraction retained in graph metadata.
    Radiographic labels remain report-derived evidence, not an image reread.
    Measurement units follow the existing Step 1 fields; no unit conversion or
    new thresholds are applied here. Zero and missing values stay distinct.
    """
    if len(stay_windows) != 4:
        raise ValueError("Audit context requires four complete windows")
    for field in ("stay_id", "subject_id", "split"):
        present = [window[field] for window in stay_windows if field in window]
        if present and (len(present) != 4 or any(value != present[0] for value in present)):
            raise ValueError("Audit windows must belong to the same stay, subject, and split")
    context = {}
    if subgraphs is not None and set(subgraphs) != {f"window_{i}" for i in range(4)}:
        raise ValueError("Subgraphs must contain exactly four window keys")
    for window in stay_windows:
        index = _window_index(window.get("window_idx"))
        if index in context:
            raise ValueError("Duplicate audit window")
        graph = subgraphs[f"window_{index}"] if subgraphs is not None else None
        if graph is not None and not isinstance(graph, nx.MultiDiGraph):
            raise ValueError("Audit graphs must preserve directed parallel relations")
        if graph is not None and graph.graph.get("window_idx", index) != index:
            raise ValueError("Graph metadata must match its audit window")
        raw_findings = window.get("findings")
        if raw_findings is None and graph is not None:
            raw_findings = graph.graph.get("findings")
        findings = []
        if raw_findings is not None:
            if not isinstance(raw_findings, list):
                raise ValueError("Findings must be an explicit list")
            for finding in raw_findings:
                if not isinstance(finding, Mapping) or not isinstance(finding.get("name"), str) or not finding["name"].strip():
                    raise ValueError("Extracted findings require names")
                findings.append({"name": finding["name"], "source": finding.get("source")})
        measurements = {}
        for field, unit in MEASUREMENT_UNITS.items():
            value = window.get(field)
            if value is None or value is pd.NA or (isinstance(value, Real) and np.isnan(value)):
                continue
            if not _finite(value):
                raise ValueError("Audit measurements must be finite or missing")
            measurements[field] = {"value": float(value), "unit": unit}
        triples = set()
        if graph is not None:
            for source, target, key, edge in graph.edges(keys=True, data=True):
                relation = edge.get("relation", key)
                if not isinstance(relation, str) or not relation.strip():
                    raise ValueError("Retrieved edges must retain named relations")
                triples.add((_node_identity(source), relation, _node_identity(target)))
        context[index] = {
            "findings": findings, "findings_available": raw_findings is not None,
            "measurements": measurements, "triples": triples,
            "graph_available": graph is not None,
        }
    return context


def _check_claim(claim: Mapping[str, Any], evidence: Mapping[str, Any]) -> tuple[str, str]:
    kind = claim["kind"]
    if kind == "observed_finding":
        if not isinstance(claim.get("finding"), str) or not claim["finding"].strip() or claim.get("assertion") not in ("present", "absent"):
            raise ValueError("Finding claims require a name and present/absent assertion")
        if claim["assertion"] == "absent":
            return "requires_review", "absence_not_established_by_positive_finding_extraction"
        if not evidence["findings_available"]:
            return "requires_review", "extracted_finding_coverage_unavailable"
        match = any(f["name"].strip().casefold() == claim["finding"].strip().casefold() for f in evidence["findings"])
        return ("matched_evidence", "observed_finding_in_same_window") if match else ("not_in_evidence", "finding_not_observed_in_same_window")
    if kind == "measurement":
        if claim.get("field") not in MEASUREMENT_UNITS or not _finite(claim.get("value")) or not isinstance(claim.get("unit"), str):
            raise ValueError("Measurement claims require a known field, finite value, and unit")
        observed = evidence["measurements"].get(claim["field"])
        if observed is None:
            return "not_in_evidence", "measurement_missing_in_same_window"
        if claim["unit"] != observed["unit"]:
            return "not_in_evidence", "measurement_unit_mismatch"
        if not np.isclose(claim["value"], observed["value"], rtol=0, atol=0.01):
            return "not_in_evidence", "measurement_value_mismatch"
        return "matched_evidence", "measurement_matches_same_window"
    if kind == "retrieved_relation":
        source, target = _node_identity(claim.get("source")), _node_identity(claim.get("target"))
        if not isinstance(claim.get("relation"), str) or not claim["relation"].strip() or claim.get("use") not in ("biomedical_context", "patient_diagnosis"):
            raise ValueError("Graph claims require a native relation and explicit use")
        if claim["use"] == "patient_diagnosis":
            return "not_in_evidence", "retrieved_context_does_not_establish_patient_diagnosis"
        if not evidence["graph_available"]:
            return "requires_review", "retrieval_evidence_unavailable"
        triple = (source, claim["relation"], target)
        return ("matched_evidence", "directed_native_relation_in_same_window") if triple in evidence["triples"] else ("not_in_evidence", "directed_relation_not_in_same_window")
    # Labels and retrieved disease nodes are deliberately not diagnostic proof.
    return "not_in_evidence", "patient_diagnosis_not_established_by_supplied_evidence"


def audit_rationale(
    rationale: str,
    claims: Sequence[Mapping[str, Any]],
    context: Mapping[int, Mapping[str, Any]],
    review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a LOCAL case audit, including quoted text; never log this object.

    A reviewer supplies claim annotations and may record one supported,
    unsupported, or uncertain decision per claim, plus a completeness attestation.
    Exact quotes and non-overlapping spans prevent duplicate annotation scoring.
    Automatic matches do not endorse negation, causality, or diagnostic wording.
    """
    if not isinstance(rationale, str) or not rationale.strip() or set(context) != set(range(4)):
        raise ValueError("Audit requires a nonempty rationale and four-window context")
    results, spans = [], []
    for claim in claims:
        if not isinstance(claim, Mapping) or claim.get("kind") not in KINDS:
            raise ValueError("Unknown claim rubric category")
        start, end = claim.get("start"), claim.get("end")
        if any(not isinstance(v, Integral) or isinstance(v, bool) for v in (start, end)) or not 0 <= start < end <= len(rationale):
            raise ValueError("Claim annotations require valid character spans")
        if rationale[start:end] != claim.get("quote") or not rationale[start:end].strip():
            raise ValueError("Claim quote must match the annotated rationale span")
        if any(start < old_end and old_start < end for old_start, old_end in spans):
            raise ValueError("Claim annotation spans must not overlap")
        spans.append((start, end))
        index = _window_index(claim.get("window_idx"))
        status, reason = _check_claim(claim, context[index])
        results.append({"kind": claim["kind"], "window_idx": index,
                        "quote": claim["quote"], "status": status, "reason": reason,
                        "review_decision": None})
    reviewed = False
    reviewer_role = None
    if review is not None:
        if set(review) != {"reviewer_role", "claims_complete", "decisions"}:
            raise ValueError("Review requires role, completeness, and claim decisions")
        if not isinstance(review["reviewer_role"], str) or not review["reviewer_role"].strip() or type(review["claims_complete"]) is not bool:
            raise ValueError("Review requires a nonempty role and explicit completeness")
        decisions = review["decisions"]
        if not isinstance(decisions, list) or len(decisions) != len(results) or any(d not in DECISIONS for d in decisions):
            raise ValueError("Review requires one rubric decision per annotated claim")
        for result, decision in zip(results, decisions):
            result["review_decision"] = decision
        reviewed = review["claims_complete"] and bool(results)
        reviewer_role = review["reviewer_role"]
    return {
        "rubric_version": RUBRIC_VERSION, "claims": results,
        "rationale_characters": len(rationale),
        "annotated_characters": sum(end - start for start, end in spans),
        "review_complete": reviewed, "reviewer_role": reviewer_role,
    }


def audit_heldout_sample(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    windows: pd.DataFrame,
    trajectories: Mapping[str, Any],
    annotations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit a frozen local sample and return aggregates only.

    Predictions need stay/subject/split metadata and reasoning_chain. Annotations
    are keyed by string stay identity and contain claims plus optional review.
    This adapter binds each rationale to the same stay's windows and graphs.
    It neither draws a new sample nor modifies the frozen manifest.
    """
    if not 20 <= len(manifest) <= 30 or set(manifest.columns) != {"stay_id", "subject_id", "split"}:
        raise ValueError("Frozen manifest must contain twenty to thirty test stays")
    if manifest.isna().any().any() or manifest.stay_id.duplicated().any() or manifest.subject_id.duplicated().any() or not manifest.split.eq("test").all():
        raise ValueError("Frozen audit manifest requires unique test subjects and stays")
    required = {"stay_id", "subject_id", "split", "reasoning_chain"}
    if required - set(predictions.columns) or {"stay_id", "subject_id", "split", "window_idx"} - set(windows.columns):
        raise ValueError("Audit inputs lack required identity or rationale fields")
    if predictions.stay_id.duplicated().any() or predictions[["stay_id", "subject_id", "split"]].isna().any().any():
        raise ValueError("Prediction identities must be unique and nonmissing")
    if not predictions.split.isin(["train", "val", "test"]).all() or (predictions.groupby("subject_id").split.nunique() > 1).any():
        raise ValueError("Prediction subject splits must remain isolated")
    expected = set(manifest.stay_id.map(str))
    if set(annotations) != expected:
        raise ValueError("Annotations must match the complete frozen sample")
    by_stay = predictions.set_index("stay_id")
    audits = []
    for row in manifest.itertuples(index=False):
        if row.stay_id not in by_stay.index:
            raise ValueError("A sampled stay has no prediction")
        prediction = by_stay.loc[row.stay_id]
        stay_windows = windows.loc[windows.stay_id == row.stay_id]
        trajectory = trajectories.get(str(row.stay_id))
        if prediction.subject_id != row.subject_id or prediction.split != "test" or trajectory is None:
            raise ValueError("Sample and prediction/trajectory metadata must match")
        if trajectory.get("subject_id") != row.subject_id or trajectory.get("split") != "test" or not stay_windows.subject_id.eq(row.subject_id).all() or not stay_windows.split.eq("test").all():
            raise ValueError("Audit evidence must belong to the sampled held-out subject")
        annotation = annotations[str(row.stay_id)]
        if "claims" not in annotation or set(annotation) - {"claims", "review"}:
            raise ValueError("Case annotations require claims and optional review only")
        context = build_audit_context(stay_windows.to_dict("records"), trajectory["windows"])
        audits.append(audit_rationale(prediction.reasoning_chain, annotation["claims"], context, annotation.get("review")))
    return aggregate_audits(audits)


def aggregate_audits(audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return only counts/rates and fixed rubric text, never case text or IDs."""
    if not audits:
        raise ValueError("At least one case audit is required")
    by_kind = {kind: {status: 0 for status in STATUSES} for kind in KINDS}
    decisions = {decision: 0 for decision in DECISIONS}
    reviewed_cases, flagged_cases, claims_count = 0, 0, 0
    for audit in audits:
        if audit.get("rubric_version") != RUBRIC_VERSION or type(audit.get("review_complete")) is not bool:
            raise ValueError("Unsupported case audit schema")
        claims = audit.get("claims")
        if not isinstance(claims, list):
            raise ValueError("Case audit requires a claim list")
        if audit["review_complete"] and (not claims or any(c.get("review_decision") not in DECISIONS for c in claims)):
            raise ValueError("Completed review requires decisions on all claims")
        reviewed_cases += int(audit["review_complete"])
        flagged_cases += int(any(c.get("status") == "not_in_evidence" for c in claims))
        for claim in claims:
            if claim.get("kind") not in KINDS or claim.get("status") not in STATUSES or claim.get("review_decision") not in (*DECISIONS, None):
                raise ValueError("Unknown audit rubric result")
            by_kind[claim["kind"]][claim["status"]] += 1
            claims_count += 1
            if claim["review_decision"] is not None:
                decisions[claim["review_decision"]] += 1
    totals = {status: sum(counts[status] for counts in by_kind.values()) for status in STATUSES}
    reviewed_claims = sum(decisions.values())
    return {
        "rubric_version": RUBRIC_VERSION, "scope": "exploratory_evidence_grounding",
        "cases": len(audits), "claims": claims_count,
        "reviewed_cases": reviewed_cases, "pending_review_cases": len(audits) - reviewed_cases,
        "cases_with_missing_evidence_flags": flagged_cases,
        "automatic_status_counts": totals, "by_claim_kind": by_kind,
        "reviewer_decisions": decisions,
        "automatic_match_fraction": totals["matched_evidence"] / claims_count if claims_count else None,
        "reviewer_unsupported_fraction": decisions["unsupported"] / reviewed_claims if reviewed_claims else None,
        "sample_size_in_planned_range": 20 <= len(audits) <= 30,
        "clinical_correctness_validated": False,
    }


def save_audit_summary(audits: Sequence[Mapping[str, Any]], output_dir: str | Path) -> Path:
    """Construct an aggregate report from local audits; do not serialize cases."""
    report = aggregate_audits(audits)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "step5_evidence_audit.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
