"""Evaluate local held-out predictions; export aggregate results only.

AUPRC is reported as scikit-learn average precision, alongside prevalence.
Subject-cluster percentile bootstrap intervals preserve multiple stays per
subject. This module does not train models or select a test-set threshold.
"""

from __future__ import annotations

import argparse
import json
from numbers import Integral, Real
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


VARIANTS = tuple(config.STEP5_MODEL_VARIANTS)
INTERVAL_METRICS = ("auroc", "auprc", "brier_score", "sensitivity", "specificity")
COUNT_METRICS = ("stays", "positives", "negatives", "true_negatives", "false_positives", "false_negatives", "true_positives")
SCALAR_METRICS = COUNT_METRICS + ("prevalence", "threshold") + INTERVAL_METRICS
PREDICTION_COLUMNS = ("stay_id", "subject_id", "split", "label", "sepsis_probability")
LOSS_FIELDS = frozenset({"step", "epoch", "train_loss", "validation_loss", "learning_rate"})


def _number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and bool(np.isfinite(value))


def _binary_arrays(labels: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        y = np.asarray(labels, dtype=float)
        p = np.asarray(probabilities, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Labels and probabilities must be numeric") from exc
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels and probabilities must be nonempty aligned vectors")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be finite binary values")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values between zero and one")
    return y.astype(int), p


def compute_metrics(labels: Any, probabilities: Any, threshold: float = 0.5) -> dict[str, Any]:
    """Use a prespecified threshold; represent undefined metrics as JSON null."""
    if not _number(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between zero and one")
    y, p = _binary_arrays(labels, probabilities)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel())
    positives = int(y.sum())
    negatives = len(y) - positives
    return {
        "stays": len(y), "positives": positives, "negatives": negatives,
        "prevalence": float(y.mean()), "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, p)) if positives and negatives else None,
        "auprc": float(average_precision_score(y, p)) if positives else None,
        "brier_score": float(brier_score_loss(y, p)),
        "sensitivity": tp / positives if positives else None,
        "specificity": tn / negatives if negatives else None,
        "true_negatives": tn, "false_positives": fp,
        "false_negatives": fn, "true_positives": tp,
    }


def bootstrap_confidence_intervals(
    labels: Any,
    probabilities: Any,
    subject_ids: Any,
    *,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = config.RANDOM_SEED,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Resample subjects with replacement, retaining all their stays per draw.

    Single-class draws contribute only to metrics defined for those draws.
    Report valid draw counts for each metric; do not replace undefined values
    with zero. Each stay remains one prediction, never four window predictions.
    """
    y, p = _binary_arrays(labels, probabilities)
    compute_metrics(y, p, threshold)  # Validate threshold even before resampling.
    subjects = np.asarray(subject_ids)
    if subjects.ndim != 1 or len(subjects) != len(y) or pd.isna(subjects).any():
        raise ValueError("Subject identities must be nonmissing and aligned")
    if not isinstance(n_resamples, Integral) or isinstance(n_resamples, bool) or n_resamples < 1:
        raise ValueError("Resample count must be a positive integer")
    if not _number(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be strictly between zero and one")
    codes, unique_subjects = pd.factorize(subjects, sort=False)
    if len(unique_subjects) < 2:
        raise ValueError("Subject bootstrap requires at least two subjects")
    groups = [np.flatnonzero(codes == i) for i in range(len(unique_subjects))]
    rng = np.random.default_rng(seed)
    values = {name: [] for name in INTERVAL_METRICS}
    for _ in range(n_resamples):
        draws = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[i] for i in draws])
        metrics = compute_metrics(y[indices], p[indices], threshold)
        for name in INTERVAL_METRICS:
            if metrics[name] is not None:
                values[name].append(metrics[name])
    tail = (1 - confidence_level) / 2
    intervals = {}
    for name, samples in values.items():
        limits = np.quantile(samples, [tail, 1 - tail]) if samples else (None, None)
        intervals[name] = {
            "lower": float(limits[0]) if samples else None,
            "upper": float(limits[1]) if samples else None,
            "valid_resamples": len(samples),
        }
    return {
        "method": "subject_cluster_percentile", "n_resamples": int(n_resamples),
        "confidence_level": float(confidence_level), "seed": int(seed),
        "subjects": len(groups), "intervals": intervals,
    }


def _test_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if set(PREDICTION_COLUMNS) - set(frame.columns) or frame.empty:
        raise ValueError("Predictions require stay, subject, split, label, and probability columns")
    data = frame.loc[:, list(PREDICTION_COLUMNS)].copy()
    if data[["stay_id", "subject_id", "split"]].isna().any().any():
        raise ValueError("Prediction metadata must be nonmissing")
    if data.stay_id.duplicated().any():
        raise ValueError("Predictions must contain one row per stay")
    if not data.split.isin(["train", "val", "test"]).all():
        raise ValueError("Unknown prediction split")
    if (data.groupby("subject_id").split.nunique() > 1).any():
        raise ValueError("Subject split isolation failed")
    y, p = _binary_arrays(data.label, data.sepsis_probability)
    data["label"], data["sepsis_probability"] = y, p
    test = data.loc[data.split == "test"].set_index("stay_id").sort_index()
    if test.empty:
        raise ValueError("No held-out test predictions")
    return test


def evaluate_model_comparison(
    predictions: Mapping[str, pd.DataFrame],
    *,
    threshold: float = 0.5,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = config.RANDOM_SEED,
) -> dict[str, Any]:
    """Compare identical test cohorts; label missing model variants explicitly.

    This verifies prediction alignment, not how upstream models were trained.
    No patient identities, labels, probabilities, or rationales enter the report.
    """
    if not predictions or set(predictions) - set(VARIANTS):
        raise ValueError("Provide one or more recognized model variants")
    prepared = {variant: _test_predictions(predictions[variant]) for variant in VARIANTS if variant in predictions}
    reference = next(iter(prepared.values()))
    for data in prepared.values():
        if not data.index.equals(reference.index) or not data[["subject_id", "label"]].equals(reference[["subject_id", "label"]]):
            raise ValueError("Model comparisons require identical test stays, subjects, and labels")
    models = {}
    for variant, data in prepared.items():
        models[variant] = {
            "metrics": compute_metrics(data.label, data.sepsis_probability, threshold),
            "bootstrap": bootstrap_confidence_intervals(
                data.label, data.sepsis_probability, data.subject_id,
                n_resamples=n_resamples, confidence_level=confidence_level,
                seed=seed, threshold=threshold,
            ),
        }
    return {
        "schema_version": 1, "validation": "passed", "scope": "prediction_evaluation_only",
        "split": "test", "unit": "stay", "observation_hours": 24,
        "auprc_definition": "average_precision", "threshold_policy": "prespecified_no_test_tuning",
        "cohort": {"stays": len(reference), "subjects": int(reference.subject_id.nunique()),
                   "positives": int(reference.label.sum()), "prevalence": float(reference.label.mean())},
        "available_variants": list(models), "missing_variants": [v for v in VARIANTS if v not in models],
        "all_variants_present": len(models) == len(VARIANTS), "models": models,
    }


def comparison_table(report: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten only known aggregate fields for a stable comparison CSV."""
    rows = []
    for variant in VARIANTS:
        if variant not in report["models"]:
            continue
        result = report["models"][variant]
        row = {"model_variant": variant, **{name: result["metrics"][name] for name in SCALAR_METRICS}}
        for name in INTERVAL_METRICS:
            for field in ("lower", "upper", "valid_resamples"):
                row[f"{name}_ci_{field}"] = result["bootstrap"]["intervals"][name][field]
        rows.append(row)
    return pd.DataFrame(rows)


def save_evaluation_report(report: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Persist aggregate results; never copy input prediction rows."""
    expected = {"schema_version", "validation", "scope", "split", "unit", "observation_hours",
                "auprc_definition", "threshold_policy", "cohort", "available_variants",
                "missing_variants", "all_variants_present", "models"}
    constants = {"schema_version": 1, "validation": "passed", "scope": "prediction_evaluation_only",
                 "split": "test", "unit": "stay", "observation_hours": 24,
                 "auprc_definition": "average_precision", "threshold_policy": "prespecified_no_test_tuning"}
    if set(report) != expected or any(report[k] != v for k, v in constants.items()):
        raise ValueError("Only the aggregate evaluation schema can be exported")
    if set(report["models"]) - set(VARIANTS) or not report["models"]:
        raise ValueError("Report contains unknown model variants")
    if report["available_variants"] != [v for v in VARIANTS if v in report["models"]] or report["missing_variants"] != [v for v in VARIANTS if v not in report["models"]]:
        raise ValueError("Report variant lists are inconsistent")
    if type(report["all_variants_present"]) is not bool or report["all_variants_present"] != (len(report["models"]) == len(VARIANTS)):
        raise ValueError("Report completeness must match available variants")
    if set(report["cohort"]) != {"stays", "subjects", "positives", "prevalence"} or any(not _number(v) for v in report["cohort"].values()):
        raise ValueError("Cohort summary must contain only numeric aggregates")
    for result in report["models"].values():
        if set(result) != {"metrics", "bootstrap"} or set(result["metrics"]) != set(SCALAR_METRICS):
            raise ValueError("Only known aggregate model metrics can be exported")
        if any(v is not None and not _number(v) for v in result["metrics"].values()):
            raise ValueError("Model metrics must be numeric aggregates or null")
        bootstrap = result["bootstrap"]
        if set(bootstrap) != {"method", "n_resamples", "confidence_level", "seed", "subjects", "intervals"} or bootstrap["method"] != "subject_cluster_percentile":
            raise ValueError("Unknown bootstrap report schema")
        if any(not _number(bootstrap[k]) for k in ("n_resamples", "confidence_level", "seed", "subjects")) or set(bootstrap["intervals"]) != set(INTERVAL_METRICS):
            raise ValueError("Bootstrap metadata must contain numeric aggregates")
        for interval in bootstrap["intervals"].values():
            if set(interval) != {"lower", "upper", "valid_resamples"} or any(v is not None and not _number(v) for v in interval.values()):
                raise ValueError("Confidence intervals must contain only numeric aggregates or null")
    output = Path(output_dir)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    table = comparison_table(report)
    output.mkdir(parents=True, exist_ok=True)
    (output / "step5_validation.json").write_text(encoded + "\n", encoding="utf-8")
    table.to_csv(output / "model_comparison.csv", index=False)
    return output


def log_aggregate_to_wandb(
    report: Mapping[str, Any],
    loss_history: Sequence[Mapping[str, Any]] = (),
    *,
    enabled: bool = False,
    wandb_client: Any = None,
) -> bool:
    """Opt-in logging of allowlisted scalars only, without files or raw tables.

    Validate the entire payload before opening a remote run. Additional report
    fields are never forwarded; loss history rejects unknown keys outright.
    """
    if not enabled:
        return False
    payload = {}
    for variant in VARIANTS:
        if variant not in report["models"]:
            continue
        result = report["models"][variant]
        for name in SCALAR_METRICS:
            value = result["metrics"][name]
            if value is not None:
                if not _number(value):
                    raise ValueError("Aggregate metrics must be finite scalars")
                payload[f"test/{variant}/{name}"] = value
        for name in INTERVAL_METRICS:
            for field in ("lower", "upper", "valid_resamples"):
                value = result["bootstrap"]["intervals"][name][field]
                if value is not None:
                    if not _number(value):
                        raise ValueError("Confidence intervals must be finite scalars")
                    payload[f"test/{variant}/{name}_ci_{field}"] = value
    history = []
    for entry in loss_history:
        if not entry or set(entry) - LOSS_FIELDS or any(not _number(v) or v < 0 for v in entry.values()):
            raise ValueError("Loss history must contain only allowlisted nonnegative scalars")
        history.append({f"training/{key}": value for key, value in entry.items()})
    if wandb_client is None:
        import wandb as wandb_client
    run = wandb_client.init(
        project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY,
        config={"evaluation_split": "test", "observation_hours": 24, "auprc_definition": "average_precision"},
        settings=wandb_client.Settings(console="off", disable_git=True, disable_code=True,
                                       x_disable_stats=True, x_disable_meta=True),
    )
    try:
        run.log(payload)
        for entry in history:
            run.log(entry)
    finally:
        run.finish()
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, metavar="VARIANT=PARQUET")
    parser.add_argument("--output", type=Path, default=config.STEP5_RESULTS_DIR / "evaluation")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)
    predictions = {}
    for item in args.prediction:
        variant, separator, path = item.partition("=")
        if not separator or not path or variant not in VARIANTS or variant in predictions:
            parser.error("Each prediction must specify a unique recognized VARIANT=PARQUET")
        predictions[variant] = pd.read_parquet(path)
    report = evaluate_model_comparison(predictions, threshold=args.threshold, n_resamples=args.resamples)
    save_evaluation_report(report, args.output)
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
