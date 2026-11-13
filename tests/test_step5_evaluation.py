"""Synthetic metric, clustered-bootstrap, export, and mocked-tracking checks."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config
from src import step5_evaluation as evaluation


@pytest.fixture
def predictions():
    # Two stays belong to the first subject; train/validation rows are excluded.
    return pd.DataFrame({
        "stay_id": [11, 12, 13, 14, 15, 16, 21, 22],
        "subject_id": [1, 1, 2, 3, 4, 5, 100, 101],
        "split": ["test"] * 6 + ["train", "val"],
        "label": [0, 1, 0, 1, 0, 1, 0, 1],
        "sepsis_probability": [.1, .8, .2, .9, .3, .7, .99, .01],
        "reasoning_chain": ["SYNTHETIC_PRIVATE_TEXT"] * 8,
    })


def test_metrics_match_sklearn_and_prespecified_threshold():
    y = [0, 0, 1, 1]
    p = [.1, .5, .5, .9]
    metrics = evaluation.compute_metrics(y, p)
    assert metrics["auroc"] == pytest.approx(roc_auc_score(y, p))
    assert metrics["auprc"] == pytest.approx(average_precision_score(y, p))
    assert metrics["brier_score"] == pytest.approx(brier_score_loss(y, p))
    assert metrics["prevalence"] == .5
    assert [metrics[k] for k in ("true_negatives", "false_positives", "false_negatives", "true_positives")] == [1, 1, 0, 2]
    assert metrics["sensitivity"] == 1
    assert metrics["specificity"] == .5
    assert evaluation.compute_metrics(y, p, threshold=.6)["sensitivity"] == .5


@pytest.mark.parametrize("labels,expected_auprc,undefined", [
    ([0, 0], None, "sensitivity"), ([1, 1], 1., "specificity"),
])
def test_single_class_metrics_are_explicitly_undefined(labels, expected_auprc, undefined):
    metrics = evaluation.compute_metrics(labels, [.1, .9])
    assert metrics["auroc"] is None
    assert metrics["auprc"] == expected_auprc
    assert metrics[undefined] is None
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("labels,probabilities,threshold", [
    ([], [], .5), ([0], [.2, .3], .5), ([[0]], [[.2]], .5),
    ([2], [.2], .5), ([np.nan], [.2], .5), ([0], [np.nan], .5),
    ([0], [np.inf], .5), ([0], [-.1], .5), ([1], [1.1], .5),
    ([0], [.2], -.1), ([0], [.2], 1.1), ([0], [.2], np.nan),
    ([0], [.2], True), (["invalid"], [.2], .5),
])
def test_invalid_metric_inputs_fail(labels, probabilities, threshold):
    with pytest.raises(ValueError):
        evaluation.compute_metrics(labels, probabilities, threshold)


def test_default_bootstrap_uses_1000_draws_and_reports_single_class_draw_counts():
    result = evaluation.bootstrap_confidence_intervals([0, 1], [.1, .9], [1, 2])
    assert result["n_resamples"] == 1000
    assert result["subjects"] == 2
    assert 0 < result["intervals"]["auroc"]["valid_resamples"] < 1000
    assert result["intervals"]["brier_score"]["valid_resamples"] == 1000
    assert result["intervals"]["auroc"]["lower"] == 1
    assert result["intervals"]["auroc"]["upper"] == 1


def test_bootstrap_keeps_subject_stays_together_and_is_reproducible(monkeypatch):
    original = evaluation.compute_metrics
    draws = []

    def capture(labels, probabilities, threshold=.5):
        draws.append(list(probabilities))
        return original(labels, probabilities, threshold)

    monkeypatch.setattr(evaluation, "compute_metrics", capture)
    first = evaluation.bootstrap_confidence_intervals([0, 1, 1], [.11, .12, .8], [1, 1, 2], n_resamples=30, seed=13)
    second = evaluation.bootstrap_confidence_intervals([0, 1, 1], [.11, .12, .8], [1, 1, 2], n_resamples=30, seed=13)
    assert first == second
    sampled = draws[1:31]
    assert any(len(draw) == 4 for draw in sampled)
    assert any(len(draw) == 2 for draw in sampled)
    assert all(draw.count(.11) == draw.count(.12) for draw in sampled)
    for interval in first["intervals"].values():
        assert interval["lower"] <= interval["upper"]


@pytest.mark.parametrize("kwargs", [
    {"n_resamples": 0}, {"n_resamples": 1.5}, {"n_resamples": True},
    {"confidence_level": 0}, {"confidence_level": 1}, {"confidence_level": np.nan},
    {"subject_ids": [1]}, {"subject_ids": [1, None]}, {"subject_ids": [1, 1]},
])
def test_invalid_bootstrap_setup(kwargs):
    subjects = kwargs.pop("subject_ids", [1, 2])
    with pytest.raises(ValueError):
        evaluation.bootstrap_confidence_intervals([0, 1], [.1, .9], subjects, **kwargs)


def test_single_class_bootstrap_has_null_auroc_and_valid_brier():
    result = evaluation.bootstrap_confidence_intervals([0, 0], [.2, .3], [1, 2], n_resamples=10)
    assert result["intervals"]["auroc"] == {"lower": None, "upper": None, "valid_resamples": 0}
    assert result["intervals"]["brier_score"]["valid_resamples"] == 10


def test_comparison_aligns_models_filters_test_and_reports_missing_variants(predictions):
    other = predictions.sample(frac=1, random_state=9)
    report = evaluation.evaluate_model_comparison({"xgboost": predictions, "trag": other}, n_resamples=12)
    assert report["cohort"] == {"stays": 6, "subjects": 5, "positives": 3, "prevalence": .5}
    assert report["available_variants"] == ["trag", "xgboost"]
    assert report["missing_variants"] == ["static_rag", "no_kg"]
    assert report["all_variants_present"] is False
    assert report["scope"] == "prediction_evaluation_only"
    assert report["models"]["trag"] == report["models"]["xgboost"]
    assert report["models"]["trag"]["metrics"]["auroc"] == 1
    assert "SYNTHETIC_PRIVATE_TEXT" not in json.dumps(report)
    table = evaluation.comparison_table(report)
    assert list(table.model_variant) == ["trag", "xgboost"]
    assert table.auprc.eq(1).all()
    assert "auroc_ci_valid_resamples" in table
    assert not {"stay_id", "subject_id", "label", "reasoning_chain"} & set(table.columns)


def test_all_four_models_are_recognized(predictions):
    inputs = {variant: predictions for variant in config.STEP5_MODEL_VARIANTS}
    report = evaluation.evaluate_model_comparison(inputs, n_resamples=8)
    assert report["all_variants_present"] is True
    assert report["missing_variants"] == []
    assert len(evaluation.comparison_table(report)) == 4


@pytest.mark.parametrize("problem", ["stays", "labels", "subjects"])
def test_refuse_mismatched_comparison_cohorts(predictions, problem):
    other = predictions.copy()
    if problem == "stays":
        other = other.iloc[1:]
    elif problem == "labels":
        other.loc[0, "label"] = 1
    else:
        other.loc[0, "subject_id"] = 999
    with pytest.raises(ValueError, match="identical test"):
        evaluation.evaluate_model_comparison({"trag": predictions, "xgboost": other}, n_resamples=2)


@pytest.mark.parametrize("problem", ["missing_column", "duplicate", "leakage", "missing_metadata", "split", "no_test", "probability"])
def test_reject_invalid_prediction_records(predictions, problem):
    if problem == "missing_column":
        predictions = predictions.drop(columns="subject_id")
    elif problem == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[:1]])
    elif problem == "leakage":
        predictions.loc[6, "subject_id"] = 1
    elif problem == "missing_metadata":
        predictions.loc[0, "stay_id"] = np.nan
    elif problem == "split":
        predictions.loc[0, "split"] = "unknown"
    elif problem == "no_test":
        predictions = predictions[predictions.split != "test"]
    else:
        predictions.loc[0, "sepsis_probability"] = np.nan
    with pytest.raises(ValueError):
        evaluation.evaluate_model_comparison({"trag": predictions}, n_resamples=2)


def test_empty_or_unknown_comparison_fails(predictions):
    for inputs in ({}, {"unknown": predictions}):
        with pytest.raises(ValueError):
            evaluation.evaluate_model_comparison(inputs)


def test_report_export_and_cli_are_aggregate_only(predictions, tmp_path, capsys):
    source = tmp_path / "predictions.parquet"
    predictions.to_parquet(source, index=False)
    output = tmp_path / "evaluation"
    assert evaluation.main(["--prediction", f"xgboost={source}", "--output", str(output), "--resamples", "12"]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((output / "step5_validation.json").read_text())
    assert printed == saved
    assert saved["cohort"]["stays"] == 6
    csv = pd.read_csv(output / "model_comparison.csv")
    assert csv.model_variant.tolist() == ["xgboost"]
    assert "SYNTHETIC_PRIVATE_TEXT" not in (output / "step5_validation.json").read_text()
    assert set(p.name for p in output.iterdir()) == {"step5_validation.json", "model_comparison.csv"}


@pytest.mark.parametrize("location", ["top", "model", "metric", "interval", "cohort"])
def test_report_export_rejects_extra_patient_fields(predictions, tmp_path, location):
    report = evaluation.evaluate_model_comparison({"trag": predictions}, n_resamples=2)
    locations = {
        "top": report, "model": report["models"]["trag"],
        "metric": report["models"]["trag"]["metrics"],
        "interval": report["models"]["trag"]["bootstrap"]["intervals"]["auroc"],
        "cohort": report["cohort"],
    }
    locations[location]["patient_id"] = "SYNTHETIC_PRIVATE_TEXT"
    with pytest.raises(ValueError):
        evaluation.save_evaluation_report(report, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


class FakeRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, payload):
        self.logged.append(payload)

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.run = FakeRun()
        self.initialized = []

    def Settings(self, **kwargs):
        return kwargs

    def init(self, **kwargs):
        self.initialized.append(kwargs)
        return self.run


def test_wandb_is_opt_in_and_forwards_only_allowlisted_scalars(predictions):
    report = evaluation.evaluate_model_comparison({"trag": predictions}, n_resamples=2)
    client = FakeWandb()
    assert not evaluation.log_aggregate_to_wandb(report, wandb_client=client)
    assert client.initialized == []
    report["patient_prompt"] = "SYNTHETIC_PRIVATE_TEXT"
    report["models"]["trag"]["raw_predictions"] = predictions
    assert evaluation.log_aggregate_to_wandb(
        report, [{"step": 1, "train_loss": .4}], enabled=True, wandb_client=client,
    )
    assert client.initialized[0]["project"] == config.WANDB_PROJECT
    assert client.initialized[0]["entity"] == config.WANDB_ENTITY
    settings = client.initialized[0]["settings"]
    assert settings["console"] == "off"
    assert settings["disable_code"] and settings["disable_git"]
    assert settings["x_disable_stats"] and settings["x_disable_meta"]
    assert client.run.finished
    assert client.run.logged[-1] == {"training/step": 1, "training/train_loss": .4}
    assert all(isinstance(value, (int, float)) for payload in client.run.logged for value in payload.values())
    assert "SYNTHETIC_PRIVATE_TEXT" not in json.dumps(client.initialized + client.run.logged)


@pytest.mark.parametrize("history", [
    [{"prompt": "SYNTHETIC_PRIVATE_TEXT"}], [{"train_loss": np.nan}],
    [{"train_loss": "SYNTHETIC_PRIVATE_TEXT"}], [{"step": -1}], [{}],
])
def test_invalid_loss_history_fails_before_remote_init(predictions, history):
    report = evaluation.evaluate_model_comparison({"trag": predictions}, n_resamples=2)
    client = FakeWandb()
    with pytest.raises(ValueError):
        evaluation.log_aggregate_to_wandb(report, history, enabled=True, wandb_client=client)
    assert client.initialized == []


def test_tracking_run_finishes_if_logging_fails(predictions):
    report = evaluation.evaluate_model_comparison({"trag": predictions}, n_resamples=2)
    client = FakeWandb()

    def fail(payload):
        raise RuntimeError("synthetic logging error")

    client.run.log = fail
    with pytest.raises(RuntimeError):
        evaluation.log_aggregate_to_wandb(report, enabled=True, wandb_client=client)
    assert client.run.finished


def test_cli_rejects_duplicate_variant_arguments(predictions, tmp_path):
    source = tmp_path / "predictions.parquet"
    predictions.to_parquet(source, index=False)
    with pytest.raises(SystemExit):
        evaluation.main(["--prediction", f"trag={source}", "--prediction", f"trag={source}"])
