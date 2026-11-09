"""Synthetic tests for the Step 5 tabular comparison; no patient data is read."""

import json

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.step5_tabular_baseline import (
    OBSERVATION_COLUMNS,
    extract_tabular_features,
    main,
    persist_baseline,
    train_tabular_baseline,
)


@pytest.fixture
def windows():
    rows = []
    for stay in range(12):
        for window in range(4):
            label = stay % 2
            row = {
                "stay_id": stay, "subject_id": 100 + stay,
                "split": ("train", "val", "test")[stay // 4],
                "label": label, "window_idx": window,
                "cohort_type": "must never become a feature",
                "t0": pd.Timestamp("2020-01-01"),
                "hadm_id": 999 + stay,
            }
            row.update({column: float(10 * label + window) for column in OBSERVATION_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_four_windows_feature_order_missingness_and_metadata_exclusion(windows):
    windows.loc[(windows.stay_id == 0) & (windows.window_idx == 2), "lactate_latest"] = np.nan
    windows["bilirubin_latest"] = np.nan
    dataset = extract_tabular_features(windows.sample(frac=1, random_state=3))
    assert dataset.features.shape == (12, 144)
    assert dataset.features.index.equals(dataset.metadata.index)
    assert dataset.features.columns[0] == "window_0__hr_mean"
    assert dataset.features.columns[-1] == "window_3__wbc_latest__missing"
    assert dataset.features.loc[0, "window_3__hr_mean"] == 3
    assert np.isnan(dataset.features.loc[0, "window_2__lactate_latest"])
    assert dataset.features.loc[0, "window_2__lactate_latest__missing"] == 1
    assert dataset.features.loc[0, "window_1__lactate_latest__missing"] == 0
    assert dataset.features["window_0__bilirubin_latest"].isna().all()
    assert dataset.features["window_0__bilirubin_latest__missing"].eq(1).all()
    assert set(dataset.metadata.columns) == {"subject_id", "split", "label"}
    pd.testing.assert_frame_equal(dataset.features, extract_tabular_features(windows).features)


@pytest.mark.parametrize("problem", [
    "duplicate", "incomplete", "bad_index", "fractional_index", "label",
    "missing_metadata", "unknown_split", "subject_leakage", "inconsistent_stay",
    "missing_column", "infinity", "nonnumeric", "empty",
])
def test_reject_invalid_cohort(windows, problem):
    if problem == "duplicate":
        windows = pd.concat([windows, windows.iloc[:1]], ignore_index=True)
    elif problem == "incomplete":
        windows = windows.iloc[1:]
    elif problem == "bad_index":
        windows.loc[0, "window_idx"] = 4
    elif problem == "fractional_index":
        windows["window_idx"] = windows["window_idx"].astype(float)
        windows.loc[0, "window_idx"] = 0.5
    elif problem == "label":
        windows.loc[0, "label"] = 2
    elif problem == "missing_metadata":
        windows.loc[0, "subject_id"] = np.nan
    elif problem == "unknown_split":
        windows.loc[0, "split"] = "holdout"
    elif problem == "subject_leakage":
        windows.loc[windows.stay_id == 8, "subject_id"] = 100
    elif problem == "inconsistent_stay":
        windows.loc[0, "label"] = 1
    elif problem == "missing_column":
        windows = windows.drop(columns="hr_mean")
    elif problem == "infinity":
        windows.loc[0, "hr_mean"] = np.inf
    elif problem == "nonnumeric":
        windows["hr_mean"] = windows["hr_mean"].astype(object)
        windows.loc[0, "hr_mean"] = "not a number"
    elif problem == "empty":
        windows = windows.iloc[:0]
    with pytest.raises((ValueError, TypeError)):
        extract_tabular_features(windows)


def test_multiple_stays_per_subject_within_one_split_are_retained(windows):
    windows.loc[windows.stay_id == 1, "subject_id"] = 100
    dataset = extract_tabular_features(windows)
    assert len(dataset.features) == 12
    assert dataset.metadata.subject_id.nunique() == 11


def test_fit_uses_train_only_and_test_labels_do_not_select_model(windows, monkeypatch):
    original_fit = XGBClassifier.fit
    fitted_indices = []

    def record_fit(model, x, y, **kwargs):
        fitted_indices.append(list(x.index))
        assert list(y.index) == list(x.index)
        assert "eval_set" not in kwargs
        return original_fit(model, x, y, **kwargs)

    monkeypatch.setattr(XGBClassifier, "fit", record_fit)
    grid = [{"n_estimators": 3, "max_depth": 1}, {"n_estimators": 5, "max_depth": 2}]
    result = train_tabular_baseline(windows, grid)
    changed = windows.copy()
    changed.loc[changed.split == "test", "label"] = 1 - changed.loc[changed.split == "test", "label"]
    second = train_tabular_baseline(changed, grid)
    assert fitted_indices == [[0, 1, 2, 3]] * 4
    assert result.summary["candidate_validation_scores"] == second.summary["candidate_validation_scores"]
    assert result.summary["selected_candidate"] == second.summary["selected_candidate"]
    np.testing.assert_array_equal(result.predictions.sepsis_probability, second.predictions.sepsis_probability)
    assert result.predictions.sepsis_probability.between(0, 1).all()
    assert result.summary["splits"]["test"]["prevalence"] == 0.5
    assert len(result.feature_importance) == 144
    assert set(result.feature_importance.feature) == set(extract_tabular_features(windows).features.columns)


@pytest.mark.parametrize("problem", ["empty_grid", "large_grid", "unsafe_parameter", "missing_split", "one_class_train", "one_class_val"])
def test_reject_invalid_training_setup(windows, problem):
    grid = [{"n_estimators": 2}]
    if problem == "empty_grid":
        grid = []
    elif problem == "large_grid":
        grid = [{}] * 11
    elif problem == "unsafe_parameter":
        grid = [{"objective": "reg:squarederror"}]
    elif problem == "missing_split":
        windows = windows[windows.split != "test"]
    elif problem.startswith("one_class_"):
        split = problem.removeprefix("one_class_")
        windows.loc[windows.split == split, "label"] = 0
    with pytest.raises(ValueError):
        train_tabular_baseline(windows, grid)


def test_model_predictions_importance_and_summary_roundtrip(windows, tmp_path):
    result = train_tabular_baseline(windows, [{"n_estimators": 3}])
    output = persist_baseline(result, tmp_path / "baseline")
    reloaded = XGBClassifier()
    reloaded.load_model(output / "xgboost_model.json")
    features = extract_tabular_features(windows).features
    np.testing.assert_allclose(reloaded.predict_proba(features)[:, 1], result.predictions.sepsis_probability)
    pd.testing.assert_frame_equal(pd.read_parquet(output / "predictions.parquet"), result.predictions)
    assert len(pd.read_csv(output / "feature_importance.csv")) == 144
    summary = json.loads((output / "baseline_summary.json").read_text())
    assert summary["validation"] == "passed"
    assert summary["features"] == 144
    assert not {"subject_id", "stay_id", "predictions", "label"} & summary.keys()


def test_cli_on_synthetic_parquet_only(windows, tmp_path, capsys):
    source = tmp_path / "windows.parquet"
    windows.to_parquet(source, index=False)
    output = tmp_path / "results"
    assert main(["--input", str(source), "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["splits"]["train"]["stays"] == 4
    assert "subject_id" not in report
    assert (output / "xgboost_model.json").exists()
