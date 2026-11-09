"""Local, stay-level XGBoost comparison using the existing four-window cohort.

No feature engineering is learned from validation/test data. Missing numerical
values remain NaN for XGBoost's native handling and have explicit indicators.
Only the allowlisted observations below become model inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


OBSERVATION_COLUMNS = (
    "hr_mean", "hr_min", "hr_max", "hr_slope",
    "map_mean", "map_min", "map_max", "map_slope", "map_latest",
    "temp_latest", "spo2_mean", "temp_mean", "resp_rate_mean",
    "lactate_latest", "platelets_latest", "creatinine_latest",
    "bilirubin_latest", "wbc_latest",
)
METADATA_COLUMNS = ("stay_id", "subject_id", "split", "label")
TUNABLE_PARAMETERS = frozenset({
    "max_depth", "learning_rate", "n_estimators", "subsample",
    "colsample_bytree", "min_child_weight", "reg_lambda", "reg_alpha", "gamma",
})


@dataclass
class TabularDataset:
    features: pd.DataFrame
    metadata: pd.DataFrame


@dataclass
class BaselineResult:
    model: XGBClassifier
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    summary: dict[str, Any]


def extract_tabular_features(windows: pd.DataFrame) -> TabularDataset:
    """Validate the cohort and produce 144 ordered features per complete stay.

    Requires all existing Step 1 observation columns, including columns that
    are entirely missing. Does not invent latest values absent from that schema.
    """
    required = set(METADATA_COLUMNS) | {"window_idx"} | set(OBSERVATION_COLUMNS)
    if required - set(windows.columns):
        raise ValueError("Input is missing required metadata or observation columns")
    if windows.empty:
        raise ValueError("Input must contain complete stays")
    data = windows.loc[:, list(METADATA_COLUMNS) + ["window_idx"] + list(OBSERVATION_COLUMNS)].copy()
    if data[list(METADATA_COLUMNS) + ["window_idx"]].isna().any().any():
        raise ValueError("Metadata and window indices must not be missing")
    if not data["label"].isin([0, 1]).all():
        raise ValueError("Labels must be binary")
    if not data["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Splits must be train, val, or test")
    if (data.groupby("subject_id")["split"].nunique() > 1).any():
        raise ValueError("Subject split isolation failed")
    if (data.groupby("stay_id")[list(METADATA_COLUMNS[1:])].nunique() > 1).any().any():
        raise ValueError("Metadata must be consistent within each stay")
    if not data["window_idx"].isin(range(4)).all():
        raise ValueError("Each stay requires window indices 0 through 3")
    if data.duplicated(["stay_id", "window_idx"]).any():
        raise ValueError("Duplicate stay/window keys")
    if not data.groupby("stay_id").size().eq(4).all():
        raise ValueError("Each stay requires exactly four windows")

    numeric = data[list(OBSERVATION_COLUMNS)].apply(pd.to_numeric, errors="raise")
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Observations must be finite or missing")
    data[list(OBSERVATION_COLUMNS)] = numeric.astype(float)
    metadata = data.drop_duplicates("stay_id").set_index("stay_id")[list(METADATA_COLUMNS[1:])].sort_index()
    wide = data.set_index(["stay_id", "window_idx"])[list(OBSERVATION_COLUMNS)].unstack("window_idx")
    columns = {}
    for window in range(4):
        for observation in OBSERVATION_COLUMNS:
            name = f"window_{window}__{observation}"
            values = wide[(observation, window)].reindex(metadata.index)
            columns[name] = values
            columns[f"{name}__missing"] = values.isna().astype("uint8")
    features = pd.DataFrame(columns, index=metadata.index)
    return TabularDataset(features=features, metadata=metadata)


def train_tabular_baseline(
    windows: pd.DataFrame,
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> BaselineResult:
    """Select by validation AUPRC, then predict with the selected train-only fit.

    Defaults to one prespecified configuration. An explicit grid is bounded to
    ten candidates; ties choose the first. Test labels never select parameters.
    """
    dataset = extract_tabular_features(windows)
    x, metadata = dataset.features, dataset.metadata
    masks = {split: metadata["split"].eq(split) for split in ("train", "val", "test")}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Training, validation, and test splits must all be nonempty")
    for split in ("train", "val"):
        if metadata.loc[masks[split], "label"].nunique() != 2:
            raise ValueError("Training and validation splits must each contain both classes")
    grid = [{}] if candidates is None else list(candidates)
    if not 1 <= len(grid) <= 10:
        raise ValueError("Provide between one and ten validation candidates")
    for candidate in grid:
        if set(candidate) - TUNABLE_PARAMETERS:
            raise ValueError("Candidate contains unsupported hyperparameters")
    base_params = {
        "max_depth": config.STEP5_XGB_MAX_DEPTH,
        "learning_rate": config.STEP5_XGB_LEARNING_RATE,
        "n_estimators": config.STEP5_XGB_N_ESTIMATORS,
        "subsample": config.STEP5_XGB_SUBSAMPLE,
        "colsample_bytree": config.STEP5_XGB_COLSAMPLE_BYTREE,
        "objective": "binary:logistic", "eval_metric": "logloss",
        "tree_method": "hist", "random_state": config.RANDOM_SEED,
        "n_jobs": 1, "missing": np.nan,
    }
    best_model = None
    best_score = -np.inf
    best_index = 0
    candidate_scores = []
    for index, overrides in enumerate(grid):
        model = XGBClassifier(**(base_params | dict(overrides)))
        model.fit(x.loc[masks["train"]], metadata.loc[masks["train"], "label"])
        val_prob = model.predict_proba(x.loc[masks["val"]])[:, 1]
        score = float(average_precision_score(metadata.loc[masks["val"], "label"], val_prob))
        candidate_scores.append(score)
        if score > best_score:
            best_model, best_score, best_index = model, score, index

    predictions = metadata.reset_index()
    predictions["sepsis_probability"] = best_model.predict_proba(x)[:, 1]
    probabilities = predictions["sepsis_probability"].to_numpy()
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model produced invalid probabilities")
    importance = pd.DataFrame({"feature": x.columns, "importance": best_model.feature_importances_})
    importance = importance.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    summary = {
        "model_variant": "xgboost", "validation": "passed",
        "observation_hours": 24, "features": len(x.columns),
        "missingness_policy": "native_nan_with_per_window_indicators",
        "selection_metric": "validation_average_precision",
        "candidate_validation_scores": candidate_scores,
        "selected_candidate": best_index, "selected_overrides": dict(grid[best_index]),
        "training_parameters": {k: v for k, v in best_model.get_params().items() if k in base_params and k != "missing"},
        "splits": {
            split: {
                "stays": int(mask.sum()),
                "subjects": int(metadata.loc[mask, "subject_id"].nunique()),
                "positives": int(metadata.loc[mask, "label"].sum()),
                "prevalence": float(metadata.loc[mask, "label"].mean()),
            }
            for split, mask in masks.items()
        },
    }
    return BaselineResult(best_model, predictions, importance, summary)


def persist_baseline(result: BaselineResult, output_dir: str | Path) -> Path:
    """Save model and local patient predictions separately from aggregate reports."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.model.save_model(output / "xgboost_model.json")
    result.predictions.to_parquet(output / "predictions.parquet", index=False)
    result.feature_importance.to_csv(output / "feature_importance.csv", index=False)
    (output / "baseline_summary.json").write_text(
        json.dumps(result.summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing experiment-cohort patient_windows.parquet")
    parser.add_argument("--output", type=Path, default=config.STEP5_RESULTS_DIR / "xgboost")
    args = parser.parse_args(argv)
    result = train_tabular_baseline(pd.read_parquet(args.input))
    persist_baseline(result, args.output)
    print(json.dumps(result.summary, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
