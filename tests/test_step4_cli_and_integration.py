"""Unit tests for Step 4 CLI, pipeline orchestration, and aggregate validation reporting."""

import json
import logging
import pickle
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from datasets import DatasetDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src import step4_serialization as step4


@pytest.fixture
def synthetic_step4_inputs(tmp_path):
    """Create small synthetic patient_windows.parquet and subgraph_trajectories.pkl."""
    # 12 stays across 10 unique subjects
    rows = []
    trajectories = {}
    for i in range(1, 11):
        subject_id = 1000 + i
        stay_id = 2000 + i
        split = "train" if i <= 6 else "val" if i <= 8 else "test"
        label = 1 if i % 2 == 0 else 0

        # Subgraphs for this stay
        stay_subgraphs = {}
        for w in range(4):
            g = nx.MultiDiGraph()
            if w == 0:
                g.add_node("c1", node_name="Tachycardia")
                g.add_node("c2", node_name="Sepsis")
                g.add_edge("c1", "c2", key="phenotype_phenotype", relation="phenotype_phenotype")
            stay_subgraphs[f"window_{w}"] = g

            rows.append({
                "stay_id": stay_id,
                "subject_id": subject_id,
                "window_idx": w,
                "split": split,
                "label": label,
                "heart_rate_mean": 85.0 + w,
                "map_latest": 72.0 - w,
                "spo2_mean": 96.0,
                "temp_latest": 37.2,
                "resp_rate_mean": 18.0,
                "lactate_latest": 1.2 + w * 0.5,
                "platelets_latest": 200.0,
                "creatinine_latest": 0.9,
                "wbc_latest": 11.0,
                "bilirubin_latest": 0.6,
                "cxr_path": f"p10/p{subject_id}/s500/img_{w}.jpg" if w == 1 and i % 3 == 0 else None,
            })

        trajectories[str(stay_id)] = {
            "subject_id": subject_id,
            "split": split,
            "windows": stay_subgraphs,
        }

    windows_df = pd.DataFrame(rows)
    parquet_path = tmp_path / "patient_windows.parquet"
    windows_df.to_parquet(parquet_path)

    pickle_path = tmp_path / "subgraph_trajectories.pkl"
    with pickle_path.open("wb") as f:
        pickle.dump(trajectories, f)

    return parquet_path, pickle_path


def test_compute_file_sha256(tmp_path):
    """Verify streaming SHA-256 calculation against known content."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Multimodal Sepsis Project", encoding="utf-8")
    digest = step4.compute_file_sha256(test_file)
    assert len(digest) == 64
    assert digest == step4.compute_file_sha256(test_file)


def test_load_trajectories_pickle_success_and_errors(tmp_path):
    """Verify loading valid trajectories and error handling on missing/corrupt files."""
    missing = tmp_path / "nonexistent.pkl"
    with pytest.raises(FileNotFoundError):
        step4.load_trajectories_pickle(missing)

    invalid_pkl = tmp_path / "invalid.pkl"
    with invalid_pkl.open("wb") as f:
        pickle.dump(["not", "a", "dict"], f)

    with pytest.raises(ValueError, match="must be a dict"):
        step4.load_trajectories_pickle(invalid_pkl)


def test_load_patient_windows_parquet_success_and_errors(tmp_path):
    """Verify loading valid parquet windows and error handling on schema mismatch."""
    missing = tmp_path / "nonexistent.parquet"
    with pytest.raises(FileNotFoundError):
        step4.load_patient_windows_parquet(missing)

    invalid_df = pd.DataFrame({"some_col": [1, 2, 3]})
    invalid_parquet = tmp_path / "invalid.parquet"
    invalid_df.to_parquet(invalid_parquet)

    with pytest.raises(ValueError, match="missing required columns"):
        step4.load_patient_windows_parquet(invalid_parquet)


def test_run_step4_pipeline_end_to_end(tmp_path, synthetic_step4_inputs):
    """Verify full Step 4 pipeline execution, disk output, and validation summary."""
    parquet_path, pickle_path = synthetic_step4_inputs
    output_dir = tmp_path / "tokenized_dataset"

    summary = step4.run_step4_pipeline(
        windows_path=parquet_path,
        trajectories_path=pickle_path,
        output_dir=output_dir,
        sample_size=10,
        full=False,
    )

    assert summary["validation"] == "passed"
    assert summary["unique_patients"] == 10
    assert summary["stays"] == 10
    assert summary["windows"] == 40
    assert "input_sha256" in summary
    assert "windows" in summary["input_sha256"]
    assert "trajectories" in summary["input_sha256"]

    # Verify saved DatasetDict on disk
    loaded_dataset = DatasetDict.load_from_disk(str(output_dir))
    assert set(loaded_dataset.keys()) == {"train", "validation", "test"}
    assert len(loaded_dataset["train"]) == 6
    assert len(loaded_dataset["validation"]) == 2
    assert len(loaded_dataset["test"]) == 2

    # Verify validation report file exists beside output_dir
    report_file = output_dir.parent / "step4_validation.json"
    assert report_file.is_file()
    saved_report = json.loads(report_file.read_text())
    assert saved_report == summary


def test_main_cli_arguments_and_smoke_run(tmp_path, synthetic_step4_inputs, caplog):
    """Verify CLI flags, argument parsing, error logging, and exit codes."""
    parquet_path, pickle_path = synthetic_step4_inputs
    output_dir = tmp_path / "cli_dataset"

    # Test invalid sample_size
    with pytest.raises(SystemExit) as caught:
        step4.main(["--sample_size", "0"])
    assert caught.value.code == 2

    # Test missing input fails with code 1 without exposing sensitive stack traces
    missing_parquet = tmp_path / "missing.parquet"
    with caplog.at_level(logging.ERROR, logger=step4.logger.name):
        code = step4.main([
            "--windows", str(missing_parquet),
            "--trajectories", str(pickle_path),
            "--output", str(output_dir),
        ])
    assert code == 1
    assert "Step 4 failed (FileNotFoundError)" in caplog.text

    # Test valid smoke CLI run
    code_success = step4.main([
        "--windows", str(parquet_path),
        "--trajectories", str(pickle_path),
        "--output", str(output_dir),
        "--sample_size", "5",
    ])
    assert code_success == 0
    assert output_dir.is_dir()
    assert (output_dir.parent / "step4_validation.json").is_file()

