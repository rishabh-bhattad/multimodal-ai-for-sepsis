"""Synthetic end-to-end suite orchestration; no GPU, network, or model downloads."""
import json
import pickle
from pathlib import Path
import subprocess

import networkx as nx
import pandas as pd
import pytest
from datasets import load_from_disk

from src import step5_run_experiments as suite


@pytest.fixture
def inputs(tmp_path):
    rows, graphs = [], {}
    for sid in range(1, 7):
        split = "train" if sid <= 2 else "val" if sid <= 4 else "test"
        windows = {}
        for index in range(4):
            start = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=6 * index)
            rows.append(dict(subject_id=sid, stay_id=sid, window_idx=index,
                window_start=start, window_end=start + pd.Timedelta(hours=6),
                split=split, label=sid % 2, cxr_path=None, heart_rate_mean=80))
            graph = nx.MultiDiGraph()
            graph.add_node("a", node_name="Finding")
            graph.add_node("b", node_name="Context")
            graph.add_edge("a", "b", relation="phenotype_phenotype")
            windows[f"window_{index}"] = graph
        graphs[str(sid)] = dict(subject_id=sid, split=split, windows=windows)
    windows = tmp_path / "windows.parquet"
    trajectories = tmp_path / "graphs.pkl"
    pd.DataFrame(rows).to_parquet(windows)
    trajectories.write_bytes(pickle.dumps(graphs))
    return windows, trajectories


def test_prepare_preserves_shared_cohort(inputs, tmp_path):
    windows, graphs = inputs
    paths, reference, digests = suite.prepare_variants(windows, graphs, suite.NEURAL, tmp_path, False)
    assert len(reference) == 24 and len(digests["windows_sha256"]) == 64
    datasets = {name: load_from_disk(str(path)) for name, path in paths.items()}
    for split in ("train", "validation", "test"):
        ids = [ds[split]["stay_id"] for ds in datasets.values()]
        assert ids[0] == ids[1] == ids[2]
        assert "Finding" in datasets["trag"][split][0]["prompt_text"]
        assert "Finding" not in datasets["no_kg"][split][0]["prompt_text"]


def test_suite_trains_selected_variants_and_compares(inputs, tmp_path, monkeypatch):
    windows, graphs = inputs
    output = tmp_path / "suite"
    calls = []
    def fake_training(command, **kwargs):
        assert kwargs["check"] is True
        variant = command[command.index("--variant") + 1]
        calls.append(variant)
        dataset = load_from_disk(command[command.index("--dataset") + 1])
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir(parents=True)
        pd.DataFrame([dict(stay_id=r["stay_id"], subject_id=r["subject_id"],
            split="test", label=r["label"], sepsis_probability=0.7 if r["label"] else 0.3)
            for r in dataset["test"]]).to_parquet(destination / "predictions.parquet")
    monkeypatch.setattr(suite.subprocess, "run", fake_training)
    assert suite.main(["--windows", str(windows), "--trajectories", str(graphs),
        "--output", str(output), "--variants", "trag", "no_kg",
        "--wandb-mode", "disabled", "--resamples", "10"]) == 0
    assert calls == ["trag", "no_kg"]
    report = json.loads((output / "comparison/step5_validation.json").read_text())
    assert report["available_variants"] == ["trag", "no_kg"]
    assert report["cohort"]["stays"] == 2


def test_failed_training_stops_later_variants(inputs, tmp_path, monkeypatch):
    windows, graphs = inputs
    call_count = []
    def fail(*args, **kwargs):
        call_count.append(1)
        raise subprocess.CalledProcessError(1, "synthetic")
    monkeypatch.setattr(suite.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        suite.main(["--windows", str(windows), "--trajectories", str(graphs),
            "--output", str(tmp_path / "failed"), "--variants", "trag", "no_kg",
            "--wandb-mode", "disabled"])
    assert len(call_count) == 1


def test_dry_run_creates_nothing(tmp_path):
    destination = tmp_path / "nothing"
    assert suite.main(["--output", str(destination), "--dry-run"]) == 0
    assert not destination.exists()


def test_reused_xgboost_alignment_rejects_other_cohort(tmp_path):
    reference = pd.DataFrame([dict(stay_id=1, subject_id=1, label=1, split="test")])
    cached = tmp_path / "cached.parquet"
    pd.DataFrame([dict(stay_id=2, subject_id=2, label=1, split="test",
                       sepsis_probability=0.5)]).to_parquet(cached)
    args = suite.parser().parse_args(["--output", str(tmp_path / "suite"), "--wandb-mode", "disabled"])
    with pytest.raises(ValueError, match="cover"):
        suite.compare(args, {"xgboost": cached}, reference)
