"""Runner checks use synthetic records and mock model loading; no weights downloaded."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from datasets import Dataset, DatasetDict

from src import step5_run_training as runner


def records(subjects):
    return Dataset.from_list([
        dict(subject_id=s, stay_id=s, prompt_text="four clinical windows",
             image_paths=[], target_completion="Sepsis", label=1)
        for s in subjects
    ])


def test_metrics_never_forward_records():
    assert runner.scalar_metrics({"loss": 0.3, "prompt_text": "private",
                                  "learning_rate": float("nan"), "epoch": True}) == {"loss": 0.3}


def test_rejects_patient_leakage():
    data = {"train": records([1, 2]), "validation": records([2, 3])}
    with pytest.raises(ValueError, match="overlap"):
        runner.validate_datasets(data)


def test_test_split_is_never_read():
    data = {"train": records([1, 2]), "validation": records([3, 4]), "test": object()}
    selected = runner.validate_datasets(data, smoke_subjects=1)
    assert len(selected["train"]) == len(selected["validation"]) == 1
    assert set(selected) == {"train", "validation"}


def test_missing_images_fail(tmp_path):
    row = dict(subject_id=1, stay_id=1, prompt_text="<image>", image_paths=[str(tmp_path / "missing.jpg")],
               target_completion="Sepsis", label=1)
    with pytest.raises(ValueError, match="missing"):
        runner.validate_datasets({"train": Dataset.from_list([row]), "validation": records([2])})


def test_vision_layers_are_not_adapter_targets():
    model = SimpleNamespace(named_modules=lambda: [
        ("model.visual.blocks.0.q_proj", None),
        ("model.language_model.layers.0.self_attn.q_proj", None)])
    assert runner.language_adapter_targets(model) == ["model.language_model.layers.0.self_attn.q_proj"]


@pytest.mark.parametrize("predict", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_smoke_orchestration_and_wandb_allowlist(tmp_path, monkeypatch, predict, full):
    import torch
    import transformers
    import wandb
    from src import step5_training as training

    dataset = tmp_path / "dataset"
    DatasetDict(train=records([1, 2]), validation=records([3]), test=records([4])).save_to_disk(dataset)
    output = tmp_path / "run"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "mock GPU")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2**31)
    model = torch.nn.Module()
    model.q_proj = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(use_cache=True)
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        Mock(return_value=SimpleNamespace(model_type="qwen3_vl", _commit_hash="pinned")))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", Mock(return_value=processor))
    loader = Mock(return_value=model)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", loader)
    monkeypatch.setattr(training, "apply_qlora_to_model", lambda model, config: model)
    monkeypatch.setattr(training, "get_training_arguments", lambda **kwargs: SimpleNamespace(**kwargs))
    trainer = Mock()
    trainer.state.best_metric = 0.4
    trainer.state.best_model_checkpoint = "checkpoint-1"
    trainer.train.return_value = SimpleNamespace(metrics={"train_loss": 0.5, "private": "never upload"})
    trainer.evaluate.return_value = {"eval_loss": 0.4}
    factory = Mock(return_value=trainer)
    monkeypatch.setattr(training, "SepsisTrainer", factory)
    tracking = Mock()
    initialize = Mock(return_value=tracking)
    monkeypatch.setattr(wandb, "init", initialize)
    monkeypatch.chdir(tmp_path)
    args = runner.parser().parse_args(["--dataset", str(dataset), "--output", str(output),
                                      "--smoke", "--wandb-mode", "offline"])
    args.full = full
    args.predict_test = predict
    scorer = Mock(return_value={"sepsis_probability": 0.7})
    monkeypatch.setattr(training, "predict_sepsis", scorer)
    runner.run(args)
    if predict:
        assert (output / "predictions.parquet").is_file()
        assert scorer.call_args.kwargs["allow_graph_reduction"] is False
        assert scorer.call_args.kwargs["generate_rationale"] is False
    else:
        scorer.assert_not_called()
    assert loader.call_args.kwargs["device_map"] == {"": 0}
    assert factory.call_args.kwargs["args"].max_steps == (-1 if full else 2)
    assert factory.call_args.kwargs["args"].load_best_model_at_end is full
    assert factory.call_args.kwargs["args"].metric_for_best_model == "eval_loss"
    callback = factory.call_args.kwargs["callbacks"][0]
    control = SimpleNamespace(should_evaluate=False, should_save=False)
    callback.on_epoch_end(args=None, state=None, control=control)
    assert control.should_evaluate is full
    assert control.should_save is full
    assert factory.call_args.kwargs["args"].report_to == "none"
    assert initialize.call_args.kwargs["mode"] == "offline"
    assert "private" not in tracking.log.call_args.args[0]
    assert "dataset" not in initialize.call_args.kwargs["config"]
    trainer.train.assert_called_once_with(resume_from_checkpoint=None)
    trainer.save_state.assert_called_once()
    tracking.finish.assert_called_once()
    assert (output / "training_summary.json").is_file()


def test_strict_context_rejects_oversized_prompt():
    import torch
    from src.step5_training import prepare_model_prompt
    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "clinical prompt"
        def __call__(self, **kwargs):
            return {"input_ids": torch.ones((1, 12), dtype=torch.long)}
    record = {"prompt_text": "clinical prompt", "image_paths": []}
    with pytest.raises(ValueError, match="Complete prompt exceeds"):
        prepare_model_prompt(record, Processor(), 10, allow_graph_reduction=False)
    _, encoded = prepare_model_prompt(record, Processor(), 12, allow_graph_reduction=False)
    assert encoded["input_ids"].shape[-1] == 12


def test_xgboost_tracking_logs_only_aggregate_counts(tmp_path, monkeypatch):
    from src import step5_tabular_baseline as baseline
    import pandas as pd
    windows = tmp_path / "windows.parquet"
    pd.DataFrame({"synthetic": [1]}).to_parquet(windows)
    result = SimpleNamespace(summary={
        "training_parameters": {"n_estimators": 100},
        "candidate_validation_scores": [0.6],
        "splits": {"train": {"stays": 10, "subjects": 10, "positives": 5, "prevalence": 0.5}},
    }, predictions="private data")
    monkeypatch.setattr(baseline, "train_tabular_baseline", lambda _: result)
    monkeypatch.setattr(baseline, "persist_baseline", lambda *args: None)
    tracking = Mock()
    monkeypatch.setattr(runner, "start_tracking", lambda *args: tracking)
    args = runner.parser().parse_args(["--variant", "xgboost", "--full", "--windows", str(windows),
                                      "--output", str(tmp_path / "out")])
    runner.run_xgboost(args)
    assert tracking.log.call_args.args[0] == {
        "validation/average_precision": 0.6, "train/stays": 10, "train/subjects": 10,
        "train/positives": 5, "train/prevalence": 0.5}
    tracking.finish.assert_called_once()


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


@pytest.mark.parametrize("fail_first", [False, True])
def test_suite_trains_selected_variants_and_compares(inputs, tmp_path, monkeypatch, fail_first):
    windows, graphs = inputs
    output = tmp_path / "suite"
    calls = []
    def fake_training(command, **kwargs):
        assert kwargs["check"] is True
        variant = command[command.index("--variant") + 1]
        calls.append(variant)
        if fail_first and variant == "trag":
            raise subprocess.CalledProcessError(1, "synthetic")
        dataset = load_from_disk(command[command.index("--dataset") + 1])
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir(parents=True)
        pd.DataFrame([dict(stay_id=r["stay_id"], subject_id=r["subject_id"],
            split="test", label=r["label"], sepsis_probability=0.7 if r["label"] else 0.3)
            for r in dataset["test"]]).to_parquet(destination / "predictions.parquet")
    monkeypatch.setattr(suite.subprocess, "run", fake_training)
    assert suite.main(["--windows", str(windows), "--trajectories", str(graphs),
        "--output", str(output), "--variants", "trag", "no_kg",
        "--wandb-mode", "disabled", "--resamples", "10", "--continue-on-error"]) == (1 if fail_first else 0)
    assert calls == ["trag", "no_kg"]
    report = json.loads((output / "comparison/step5_validation.json").read_text())
    assert report["available_variants"] == (["no_kg"] if fail_first else ["trag", "no_kg"])
    state = json.loads((output / "status.json").read_text())
    assert state["stage"] == ("partial_completion" if fail_first else "completed")
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
