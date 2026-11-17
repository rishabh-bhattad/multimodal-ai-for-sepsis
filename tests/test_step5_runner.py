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
def test_smoke_orchestration_and_wandb_allowlist(tmp_path, monkeypatch, predict):
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
    assert factory.call_args.kwargs["args"].max_steps == 2
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
