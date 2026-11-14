"""Software integration gate on synthetic data, not the research exit gate.

No model downloads, GPU jobs, patient data, or external service calls. Real
Qwen image processing, GPU training, frozen experimental runs, and held-out
performance remain required before declaring the experiment complete.
"""

from copy import deepcopy
import json

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.step4_serialization import assemble_multimodal_prompt, tokenize_and_mask_labels
from src.step5_ablations import build_ablation_dataset_dict
from src.step5_evaluation import evaluate_model_comparison, save_evaluation_report
from src.step5_evidence_audit import (
    audit_heldout_sample, audit_rationale, build_audit_context,
    save_audit_summary, select_audit_sample,
)
from src.step5_tabular_baseline import OBSERVATION_COLUMNS, persist_baseline, train_tabular_baseline
from src.step5_training import (
    MultimodalDataCollator, apply_qlora_to_model, extract_sepsis_probability,
    format_inference_output, get_qlora_config, save_adapter_checkpoint,
    validate_inference_output, verify_adapter_reload,
)


class LocalTokenizer:
    """Single-token synthetic classes; does not emulate Qwen tokenization."""

    def __init__(self):
        self.vocabulary = {"<pad>": 0, "<unk>": 1, "Sepsis": 2, "Non-Sepsis": 3}

    def encode(self, text):
        return [self.vocabulary.setdefault(word, len(self.vocabulary)) for word in text.split()]

    def __call__(self, text, add_special_tokens=True):
        ids = self.encode(text)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


@pytest.fixture
def synthetic_pipeline(tmp_path):
    image = tmp_path / "synthetic.jpg"
    Image.new("RGB", (8, 8), "gray").save(image)
    rows, trajectories, predictions, annotations = [], {}, [], {}
    rationale = "Lactate is 4 mmol/L in window 1."
    for stay in range(40):
        split = "train" if stay < 8 else "val" if stay < 12 else "test"
        graphs = {}
        for window in range(4):
            row = {"stay_id": stay, "subject_id": 100 + stay, "split": split,
                   "label": stay % 2, "window_idx": window,
                   "cxr_path": "synthetic.jpg" if window == 1 else None}
            row.update({field: float(2 + stay % 2 + window) for field in OBSERVATION_COLUMNS})
            row["lactate_latest"] = 4. if window == 1 else 1.
            graph = nx.MultiDiGraph()
            graph.graph.update(window_idx=window, findings=[])
            if window == 1:
                graph.add_node(("disease", "d"), node_name="Synthetic disease")
                graph.add_node(("effect/phenotype", "p"), node_name="Synthetic finding")
                graph.add_edge(("disease", "d"), ("effect/phenotype", "p"),
                               key="disease_phenotype_positive", relation="disease_phenotype_positive")
                graph.graph["findings"] = [{"name": "Elevated Lactate", "source": "lactate_latest"}]
            graphs[f"window_{window}"] = graph
            rows.append(row)
        trajectories[str(stay)] = {"subject_id": 100 + stay, "split": split, "windows": graphs}
        predictions.append({"stay_id": stay, "subject_id": 100 + stay, "split": split,
                            "label": stay % 2, "sepsis_probability": .75 if stay % 2 else .25,
                            "reasoning_chain": rationale})
        annotations[str(stay)] = {
            "claims": [{"start": 0, "end": len(rationale), "quote": rationale,
                        "kind": "measurement", "window_idx": 1, "field": "lactate_latest",
                        "value": 4., "unit": "mmol/L"}],
            # These are synthetic reviewer decisions, not a clinical review.
            "review": {"reviewer_role": "synthetic_test_reviewer", "claims_complete": True,
                       "decisions": ["supported"]},
        }
    return pd.DataFrame(rows), trajectories, pd.DataFrame(predictions), annotations, tmp_path


def test_prompt_collation_adapter_gradient_checkpoint_and_score_integration(synthetic_pipeline):
    windows, trajectories, _, _, root = synthetic_pipeline
    selected = windows[windows.stay_id == 0].to_dict("records")
    prompt, images = assemble_multimodal_prompt(selected, trajectories["0"]["windows"], cxr_dir=root)
    assert prompt.count("<image>") == len(images) == 1
    assert [prompt.index(f"T - {24 - 6 * i} hours") for i in range(4)] == sorted(prompt.index(f"T - {24 - 6 * i} hours") for i in range(4))
    tokenizer = LocalTokenizer()
    feature = tokenize_and_mask_labels(prompt, "Sepsis", tokenizer)
    batch = MultimodalDataCollator()([feature])
    assert batch["labels"][0, -1].item() == 2
    assert batch["labels"][0, :-1].eq(-100).all()
    assert len(tokenizer.vocabulary) < 512
    torch.manual_seed(42)
    settings = GPT2Config(vocab_size=512, n_positions=2048, n_embd=16, n_layer=1, n_head=2,
                         resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
    base = GPT2LMHeadModel(settings)
    base_state = deepcopy(base.state_dict())
    model = apply_qlora_to_model(base, get_qlora_config(r=2, lora_alpha=4, lora_dropout=0., target_modules=["c_attn"]), prepare_for_kbit=False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=.01)
    model.train()
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    optimizer.step()
    model.eval()
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    with torch.no_grad():
        expected = model(**inputs).logits
    checkpoint = save_adapter_checkpoint(model, root / "tiny_adapter")
    fresh = GPT2LMHeadModel(settings)
    fresh.load_state_dict(base_state)
    assert verify_adapter_reload(fresh, checkpoint, inputs, expected_logits=expected)
    # Score the position immediately before the synthetic target completion.
    probability = extract_sepsis_probability(expected[0, -2], pos_token_id=2, neg_token_id=3)
    output = format_inference_output(probability, "Lactate is 4 mmol/L in window 1.")
    assert validate_inference_output(output)
    assert 0 <= output["sepsis_probability"] <= 1


def test_all_comparison_variants_and_frozen_audit_sample_integrate(synthetic_pipeline):
    windows, trajectories, predictions, annotations, root = synthetic_pipeline
    by_split = {"train": [], "validation": [], "test": []}
    for stay in (0, 8, 12):
        block = windows[windows.stay_id == stay]
        split = "validation" if block.iloc[0].split == "val" else block.iloc[0].split
        by_split[split].append({"stay_id": stay, "subject_id": 100 + stay,
                               "label": stay % 2, "windows": block.to_dict("records"),
                               "subgraphs": trajectories[str(stay)]["windows"]})
    for variant in ("static_rag", "no_kg"):
        dataset = build_ablation_dataset_dict(by_split, variant, tokenizer=LocalTokenizer(), cxr_dir=root)
        assert set(dataset) == {"train", "validation", "test"}
        for split in dataset:
            example = dataset[split][0]
            assert example["prompt_text"].count("<image>") == len(example["image_paths"]) == 1
            assert example["labels"][-1] in (2, 3)
            assert all(value == -100 for value in example["labels"][:-1])
    baseline = train_tabular_baseline(windows, [{"n_estimators": 3, "max_depth": 2}])
    persist_baseline(baseline, root / "tabular")
    inputs = {variant: predictions for variant in ("trag", "static_rag", "no_kg")}
    inputs["xgboost"] = baseline.predictions
    report = evaluate_model_comparison(inputs, n_resamples=25)
    assert report["all_variants_present"]
    assert report["cohort"]["stays"] == 28
    save_evaluation_report(report, root / "evaluation")
    manifest = select_audit_sample(predictions, size=25)
    manifest.to_parquet(root / "local_audit_manifest.parquet", index=False)
    frozen = pd.read_parquet(root / "local_audit_manifest.parquet")
    selected_annotations = {str(stay): annotations[str(stay)] for stay in frozen.stay_id}
    summary = audit_heldout_sample(frozen, predictions, windows, trajectories, selected_annotations)
    assert summary["cases"] == summary["reviewed_cases"] == 25
    assert summary["pending_review_cases"] == 0
    assert summary["automatic_match_fraction"] == 1
    assert summary["clinical_correctness_validated"] is False
    case_audits = []
    for row in frozen.itertuples(index=False):
        context = build_audit_context(windows[windows.stay_id == row.stay_id].to_dict("records"), trajectories[str(row.stay_id)]["windows"])
        annotation = selected_annotations[str(row.stay_id)]
        rationale = predictions.set_index("stay_id").loc[row.stay_id, "reasoning_chain"]
        case_audits.append(audit_rationale(rationale, annotation["claims"], context, annotation["review"]))
    saved = json.loads(save_audit_summary(case_audits, root / "audit").read_text())
    assert saved == summary
    assert not {"stay_id", "subject_id", "reasoning_chain", "reviewer_role"} & set(saved)


@pytest.mark.parametrize("problem", ["prediction_subject", "window_subject", "trajectory_split", "annotation_shortfall", "manifest_train"])
def test_audit_adapter_rejects_wrong_stay_or_unfrozen_annotations(synthetic_pipeline, problem):
    windows, trajectories, predictions, annotations, _ = synthetic_pipeline
    manifest = select_audit_sample(predictions, size=20)
    selected = {str(stay): annotations[str(stay)] for stay in manifest.stay_id}
    stay = manifest.iloc[0].stay_id
    if problem == "prediction_subject":
        predictions.loc[predictions.stay_id == stay, "subject_id"] = 999
    elif problem == "window_subject":
        windows.loc[windows.stay_id == stay, "subject_id"] = 999
    elif problem == "trajectory_split":
        trajectories[str(stay)]["split"] = "train"
    elif problem == "annotation_shortfall":
        del selected[str(stay)]
    else:
        manifest.loc[0, "split"] = "train"
    with pytest.raises(ValueError):
        audit_heldout_sample(manifest, predictions, windows, trajectories, selected)
