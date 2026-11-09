"""Step 4 Exit Gate & End-to-End Verification Test Suite.

Verifies all technical contracts specified in ARCHITECTURE_SPEC.md and TASKS.md:
- S4-01: Canonical binary completions and probability normalization.
- S4-02: 24-hour chronological timeline serialization and strict 1-to-1 multimodal CXR alignment.
- S4-03: Target-only causal loss masking, DatasetDict persistence, and split isolation without PHI.
"""

import json
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


class MockTokenizer:
    """Deterministic word-level tokenizer for loss masking and length tests."""

    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1, "<image>": 2, "Sepsis": 3, "Non-Sepsis": 4}

    def _get_id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def encode(self, text: str) -> list[int]:
        return [self._get_id(t) for t in text.split()]

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        ids = self.encode(text)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


@pytest.fixture
def cohort_pipeline_fixture(tmp_path):
    """Build a realistic multi-patient cohort fixture with mixed image modalities and PrimeKG subgraphs."""
    num_patients = 10
    rows = []
    trajectories = {}

    for i in range(1, num_patients + 1):
        subject_id = 10000 + i
        stay_id = 20000 + i
        split = "train" if i <= 6 else "val" if i <= 8 else "test"
        label = 1 if i % 2 == 1 else 0

        stay_subgraphs = {}
        for w in range(config.NUM_WINDOWS):
            g = nx.MultiDiGraph()
            if w == 1:
                g.add_node("d1", node_name="Consolidation")
                g.add_node("d2", node_name="Bacterial Pneumonia")
                g.add_edge("d1", "d2", key="disease_disease", relation="disease_disease")
            elif w == 3 and label == 1:
                g.add_node("d3", node_name="Septic Shock")
                g.add_node("p1", node_name="Microvascular Hypoperfusion")
                g.add_edge("d3", "p1", key="disease_phenotype_positive", relation="disease_phenotype_positive")

            stay_subgraphs[f"window_{w}"] = g

            # Simulate radiograph in window 2 for even patients, window 0 for patients divisible by 3
            cxr = None
            if i % 2 == 0 and w == 2:
                cxr = f"p10/p{subject_id}/s500{i}/img_w2.jpg"
            elif i % 3 == 0 and w == 0:
                cxr = f"p10/p{subject_id}/s500{i}/img_w0.jpg"

            rows.append({
                "stay_id": stay_id,
                "subject_id": subject_id,
                "window_idx": w,
                "split": split,
                "label": label,
                "heart_rate_mean": 75.0 + w * 10,
                "map_latest": 85.0 - w * 7,
                "spo2_mean": 98.0 - w,
                "temp_latest": 37.0 + w * 0.4,
                "resp_rate_mean": 16.0 + w * 2,
                "lactate_latest": 1.1 + w * 0.9,
                "platelets_latest": 240.0 - w * 30,
                "creatinine_latest": 0.8 + w * 0.3,
                "wbc_latest": 9.0 + w * 2,
                "bilirubin_latest": 0.6,
                "cxr_path": cxr,
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


def test_step4_exit_gate_contract(tmp_path, cohort_pipeline_fixture):
    """Verify that Step 4 produces a valid Hugging Face dataset meeting all exit-gate contracts."""
    parquet_path, pickle_path = cohort_pipeline_fixture
    output_dir = tmp_path / "exit_gate_dataset"

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
    assert summary["stays_per_split"] == {"train": 6, "validation": 2, "test": 2}

    # Verify on-disk dataset reload
    dataset = DatasetDict.load_from_disk(str(output_dir))
    assert set(dataset.keys()) == {"train", "validation", "test"}

    # Verify column schemas across all splits
    expected_columns = {
        "stay_id", "subject_id", "prompt_text", "image_paths", "target_completion", "label",
    }
    for split_name, split_ds in dataset.items():
        assert expected_columns.issubset(set(split_ds.column_names))
        for row in split_ds:
            assert row["target_completion"] in ("Sepsis", "Non-Sepsis")
            assert row["label"] in (0, 1)
            # Verify 1-to-1 multimodal alignment
            assert row["prompt_text"].count("<image>") == len(row["image_paths"])
            # Verify 4 chronological timeline headers
            for w in range(4):
                expected_hdr = f"[TIMELINE: T - {24 - 6*w} hours to T - {24 - 6*(w+1)} hours]"
                assert expected_hdr in row["prompt_text"]

    # Verify strict subject-level split disjointness
    train_subjects = set(dataset["train"]["subject_id"])
    val_subjects = set(dataset["validation"]["subject_id"])
    test_subjects = set(dataset["test"]["subject_id"])

    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)


def test_step4_deterministic_serialization(tmp_path, cohort_pipeline_fixture):
    """Verify that prompt serialization is bit-for-bit deterministic across repeated pipeline runs."""
    parquet_path, pickle_path = cohort_pipeline_fixture
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    step4.run_step4_pipeline(
        windows_path=parquet_path,
        trajectories_path=pickle_path,
        output_dir=out1,
        sample_size=10,
    )
    step4.run_step4_pipeline(
        windows_path=parquet_path,
        trajectories_path=pickle_path,
        output_dir=out2,
        sample_size=10,
    )

    ds1 = DatasetDict.load_from_disk(str(out1))
    ds2 = DatasetDict.load_from_disk(str(out2))

    for split in ("train", "validation", "test"):
        assert ds1[split]["prompt_text"] == ds2[split]["prompt_text"]
        assert ds1[split]["image_paths"] == ds2[split]["image_paths"]
        assert ds1[split]["target_completion"] == ds2[split]["target_completion"]


def test_step4_target_loss_masking_integrity():
    """Verify that causal language modeling loss is strictly applied only to the target completion."""
    tokenizer = MockTokenizer()
    prompt = "Timeline of vitals and retrieved pathways... [TARGET TASK] Respond with JSON schema."
    target = "Sepsis"

    tokenized = step4.tokenize_and_mask_labels(prompt, target, tokenizer=tokenizer)

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = tokenized["labels"]

    assert len(input_ids) == len(attention_mask) == len(labels)

    prompt_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt_ids)

    # Every single prompt token must be masked with -100
    assert labels[:prompt_len] == [-100] * prompt_len

    # Target completion tokens must NOT be masked and must match input_ids
    assert labels[prompt_len:] == input_ids[prompt_len:]
    assert all(lbl != -100 for lbl in labels[prompt_len:])


def test_step4_validation_report_has_no_phi(tmp_path, cohort_pipeline_fixture):
    """Verify that step4_validation.json contains only aggregate counts and file digests with zero PHI."""
    parquet_path, pickle_path = cohort_pipeline_fixture
    output_dir = tmp_path / "phi_check_dataset"

    step4.run_step4_pipeline(
        windows_path=parquet_path,
        trajectories_path=pickle_path,
        output_dir=output_dir,
        sample_size=10,
    )

    report_path = output_dir.parent / "step4_validation.json"
    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)

    # Required aggregate fields
    required_fields = {
        "validation", "mode", "unique_patients", "stays", "windows",
        "stays_per_label", "patients_per_split", "stays_per_split",
        "stays_with_images", "total_images_referenced", "input_sha256",
    }
    assert required_fields.issubset(set(report.keys()))

    # Zero clinical text, prompt text, or individual subject identifiers in the validation file
    assert "prompt_text" not in report_text
    assert "[TIMELINE:" not in report_text
    assert "10001" not in report_text  # subject ID
    assert "20001" not in report_text  # stay ID
