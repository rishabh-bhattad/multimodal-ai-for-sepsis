"""Unit tests for Step 4 multimodal prompt assembly and CXR image alignment."""

import math
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.step4_serialization import (
    assemble_multimodal_prompt,
    resolve_image_path,
    select_stays_for_dataset,
    validate_stay_alignment,
)


def test_sample_selects_unique_subjects_and_images_from_later_windows():
    rows = [
        {"stay_id": stay, "subject_id": subject, "window_idx": index,
         "cxr_path": "later.jpg" if stay == 11 and index == 3 else ""}
        for stay, subject in ((10, 1), (11, 1), (12, 2)) for index in range(4)
    ]
    windows = pd.DataFrame(rows)
    trajectories = {str(stay): {} for stay in (10, 11, 12)}
    assert select_stays_for_dataset(trajectories, windows, 2) == [11, 12]
    assert select_stays_for_dataset(trajectories, windows, None) == [10, 11, 12]
    with pytest.raises(ValueError, match="only 2 are eligible"):
        select_stays_for_dataset(trajectories, windows, 3)
    del trajectories["10"]
    with pytest.raises(ValueError, match="identical stay sets"):
        select_stays_for_dataset(trajectories, windows, None)
    assert select_stays_for_dataset(trajectories, windows, 2) == [11, 12]


@pytest.fixture
def aligned_stay():
    rows, graphs = [], {}
    for index in range(4):
        start = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=6 * index)
        end = start + pd.Timedelta(hours=6)
        rows.append({"subject_id": 1, "split": "train", "label": 0,
                     "window_idx": index, "window_start": start, "window_end": end})
        graphs[f"window_{index}"] = nx.MultiDiGraph(
            window_idx=index, start=start.isoformat(), end=end.isoformat()
        )
    return pd.DataFrame(rows), {"subject_id": 1, "split": "train", "windows": graphs}


def test_alignment_accepts_explicit_empty_graphs(aligned_stay):
    validate_stay_alignment(*aligned_stay)


@pytest.mark.parametrize("field,value", [
    ("subject_id", 2), ("split", "test"), ("label", 1), ("window_idx", 2),
    ("window_start", pd.Timestamp("2020-01-02")),
])
def test_alignment_rejects_inconsistent_windows(aligned_stay, field, value):
    rows, trajectory = aligned_stay
    rows.loc[0, field] = value
    with pytest.raises(ValueError):
        validate_stay_alignment(rows, trajectory)


@pytest.mark.parametrize("field,value", [("subject_id", 2), ("split", "test")])
def test_alignment_rejects_wrong_patient_or_split(aligned_stay, field, value):
    rows, trajectory = aligned_stay
    trajectory[field] = value
    with pytest.raises(ValueError, match="Trajectory subject or split"):
        validate_stay_alignment(rows, trajectory)


@pytest.mark.parametrize("field,value", [
    ("window_idx", 1), ("start", "2020-01-02"), ("end", None),
])
def test_alignment_rejects_wrong_graph_window(aligned_stay, field, value):
    rows, trajectory = aligned_stay
    trajectory["windows"]["window_0"].graph[field] = value
    with pytest.raises(ValueError):
        validate_stay_alignment(rows, trajectory)


def test_alignment_rejects_missing_graph(aligned_stay):
    rows, trajectory = aligned_stay
    del trajectory["windows"]["window_0"]
    with pytest.raises(ValueError, match="all four window graphs"):
        validate_stay_alignment(rows, trajectory)


def make_synthetic_windows(cxr_paths=(None, None, None, None)):
    """Helper to generate 4 contiguous synthetic 6-hour windows."""
    windows = []
    for idx, cxr in enumerate(cxr_paths):
        windows.append({
            "stay_id": 10001,
            "subject_id": 20001,
            "window_idx": idx,
            "heart_rate_mean": 80.0 + idx * 5,
            "map_latest": 75.0 - idx * 4,
            "spo2_mean": 97.0,
            "temp_latest": 37.0 + idx * 0.3,
            "resp_rate_mean": 16.0 + idx,
            "lactate_latest": 1.0 + idx * 0.8,
            "platelets_latest": 250.0 - idx * 20,
            "creatinine_latest": 0.8 + idx * 0.2,
            "wbc_latest": 10.0 + idx * 2,
            "bilirubin_latest": 0.5,
            "cxr_path": cxr,
        })
    return windows


def test_resolve_image_path():
    """Verify relative and absolute path resolution and missing-value handling."""
    assert resolve_image_path(None) is None
    assert resolve_image_path("") is None
    assert resolve_image_path("   ") is None
    assert resolve_image_path(float("nan")) is None
    assert resolve_image_path("nan") is None
    assert resolve_image_path("None") is None

    # Relative path resolution with custom cxr_dir
    cxr_dir = Path("/mock/cxr/root")
    resolved = resolve_image_path("p10/p10001/s501/image.jpg", cxr_dir=cxr_dir)
    assert resolved.endswith("image.jpg")
    assert "/mock/cxr/root" in resolved.replace("\\", "/")

    # Absolute path remains unchanged
    abs_path = Path("/already/absolute/image.jpg").resolve()
    assert resolve_image_path(str(abs_path)) == str(abs_path)


def test_assemble_multimodal_prompt_zero_images():
    """Verify prompt formatting when no radiograph is aligned for any window."""
    windows = make_synthetic_windows([None, None, None, None])
    prompt, image_paths = assemble_multimodal_prompt(windows)

    assert len(image_paths) == 0
    assert prompt.count("<image>") == 0
    assert prompt.startswith(config.STEP4_SYSTEM_CONTEXT)
    assert prompt.endswith(config.STEP4_TARGET_TASK)

    for i in range(4):
        assert f"[TIMELINE: T - {24 - 6*i} hours to T - {24 - 6*(i+1)} hours]" in prompt

    # All windows report no radiograph
    assert prompt.count("No chest radiograph acquired in this window.") == 4


def test_assemble_multimodal_prompt_single_image():
    """Verify exact 1-to-1 alignment when a radiograph is present in window 2."""
    img_rel = "p10/p10001/s501/window2.jpg"
    windows = make_synthetic_windows([None, None, img_rel, None])
    prompt, image_paths = assemble_multimodal_prompt(windows, cxr_dir="/mock/cxr")

    assert len(image_paths) == 1
    assert prompt.count("<image>") == 1
    assert image_paths[0].endswith("window2.jpg")

    # Window 2 has the placeholder; the other three report no radiograph
    assert prompt.count("Frontal Chest Radiograph acquired.") == 1
    assert prompt.count("No chest radiograph acquired in this window.") == 3


def test_assemble_multimodal_prompt_multiple_images_chronological_order():
    """Verify that multiple radiographs maintain strict chronological sequence in image_paths."""
    img0 = "p10/p10001/s501/win0.jpg"
    img2 = "p10/p10001/s502/win2.jpg"
    windows = make_synthetic_windows([img0, None, img2, None])
    prompt, image_paths = assemble_multimodal_prompt(windows, cxr_dir="/mock/cxr")

    assert len(image_paths) == 2
    assert prompt.count("<image>") == 2
    # Verify strict chronological ordering matching window 0 then window 2
    assert image_paths[0].endswith("win0.jpg")
    assert image_paths[1].endswith("win2.jpg")


def test_assemble_multimodal_prompt_with_subgraphs():
    """Verify integration of retrieved PrimeKG subgraphs in the multimodal prompt."""
    g0 = nx.MultiDiGraph()
    g0.add_node("n1", node_name="Consolidation")
    g0.add_node("n2", node_name="Pneumonia")
    g0.add_edge("n1", "n2", key="disease_disease", relation="disease_disease")

    subgraphs = {"window_0": g0}  # windows 1..3 will have None
    windows = make_synthetic_windows(["p10/p10001/s501/img.jpg", None, None, None])

    prompt, image_paths = assemble_multimodal_prompt(windows, subgraphs=subgraphs)
    assert len(image_paths) == 1
    from src.compact_kg import decode_prompt
    assert decode_prompt(prompt)[0] == [("Consolidation", "disease_disease", "Pneumonia")]
    # Windows 1..3 fallback to baseline
    assert prompt.count(config.STEP4_BASELINE_KG_TEXT) == 3


def test_assemble_multimodal_prompt_rejects_invalid_window_counts_or_indices():
    """Verify error checking on incomplete or unordered window collections."""
    # 3 windows instead of 4
    with pytest.raises(ValueError, match="Expected exactly 4 windows"):
        assemble_multimodal_prompt(make_synthetic_windows()[:3])

    # 5 windows
    five_windows = make_synthetic_windows() + [{"window_idx": 4}]
    with pytest.raises(ValueError, match="Expected exactly 4 windows"):
        assemble_multimodal_prompt(five_windows)

    # Duplicated index
    broken_windows = make_synthetic_windows()
    broken_windows[3]["window_idx"] = 1
    with pytest.raises(ValueError, match="sequential indices"):
        assemble_multimodal_prompt(broken_windows)

