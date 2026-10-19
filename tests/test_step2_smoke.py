"""Smoke test for chest radiograph study identifier parsing and validation."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.step2_extraction import extract_study_id


def test_study_id_extraction():
    path = "p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384082.jpg"
    assert extract_study_id(path) == 50414267
    assert extract_study_id(None) is None
    assert extract_study_id("") is None
    print("[PASS] test_study_id_extraction")


if __name__ == "__main__":
    test_study_id_extraction()
    print("[SUCCESS] Study ID extraction smoke test passed.")
