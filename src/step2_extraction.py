"""
src/step2_extraction.py — Radiographic Finding Extraction

Extracts active positive radiographic findings (label == 1.0)
from MIMIC-CXR CheXpert and NegBio tabular reports for cohort studies.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import config
from src.utils import setup_logger

logger = setup_logger("Step2-CXR-Extraction")


def extract_study_id(cxr_path: Optional[str]) -> Optional[int]:
    """Extracts the integer study_id from a relative CXR image path."""
    if not cxr_path or pd.isna(cxr_path):
        return None
    match = re.search(r"/s(\d+)/", str(cxr_path))
    return int(match.group(1)) if match else None


def load_radiographic_findings(study_ids: List[int]) -> Dict[int, List[str]]:
    """Loads active positive radiographic findings (label == 1.0) from CheXpert and NegBio tabular reports."""
    logger.info("Extracting active radiographic findings for %d studies...", len(study_ids))
    if not study_ids:
        return {}

    study_set = set(study_ids)
    findings_per_study: Dict[int, set] = {sid: set() for sid in study_ids}

    # Parse official CheXpert labels, retaining only positive mentions (label == 1.0)
    chex_path = config.MIMIC_CXR_DIR / "mimic-cxr-2.0.0-chexpert.csv"
    if not chex_path.exists():
        chex_path = config.MIMIC_CXR_DIR / "mimic-cxr-2.0.0-chexpert.csv.gz"

    if chex_path.exists():
        for chunk in pd.read_csv(chex_path, chunksize=50000):
            matched = chunk[chunk["study_id"].isin(study_set)]
            for _, row in matched.iterrows():
                sid = int(row["study_id"])
                for pathology in config.TARGET_PATHOLOGIES:
                    if pathology in row and row[pathology] == 1.0:
                        findings_per_study[sid].add(pathology)

    # Union with NegBio labels to maximize sensitivity for pneumonia and consolidation
    negbio_path = config.MIMIC_CXR_DIR / "mimic-cxr-2.0.0-negbio.csv"
    if not negbio_path.exists():
        negbio_path = config.MIMIC_CXR_DIR / "mimic-cxr-2.0.0-negbio.csv.gz"

    if negbio_path.exists():
        for chunk in pd.read_csv(negbio_path, chunksize=50000):
            matched = chunk[chunk["study_id"].isin(study_set)]
            for _, row in matched.iterrows():
                sid = int(row["study_id"])
                for pathology in config.TARGET_PATHOLOGIES:
                    if pathology in row and row[pathology] == 1.0:
                        findings_per_study[sid].add(pathology)

    result = {sid: sorted(list(findings)) for sid, findings in findings_per_study.items()}
    logger.info("Completed radiographic extraction. Found active findings for %d/%d studies.",
                sum(1 for f in result.values() if f), len(study_ids))
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract active CXR findings for cohort studies.")
    parser.add_argument("--input", type=str, default=str(config.PATIENT_WINDOWS_PARQUET))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file %s not found.", input_path)
        sys.exit(1)

    df = pd.read_parquet(input_path)
    cxr_paths = df["cxr_path"].dropna().unique().tolist()
    study_ids = [extract_study_id(p) for p in cxr_paths if extract_study_id(p) is not None]

    findings = load_radiographic_findings(study_ids)
    for sid, active in list(findings.items())[:5]:
        logger.info("Study %s: %s", sid, active if active else "No active target pathologies")


if __name__ == "__main__":
    main()
