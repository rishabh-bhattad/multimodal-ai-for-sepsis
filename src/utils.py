"""
src/utils.py — Shared Utilities, Logging, I/O, & Clinical SOFA Calculators
==========================================================================
Provides reusable logging, safe serialization/deserialization helpers,
and clinically validated SOFA sub-score calculation tables for MIMIC-IV.
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


# ==============================================================================
# 1. STRUCTURED LOGGING
# ==============================================================================
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger("Utils")


# ==============================================================================
# 2. SAFE FILE I/O HELPERS
# ==============================================================================
def save_parquet(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """Safely saves a Pandas DataFrame to Parquet format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved %d rows to Parquet: %s", len(df), path)


def load_parquet(file_path: Union[str, Path]) -> pd.DataFrame:
    """Loads a Parquet file into a Pandas DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded %d rows from Parquet: %s", len(df), path)
    return df


def save_json(data: Any, file_path: Union[str, Path]) -> None:
    """Saves arbitrary data structure to JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved JSON file: %s", path)


def load_json(file_path: Union[str, Path]) -> Any:
    """Loads a JSON file into memory."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded JSON file: %s", path)
    return data


def save_pickle(obj: Any, file_path: Union[str, Path]) -> None:
    """Saves an object to a binary pickle file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved Pickle file: %s", path)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """Loads a pickled object from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("Loaded Pickle file: %s", path)
    return obj


# ==============================================================================
# 3. CLINICAL SOFA SUB-SCORE CALCULATION TABLES (Sepsis-3)
# ==============================================================================
def score_respiration_sofa(pao2_fio2_ratio: Optional[float], ventilated: bool = False) -> int:
    """Calculates SOFA Respiration score based on PaO2/FiO2 ratio."""
    if pao2_fio2_ratio is None or pd.isna(pao2_fio2_ratio):
        return 0
    if pao2_fio2_ratio < 100 and ventilated:
        return 4
    if pao2_fio2_ratio < 200 and ventilated:
        return 3
    if pao2_fio2_ratio < 300:
        return 2
    if pao2_fio2_ratio < 400:
        return 1
    return 0


def score_coagulation_sofa(platelets: Optional[float]) -> int:
    """Calculates SOFA Coagulation score based on Platelets count (x10^3/uL)."""
    if platelets is None or pd.isna(platelets):
        return 0
    if platelets < 20:
        return 4
    if platelets < 50:
        return 3
    if platelets < 100:
        return 2
    if platelets < 150:
        return 1
    return 0


def score_liver_sofa(bilirubin: Optional[float]) -> int:
    """Calculates SOFA Liver score based on Total Bilirubin (mg/dL)."""
    if bilirubin is None or pd.isna(bilirubin):
        return 0
    if bilirubin >= 12.0:
        return 4
    if bilirubin >= 6.0:
        return 3
    if bilirubin >= 2.0:
        return 2
    if bilirubin >= 1.2:
        return 1
    return 0


def score_cardiovascular_sofa(
    mean_arterial_pressure: Optional[float],
    dopamine_rate: float = 0.0,
    epinephrine_rate: float = 0.0,
    norepinephrine_rate: float = 0.0,
) -> int:
    """Calculates SOFA Cardiovascular score based on MAP and vasopressors."""
    if dopamine_rate > 15 or epinephrine_rate > 0.1 or norepinephrine_rate > 0.1:
        return 4
    if dopamine_rate > 5 or (epinephrine_rate > 0 and epinephrine_rate <= 0.1) or (norepinephrine_rate > 0 and norepinephrine_rate <= 0.1):
        return 3
    if dopamine_rate > 0:
        return 2
    if mean_arterial_pressure is not None and not pd.isna(mean_arterial_pressure):
        if mean_arterial_pressure < 70:
            return 1
    return 0


def score_cns_sofa(gcs_total: Optional[float]) -> int:
    """Calculates SOFA Central Nervous System score based on Glasgow Coma Scale."""
    if gcs_total is None or pd.isna(gcs_total):
        return 0
    if gcs_total < 6:
        return 4
    if gcs_total <= 9:
        return 3
    if gcs_total <= 12:
        return 2
    if gcs_total <= 14:
        return 1
    return 0


def score_renal_sofa(creatinine: Optional[float], urine_output_24h: Optional[float] = None) -> int:
    """Calculates SOFA Renal score based on Creatinine (mg/dL) or 24h Urine Output (mL)."""
    score = 0
    if creatinine is not None and not pd.isna(creatinine):
        if creatinine >= 5.0:
            score = max(score, 4)
        elif creatinine >= 3.5:
            score = max(score, 3)
        elif creatinine >= 2.0:
            score = max(score, 2)
        elif creatinine >= 1.2:
            score = max(score, 1)

    if urine_output_24h is not None and not pd.isna(urine_output_24h):
        if urine_output_24h < 200:
            score = max(score, 4)
        elif urine_output_24h < 500:
            score = max(score, 3)

    return score


def calculate_total_sofa(
    pao2_fio2: Optional[float] = None,
    ventilated: bool = False,
    platelets: Optional[float] = None,
    bilirubin: Optional[float] = None,
    map_val: Optional[float] = None,
    dopamine: float = 0.0,
    epinephrine: float = 0.0,
    norepinephrine: float = 0.0,
    gcs: Optional[float] = None,
    creatinine: Optional[float] = None,
    urine_output: Optional[float] = None,
) -> int:
    """Computes total SOFA score sum across all 6 organ systems."""
    return (
        score_respiration_sofa(pao2_fio2, ventilated)
        + score_coagulation_sofa(platelets)
        + score_liver_sofa(bilirubin)
        + score_cardiovascular_sofa(map_val, dopamine, epinephrine, norepinephrine)
        + score_cns_sofa(gcs)
        + score_renal_sofa(creatinine, urine_output)
    )
