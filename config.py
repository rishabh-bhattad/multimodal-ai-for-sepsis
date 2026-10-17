"""
Configuration for data paths, exploratory checks, and active pipeline settings.
Additional parameters for future steps are maintained in config_reference.py.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
MIMIC_IV_DIR = DATA_RAW / "mimic_iv"
MIMIC_CXR_DIR = DATA_RAW / "mimic_cxr"
PRIMEKG_DIR = DATA_RAW / "primekg"
PRIMEKG_CSV = PRIMEKG_DIR / "kg.csv"

DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_INTERIM.mkdir(parents=True, exist_ok=True)

# Experiment tracking
WANDB_PROJECT = "sepsis-temporal-graph-rag"
WANDB_ENTITY = "team-ark"

# PrimeKG exploration settings
PRIMEKG_ALLOWED_NODE_TYPES = ["disease", "effect/phenotype"]
PRIMEKG_RELATION_WHITELIST = [
    "disease_phenotype_positive",
    "disease_disease",
    "phenotype_phenotype",
    "indication",
    "contraindication",
    "disease_phenotype_negative",
]

# MIMIC-IV item IDs for initial data inspection
VITAL_ITEM_IDS = {
    "heart_rate": [220045],
    "map": [220052, 220181, 225312],
    "spo2": [220277],
    "temperature": [223761, 223762],
    "resp_rate": [220210, 224690],
}

LAB_ITEM_IDS = {
    "lactate": [50813, 52442, 53154],
    "platelets": [51265],
    "creatinine": [50912],
    "bilirubin": [50885],
    "wbc": [51300, 51301],
}
