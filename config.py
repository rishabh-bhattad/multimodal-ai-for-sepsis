"""
Configuration for data paths, exploratory checks, and active pipeline settings.
Additional parameters for future steps are maintained in config_reference.py.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
MIMIC_IV_DIR = DATA_RAW / "mimic_iv"
MIMIC_IV_HOSP_DIR = MIMIC_IV_DIR / "hosp"
MIMIC_IV_ICU_DIR = MIMIC_IV_DIR / "icu"
MIMIC_CXR_DIR = DATA_RAW / "mimic_cxr"
PRIMEKG_DIR = DATA_RAW / "primekg"
PRIMEKG_CSV = PRIMEKG_DIR / "kg.csv"

# Cohort CXR downloads; window paths are relative to the remote files/ folder.
CXR_DOWNLOAD_BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files"
CXR_DOWNLOAD_COMMAND = "wget"
CXR_DOWNLOAD_TIMEOUT_SECONDS = (10, 60)
CXR_DOWNLOAD_TOTAL_TIMEOUT_SECONDS = 120
CXR_DOWNLOAD_SAMPLE_SIZE = 1

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

# Step 2: Feature extraction & entity linking
PATIENT_WINDOWS_PARQUET = DATA_INTERIM / "patient_windows.parquet"
TEMPORAL_QUADRUPLES_JSON = DATA_INTERIM / "temporal_quadruples.json"
PRIMEKG_CLINICAL_EMBEDDINGS = DATA_INTERIM / "primekg_clinical_embeddings.pt"

TARGET_PATHOLOGIES = [
    "Consolidation",
    "Pleural Effusion",
    "Atelectasis",
    "Pneumonia",
    "Lung Opacity",
    "Edema",
    "Cardiomegaly",
]

LAB_SEVERITY_THRESHOLDS = {
    "lactate": [
        (4.0, ">", "Lactic Acidosis"),
        (2.0, ">", "Hyperlactatemia"),
    ],
    "map": [
        (65.0, "<", "Systemic Hypotension"),
    ],
    "platelets": [
        (100.0, "<", "Thrombocytopenia"),
    ],
    "creatinine": [
        (1.2, ">", "Acute Kidney Injury"),
    ],
    "wbc": [
        (12.0, ">", "Leukocytosis"),
        (4.0, "<", "Leukopenia"),
    ],
    "temp": [
        (38.3, ">", "Pyrexia"),
        (36.0, "<", "Hypothermia"),
    ],
}

SAPBERT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
SAPBERT_SIMILARITY_THRESHOLD = 0.70

# Vision-Language Model backbone (Step 5)
BASE_LVLM_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# local MIT-LCP derivations and 24-hour observations
MIT_LCP_REVISION = "303d26c623dcc9c49cc0f204468d4acc2f063797"
STEP1_SQL_DIR = PROJECT_ROOT / "src" / "sql"
MIT_LCP_SQL_DIR = STEP1_SQL_DIR / "mit_lcp"
MIT_LCP_CONCEPTS = [
    "icustay_times.sql",
    "icustay_hourly.sql",
    "vitalsign.sql",
    "gcs.sql",
    "bg.sql",
    "enzyme.sql",
    "chemistry.sql",
    "complete_blood_count.sql",
    "ventilator_setting.sql",
    "oxygen_delivery.sql",
    "ventilation.sql",
    "weight_durations.sql",
    "urine_output.sql",
    "urine_output_rate.sql",
    "dobutamine.sql",
    "dopamine.sql",
    "epinephrine.sql",
    "norepinephrine.sql",
    "antibiotic.sql",
    "suspicion_of_infection.sql",
    "sofa.sql",
    "sepsis3.sql",
]
OBSERVATION_HOURS = 24
WINDOW_SIZE_HOURS = 6
NUM_WINDOWS = 4
RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
STEP1_SAMPLE_SIZE = 10
STEP1_CANDIDATE_MULTIPLIER = 20
DUCKDB_THREADS = 4
DUCKDB_MEMORY_LIMIT = "4GB"
COHORT_PARQUET = DATA_INTERIM / "step1_cohort.parquet"
STEP1_VALIDATION_JSON = DATA_INTERIM / "step1_validation.json"
STEP1_CHART_ITEM_IDS = [
    220045, 220050, 220051, 220052, 220179, 220180, 220181, 220210,
    220277, 220339, 220621, 220739, 223761, 223762, 223834, 223835,
    223848, 223849, 223900, 223901, 224639, 224642, 224684, 224685,
    224686, 224687, 224688, 224689, 224690, 224691, 224696, 224700,
    225309, 225310, 225312, 225664, 226512, 226537, 226732, 227287,
    227582, 229314,
]
STEP1_LAB_ITEM_IDS = [
    50801, 50802, 50803, 50804, 50805, 50806, 50807, 50808,
    50809, 50810, 50811, 50813, 50814, 50815, 50816, 50817,
    50818, 50819, 50820, 50821, 50822, 50823, 50824, 50825,
    50861, 50862, 50863, 50867, 50868, 50878, 50882, 50883,
    50884, 50885, 50893, 50902, 50910, 50911, 50912, 50927,
    50930, 50931, 50954, 50971, 50976, 50983, 51006, 51221,
    51222, 51248, 51249, 51250, 51265, 51277, 51279, 51300,
    51301, 52033, 52159, 52442, 53154,
]
STEP1_VASO_ITEM_IDS = [221653, 221662, 221289, 221906]
