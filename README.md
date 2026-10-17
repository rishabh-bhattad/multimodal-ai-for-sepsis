# Multimodal Temporal Graph-RAG for Early Sepsis Detection

A multimodal research project combining ICU electronic health records, chest radiographs, and biomedical knowledge graphs to predict Sepsis-3 onset up to 24 hours in advance.

---

## Overview

Clinical trajectories in the ICU are dynamic. This project investigates whether retrieving time-sliced, evolving subgraphs from a biomedical knowledge graph (**PrimeKG**) and integrating them with windowed vitals, labs, and chest radiographs improves early Sepsis-3 prediction and reasoning in vision-language models.

### Datasets
* **MIMIC-IV (v2.2+)**: ICU vitals, laboratory tests, microbiology cultures, and antibiotic prescriptions (requires credentialed access via PhysioNet).
* **MIMIC-CXR-JPG (v2.0.0)**: Frontal chest radiographs and associated study metadata (requires credentialed access via PhysioNet).
* **PrimeKG**: Precision medicine knowledge graph linking diseases, phenotypes, and biological pathways (Harvard Dataverse).

---

## Repository Structure

```text
multimodal-ai-for-sepsis/
├── config.py                 # Active project configuration and paths
├── requirements.txt          # Python dependencies
├── README.md
├── data/
│   ├── raw/                  # Raw datasets (mimic_iv/, mimic_cxr/, primekg/)
│   └── interim/              # Cleaned and intermediate artifacts
├── notebooks/
│   ├── 01_mimic_eda.ipynb    # MIMIC-IV vitals, labs, and CXR metadata inspection
│   └── 02_primekg_eda.ipynb  # PrimeKG graph exploration and relation checks
└── src/
    └── utils.py              # Shared helpers (logging, safe I/O, SOFA scoring tables)
```

---

## Setup

### 1. Environment

Python 3.10 or 3.11 is recommended:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Placement

Place downloaded datasets into `data/raw/` (gitignored):

* `data/raw/primekg/kg.csv` — PrimeKG knowledge graph.
* `data/raw/mimic_iv/` — MIMIC-IV tables (`icustays`, `chartevents`, `labevents`, `microbiologyevents`, `prescriptions`).
* `data/raw/mimic_cxr/` — MIMIC-CXR metadata and frontal JPEG images.

### 3. Experiment Tracking

Experiments and metrics are tracked collaboratively on Weights & Biases under team `team-ark`:

```bash
wandb login
```

---

## Current Status

We are currently in the initial data exploration and setup phase:
* **PrimeKG:** Ingested and verified (`data/raw/primekg/`). Exploratory graph analysis in `notebooks/02_primekg_eda.ipynb`.
* **MIMIC Data:** Initial inspection notebook ready in `notebooks/01_mimic_eda.ipynb` (awaiting PhysioNet data approval).