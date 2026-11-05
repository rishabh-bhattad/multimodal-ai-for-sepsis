# Multimodal Temporal Graph-RAG for Sepsis Detection

A multimodal research project combining ICU electronic health records, chest radiographs, and biomedical knowledge graphs to estimate sepsis probability from 24 hours of observations.

The planned model output is a sepsis probability and a short, evidence-based clinical rationale.

---

## Overview

This project investigates whether preserving the temporal evolution of patient findings and retrieving window-specific biomedical knowledge from **PrimeKG** improves sepsis probability estimation compared with static or no knowledge-graph retrieval. It combines ICU vitals, laboratory measurements, and chest radiographs across four six-hour windows, supporting research toward early sepsis detection.

### Datasets

* **MIMIC-IV (v2.2+)**: ICU vitals, laboratory tests, microbiology cultures, and antibiotic prescriptions (requires credentialed access via PhysioNet).
* **MIMIC-CXR-JPG (v2.1.0)**: Frontal chest radiographs and associated study metadata, both sourced from the v2.1.0 release (requires credentialed access via PhysioNet).
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
├── src/
│   ├── step1_windowing.py    # MIT-LCP cohort derivation and four six-hour windows
│   ├── step2_extraction.py   # Latest findings, CXR labels, and local SapBERT linking
│   ├── sql/                 # Cohort queries and MIT-LCP clinical derivations
│   └── utils.py             # Shared logging and image-download helpers
└── tests/                   # Pipeline tests
```

---

## Setup

### 1. Environment

Use Python 3.11. Create and activate a virtual environment with:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Placement

Download the required datasets separately from their providers and place them in `data/raw/` (gitignored). MIMIC datasets require approved PhysioNet access.

* `data/raw/primekg/kg.csv` — PrimeKG knowledge graph.
* `data/raw/mimic_iv/` — Compressed MIMIC-IV tables under `hosp/` and `icu/`.
* `data/raw/mimic_cxr/` — MIMIC-CXR metadata, CheXpert/NegBio labels, and frontal JPEG images.

SapBERT runs locally for biomedical entity linking.

## Usage

With the environment active and Step 1 window data prepared, run a ten-patient extraction check from the repository root:

```bash
python src/step2_extraction.py \
  --input data/interim/full_cohort/patient_windows.parquet \
  --sample_size 10 \
  --output data/interim/step2_smoke/temporal_quadruples.json
```

Use `--full` instead of `--sample_size 10` for full processing after a successful check. Patient data must remain on local or institutional hardware.
