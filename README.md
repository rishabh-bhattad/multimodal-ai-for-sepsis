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
│   ├── step3_graph_rag.py    # Bounded retrieval of four independent window graphs
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

SapBERT runs locally for biomedical entity linking. On the first run, Step 2
downloads the pinned public model and tokenizer from Hugging Face if they are
not cached. Later runs prefer the local cache. Set `SAPBERT_OFFLINE=1` to
require cached files; a missing cache then produces an explicit error.
`HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are also respected. Patient data
is processed locally.

Step 2 automatically uses CUDA when available and otherwise uses CPU. Set
`SAPBERT_DEVICE=cpu` to force CPU execution. Embedding caches remain CPU tensors
for portability. Current resource defaults target the 32-CPU, 115-GiB RAM host:
16 DuckDB threads, a 64-GB DuckDB memory limit, up to 200 GB of temporary disk,
and 8 SapBERT CPU threads. These limits do not reserve resources or cap total
Python process memory.

To run Steps 1–4 on the full eligible cohort, activate the project environment
and run `./run_full_pipeline.sh`. Outputs and logs go to
`data/interim/full_cohort/`; existing full-run outputs are overwritten. The
script requires actual cohort CXR JPEGs, checks tokenization, and stops on errors.
Pass `data/interim/full_cohort/tokenized_dataset` explicitly to Step 5 rather
than using its default dataset path. Set `PYTHON=./capstone/bin/python` if you
want to select the project environment without activating it.

Step 3 retrieves clinical PrimeKG context independently for each of the four
observation windows. Native directed relations are preserved, including
explicitly negative associations. Each window is capped at 200 edges, with
connecting paths prioritized over one-hop neighbors. Empty windows and
unresolved findings remain explicit.

## Usage

With the environment active and Step 1 window data prepared, run a ten-patient extraction check from the repository root:

```bash
python src/step2_extraction.py \
  --input data/interim/full_cohort/patient_windows.parquet \
  --sample_size 10 \
  --output data/interim/step2_smoke/temporal_quadruples.json
```

Use `--full` instead of `--sample_size 10` for full processing after a successful check. Patient data must remain on local or institutional hardware.

### Step 3: temporal subgraph retrieval

With Step 2 output available, run a ten-subject retrieval check:

```bash
python src/step3_graph_rag.py \
  --input data/interim/full_cohort/temporal_quadruples.json \
  --primekg data/raw/primekg/kg.csv \
  --sample_size 10 \
  --output data/interim/step3_smoke/subgraph_trajectories.pkl
```

Sampling selects ten unique subjects with one stay each and includes image and
no-image cases when available. It preserves existing splits and fails if fewer
than ten subjects are available. Omitting the sampling flags uses the configured
ten-subject default. Full retrieval requires the explicit `--full` flag:

```bash
python src/step3_graph_rag.py \
  --input data/interim/full_cohort/temporal_quadruples.json \
  --primekg data/raw/primekg/kg.csv \
  --full \
  --output data/interim/full_cohort/subgraph_trajectories.pkl
```

The pickle is keyed by native stay-ID strings. Each stay retains `subject_id`,
`split`, and a `windows` dictionary containing four separate NetworkX
`MultiDiGraph` objects (`window_0` through `window_3`). Each graph retains its
window timestamps, original findings, separate CXR coverage, retrieval counts,
and graph provenance. Node keys are `(node_type, native_node_id_string)`;
edge keys retain the native relation. Retrieved diseases are biomedical
context, not confirmed patient diagnoses.

`step3_validation.json` is written beside the selected pickle. It contains
aggregate coverage, mapping/retrieval counts, pruning totals, active limits,
and input/graph/pickle checksums. Keep smoke and full outputs in separate
directories so their reports remain separate. Patient records stay in local,
gitignored artifacts; console diagnostics contain aggregate results only.

Run the regression suite from the repository root with:

```bash
python -m pytest tests/ -q
```

November 6, 2026 checkpoint: all 128 tests passed, including Step 2-to-Step 3
integration and retrieval consistency across independent processes. The real
ten-subject check produced 40 independent graphs, preserving 22 empty-finding
windows and the existing splits. It retained 1,504 edges; three windows reached
the 200-edge cap, with no connecting-path edges pruned. A separate saved-output
audit verified checksums and confirmed that 42 protected source/input files
were unchanged. Reports are local under `data/interim/step3_smoke/`.

These are functional checks, not clinical performance or SapBERT mapping
accuracy results. The Step 2 mapping cutoff remains provisional. Full-cohort
Step 3 retrieval has not been run.
