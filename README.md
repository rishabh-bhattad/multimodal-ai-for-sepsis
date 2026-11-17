# Multimodal Temporal Graph-RAG for Retrospective Sepsis Detection

This project studies whether temporal knowledge graph retrieval improves
retrospective sepsis classification using ICU records and chest radiographs.
It compares temporal-RAG, static-RAG, and no-KG variants, with XGBoost as a
structured-data baseline.

Each example covers four six-hour windows before a retrospectively selected
assessment time. The cohort uses Sepsis-3 labels and patient-level train,
validation, and test splits. Scores are not established as calibrated clinical
probabilities, and generated evidence summaries require review. This study does
not establish prospective early-warning performance or clinical readiness.

## Overview

The research question is whether biomedical context retrieved separately for
each observation window improves sepsis classification compared with a shared
static graph or no graph context. The pipeline combines vitals, laboratory
measurements, available chest radiographs, and associations from PrimeKG.
SapBERT links clinical findings to graph entities.

| Stage | Purpose |
| --- | --- |
| 1. Cohort and windowing | Derive the retrospective Sepsis-3 cohort and divide the preceding 24 hours into four six-hour windows. |
| 2. Finding extraction | Extract window findings, associate CXR metadata, and link biomedical entities with SapBERT. |
| 3. Graph retrieval | Retrieve a separate directed PrimeKG subgraph for each window. |
| 4. Serialization | Assemble chronological multimodal prompts and prepare model-specific datasets. |
| 5. Experiments | Train variants, predict on held-out patients, and compare performance. |

Retrieved graph associations provide biomedical context; they are not confirmed
patient diagnoses. Assessment anchors use retrospective information, and the task
has no specified future prediction horizon. Event versus result/documentation
availability timestamps still require auditing before prospective claims.

## Experiments

All variants use the same cohort and patient splits. The neural variants retain
the same observations and images while changing graph context:

| Variant | Inputs and graph treatment |
| --- | --- |
| Temporal-RAG (`trag`) | Window-specific findings, images, and separately retrieved graphs. |
| Static-RAG (`static_rag`) | The window graphs are merged, and the bounded shared graph is stored once for all four windows. |
| No-KG (`no_kg`) | The same clinical observations and images, with graph triples omitted. |
| XGBoost (`xgboost`) | A baseline trained on structured measurements. |

Neural training uses Qwen3-VL with NF4 QLoRA and language-layer adapters.
Evaluation reports AUROC, AUPRC (average precision), Brier score, sensitivity,
and specificity, with subject-level bootstrap confidence intervals. Test
predictions follow training and validation; the classification threshold is
0.5 and is not tuned on the test set. Optional generated rationales require
separate evidence review.

## Datasets

- **MIMIC-IV (v2.2+)** supplies ICU measurements, laboratory results, microbiology,
  and antibiotic records for cohort derivation and clinical observations.
- **MIMIC-CXR-JPG (v2.1.0)** supplies frontal chest radiographs, study metadata,
  and CheXpert/NegBio labels from the same release.
- **PrimeKG** supplies biomedical associations between diseases, phenotypes,
  and other biological entities for graph retrieval.

MIMIC datasets require credentialed access through PhysioNet. Download datasets
separately from their providers; they are not included in the repository.

## Repository structure

```text
multimodal-ai-for-sepsis/
├── config.py                    # Paths, model, retrieval, and training settings
├── requirements.txt             # Python dependencies
├── run_full_pipeline.sh         # Preprocessing launcher (Steps 1–4)
├── run_training.sh              # Single-run and experiment-suite launcher
├── data/
│   ├── raw/                     # Source datasets (gitignored)
│   └── interim/                 # Processed artifacts and reports (gitignored)
├── notebooks/
│   ├── 01_mimic_eda.ipynb        # Clinical data and CXR exploration
│   └── 02_primekg_eda.ipynb      # Knowledge graph exploration
├── src/
│   ├── step1_windowing.py        # Cohort derivation and observation windows
│   ├── step2_extraction.py       # Clinical findings and SapBERT entity linking
│   ├── step3_graph_rag.py        # Window-specific graph retrieval
│   ├── step4_serialization.py    # Multimodal prompts and dataset preparation
│   ├── step5_run_training.py     # Individual training runs
│   ├── step5_run_experiments.py  # Variant preparation and experiment orchestration
│   ├── step5_ablations.py        # Static-RAG and no-KG datasets
│   ├── step5_training.py         # Model loading, collators, and inference
│   ├── step5_tabular_baseline.py # XGBoost baseline
│   ├── step5_evaluation.py       # Held-out metrics and comparisons
│   ├── step5_evidence_audit.py   # Evidence review utilities
│   ├── sql/                     # Cohort SQL and MIT-LCP derivations
│   └── utils.py                 # Shared helpers and image downloading
└── tests/                       # Pipeline and training tests
```

## Setup

Use Python 3.11. From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHON="$PWD/venv/bin/python"
```

The shell launchers use `capstone/bin/python` by default. Set `PYTHON` as above
to use the environment you created. Neural training requires a CUDA GPU; the
current runner supports Qwen3-VL with NF4 QLoRA.

Download the datasets from their providers and place them under `data/raw/`:

| Dataset | Location | Contents |
| --- | --- | --- |
| MIMIC-IV v2.2+ | `data/raw/mimic_iv/` | Compressed tables under `hosp/` and `icu/` |
| MIMIC-CXR-JPG v2.1.0 | `data/raw/mimic_cxr/` | Study metadata, CheXpert/NegBio labels, and frontal JPEGs |
| PrimeKG | `data/raw/primekg/kg.csv` | Biomedical knowledge graph |

MIMIC datasets require credentialed PhysioNet access. Keep patient data on local
or institutional hardware and out of version control. Step 2 downloads the
SapBERT model from Hugging Face on first use; subsequent runs use cached files.

## Usage

### Preprocessing

Build the cohort, extract findings, and retrieve window-specific graphs:

```bash
./run_full_pipeline.sh --from 1 --to 3
```

Download the cohort images using your S3 prefix and region when prompted:

```bash
"$PYTHON" src/utils.py \
  --input data/interim/full_cohort/patient_windows.parquet --all --workers 8
```

Build the model-specific dataset:

```bash
./run_full_pipeline.sh --from 4 --to 4
```

Outputs and logs go to `data/interim/full_cohort/`. Selected steps overwrite
existing outputs; prerequisites for skipped steps must already exist. Steps
2–3 use CXR tables, while Step 4 requires the JPEGs.

### Training and comparison

The experiment suite reuses Steps 1–3 to prepare variant datasets, train models,
generate held-out predictions, and compare results. It does not require Step 4.

Preview the suite, then run a smoke check:

```bash
./run_training.sh --suite --dry-run \
  --output data/interim/step5_results/suite_plan

./run_training.sh --suite \
  --output data/interim/step5_results/suite_smoke
```

Smoke mode uses up to ten subjects per training/validation split and two
optimizer steps per neural variant. It checks execution, not model performance
or worst-case GPU memory use.

The default context limit is 8192 tokens, including image tokens. Training stops
if a prompt exceeds it. Some full-cohort graph prompts exceed 8K; increase
`--max-seq-length` or enable `--allow-kg-reduction` to process them. Check GPU
memory with the chosen settings before full training.

After checking context lengths and memory, launch the full comparison with a
new output directory and the chosen context settings:

```bash
./run_training.sh --suite --full \
  --variants trag static_rag no_kg xgboost \
  --output data/interim/step5_results/suite_full
```

The suite defaults to the three neural variants; the command above also trains
XGBoost. Hold model revision, image resolution, and context policy constant
across neural comparisons. Use `--model`, `--revision`, `--max-image-pixels`,
and `--max-seq-length` to select these settings.

For an individual temporal-RAG smoke run using the Step 4 dataset:

```bash
./run_training.sh --smoke \
  --dataset data/interim/full_cohort/tokenized_dataset \
  --output checkpoints/trag_smoke
```

Use `--variant` to select another variant with its matching dataset.
`--resume /path/to/checkpoint-N` resumes an individual training run with matching
model and context settings. The experiment suite does not resume automatically.

W&B logging defaults to offline. For online logging, run `wandb login` locally
and pass `--wandb-mode online`. Entity and project defaults are defined in
`config.py`; override them with `--wandb-entity` and `--wandb-project`.
Logging sends aggregate metrics; patient records, prompts, images, predictions,
and checkpoints remain local.

For all available options:

```bash
./run_full_pipeline.sh --help
./run_training.sh --help
./run_training.sh --suite --help
```

### Outputs

Preprocessing writes cohort windows, extracted findings, graphs, and the Step 4
dataset under `data/interim/full_cohort/`. A suite output directory contains:

| Path | Contents |
| --- | --- |
| `datasets/VARIANT/` | Prepared variant datasets |
| `runs/VARIANT/` | Adapters, training metrics, and held-out `predictions.parquet` |
| `comparison/model_comparison.csv` | Aggregate comparison metrics and confidence intervals |
| `comparison/step5_validation.json` | Evaluation report |
| `suite_config.json` | Experiment settings and input checksums |

Use a new suite directory for each experiment; nonempty output directories are
rejected. Add `--rationales` to save generated test rationales locally.

## Tests

```bash
python -m pytest tests/ -q
```

Compact KG serialization preserves the existing displayed edge evidence. Full-cohort checks reduced 8K overflows to 200 temporal-RAG prompts and zero static-RAG/no-KG prompts. See [compact-context measurements](docs/experiments/compact-context-check.md) for the method, image settings, and remaining limits.
