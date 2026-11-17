# Multimodal Temporal Graph-RAG for Retrospective Sepsis Detection

**Estimating sepsis probability from 24-hour multimodal observations in a retrospective Sepsis-3 cohort.**

This research study combines ICU electronic health records, chest radiographs, and biomedical knowledge graphs from the 24 hours preceding a retrospectively selected assessment anchor (`t0`).

The planned model output is a model-derived sepsis classification score and a short evidence summary. The score is not established as a calibrated clinical probability, and generated summaries require evidence-grounding review.

---

## Overview

This project investigates whether preserving the temporal evolution of patient findings and retrieving window-specific biomedical knowledge from **PrimeKG** improves sepsis probability estimation in a retrospective Sepsis-3 cohort compared with static or no knowledge-graph retrieval. ICU vitals, laboratory measurements, and available chest radiographs are represented across four consecutive six-hour windows.

### Study scope and interpretation

The observation interval is `[t0 - 24 hours, t0)`. Positive assessment anchors
come from the MIT-LCP Sepsis-3 derivation; control anchors use suspected infection
or the maximum eligible SOFA score, depending on the control cohort. Anchor and
control selection use retrospective stay/admission information. The task has no
specified future prediction horizon and does not establish how many hours before
sepsis onset an alert could be issued.

The primary research question is whether temporal KG retrieval improves
sepsis probability estimation on held-out patients within this retrospectively
constructed cohort. Patient-level training/validation/test separation supports that internal
comparison; it does not establish prospective or external generalization.

This study does not demonstrate a validated real-time early-warning system or
readiness for clinical deployment. Before making prospective claims, the protocol
would need prediction-time-available eligibility/anchors, an explicit future
outcome horizon, and verification that every input was available at prediction
time. An audit of event versus result/documentation availability timestamps
remains outstanding. These limitations also need to accompany retrospective
results; changing the wording does not resolve them.

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

## Step 5 training runner and experiment tracking

Activate the project environment. Authenticate locally with `wandb login` before
online runs; never put the API key in source, commands shared with others, or chat.
Runs use entity `team-ark`, project `sepsis-temporal-graph-rag`, and group
`step5-comparison` by default. The variant is recorded as the W&B job type,
tag, and run configuration; automatic names also include model, context limit,
and smoke/full mode. Override these using the corresponding CLI options.

First check the CLI:

```bash
python src/step5_run_training.py --help
```

A smoke run uses up to ten subjects from each training/validation split and
two optimizer steps, followed by validation. It is a wiring/memory check, not a
representative performance estimate. It prefers image and no-image cases when
available but is not a worst-case memory benchmark.

```bash
python src/step5_run_training.py --smoke \
  --output checkpoints/trag_8b_8k_smoke \
  --max-seq-length 8192 --wandb-mode online
```

The default context policy preserves complete saved prompt text and actual image
tokens. If an example does not fit, training stops instead of silently dropping
context. Increasing the limit does not guarantee every example fits. The runner
reprocesses `prompt_text` and images rather than using Step 4's already budgeted
token IDs. This cannot restore text removed before saving a dataset.
`--allow-kg-reduction` explicitly enables the existing KG fitting policy;
use that only as a deliberate experiment choice. The default remains 4096 unless
`--max-seq-length` is specified. Neither 8K GPU fit nor real-model training has
been verified by the runner's synthetic tests.

After a successful smoke test and context/memory assessment, a full run is:

```bash
python src/step5_run_training.py --full \
  --output checkpoints/trag_8b_8k_full \
  --max-seq-length 8192 --wandb-mode online
```

The runner currently supports Qwen3-VL on one visible CUDA GPU, with NF4 QLoRA,
BF16 when supported, batch size 1, gradient accumulation, gradient checkpointing,
SDPA, and language-only adapters. Qwen3.5/3.8 are rejected until their hybrid
adapter coverage is verified. An optional `--max-image-pixels` changes the image
resolution ceiling and must be held constant across comparisons.

Use `--variant static_rag` or `--variant no_kg` with an explicitly prepared
matching `--dataset`. The runner does not build ablation datasets automatically.
All variants must use the same cohort and patient splits.

```bash
python src/step5_run_training.py --variant xgboost --full \
  --windows data/interim/full_cohort/patient_windows.parquet \
  --output data/interim/step5_results/xgboost_run1 --wandb-mode online
```

XGBoost uses CPU and logs aggregate validation/count metrics. Neural runs log
loss, validation loss, learning rate, gradient norm, epoch, runtime, and peak
GPU memory. Neither path uploads patient records, prompts, images, predictions,
source code, or checkpoint artifacts. W&B logging uses an allowlisted callback,
not the default Trainer integration. `--wandb-mode offline` (default) stores
tracking locally; `disabled` disables it.

Checkpoints, the final adapter/processor, run configuration, and training summary
stay under `--output`. Select a new output directory for each experiment.
To resume use `--resume /path/to/checkpoint-N` with matching model/context
settings; the checkpoint must include Trainer state. W&B starts a new tracking
run for each invocation, including a resumed training invocation.
The test split is not used during training or validation. Held-out prediction,
comparison metrics, and evidence auditing remain separate Step 5 operations.

## Shell launchers and stage boundaries

Use `run_full_pipeline.sh` for preprocessing and `run_training.sh` for
experiments. Both use `capstone/bin/python` automatically; set `PYTHON` to
override the interpreter. Both change to the repository root, so relative paths
are interpreted from there.

Recommended first-time sequence:

```bash
# Stable, reusable cohort/findings/graph artifacts.
./run_full_pipeline.sh --from 1 --to 3

# Download images referenced by the cohort; enter your S3 prefix and region.
./capstone/bin/python src/utils.py \
  --input data/interim/full_cohort/patient_windows.parquet --all --workers 8

# Model-specific serialization and tokenization.
./run_full_pipeline.sh --from 4 --to 4
```

Steps 2–3 use CXR tables, not JPEGs, so image downloading can follow Step 3.
For earlier image access, run `--from 1 --to 1`, download images, then
`--from 2 --to 3`. Running the preprocessing launcher without options retains
the original Steps 1–4 behavior; it stops before Step 4 if images are missing.
Selected steps overwrite their artifacts and logs; prerequisites for skipped
steps must already exist.

When changing the tokenizer or serialization length, rerun only Step 4:

```bash
MAX_SEQ_LENGTH=8192 ./run_full_pipeline.sh --from 4 --to 4
```

This prepares an 8K dataset but does not establish GPU training fit. It also
does not change the configured model or the training runner's length default.
A model change may require training-code compatibility work as well.

Authenticate once with `./capstone/bin/wandb login`. Then launch an experiment:

```bash
./run_training.sh --smoke --max-seq-length 8192 \
  --output checkpoints/trag_8b_8k_smoke --wandb-mode online
```

The shell launcher checks W&B authentication before loading weights for online
runs. Offline and disabled modes do not contact W&B during this preflight.
Full training requires `--full`; the default is smoke mode. Every training
argument is forwarded to the Python runner, including `--variant`, `--dataset`,
`--resume`, and W&B naming/group options. Use `./run_training.sh --help`.
Training never reruns preprocessing or downloads cohort images.

## Master experiment suite

`run_training.sh --suite` orchestrates dataset preparation, sequential training,
held-out prediction, and aggregate comparison. It reuses Steps 1–3 and builds
fresh raw prompt datasets directly from their artifacts. Model-specific
processing happens in the training/prediction collator, so the suite does not
reuse truncated Step 4 token IDs.

Edit `VARIANTS=(trag static_rag no_kg)` near the top of `run_training.sh`,
or select runs explicitly:

```bash
# Plan only: no data/model loading, file creation, or W&B requests.
./run_training.sh --suite --output data/interim/step5_results/suite_8b_8k \
  --variants trag static_rag no_kg --max-seq-length 8192 --dry-run

# Smoke: ten subjects per split, two optimizer steps per neural variant.
./run_training.sh --suite --output data/interim/step5_results/suite_8b_8k_smoke \
  --variants trag static_rag no_kg --max-seq-length 8192 --wandb-mode online

# Full experiment: use a new directory.
./run_training.sh --suite --output data/interim/step5_results/suite_8b_8k_full \
  --variants trag static_rag no_kg --max-seq-length 8192 --full --wandb-mode online
```

Smoke comparisons are integration checks, not research performance estimates.
A smoke selection is not a worst-case GPU memory benchmark. The context limit
must fit complete prompts including image tokens, or the suite stops;
`--allow-kg-reduction` is an explicit experimental choice.

The suite defaults to excluding XGBoost. Add `xgboost` to `--variants` to train
it once. Since it uses structured measurements rather than Qwen, it need not
be retrained when only the Qwen model changes. Reuse its predictions with:

```bash
./run_training.sh --suite --output data/interim/step5_results/new_suite \
  --variants trag static_rag no_kg --full \
  --xgboost-predictions /absolute/path/to/xgboost/predictions.parquet
```

Reused predictions must come from the same cohort and split definition; regenerate
them if cohort construction, structured features, or baseline settings changed.
The suite checks held-out stay/subject/label alignment, but cannot infer the
training provenance of an arbitrary cached prediction file.

Artifacts under the suite directory:

- `datasets/VARIANT/`: identical cohorts with the chosen graph treatment.
- `runs/VARIANT/`: local adapters, training metrics, and `predictions.parquet`.
- `comparison/model_comparison.csv` and `comparison/step5_validation.json`:
  metrics and subject-bootstrap intervals for the selected variants.
- `suite_config.json`: selected settings and cohort/graph checksums.

Each variant has a separate W&B run in one group, and the comparison has its
own aggregate-only run. Failed stages stop the suite. Existing nonempty suite
directories are rejected; individual training runs can still use the training
runner's checkpoint-resume support. The suite itself does not resume automatically.

Use `--rationales` to also save generated test rationales locally; otherwise
only probabilities are produced. Reviewer-assisted evidence auditing remains a
separate action because it needs annotations. The default threshold is 0.5 and
is not tuned on test data. Test predictions are generated only after each
variant finishes training and validation.

`--model`, `--revision`, and `--max-image-pixels` are shared across selected
neural variants. Pin a model revision for reproducible comparisons. The runner
currently supports Qwen3-VL; Qwen3.5/3.8 still need model/adapter compatibility
work before using the same orchestration. Model changes do not require rerunning
Steps 1–3 if the cohort/retrieval policy is unchanged.
