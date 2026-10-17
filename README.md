# Multimodal Temporal Graph-RAG for Early Sepsis Detection

A multimodal AI framework combining Intensive Care Unit (ICU) Electronic Health Records (EHR), chest radiographs, and biomedical knowledge graphs to predict prospective **Sepsis-3 onset up to 24 hours in advance**.

---

## 1. Overview

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. Because clinical trajectories in the ICU evolve continuously, static models and flat knowledge graphs fail to capture disease sequencing, acute physiological shifts, and temporal validity.

This project implements a **Multimodal Temporal Graph-RAG (T-RAG)** system:
1. **Temporal Slicing:** Discretizes the 24-hour observation window preceding sepsis onset into 4 discrete 6-hour rolling slices ($T_{-24h \to -18h}$, $T_{-18h \to -12h}$, $T_{-12h \to -6h}$, $T_{-6h \to T_0}$).
2. **Clinical Entity Linking:** Maps radiological findings (MIMIC-CXR-JPG) and physiological/biochemical derangements (MIMIC-IV) to standardized disease and phenotype concepts via SapBERT.
3. **Trajectory Subgraph Retrieval:** Extracts evolving multi-hop pathophysiological pathways from the **PrimeKG** knowledge graph independently for each temporal window.
4. **Chronological Prompt Serialization:** Serializes multimodal trajectories (vitals, labs, image tokens, and linearized knowledge graph triples) into structured clinical timelines.
5. **Time-Respecting Fine-Tuning:** Employs parameter-efficient fine-tuning (QLoRA 4-bit) on an open-weight Vision-Language Model (`Qwen2-VL-7B-Instruct`) to predict Sepsis-3 probability, infection phenotype, and grounded clinical rationales.

---

## 2. Pipeline Architecture

```text
[MIMIC-IV EHR]                                      [MIMIC-CXR-JPG]
 (chartevents, labevents, microbiology, meds)        (Frontal radiographs)
       │                                                   │
       ▼                                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 1: Temporal Windowing & Alignment (src/step1_windowing.py)        │
│ - Cohort selection (Sepsis-3 consensus) & matched control groups       │
│ - 4 rolling 6h windows (T-24h to T0) with aggregated vitals and labs   │
│ - Output: data/interim/patient_windows.parquet                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 2: Feature Extraction & Entity Linking (src/step2_extraction.py) │
│ - TorchXRayVision (DenseNet-121) for pulmonary finding extraction      │
│ - Numerical lab discretizer for critical severity labels               │
│ - SapBERT embedding matching against PrimeKG concepts                  │
│ - Output: data/interim/temporal_quadruples.json                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 3: Trajectory Subgraph Retrieval (src/step3_graph_rag.py)        │
│ - PrimeKG multi-hop pathfinding (k-hop neighbor & shortest paths)      │
│ - Isolated per-window subgraphs preserved across time slices           │
│ - Output: data/interim/subgraph_trajectories.pkl                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 4: Chronological Prompt Serialization (src/step4_serialization.py)│
│ - Linearized semantic triples formatted by timestamp interval          │
│ - Interleaved image visual tokens and structured timeline formatting   │
│ - Output: data/interim/tokenized_dataset/                              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 5: Fine-Tuning & Evaluation (src/step5_training.py & evaluate.py) │
│ - Qwen2-VL-7B QLoRA 4-bit instruction tuning                           │
│ - Structured outputs: sepsis_probability, phenotype, reasoning_chain   │
│ - Evaluation: AUROC, AUPRC, Graph Path Adherence (T-RAG vs Static-RAG) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```text
multimodal-ai-for-sepsis/
├── config.py                 # Single source of truth for paths, thresholds & parameters
├── requirements.txt          # Python environment dependencies
├── README.md                 # Project documentation
├── data/
│   ├── raw/                  # Raw datasets (mimic_iv/, mimic_cxr/, primekg/)
│   └── interim/              # Pipeline stage outputs (.parquet, .json, .pkl)
├── src/
│   ├── utils.py              # Shared helpers (logging, I/O, SOFA tables)
│   ├── step1_windowing.py    # Sepsis-3 cohorting & 6h temporal slicing
│   ├── step2_extraction.py   # Vision encoder + lab discretizer + SapBERT linking
│   ├── step3_graph_rag.py    # PrimeKG loader + per-window subgraph traversal
│   ├── step4_serialization.py# Chronological prompt formatter & HF dataset builder
│   ├── step5_training.py     # Qwen2-VL-7B QLoRA training pipeline
│   └── evaluate.py           # Evaluation suite & benchmark baselines
└── tests/
    ├── test_windowing.py     # SOFA calculation & window boundary tests
    ├── test_entity_linking.py# SapBERT mapping tests
    └── test_graph_rag.py     # Subgraph retrieval tests
```

---

## 4. Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU with $\ge$ 24 GB VRAM (RTX 3090/4090, A10G, A100) or Google Colab Pro+
- CUDA 12.1+

### Installation

```bash
# Clone the repository
git clone https://github.com/rishabhbhattad/multimodal-ai-for-sepsis.git
cd multimodal-ai-for-sepsis

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Data Requirements

This project relies on three datasets:

1. **MIMIC-IV (v2.2+)**: Clinical EHR data from ICU stays (demographics, chartevents, labevents, microbiology, prescriptions). Requires credentialed access on [PhysioNet](https://physionet.org/).
2. **MIMIC-CXR-JPG (v2.0.0)**: Chest radiograph images in JPEG format with study metadata. Requires credentialed access on [PhysioNet](https://physionet.org/).
3. **PrimeKG**: Precision medicine knowledge graph linking diseases, drugs, phenotypes, and pathways. Available openly from the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXACMR).

Place raw files inside `data/raw/` according to the structure specified in `config.py`.

---

## 6. Running the Pipeline

Each step is executable independently from the project root:

```bash
# Step 1: Temporal Windowing (run 10-patient smoke test first)
python src/step1_windowing.py --sample_size 10

# Step 2: Feature Extraction & Entity Linking
python src/step2_extraction.py --sample_size 10

# Step 3: Trajectory Subgraph Retrieval
python src/step3_graph_rag.py --sample_size 10

# Step 4: Prompt Serialization & Dataset Preparation
python src/step4_serialization.py --sample_size 10

# Step 5: QLoRA Fine-Tuning
python src/step5_training.py

# Evaluation & Baselines
python src/evaluate.py
```

---

## 7. License

This repository is licensed under the Apache 2.0 License. Refer to `LICENSE` for details.