#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
OUT="$ROOT/data/interim/full_cohort"
export PYTHONUNBUFFERED=1
if [[ ${1:-} == --help ]]; then
  echo "Usage: [PYTHON=/path/to/python] [TOKENIZER=name-or-path] ./run_full_pipeline.sh"
  echo "Runs Steps 1–4 with --full into data/interim/full_cohort/. Existing full outputs are overwritten."
  exit 0
fi
if (( $# )); then echo "Unexpected arguments; use --help." >&2; exit 2; fi
TOKENIZER="${TOKENIZER:-$("$PYTHON" -c 'import config; print(config.BASE_LVLM_NAME)')}"
trap 'echo "Pipeline failed at line $LINENO. Logs: $OUT/logs" >&2' ERR
"$PYTHON" - <<'PY'
import importlib.util
import config
modules = ["duckdb", "pandas", "pyarrow", "torch", "transformers", "datasets", "networkx", "scipy"]
missing = [m for m in modules if importlib.util.find_spec(m) is None]
if missing: raise SystemExit("Install requirements.txt; missing: " + ", ".join(missing))
for path in [config.MIMIC_IV_HOSP_DIR, config.MIMIC_IV_ICU_DIR, config.PRIMEKG_CSV]:
    if not path.exists(): raise SystemExit(f"Required input missing: {path}")
for name in ["metadata", "chexpert", "negbio"]:
    p = config.MIMIC_CXR_DIR / f"mimic-cxr-2.0.0-{name}.csv"
    if not p.is_file() and not p.with_suffix(".csv.gz").is_file():
        raise SystemExit(f"Required CXR table missing: {p.name} (or .gz)")
PY
mkdir -p "$OUT/logs"
run_step() {
  local name="$1"
  shift
  echo "Starting $name (full cohort)"
  "$PYTHON" "$@" 2>&1 | tee "$OUT/logs/$name.log"
}
run_step step1 src/step1_windowing.py --full --output "$OUT/patient_windows.parquet"
# Metadata alone is insufficient: verify actual cohort JPEGs before proceeding.
"$PYTHON" - "$OUT/patient_windows.parquet" <<'PY'
import sys
from pathlib import Path
import pandas as pd
import config
paths = pd.read_parquet(sys.argv[1], columns=["cxr_path"]).cxr_path.dropna().unique()
missing = sum(not (config.MIMIC_CXR_DIR / Path(str(p))).is_file() for p in paths)
if missing:
    raise SystemExit(f"{missing} of {len(paths)} CXR images missing. Place cohort JPEGs under data/raw/mimic_cxr/ and rerun.")
print(f"Verified {len(paths)} unique image files.")
PY
run_step step2 src/step2_extraction.py --full --input "$OUT/patient_windows.parquet" --output "$OUT/temporal_quadruples.json"
run_step step3 src/step3_graph_rag.py --full --input "$OUT/temporal_quadruples.json" --primekg "$ROOT/data/raw/primekg/kg.csv" --output "$OUT/subgraph_trajectories.pkl"
run_step step4 src/step4_serialization.py --full --windows "$OUT/patient_windows.parquet" --trajectories "$OUT/subgraph_trajectories.pkl" --cxr_dir "$ROOT/data/raw/mimic_cxr" --tokenizer "$TOKENIZER" --output "$OUT/tokenized_dataset"
# Step 4 can fall back to untokenized output; reject that for Step 5.
"$PYTHON" - "$OUT/tokenized_dataset" <<'PY'
import sys
from datasets import load_from_disk
splits = load_from_disk(sys.argv[1])
for name in ["train", "validation", "test"]:
    if name not in splits or not len(splits[name]):
        raise SystemExit(f"Step 5 readiness failed: empty or missing {name} split.")
    if not {"input_ids", "attention_mask", "labels"}.issubset(splits[name].column_names):
        raise SystemExit("Tokenization missing. Check tokenizer access and Step 4 log.")
    print(f"{name}: {len(splits[name])} tokenized examples")
PY
echo "Steps 1–4 complete. Step 5 dataset: $OUT/tokenized_dataset"
echo "Pass this path explicitly to Step 5; config defaults use data/interim/tokenized_dataset."
echo "Validation reports and logs: $OUT"
