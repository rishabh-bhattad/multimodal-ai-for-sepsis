#!/usr/bin/env bash
# Full-cohort preprocessing. Select ranges to reuse existing artifacts.
set -Eeuo pipefail
umask 077
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/capstone/bin/python}"
OUT="$ROOT/data/interim/full_cohort"
START=1
END=4
export PYTHONUNBUFFERED=1
usage() {
  cat <<'HELP'
Usage: ./run_full_pipeline.sh [--from 1..4] [--to 1..4]
Default: Steps 1–4, full cohort.
  --from 1 --to 3   Build reusable cohort, findings, and graphs.
  --from 4 --to 4   Rebuild model-specific dataset from existing artifacts.
  --from 1 --to 1   Build patient windows for the image downloader.
Environment:
  PYTHON       Python executable (default: capstone/bin/python).
  TOKENIZER    Tokenizer name/path (default: configured model).
  MAX_SEQ_LENGTH  Optional Step 4 token limit (config default otherwise).
Outputs: data/interim/full_cohort/. Selected steps overwrite their outputs.
Images are required only for Step 4; downloading is a separate action.
HELP
}
while (( $# )); do
  case "$1" in
    --from|--to)
      if (( $# < 2 )); then echo "Missing value for $1" >&2; exit 2; fi
      case "$2" in [1-4]) ;; *) echo "Step must be 1–4" >&2; exit 2 ;; esac
      if [[ $1 == --from ]]; then START="$2"; else END="$2"; fi
      shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if (( START > END )); then echo "--from must not exceed --to" >&2; exit 2; fi
if ! command -v "$PYTHON" >/dev/null; then echo "Python unavailable: set PYTHON to your project interpreter." >&2; exit 1; fi
trap 'echo "Preprocessing failed at line $LINENO. Logs: $OUT/logs" >&2' ERR
mkdir -p "$OUT/logs"
run_step() {
  local name="$1"
  shift
  echo "Starting $name (full cohort)"
  "$PYTHON" "$@" 2>&1 | tee "$OUT/logs/$name.log"
}
if (( START <= 1 && END >= 1 )); then
  run_step step1 src/step1_windowing.py --full --output "$OUT/patient_windows.parquet"
fi
if (( START <= 2 && END >= 2 )); then
  run_step step2 src/step2_extraction.py --full --input "$OUT/patient_windows.parquet" --output "$OUT/temporal_quadruples.json"
fi
if (( START <= 3 && END >= 3 )); then
  run_step step3 src/step3_graph_rag.py --full --input "$OUT/temporal_quadruples.json" --primekg "$ROOT/data/raw/primekg/kg.csv" --output "$OUT/subgraph_trajectories.pkl"
fi
if (( START <= 4 && END >= 4 )); then
  "$PYTHON" - "$OUT/patient_windows.parquet" <<'PY'
import sys
from pathlib import Path
import pandas as pd
import config
paths = pd.read_parquet(sys.argv[1], columns=["cxr_path"]).cxr_path.dropna().unique()
missing = sum(not (config.MIMIC_CXR_DIR / Path(str(p))).is_file() for p in paths)
if missing:
    raise SystemExit(f"{missing} cohort images missing. Run src/utils.py with this windows file before Step 4.")
print(f"Found {len(paths)} unique image files.")
PY
  TOKENIZER="${TOKENIZER:-$("$PYTHON" -c 'import config; print(config.BASE_LVLM_NAME)')}"
  length_args=()
  if [[ -n ${MAX_SEQ_LENGTH:-} ]]; then length_args=(--max_seq_length "$MAX_SEQ_LENGTH"); fi
  run_step step4 src/step4_serialization.py --full --windows "$OUT/patient_windows.parquet" --trajectories "$OUT/subgraph_trajectories.pkl" --cxr_dir "$ROOT/data/raw/mimic_cxr" --tokenizer "$TOKENIZER" --output "$OUT/tokenized_dataset" "${length_args[@]}"
  "$PYTHON" - "$OUT/tokenized_dataset" <<'PY'
import sys
from datasets import load_from_disk
splits = load_from_disk(sys.argv[1])
for name in ("train", "validation", "test"):
    if name not in splits or not len(splits[name]):
        raise SystemExit(f"Missing/empty {name} split")
    if not {"input_ids", "attention_mask", "labels"}.issubset(splits[name].column_names):
        raise SystemExit("Tokenization missing; check tokenizer access and Step 4 log.")
    print(f"{name}: {len(splits[name])} tokenized examples")
PY
fi
echo "Steps $START–$END complete. Artifacts and logs: $OUT"
