#!/usr/bin/env bash
# Single experiment entry point: one variant by default, selected variants with --suite.
set -Eeuo pipefail
umask 077
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/capstone/bin/python}"
VARIANTS=(trag static_rag no_kg)  # Suite defaults; optionally add xgboost.
export PYTHONUNBUFFERED=1
if ! command -v "$PYTHON" >/dev/null; then
  echo "Python unavailable: set PYTHON to your project interpreter." >&2
  exit 1
fi
SUITE=0
ARGS=()
for arg in "$@"; do
  if [[ $arg == --suite ]]; then SUITE=1; else ARGS+=("$arg"); fi
done
MODULE=src.step5_run_training
SCRIPT=src/step5_run_training.py
if (( SUITE )); then
  MODULE=src.step5_run_experiments
  SCRIPT=src/step5_run_experiments.py
  ARGS=(--variants "${VARIANTS[@]}" "${ARGS[@]}")
fi
# Optional shared model/context overrides; explicit CLI arguments take precedence.
if [[ -n ${MODEL:-} ]]; then ARGS=(--model "$MODEL" "${ARGS[@]}"); fi
if [[ -n ${CONTEXT_LENGTH:-} ]]; then ARGS=(--max-seq-length "$CONTEXT_LENGTH" "${ARGS[@]}"); fi
for arg in "${ARGS[@]}"; do
  if [[ $arg == --help || $arg == -h ]]; then
    echo "Use --suite to prepare, train, predict, and compare selected variants."
    exec "$PYTHON" "$SCRIPT" "${ARGS[@]}"
  fi
done
# Reuse the selected Python CLI; dry-run and offline mode never check W&B.
"$PYTHON" - "$MODULE" "${ARGS[@]}" <<'PY'
import importlib
import sys
args = importlib.import_module(sys.argv[1]).parser().parse_args(sys.argv[2:])
if args.wandb_mode == "online" and not getattr(args, "dry_run", False):
    import wandb
    try:
        if not wandb.Api(timeout=15).viewer:
            raise RuntimeError("No authenticated viewer")
    except Exception:
        raise SystemExit("W&B authentication check failed. Run capstone/bin/wandb login locally, verify network access, and retry. Never share your API key.")
    print("W&B authentication verified. The run will also verify project write access.")
PY
exec "$PYTHON" "$SCRIPT" "${ARGS[@]}"
