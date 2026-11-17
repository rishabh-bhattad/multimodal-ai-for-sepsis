"""Prepare, train, predict, and compare selected variants sequentially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

NEURAL = ("trag", "static_rag", "no_kg")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", choices=(*NEURAL, "xgboost"), default=list(NEURAL))
    p.add_argument("--output", type=Path, required=True, help="New experiment-suite directory")
    p.add_argument("--windows", type=Path, default=config.DATA_INTERIM / "full_cohort/patient_windows.parquet")
    p.add_argument("--trajectories", type=Path, default=config.DATA_INTERIM / "full_cohort/subgraph_trajectories.pkl")
    p.add_argument("--model", default=config.BASE_LVLM_NAME)
    p.add_argument("--revision", default="main")
    p.add_argument("--max-seq-length", type=int, default=config.STEP4_MAX_SEQ_LENGTH)
    p.add_argument("--max-image-pixels", type=int)
    p.add_argument("--epochs", type=float, default=config.STEP5_EPOCHS)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true", help="Default: ten subjects per split; two optimizer steps")
    p.add_argument("--allow-kg-reduction", action="store_true")
    p.add_argument("--rationales", action="store_true")
    p.add_argument("--offline", action="store_true", help="Require cached models")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    p.add_argument("--wandb-entity", default=config.WANDB_ENTITY)
    p.add_argument("--wandb-project", default=config.WANDB_PROJECT)
    p.add_argument("--wandb-group", help="Default: suite output directory name")
    p.add_argument("--xgboost-predictions", type=Path, help="Reuse existing XGBoost predictions instead of retraining")
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true", help="Show plan without reading data, writing files, or training")
    return p


def prepare_variants(windows, trajectories, variants, output, full):
    """Use identical stays, labels, and images for every selected neural variant."""
    from datasets import Dataset, DatasetDict
    from src.step4_serialization import build_stay_example, load_patient_windows_parquet, load_trajectories_pickle
    from src.step5_ablations import build_ablation_dataset_dict
    frame, windows_digest = load_patient_windows_parquet(windows)
    if frame.groupby("subject_id").split.nunique().max() != 1:
        raise ValueError("Patient splits overlap")
    if not full:
        subjects = []
        for split in ("train", "val", "test"):
            part = frame[frame.split == split]
            # Stable across models/variants; include image and no-image subjects.
            representatives = part.assign(has_image=part.cxr_path.notna()).groupby("has_image", sort=True).subject_id.first().tolist()
            ids = list(dict.fromkeys([*representatives, *sorted(part.subject_id.unique())]))
            subjects.extend(ids[:config.STEP5_SMOKE_SAMPLE_SIZE])
        frame = frame[frame.subject_id.isin(subjects)]
    graph_data = {}
    graph_digest = None
    if any(v in ("trag", "static_rag") for v in variants):
        graph_data, graph_digest = load_trajectories_pickle(trajectories)
    stays = {"train": [], "validation": [], "test": []}
    for sid, rows in frame.groupby("stay_id", sort=True):
        records = rows.sort_values("window_idx").to_dict("records")
        first = records[0]
        name = "validation" if first["split"] == "val" else first["split"]
        graph = graph_data.get(str(sid), graph_data.get(sid))
        if graph_data and graph is None:
            raise ValueError("Cohort stay missing from Step 3 trajectories")
        if graph is not None and (int(graph["subject_id"]) != int(first["subject_id"])
                                  or graph["split"] != first["split"]):
            raise ValueError("Trajectory subject/split differs from cohort")
        stays[name].append(dict(stay_id=int(sid), subject_id=int(first["subject_id"]),
                                label=int(first["label"]), windows=records,
                                subgraphs=graph["windows"] if graph else None))
    if any(not items for items in stays.values()):
        raise ValueError("Every experiment split must be nonempty")
    destinations = {}
    for variant in variants:
        if variant not in NEURAL:
            continue
        if variant == "trag":
            dataset = DatasetDict({
                name: Dataset.from_list([
                    build_stay_example(
                        stay_id=r["stay_id"], subject_id=r["subject_id"], split=name,
                        stay_windows=r["windows"], label=r["label"], subgraphs=r["subgraphs"],
                        cxr_dir=config.MIMIC_CXR_DIR,
                    ) for r in items
                ]) for name, items in stays.items()
            })
        else:
            dataset = build_ablation_dataset_dict(stays, variant=variant, cxr_dir=config.MIMIC_CXR_DIR)
        destination = output / "datasets" / variant
        dataset.save_to_disk(str(destination))
        destinations[variant] = destination
    return destinations, frame, {"windows_sha256": windows_digest, "trajectories_sha256": graph_digest}


def training_command(args, variant, dataset):
    cmd = [str(ROOT / "run_training.sh"), "--variant", variant, "--output", str(args.output / "runs" / variant),
           "--full" if args.full or variant == "xgboost" else "--smoke",
           "--wandb-mode", args.wandb_mode, "--wandb-project", args.wandb_project,
           "--wandb-entity", args.wandb_entity, "--wandb-group", args.wandb_group]
    if variant == "xgboost":
        return [*cmd, "--windows", str(args.windows)]
    cmd += ["--dataset", str(dataset), "--model", args.model, "--revision", args.revision,
            "--max-seq-length", str(args.max_seq_length), "--epochs", str(args.epochs), "--predict-test"]
    for enabled, flag in ((args.allow_kg_reduction, "--allow-kg-reduction"),
                          (args.rationales, "--rationales"), (args.offline, "--offline")):
        if enabled:
            cmd.append(flag)
    if args.max_image_pixels is not None:
        cmd += ["--max-image-pixels", str(args.max_image_pixels)]
    return cmd


def compare(args, prediction_files, reference):
    import pandas as pd
    from src.step5_evaluation import evaluate_model_comparison, save_evaluation_report
    from src.step5_run_training import start_tracking
    predictions = {}
    test_stays = set(reference.loc[reference.split == "test", "stay_id"])
    for variant, path in prediction_files.items():
        frame = pd.read_parquet(path)
        predictions[variant] = frame[(frame.split == "test") & frame.stay_id.isin(test_stays)].copy()
        if set(predictions[variant].stay_id) != test_stays:
            raise ValueError("Predictions do not cover the shared held-out cohort")
        truth = reference[reference.split == "test"].drop_duplicates("stay_id").set_index("stay_id")
        aligned = predictions[variant].set_index("stay_id")
        if not aligned[["subject_id", "label"]].sort_index().equals(truth[["subject_id", "label"]].sort_index()):
            raise ValueError("Predictions differ from the current cohort")
    report = evaluate_model_comparison(predictions, n_resamples=args.resamples)
    save_evaluation_report(report, args.output / "comparison")
    # Explicitly label smoke results; they are wiring checks, not research estimates.
    tracking_args = argparse.Namespace(**vars(args), variant="comparison",
        run_name=f"{args.output.name}-comparison")
    run = start_tracking(tracking_args, {
        "mode": "full" if args.full else "smoke", "variants": list(predictions),
        "model": args.model, "max_seq_length": args.max_seq_length,
    })
    if run:
        try:
            payload = {}
            for variant, result in report["models"].items():
                for metric, value in result["metrics"].items():
                    if value is not None:
                        payload[f"test/{variant}/{metric}"] = value
            run.log(payload)
        finally:
            run.finish()
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("Duplicate variants")
    if args.max_seq_length < 256 or args.epochs <= 0 or args.resamples < 1:
        raise ValueError("Invalid context length, epochs, or bootstrap resamples")
    for name in ("output", "windows", "trajectories", "xgboost_predictions"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.wandb_group = args.wandb_group or args.output.name
    plan = {"kg_serialization": config.STEP4_KG_FORMAT, "variants": args.variants, "mode": "full" if args.full else "smoke",
            "model": args.model, "max_seq_length": args.max_seq_length,
            "xgboost_reused": bool(args.xgboost_predictions),
            "output": str(args.output), "wandb_mode": args.wandb_mode,
            "stages": ["prepare selected datasets", "train sequentially", "held-out predictions", "comparison"]}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.xgboost_predictions and not args.xgboost_predictions.is_file():
        raise ValueError("Cached XGBoost predictions are missing")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a new, empty suite output directory")
    os.chdir(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    destinations, reference, digests = prepare_variants(
        args.windows, args.trajectories, args.variants, args.output, args.full)
    manifest = {**plan, **digests}
    (args.output / "suite_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    prediction_files = {}
    for variant in args.variants:
        if variant == "xgboost" and args.xgboost_predictions:
            prediction_files[variant] = args.xgboost_predictions
            continue
        subprocess.run(training_command(args, variant, destinations.get(variant)),
                       check=True, env={**os.environ, "PYTHON": sys.executable})
        prediction_files[variant] = args.output / "runs" / variant / "predictions.parquet"
    if args.xgboost_predictions:
        prediction_files["xgboost"] = args.xgboost_predictions
    report = compare(args, prediction_files, reference)
    print(f"Suite complete: {len(report['models'])} variants. Comparison: {args.output / 'comparison'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Experiment suite failed ({type(exc).__name__}); downstream steps stopped. Inspect local logs without uploading patient data.", file=sys.stderr)
        sys.exit(1)
