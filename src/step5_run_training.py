"""Run local single-GPU multimodal QLoRA training with aggregate-only W&B metrics."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

METRICS = {
    "loss", "grad_norm", "learning_rate", "epoch", "eval_loss",
    "train_loss", "train_runtime", "train_samples_per_second",
    "train_steps_per_second", "eval_runtime", "eval_samples_per_second",
    "eval_steps_per_second", "total_flos",
}


def scalar_metrics(values):
    """Never forward arbitrary Trainer payloads or patient content."""
    return {key: float(value) for key, value in values.items()
            if key in METRICS and isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value)}


def validate_datasets(dataset, smoke_subjects=None):
    """Use train/validation only; reject patient leakage without reading test rows."""
    required = {"subject_id", "stay_id", "prompt_text", "image_paths",
                "target_completion", "label"}
    subjects = {}
    selected = {}
    for name in ("train", "validation"):
        if name not in dataset or not len(dataset[name]):
            raise ValueError(f"Missing or empty {name} split")
        split = dataset[name]
        if not required.issubset(split.column_names):
            raise ValueError(f"{name} lacks required raw multimodal columns")
        subjects[name] = set(split["subject_id"])
        if smoke_subjects:
            # Include image-bearing and image-free cases when present.
            representatives = {}
            for index, row in enumerate(split):
                representatives.setdefault(bool(row["image_paths"]), row["subject_id"])
                if len(representatives) == 2:
                    break
            ids = list(dict.fromkeys([*representatives.values(), *sorted(subjects[name])]))[:smoke_subjects]
            keep = set(ids)
            split = split.select([i for i, subject in enumerate(split["subject_id"]) if subject in keep])
        for row in split:
            if row["label"] not in (0, 1) or row["target_completion"] != config.STEP4_CLASS_LABELS[row["label"]]:
                raise ValueError("Binary target and label disagree")
            if row["prompt_text"].count("<image>") != len(row["image_paths"]):
                raise ValueError("Image markers and paths disagree")
            if any(not Path(path).is_file() for path in row["image_paths"]):
                raise ValueError("Referenced images are missing")
        selected[name] = split
    if subjects["train"] & subjects["validation"]:
        raise ValueError("Train and validation subjects overlap")
    return selected


def language_adapter_targets(model):
    """Adapt matching language layers only, keeping the vision encoder frozen."""
    targets = [
        name for name, _ in model.named_modules()
        if name.rsplit(".", 1)[-1] in config.STEP5_LORA_TARGET_MODULES
        and not any(part in name.split(".") for part in ("visual", "vision_model", "vision_tower"))
    ]
    if not targets:
        raise ValueError("No configured language adapter targets matched this model")
    return targets


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True, help="New local run directory")
    p.add_argument("--variant", choices=("trag", "static_rag", "no_kg", "xgboost"), default="trag")
    p.add_argument("--windows", type=Path, default=config.DATA_INTERIM / "full_cohort/patient_windows.parquet",
                   help="XGBoost input")
    p.add_argument("--allow-kg-reduction", action="store_true",
                   help="Explicitly permit dropping retrieved edges to fit context; default preserves complete prompts")
    p.add_argument("--wandb-group", default="step5-comparison")
    p.add_argument("--model", default=config.BASE_LVLM_NAME)
    p.add_argument("--revision", default="main", help="Prefer a pinned model commit for reproducible runs")
    p.add_argument("--max-seq-length", type=int, default=config.STEP4_MAX_SEQ_LENGTH,
                   help="Includes actual image tokens and completion")
    p.add_argument("--max-image-pixels", type=int, help="Optional processor image resolution ceiling")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Train the entire training split")
    mode.add_argument("--smoke", action="store_true", help="Default: ten subjects/split, two optimizer steps")
    p.add_argument("--max-steps", type=int, help="Override optimizer steps; positive")
    p.add_argument("--predict-test", action="store_true", help="After training, save local held-out probabilities")
    p.add_argument("--rationales", action="store_true", help="Also generate local test rationales (slower)")
    p.add_argument("--epochs", type=float, default=config.STEP5_EPOCHS)
    p.add_argument("--resume", type=Path, help="Existing Trainer checkpoint, including optimizer state")
    p.add_argument("--offline", action="store_true", help="Require cached model and processor")
    p.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    p.add_argument("--wandb-project", default=config.WANDB_PROJECT)
    p.add_argument("--wandb-entity", default=config.WANDB_ENTITY)
    p.add_argument("--run-name", help="Default includes variant, model, context, and smoke/full mode")
    return p


def start_tracking(args, settings):
    if args.wandb_mode == "disabled":
        return None
    import wandb
    return wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
        group=args.wandb_group, job_type=args.variant, tags=[args.variant, settings["mode"]],
        mode=args.wandb_mode, dir=str(args.output), config=settings,
        settings=wandb.Settings(console="off", disable_git=True, disable_code=True,
                                x_disable_stats=True, x_disable_meta=True),
    )


def run_xgboost(args):
    import pandas as pd
    from src.step5_tabular_baseline import train_tabular_baseline, persist_baseline
    if not args.full:
        raise ValueError("XGBoost uses the existing full cohort; specify --full")
    args.output.mkdir(parents=True, exist_ok=True)
    result = train_tabular_baseline(pd.read_parquet(args.windows))
    persist_baseline(result, args.output)
    settings = {"variant": "xgboost", "mode": "full", "seed": config.RANDOM_SEED,
                **result.summary["training_parameters"]}
    tracking = start_tracking(args, settings)
    try:
        if tracking:
            payload = {"validation/average_precision": max(result.summary["candidate_validation_scores"])}
            for split, counts in result.summary["splits"].items():
                for field in ("stays", "subjects", "positives", "prevalence"):
                    payload[f"{split}/{field}"] = counts[field]
            tracking.log(payload)
    finally:
        if tracking:
            tracking.finish()
    print("XGBoost complete; model and predictions saved locally.")


def run(args):
    import torch
    from datasets import load_from_disk
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, TrainerCallback, set_seed
    from src.step5_training import (
        MultimodalDataCollator, SepsisTrainer, apply_qlora_to_model,
        count_trainable_parameters, get_qlora_config, get_quantization_config,
        get_training_arguments,
    )
    if args.variant in ("static_rag", "no_kg") and args.dataset is None:
        raise ValueError("Provide the matching ablation dataset explicitly with --dataset")
    args.dataset = (args.dataset or config.DATA_INTERIM / "full_cohort/tokenized_dataset").resolve()
    args.windows = args.windows.resolve()
    if not args.run_name:
        args.run_name = (f"{args.variant}-xgboost" if args.variant == "xgboost" else
                         f"{args.variant}-{args.model.rsplit('/', 1)[-1]}-{args.max_seq_length}")
        args.run_name += "-full" if args.full else "-smoke"
    args.output = args.output.resolve()
    if args.resume:
        args.resume = args.resume.resolve()
    if args.max_seq_length < 256 or args.epochs <= 0:
        raise ValueError("Sequence length must be >=256 and epochs must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.max_image_pixels is not None and args.max_image_pixels < 1:
        raise ValueError("max-image-pixels must be positive")
    if args.variant == "xgboost":
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Choose an empty XGBoost output directory")
        run_xgboost(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This 4-bit training runner requires a CUDA GPU")
    if torch.cuda.device_count() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Runner currently supports one visible GPU and one process")
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        raise ValueError("Output is not empty; choose a new directory or specify --resume")
    if args.resume and not (args.resume / "trainer_state.json").is_file():
        raise ValueError("Resume requires a Trainer checkpoint containing trainer_state.json")
    os.chdir(ROOT)  # Existing artifacts may contain repo-relative image paths.
    data = validate_datasets(load_from_disk(str(args.dataset.resolve())),
                             None if args.full else config.STEP5_SMOKE_SAMPLE_SIZE)
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(config.RANDOM_SEED)
    torch.set_num_threads(config.SAPBERT_THREADS)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    loading = dict(revision=args.revision, local_files_only=args.offline)
    model_config = AutoConfig.from_pretrained(args.model, **loading)
    if model_config.model_type != "qwen3_vl":
        raise ValueError("This first runner supports Qwen3-VL only; hybrid Qwen3.5/3.8 adapter coverage needs a separate check")
    processor = AutoProcessor.from_pretrained(args.model, **loading)
    if args.max_image_pixels is not None:
        processor.image_processor.max_pixels = args.max_image_pixels
        if isinstance(getattr(processor.image_processor, "size", None), dict):
            processor.image_processor.size["longest_edge"] = args.max_image_pixels
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, config=model_config, quantization_config=get_quantization_config(bnb_4bit_compute_dtype=dtype),
        dtype=dtype, device_map={"": 0}, attn_implementation="sdpa", **loading,
    )
    targets = language_adapter_targets(model)
    model.config.use_cache = False
    model = apply_qlora_to_model(model, get_qlora_config(target_modules=targets))
    trainable, total, percent = count_trainable_parameters(model)
    if not trainable:
        raise RuntimeError("No trainable adapter parameters")
    max_steps = args.max_steps if args.max_steps is not None else (-1 if args.full else 2)
    settings = {
        "variant": args.variant, "allow_kg_reduction": args.allow_kg_reduction,
        "kg_serialization": sorted({
            "compact_v1" if "- Compact KG dictionary v1:" in row["prompt_text"]
            else "verbose" if "  [" in row["prompt_text"] else "no_edges"
            for split in data.values() for row in split
        }),
        "model": args.model, "model_revision": getattr(model_config, "_commit_hash", args.revision),
        "max_seq_length": args.max_seq_length, "max_image_pixels": args.max_image_pixels,
        "mode": "full" if args.full else "smoke", "max_steps": max_steps,
        "train_stays": len(data["train"]), "validation_stays": len(data["validation"]),
        "epochs": args.epochs, "seed": config.RANDOM_SEED,
        "loss_projection": "supervised_positions_only",
        "quantization": "NF4-double", "dtype": str(dtype),
        "lora_r": config.STEP5_LORA_R, "lora_alpha": config.STEP5_LORA_ALPHA,
        "lora_dropout": config.STEP5_LORA_DROPOUT,
        "learning_rate": config.STEP5_LEARNING_RATE, "batch_size": 1,
        "gradient_accumulation_steps": config.STEP5_GRADIENT_ACCUMULATION_STEPS,
        "gradient_checkpointing": True, "adapter_target_count": len(targets),
        "trainable_parameters": trainable, "total_parameters": total,
        "trainable_percent": percent, "gpu": torch.cuda.get_device_name(0),
    }
    (args.output / "run_config.json").write_text(json.dumps(settings, indent=2) + "\n")
    tracking = None
    try:
        tracking = start_tracking(args, settings)

        class AggregateCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if tracking:
                    payload = scalar_metrics(logs or {})
                    if payload:
                        tracking.log(payload, step=state.global_step)

        training_args = get_training_arguments(
            output_dir=args.output, run_name=args.run_name, num_train_epochs=args.epochs,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            max_steps=max_steps, report_to="none", prediction_loss_only=True,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="epoch" if args.full else "no",
            save_strategy="epoch" if args.full else "no", save_total_limit=2,
            logging_steps=config.STEP5_LOGGING_STEPS if args.full else 1,
            bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
            dataloader_num_workers=0, seed=config.RANDOM_SEED,
        )
        trainer = SepsisTrainer(
            model=model, args=training_args, processor=processor,
            train_dataset=data["train"], eval_dataset=data["validation"],
            data_collator=MultimodalDataCollator(
                processor=processor, pad_token_id=processor.tokenizer.pad_token_id,
                max_seq_length=args.max_seq_length,
                allow_graph_reduction=args.allow_kg_reduction),
            callbacks=[AggregateCallback()],
        )
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        validation = trainer.evaluate()
        trainer.save_model(str(args.output / "adapter"))
        trainer.save_state()
        metrics = {**scalar_metrics(result.metrics), **scalar_metrics(validation),
                   "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        (args.output / "training_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if args.predict_test:
            import pandas as pd
            from src.step5_training import predict_sepsis
            heldout = load_from_disk(str(args.dataset))["test"]
            rows = []
            for record in heldout:
                prediction = predict_sepsis(
                    model, processor, record, max_seq_length=args.max_seq_length,
                    generate_rationale=args.rationales, allow_graph_reduction=args.allow_kg_reduction,
                )
                rows.append({
                    "stay_id": record["stay_id"], "subject_id": record["subject_id"],
                    "split": "test", "label": record["label"], **prediction,
                })
            pd.DataFrame(rows).to_parquet(args.output / "predictions.parquet", index=False)
            print(f"Saved {len(rows)} held-out predictions locally.")
        if tracking:
            tracking.log(metrics)
        print(json.dumps({"status": "complete", **metrics}, indent=2))
    finally:
        if tracking:
            tracking.finish()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        # Model/processor/parser errors can contain record contents; don't print raw exceptions.
        print(f"Training failed ({type(exc).__name__}); inspect locally without uploading patient records.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
