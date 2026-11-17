"""Prepare QLoRA training batches and produce sepsis scores with evidence summaries."""

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

import config
from src.step4_serialization import compute_sepsis_probability, fit_graph_context


def get_qlora_config(
    r: int = config.STEP5_LORA_R,
    lora_alpha: int = config.STEP5_LORA_ALPHA,
    lora_dropout: float = config.STEP5_LORA_DROPOUT,
    target_modules: Optional[List[str]] = None,
    bias: str = config.STEP5_LORA_BIAS,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """Builds and returns the PEFT LoRA configuration for causal language modeling."""
    if target_modules is None:
        target_modules = list(config.STEP5_LORA_TARGET_MODULES)
    return LoraConfig(
        r=int(r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
    )


def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: Any = torch.bfloat16,
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """Builds and returns the 4-bit BitsAndBytes quantization configuration (NF4)."""
    return BitsAndBytesConfig(
        load_in_4bit=bool(load_in_4bit),
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bool(bnb_4bit_use_double_quant),
    )


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """Calculates trainable parameters, total parameters, and the percentage trainable."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    percentage = 100.0 * trainable_params / all_params if all_params > 0 else 0.0
    return trainable_params, all_params, percentage


def apply_qlora_to_model(
    model: nn.Module,
    lora_config: Optional[LoraConfig] = None,
    prepare_for_kbit: bool = True,
) -> PeftModel:
    """Wraps a base model with PEFT LoRA adapters for causal LM training."""
    if lora_config is None:
        lora_config = get_qlora_config()

    if prepare_for_kbit and getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    peft_model = get_peft_model(model, lora_config)
    return peft_model


def prepare_model_prompt(record, processor, max_prompt_tokens, allow_graph_reduction=True):
    """Decode aligned radiographs and fit the actual Qwen input, including visual tokens."""
    prompt = record["prompt_text"]
    parts = prompt.split("<image>")
    paths = record["image_paths"]
    if len(parts) != len(paths) + 1:
        raise ValueError("Image placeholders and paths are not aligned")
    images = []
    for path in paths:
        try:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        except (OSError, ValueError) as error:
            raise ValueError("An aligned radiograph could not be decoded") from error

    def encode(text):
        content = []
        for index, part in enumerate(text.split("<image>")):
            if index:
                content.append({"type": "image"})
            if part:
                content.append({"type": "text", "text": part})
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
        )
        result = dict(processor(
            text=[rendered], images=images or None, return_tensors="pt",
            add_special_tokens=False, truncation=False,
        ))
        if images and ("pixel_values" not in result or "image_grid_thw" not in result
                       or result["image_grid_thw"].shape[0] != len(images)):
            raise ValueError("Processor did not produce visual inputs for every aligned radiograph")
        return result

    if not allow_graph_reduction:
        result = encode(prompt)
        if result["input_ids"].shape[-1] > max_prompt_tokens:
            raise ValueError("Complete prompt exceeds context limit; increase the limit or explicitly allow KG reduction")
        return prompt, result
    return fit_graph_context(prompt, encode, max_prompt_tokens)


def append_completion(prompt_inputs, completion_ids):
    """Append text tokens without altering the vision tensors or supervising the prompt."""
    ids = prompt_inputs["input_ids"]
    completion = ids.new_tensor([completion_ids])
    example = {**prompt_inputs, "input_ids": torch.cat((ids, completion), dim=1)}
    for key in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
        if key in example:
            values = example[key]
            fill = 1 if key == "attention_mask" else 0
            example[key] = torch.cat((values, values.new_full(completion.shape, fill)), dim=1)
    example["labels"] = torch.cat((torch.full_like(ids, -100), completion), dim=1)
    return example


class MultimodalDataCollator:
    """Collate multimodal examples while preserving target-only loss masking."""

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        padding_side: str = "right",
        processor: Optional[Any] = None,
        max_seq_length: int = config.STEP4_MAX_SEQ_LENGTH,
        allow_graph_reduction: bool = True,
    ):
        if padding_side not in ("right", "left"):
            raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side}")
        self.pad_token_id = int(pad_token_id)
        self.label_pad_token_id = int(label_pad_token_id)
        self.padding_side = padding_side
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.allow_graph_reduction = allow_graph_reduction

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collates and pads a list of sample feature dicts into batched tensors."""
        if not features:
            raise ValueError("Cannot collate an empty list of features.")
        if self.processor is not None:
            return self._collate_model_inputs(features)
        if any(feature.get("image_paths") for feature in features):
            raise ValueError("A vision-language processor is required for image-bearing records")

        required_keys = ["input_ids", "attention_mask", "labels"]
        for key in required_keys:
            if key not in features[0]:
                raise KeyError(f"Feature dictionary missing required key: '{key}'")

        batch_size = len(features)
        seq_lengths = [len(f["input_ids"]) for f in features]
        max_len = max(seq_lengths)

        padded_input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        padded_labels = torch.full(
            (batch_size, max_len), self.label_pad_token_id, dtype=torch.long
        )

        for i, feat in enumerate(features):
            ids = feat["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            else:
                ids = ids.to(dtype=torch.long)

            mask = feat["attention_mask"]
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.long)
            else:
                mask = mask.to(dtype=torch.long)

            lbl = feat["labels"]
            if not isinstance(lbl, torch.Tensor):
                lbl = torch.tensor(lbl, dtype=torch.long)
            else:
                lbl = lbl.to(dtype=torch.long)

            cur_len = ids.shape[0]
            if mask.shape[0] != cur_len or lbl.shape[0] != cur_len:
                raise ValueError(
                    f"Feature index {i} has mismatched tensor lengths: input_ids={cur_len}, "
                    f"attention_mask={mask.shape[0]}, labels={lbl.shape[0]}"
                )

            if self.padding_side == "right":
                padded_input_ids[i, :cur_len] = ids
                padded_attention_mask[i, :cur_len] = mask
                padded_labels[i, :cur_len] = lbl
            else:
                padded_input_ids[i, max_len - cur_len :] = ids
                padded_attention_mask[i, max_len - cur_len :] = mask
                padded_labels[i, max_len - cur_len :] = lbl

        batch: Dict[str, Any] = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }

        for key in features[0]:
            if key in required_keys:
                continue
            values = [f[key] for f in features]
            if all(isinstance(v, torch.Tensor) for v in values):
                try:
                    batch[key] = torch.stack(values, dim=0)
                except Exception:
                    batch[key] = values
            else:
                batch[key] = values

        return batch

    def _collate_model_inputs(self, features):
        """Pad text tensors and concatenate image patches in sample and window order."""
        positive, negative = get_target_token_ids(self.processor.tokenizer)
        budget = self.max_seq_length - max(len(positive), len(negative))
        examples = []
        for feature in features:
            target = feature["target_completion"]
            if target not in config.STEP4_CLASS_LABELS.values():
                raise ValueError("Expected a binary target completion")
            completion = positive if target == config.STEP4_TARGET_COMPLETION_POSITIVE else negative
            _, prompt_inputs = prepare_model_prompt(feature, self.processor, budget,
                                                          allow_graph_reduction=self.allow_graph_reduction)
            examples.append(append_completion(prompt_inputs, completion))

        batch = {}
        pad_values = {"input_ids": self.pad_token_id, "attention_mask": 0,
                      "labels": self.label_pad_token_id, "mm_token_type_ids": 0, "token_type_ids": 0}
        width = max(example["input_ids"].shape[-1] for example in examples)
        for key, value in pad_values.items():
            if not any(key in example for example in examples):
                continue
            sequences = []
            for example in examples:
                ids = example["input_ids"]
                sequence = example.get(key, torch.zeros_like(ids))
                padding = width - ids.shape[-1]
                sides = (padding, 0) if self.padding_side == "left" else (0, padding)
                sequences.append(torch.nn.functional.pad(sequence, sides, value=value))
            batch[key] = torch.cat(sequences)
        for key in ("pixel_values", "image_grid_thw"):
            tensors = [example[key] for example in examples if key in example]
            if tensors:
                batch[key] = torch.cat(tensors)
        return batch


def get_training_arguments(
    output_dir: Union[str, Path] = config.CHECKPOINTS_DIR,
    run_name: Optional[str] = None,
    learning_rate: float = config.STEP5_LEARNING_RATE,
    per_device_train_batch_size: int = config.STEP5_BATCH_SIZE,
    gradient_accumulation_steps: int = config.STEP5_GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs: float = config.STEP5_EPOCHS,
    warmup_ratio: float = config.STEP5_WARMUP_RATIO,
    weight_decay: float = config.STEP5_WEIGHT_DECAY,
    max_grad_norm: float = config.STEP5_MAX_GRAD_NORM,
    logging_steps: int = config.STEP5_LOGGING_STEPS,
    eval_strategy: str = config.STEP5_EVAL_STRATEGY,
    save_strategy: str = config.STEP5_SAVE_STRATEGY,
    report_to: Optional[str] = "none",
    bf16: Optional[bool] = None,
    fp16: Optional[bool] = None,
    **kwargs: Any,
) -> TrainingArguments:
    """Build the training arguments for the QLoRA run."""
    if bf16 is None:
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if fp16 is None:
        fp16 = False if bf16 else torch.cuda.is_available()

    args_dict: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(learning_rate),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "num_train_epochs": float(num_train_epochs),
        "warmup_ratio": float(warmup_ratio),
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "logging_steps": int(logging_steps),
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "report_to": report_to,
        "bf16": bool(bf16),
        "fp16": bool(fp16),
        "remove_unused_columns": False,
    }
    if run_name is not None:
        args_dict["run_name"] = run_name

    if "warmup_ratio" not in inspect.signature(TrainingArguments).parameters:
        # New Transformers versions accept a fractional warmup_steps instead.
        args_dict["warmup_steps"] = args_dict.pop("warmup_ratio")
    args_dict.update(kwargs)
    return TrainingArguments(**args_dict)


class SepsisTrainer(Trainer):
    """Custom Trainer subclass for multimodal sepsis fine-tuning with adapter saving."""

    def __init__(
        self,
        model: Union[nn.Module, PeftModel] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        processor: Optional[Any] = None,
        **kwargs: Any,
    ):
        if data_collator is None:
            if processor is None:
                processor = AutoProcessor.from_pretrained(config.BASE_LVLM_NAME, local_files_only=True)
            data_collator = MultimodalDataCollator(
                processor=processor, pad_token_id=processor.tokenizer.pad_token_id,
            )
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor if processor is not None else tokenizer,
            **kwargs,
        )

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Overrides save_model to ensure PEFT adapters are saved properly."""
        if output_dir is None:
            output_dir = self.args.output_dir
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(str(save_path))
        else:
            super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(str(save_path))


def save_adapter_checkpoint(
    model: PeftModel,
    save_directory: Union[str, Path],
    tokenizer: Optional[Any] = None,
) -> Path:
    """Persists PEFT LoRA adapter weights and optional tokenizer configuration."""
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(save_path))
    return save_path


def verify_adapter_reload(
    base_model: nn.Module,
    adapter_directory: Union[str, Path],
    sample_input: Dict[str, torch.Tensor],
    expected_logits: Optional[torch.Tensor] = None,
    tolerance: float = 1e-4,
) -> bool:
    """Verifies that an adapter reloaded from disk produces valid and consistent logits."""
    reloaded_model = PeftModel.from_pretrained(base_model, str(adapter_directory))
    reloaded_model.eval()
    with torch.no_grad():
        out = reloaded_model(**sample_input)
        logits = out.logits if hasattr(out, "logits") else out

    if expected_logits is not None:
        if not torch.allclose(logits, expected_logits, atol=tolerance):
            return False

    return bool(torch.isfinite(logits).all().item())


def extract_sepsis_probability(
    logits: Union[torch.Tensor, Any],
    pos_token_id: int,
    neg_token_id: int,
    target_pos_idx: int = -1,
    is_sequence: bool = False,
) -> Union[float, List[float]]:
    """Normalize scores for single-token labels; use sequence scoring for Qwen."""
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)

    if logits.ndim == 3:
        target_logits = logits[:, target_pos_idx, :]
    elif logits.ndim == 2:
        if is_sequence:
            target_logits = logits[target_pos_idx, :].unsqueeze(0)
        else:
            target_logits = logits
    elif logits.ndim == 1:
        target_logits = logits.unsqueeze(0)
    else:
        raise ValueError(f"Expected logits tensor with 1, 2, or 3 dims, got {logits.ndim}")

    z_pos = target_logits[:, pos_token_id].float()
    z_neg = target_logits[:, neg_token_id].float()

    max_z = torch.maximum(z_pos, z_neg)
    exp_pos = torch.exp(z_pos - max_z)
    exp_neg = torch.exp(z_neg - max_z)
    probs = exp_pos / (exp_pos + exp_neg)

    probs = torch.clamp(probs, 0.0, 1.0)
    prob_list = probs.tolist()

    if (
        (logits.ndim == 1)
        or (logits.ndim == 2 and is_sequence)
        or (logits.ndim == 2 and logits.shape[0] == 1 and not is_sequence)
    ):
        return float(prob_list[0])
    return [float(p) for p in prob_list]


def get_target_token_ids(tokenizer: Any) -> Tuple[List[int], List[int]]:
    """Return all token IDs for both binary completions."""
    pos_text = config.STEP4_TARGET_COMPLETION_POSITIVE
    neg_text = config.STEP4_TARGET_COMPLETION_NEGATIVE

    pos_ids = tokenizer.encode(pos_text, add_special_tokens=False)
    neg_ids = tokenizer.encode(neg_text, add_special_tokens=False)

    if not pos_ids:
        raise ValueError(f"Tokenizer produced empty token IDs for '{pos_text}'")
    if not neg_ids:
        raise ValueError(f"Tokenizer produced empty token IDs for '{neg_text}'")

    if pos_ids == neg_ids:
        raise ValueError("Binary completions must tokenize to distinct sequences")
    return pos_ids, neg_ids


@torch.inference_mode()
def predict_sepsis(model, processor, record, max_seq_length=config.STEP4_MAX_SEQ_LENGTH,
                   generate_rationale=True, allow_graph_reduction=True):
    """Score both complete labels on the same observation prompt and generate a separate rationale."""
    model.eval()
    device = model.get_input_embeddings().weight.device
    targets = get_target_token_ids(processor.tokenizer)
    fitted_prompt, prompt_inputs = prepare_model_prompt(
        record, processor, max_seq_length - max(map(len, targets)),
        allow_graph_reduction=allow_graph_reduction,
    )
    prompt_length = prompt_inputs["input_ids"].shape[-1]
    scores = []
    for target in targets:
        inputs = append_completion(prompt_inputs, target)
        labels = inputs.pop("labels").to(device)
        logits = model(**{key: value.to(device) for key, value in inputs.items()}).logits
        # The position immediately before each completion token predicts that token.
        log_probs = logits[:, prompt_length - 1 : -1].float().log_softmax(dim=-1)
        scores.append(log_probs.gather(-1, labels[:, prompt_length:, None]).sum().item())
    probability = compute_sepsis_probability(*scores)
    if not generate_rationale:
        return {"sepsis_probability": probability}

    if not fitted_prompt.endswith(config.STEP4_TARGET_TASK):
        raise ValueError("Expected the binary task instruction at the end of the prompt")
    rationale_prompt = fitted_prompt.removesuffix(config.STEP4_TARGET_TASK) + config.STEP5_RATIONALE_TASK
    _, rationale_inputs = prepare_model_prompt(
        {**record, "prompt_text": rationale_prompt}, processor,
        max_seq_length - config.STEP5_RATIONALE_MAX_NEW_TOKENS,
        allow_graph_reduction=allow_graph_reduction,
    )
    generated = model.generate(
        **{key: value.to(device) for key, value in rationale_inputs.items()},
        max_new_tokens=config.STEP5_RATIONALE_MAX_NEW_TOKENS, do_sample=False,
    )
    rationale = processor.tokenizer.decode(
        generated[0, rationale_inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
    )
    return format_inference_output(probability, rationale)


def format_inference_output(
    sepsis_probability: float,
    reasoning_chain: str,
) -> Dict[str, Any]:
    """Validate and package the probability and clinical rationale."""
    prob = float(sepsis_probability)
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"sepsis_probability must be in [0.0, 1.0], got {prob}")
    if not isinstance(reasoning_chain, str) or not reasoning_chain.strip():
        raise ValueError("reasoning_chain must be a non-empty string.")

    return {
        "sepsis_probability": round(prob, 4),
        "reasoning_chain": reasoning_chain.strip(),
    }


def validate_inference_output(output_dict: Any) -> bool:
    """Validates that a dictionary adheres strictly to the two-field inference schema."""
    if not isinstance(output_dict, dict):
        return False
    required_keys = {"sepsis_probability", "reasoning_chain"}
    if set(output_dict.keys()) != required_keys:
        return False
    prob = output_dict["sepsis_probability"]
    if not isinstance(prob, (int, float)) or isinstance(prob, bool):
        return False
    if not (0.0 <= float(prob) <= 1.0):
        return False
    chain = output_dict["reasoning_chain"]
    if not isinstance(chain, str) or not chain.strip():
        return False
    return True
