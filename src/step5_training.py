"""Step 5: Model Training, QLoRA Configuration & Training Harness.

Implements model quantization, PEFT LoRA adapter initialization, parameter
counting, dynamic multimodal sequence collation, Hugging Face Trainer orchestration,
adapter checkpointing/reload verification, conditional logit probability extraction,
and two-field inference JSON generation for the vision-language backbone.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

import config


def get_qlora_config(
    r: int = config.STEP5_LORA_R,
    lora_alpha: int = config.STEP5_LORA_ALPHA,
    lora_dropout: float = config.STEP5_LORA_DROPOUT,
    target_modules: Optional[List[str]] = None,
    bias: str = config.STEP5_LORA_BIAS,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """Builds and returns the PEFT LoRA configuration for causal language modeling.

    Args:
        r: LoRA attention dimension (rank).
        lora_alpha: Alpha parameter for LoRA scaling.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: Names of module targets to apply LoRA.
        bias: Bias type for LoRA ('none', 'all', or 'lora_only').
        task_type: Task type, default CAUSAL_LM.

    Returns:
        Configured LoraConfig instance.
    """
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
    """Builds and returns the 4-bit BitsAndBytes quantization configuration (NF4).

    Args:
        load_in_4bit: Enable 4-bit quantization.
        bnb_4bit_quant_type: Quantization data type ('nf4' or 'fp4').
        bnb_4bit_compute_dtype: Compute dtype for linear layers (torch.bfloat16).
        bnb_4bit_use_double_quant: Enable nested/double quantization.

    Returns:
        Configured BitsAndBytesConfig instance.
    """
    return BitsAndBytesConfig(
        load_in_4bit=bool(load_in_4bit),
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bool(bnb_4bit_use_double_quant),
    )


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """Calculates trainable parameters, total parameters, and the percentage trainable.

    Args:
        model: PyTorch Module or PeftModel.

    Returns:
        Tuple of (trainable_params, all_params, percentage).
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    percentage = 100.0 * trainable_params / all_params if all_params > 0 else 0.0
    return trainable_params, all_params, percentage


def apply_qlora_to_model(
    model: nn.Module,
    lora_config: Optional[LoraConfig] = None,
    prepare_for_kbit: bool = True,
) -> PeftModel:
    """Wraps a base model with PEFT LoRA adapters for causal LM training.

    Args:
        model: Base PyTorch neural network.
        lora_config: Optional LoraConfig. If None, uses default get_qlora_config().
        prepare_for_kbit: Whether to call prepare_model_for_kbit_training if 4-bit loaded.

    Returns:
        Configured PeftModel instance.
    """
    if lora_config is None:
        lora_config = get_qlora_config()

    if prepare_for_kbit and getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    peft_model = get_peft_model(model, lora_config)
    return peft_model


class MultimodalDataCollator:
    """Dynamic padding data collator for multimodal causal sequence training.

    Applies right-padding (or left-padding) across input_ids, attention_mask,
    and labels. Ensures target-only causal loss masking is preserved:
    - input_ids padded with pad_token_id.
    - attention_mask padded with 0.
    - labels padded with label_pad_token_id (-100).
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        padding_side: str = "right",
    ):
        if padding_side not in ("right", "left"):
            raise ValueError(f"padding_side must be 'right' or 'left', got {padding_side}")
        self.pad_token_id = int(pad_token_id)
        self.label_pad_token_id = int(label_pad_token_id)
        self.padding_side = padding_side

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collates and pads a list of sample feature dicts into batched tensors.

        Args:
            features: List of dicts, each containing 'input_ids', 'attention_mask', 'labels'.

        Returns:
            Dict containing 2D LongTensors for input_ids, attention_mask, labels,
            plus any preserved non-token fields.
        """
        if not features:
            raise ValueError("Cannot collate an empty list of features.")

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

        # Retain auxiliary fields (e.g., stay_id, image_paths, metadata)
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
    """Builds Hugging Face TrainingArguments configured for Step 5 QLoRA fine-tuning."""
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
        **kwargs: Any,
    ):
        if data_collator is None:
            data_collator = MultimodalDataCollator()
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
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
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(save_path))


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
    """Derives normalized sepsis probability P(sepsis) from conditional class logits.

    P(sepsis) = exp(z_pos - max_z) / (exp(z_pos - max_z) + exp(z_neg - max_z))

    Args:
        logits: Model output logits tensor (1D, 2D, or 3D).
        pos_token_id: Vocabulary ID for the positive token ('Sepsis').
        neg_token_id: Vocabulary ID for the negative token ('Non-Sepsis').
        target_pos_idx: Sequence index of target completion token (default -1).
        is_sequence: If True and logits is 2D, treats dim 0 as sequence length.
                     If False and logits is 2D, treats dim 0 as batch size.

    Returns:
        float if single sample, or List[float] if batched.
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)

    if logits.ndim == 3:
        target_logits = logits[:, target_pos_idx, :]
    elif logits.ndim == 2:
        if is_sequence:
            target_logits = logits[target_pos_idx : target_pos_idx + 1, :]
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


def get_target_token_ids(tokenizer: Any) -> Tuple[int, int]:
    """Retrieves token IDs for positive ('Sepsis') and negative ('Non-Sepsis') completions."""
    pos_text = config.STEP4_TARGET_COMPLETION_POSITIVE
    neg_text = config.STEP4_TARGET_COMPLETION_NEGATIVE

    pos_ids = tokenizer.encode(pos_text, add_special_tokens=False)
    neg_ids = tokenizer.encode(neg_text, add_special_tokens=False)

    if not pos_ids:
        raise ValueError(f"Tokenizer produced empty token IDs for '{pos_text}'")
    if not neg_ids:
        raise ValueError(f"Tokenizer produced empty token IDs for '{neg_text}'")

    return pos_ids[0], neg_ids[0]


def format_inference_output(
    sepsis_probability: float,
    reasoning_chain: str,
) -> Dict[str, Any]:
    """Builds a validated dictionary according to the Step 5 inference contract.

    Schema:
    {
        "sepsis_probability": <float between 0.0 and 1.0>,
        "reasoning_chain": "<short evidence-based clinical rationale>"
    }
    """
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


def parse_inference_json(raw_text: str) -> Dict[str, Any]:
    """Parses and validates a raw model completion string into structured inference output.

    Handles optional markdown JSON code blocks (e.g. ```json ... ```) or embedded JSON objects.
    """
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to decode valid JSON from text: {raw_text}") from exc

    if not validate_inference_output(data):
        raise ValueError(f"Parsed JSON does not conform to inference schema: {data}")
    return data

