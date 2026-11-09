"""Step 5: Model Training, QLoRA Configuration & Training Harness.

Implements model quantization, PEFT LoRA adapter initialization, parameter
counting, and dynamic multimodal sequence collation for the vision-language backbone.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig
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

