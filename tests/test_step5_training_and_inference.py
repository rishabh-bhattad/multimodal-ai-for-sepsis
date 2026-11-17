"""Unit tests for Step 5 training harness, adapter reload verification, and inference scoring."""

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import config
from src.step5_training import (
    apply_qlora_to_model,
    extract_sepsis_probability,
    format_inference_output,
    get_target_token_ids,
    get_training_arguments,
    save_adapter_checkpoint,
    validate_inference_output,
    verify_adapter_reload,
    SepsisTrainer,
    MultimodalDataCollator,
)


class DummyConfig(PretrainedConfig):
    model_type = "dummy"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DummyAttentionLM(nn.Module):
    """Minimal model mimicking Transformer linear projections for LoRA adapter attachment."""

    def __init__(self, hidden_dim: int = 32, vocab_size: int = 100):
        super().__init__()
        self.config = DummyConfig()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        # input_ids: (B, L)
        B, L = input_ids.shape
        x = torch.zeros((B, L, 32), dtype=torch.float32, device=input_ids.device)
        for i in range(min(32, input_ids.max().item() + 1)):
            x[..., i % 32] += (input_ids == i).float()

        h = self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
        logits = self.lm_head(self.o_proj(h))  # (B, L, vocab_size)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, 100), shift_labels.view(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits)


class MockTokenizer:
    """Mock tokenizer providing encode/save_pretrained."""

    def __init__(self, vocab_map=None):
        self.vocab_map = vocab_map or {
            config.STEP4_TARGET_COMPLETION_POSITIVE: [42],
            config.STEP4_TARGET_COMPLETION_NEGATIVE: [84],
        }

    def encode(self, text, add_special_tokens=False):
        return self.vocab_map.get(text, [1])

    def save_pretrained(self, save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "tokenizer_config.json", "w") as f:
            json.dump({"mock": True}, f)


def test_get_training_arguments(tmp_path):
    args = get_training_arguments(
        output_dir=tmp_path / "checkpoints",
        run_name="smoke_test_run",
        report_to="none",
        bf16=False,
        fp16=False,
    )
    assert args.output_dir == str(tmp_path / "checkpoints")
    assert args.learning_rate == config.STEP5_LEARNING_RATE  # 2e-4
    assert args.per_device_train_batch_size == config.STEP5_BATCH_SIZE  # 1
    assert args.gradient_accumulation_steps == config.STEP5_GRADIENT_ACCUMULATION_STEPS  # 16
    assert args.num_train_epochs == config.STEP5_EPOCHS  # 3
    assert args.run_name == "smoke_test_run"


def test_save_and_verify_adapter_reload(tmp_path):
    base_model = DummyAttentionLM(hidden_dim=32, vocab_size=100)
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    peft_model = apply_qlora_to_model(base_model)
    tokenizer = MockTokenizer()

    adapter_dir = tmp_path / "adapter_ckpt"
    saved_path = save_adapter_checkpoint(peft_model, adapter_dir, tokenizer=tokenizer)

    assert (saved_path / "adapter_config.json").exists()
    assert (saved_path / "tokenizer_config.json").exists()

    # Create fresh base model with identical base weights and verify reload
    fresh_base = DummyAttentionLM(hidden_dim=32, vocab_size=100)
    fresh_base.load_state_dict(base_state)
    sample_input = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    # Obtain expected logits from original peft_model
    with torch.no_grad():
        expected_out = peft_model(**sample_input)
        expected_logits = expected_out.logits

    is_valid = verify_adapter_reload(
        fresh_base,
        saved_path,
        sample_input,
        expected_logits=expected_logits,
        tolerance=1e-4,
    )
    assert is_valid is True


def test_extract_sepsis_probability_1d():
    # 1D logits: length 10
    logits = torch.zeros(10)
    logits[2] = 5.0   # pos_token_id = 2
    logits[5] = 5.0   # neg_token_id = 5
    prob_equal = extract_sepsis_probability(logits, pos_token_id=2, neg_token_id=5)
    assert pytest.approx(prob_equal, rel=1e-4) == 0.5

    # Positive dominates
    logits[2] = 20.0
    logits[5] = 0.0
    prob_pos = extract_sepsis_probability(logits, pos_token_id=2, neg_token_id=5)
    assert pytest.approx(prob_pos, abs=1e-4) == 1.0

    # Negative dominates
    logits[2] = 0.0
    logits[5] = 20.0
    prob_neg = extract_sepsis_probability(logits, pos_token_id=2, neg_token_id=5)
    assert pytest.approx(prob_neg, abs=1e-4) == 0.0


def test_extract_sepsis_probability_2d_and_3d():
    # 2D batch of 2
    logits_2d = torch.tensor([
        [0.0, 2.0, 0.0],   # sample 0: pos=2.0 (idx 1), neg=0.0 (idx 0)
        [1.0, -1.0, 0.0],  # sample 1: pos=-1.0 (idx 1), neg=1.0 (idx 0)
    ])
    probs = extract_sepsis_probability(logits_2d, pos_token_id=1, neg_token_id=0)
    assert isinstance(probs, list)
    assert len(probs) == 2
    # sample 0: exp(2)/(exp(2)+1) ~ 0.8808
    expected_0 = torch.sigmoid(torch.tensor(2.0)).item()
    assert pytest.approx(probs[0], rel=1e-3) == expected_0

    # 3D tensor: (batch_size=2, seq_len=3, vocab_size=5)
    logits_3d = torch.zeros((2, 3, 5))
    logits_3d[0, -1, 1] = 3.0  # pos at last token
    logits_3d[0, -1, 0] = 0.0  # neg at last token
    logits_3d[1, -1, 1] = 0.0
    logits_3d[1, -1, 0] = 3.0
    probs_3d = extract_sepsis_probability(logits_3d, pos_token_id=1, neg_token_id=0, target_pos_idx=-1)
    assert len(probs_3d) == 2
    assert probs_3d[0] > 0.95
    assert probs_3d[1] < 0.05


def test_get_target_token_ids():
    tokenizer = MockTokenizer({
        config.STEP4_TARGET_COMPLETION_POSITIVE: [101],
        config.STEP4_TARGET_COMPLETION_NEGATIVE: [102, 103],
    })
    pos_id, neg_id = get_target_token_ids(tokenizer)
    assert pos_id == [101]
    assert neg_id == [102, 103]

    # Empty token error
    broken_tokenizer = MockTokenizer({config.STEP4_TARGET_COMPLETION_POSITIVE: []})
    with pytest.raises(ValueError, match="empty token IDs"):
        get_target_token_ids(broken_tokenizer)


def test_format_inference_output():
    out = format_inference_output(
        sepsis_probability=0.87654,
        reasoning_chain="Heart rate 130, MAP 52, bilateral opacities.",
    )
    assert out["sepsis_probability"] == 0.8765
    assert out["reasoning_chain"] == "Heart rate 130, MAP 52, bilateral opacities."

    # Out of range probability
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        format_inference_output(1.5, "Some rationale")

    # Empty reasoning chain
    with pytest.raises(ValueError, match="non-empty string"):
        format_inference_output(0.5, "   ")


def test_validate_inference_output():
    valid = {"sepsis_probability": 0.72, "reasoning_chain": "Clinical rationale"}
    assert validate_inference_output(valid) is True

    # Missing keys
    assert validate_inference_output({"sepsis_probability": 0.72}) is False

    # Extra keys
    assert validate_inference_output({"sepsis_probability": 0.72, "reasoning_chain": "x", "extra": 1}) is False

    # Probability boolean (which is subclass of int in Python)
    assert validate_inference_output({"sepsis_probability": True, "reasoning_chain": "x"}) is False

    # Invalid probability range
    assert validate_inference_output({"sepsis_probability": -0.1, "reasoning_chain": "x"}) is False

    # Non-string reasoning
    assert validate_inference_output({"sepsis_probability": 0.5, "reasoning_chain": 123}) is False


class DummyDataset(torch.utils.data.Dataset):
    """Toy dataset for SepsisTrainer test."""

    def __init__(self):
        self.data = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, -100, 42]},
            {"input_ids": [4, 5, 6, 7], "attention_mask": [1, 1, 1, 1], "labels": [-100, -100, -100, 84]},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def test_sepsis_trainer_smoke(tmp_path):
    base_model = DummyAttentionLM(hidden_dim=32, vocab_size=100)
    peft_model = apply_qlora_to_model(base_model)
    dataset = DummyDataset()
    output_dir = tmp_path / "trainer_ckpt"

    training_args = get_training_arguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        report_to="none",
        bf16=False,
        fp16=False,
        save_strategy="no",
        eval_strategy="no",
    )

    trainer = SepsisTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=MultimodalDataCollator(pad_token_id=0),
    )

    # Execute 1 training step
    train_result = trainer.train()
    assert train_result is not None
    assert train_result.global_step > 0

    # Save model via trainer
    trainer.save_model(str(output_dir / "final_adapter"))
    assert (output_dir / "final_adapter" / "adapter_config.json").exists()


def _tiny_qwen_lora():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from src.step5_training import get_qlora_config
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, rope_parameters={"rope_type":"default","mrope_section":[1,1,2]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
            patch_size=2, spatial_merge_size=1, temporal_patch_size=1, out_hidden_size=32,
            num_position_embeddings=16, deepstack_visual_indexes=[]),
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    model = Qwen3VLForConditionalGeneration(cfg)
    return apply_qlora_to_model(model, get_qlora_config(
        r=2, lora_alpha=4, lora_dropout=0., target_modules=["q_proj","v_proj"]),
        prepare_for_kbit=False)


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("normalizer", [None, 9])
def test_qwen_selected_logits_match_full_loss_and_adapter_gradients(with_image, normalizer):
    from copy import deepcopy
    from src.step5_training import supervised_position_loss
    torch.manual_seed(9)
    full = _tiny_qwen_lora()
    selected = deepcopy(full)
    ids = torch.tensor([[1,62,60,63,3,4,5,6], [1,2,3,4,5,6,0,0]])
    if not with_image:
        ids[0,1:4] = torch.tensor([7,8,9])
    labels = torch.full_like(ids,-100)
    labels[0,6:] = ids[0,6:]  # Two-token completion.
    labels[1,5] = ids[1,5]   # Different completion length and right padding.
    batch = dict(input_ids=ids, attention_mask=ids.ne(0).long(), labels=labels)
    if with_image:
        mm_types = torch.zeros_like(ids)
        mm_types[0,2] = 1
        batch.update(pixel_values=torch.randn(1,12), image_grid_thw=torch.tensor([[1,1,1]]),
                     mm_token_type_ids=mm_types)
    kwargs = {} if normalizer is None else {"num_items_in_batch":normalizer}
    reference = full(**batch, use_cache=False, **kwargs).loss
    observed, output = supervised_position_loss(selected,batch,normalizer)
    torch.testing.assert_close(observed,reference,rtol=1e-5,atol=1e-6)
    assert output.logits.shape == (2,3,64)  # Union of positions predicting supervised tokens.
    reference.backward()
    observed.backward()
    grad_count = 0
    for (name,a),(other,b) in zip(full.named_parameters(), selected.named_parameters()):
        assert name == other
        if a.grad is not None:
            assert b.grad is not None
            torch.testing.assert_close(b.grad,a.grad,rtol=2e-4,atol=1e-6)
            grad_count += 1
    assert grad_count > 0


def test_selected_loss_gradient_accumulation_matches_full():
    from copy import deepcopy
    from src.step5_training import supervised_position_loss
    torch.manual_seed(12)
    full = _tiny_qwen_lora()
    selected = deepcopy(full)
    batches = []
    for target_length in [1,3]:
        ids = torch.tensor([[1,2,3,4,5,6]])
        labels = torch.full_like(ids,-100)
        labels[:,-target_length:] = ids[:,-target_length:]
        batches.append(dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels))
    for batch in batches:
        full(**batch,use_cache=False,num_items_in_batch=4).loss.backward()
        supervised_position_loss(selected,batch,4)[0].backward()
    for a,b in zip(full.parameters(),selected.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,rtol=2e-4,atol=1e-6)


def test_selected_trainer_train_eval_train_on_cpu(tmp_path):
    model = _tiny_qwen_lora()
    shapes = []
    hook = model.get_base_model().lm_head.register_forward_pre_hook(
        lambda module,args: shapes.append(args[0].shape[1]))
    data = [
        {"input_ids":[1,2,3,4,5,6],"attention_mask":[1]*6,"labels":[-100]*5+[6]},
        {"input_ids":[1,2,3,4,5,6],"attention_mask":[1]*6,"labels":[-100]*4+[5,6]},
    ]
    args = get_training_arguments(output_dir=tmp_path, use_cpu=True, bf16=False, fp16=False,
        max_steps=2, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=2, eval_strategy="steps", eval_steps=1,
        save_strategy="no", report_to="none", logging_steps=1)
    trainer = SepsisTrainer(model=model,args=args,train_dataset=data,eval_dataset=data,
                            data_collator=MultimodalDataCollator())
    assert trainer.supervised_logits_only
    assert trainer.train().global_step == 2
    hook.remove()
    assert shapes and max(shapes) <= 2
    assert len([x for x in trainer.state.log_history if "eval_loss" in x]) == 2
