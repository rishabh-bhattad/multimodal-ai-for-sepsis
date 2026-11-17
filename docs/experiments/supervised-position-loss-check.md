# Supervised-position loss implementation check


Qwen3-VL training and validation project vocabulary logits only at positions
that predict supervised completion tokens. All prompt and image tokens still
pass through the transformer. Causal shifting, ignored/padded positions,
multi-token completions, and accumulated-token loss normalization are retained.
Model parameters, adapters, saved checkpoint shapes, inference generation, and
sepsis probability scoring are unchanged. Other model types keep the existing
Trainer loss path. W&B run configuration records
`loss_projection=supervised_positions_only`.

CPU tests on a tiny real Qwen3-VL model with LoRA matched full-logit loss and
adapter gradients within numerical tolerances, including image inputs, padding,
and gradient accumulation. A CPU train–validation–train cycle also passed.
Actual A100 memory savings and sustained resumed training remain unverified;
no GPU training was launched for these code checks.
