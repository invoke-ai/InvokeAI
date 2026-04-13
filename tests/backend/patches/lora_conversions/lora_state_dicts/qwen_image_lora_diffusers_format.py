# Diffusers/PEFT-format Qwen Image LoRA state dict keys.
# Keys use the pattern: transformer_blocks.{N}.{sub_module}.{param}

state_dict_keys: dict[str, list[int]] = {
    # Block 0 - standard LoRA (lora_down/lora_up)
    "transformer_blocks.0.attn.to_k.lora_down.weight": [64, 3072],
    "transformer_blocks.0.attn.to_k.lora_up.weight": [3072, 64],
    "transformer_blocks.0.attn.to_k.alpha": [],
    "transformer_blocks.0.attn.to_q.lora_down.weight": [64, 3072],
    "transformer_blocks.0.attn.to_q.lora_up.weight": [3072, 64],
    "transformer_blocks.0.attn.to_q.alpha": [],
    # Block 1
    "transformer_blocks.1.attn.to_k.lora_down.weight": [64, 3072],
    "transformer_blocks.1.attn.to_k.lora_up.weight": [3072, 64],
    "transformer_blocks.1.attn.to_k.alpha": [],
}
