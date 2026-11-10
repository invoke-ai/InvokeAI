"""Key layout of a diffusers-trained Ideogram 4 LoRA (`linoyts/ideogram4-yarn-art-bf16`, rank 32).

Captured on 2026-10-11 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`linoyts/ideogram4-yarn-art-bf16/pytorch_lora_weights.safetensors`.

Subsetting rule: `transformer.layers` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

diffusers' Ideogram 4 transformer splits the attention projections (`attention.to_{q,k,v}`, `to_out.0`), so the
conversion fuses q/k/v into one low-rank layer on the vendored `attention.qkv`. No `alpha` tensors; the embedded
`lora_adapter_metadata` has `lora_alpha` = `r`, i.e. scale 1.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "transformer.layers.0.attention.to_k.lora_A.weight": ([32, 4608], "BF16"),
    "transformer.layers.0.attention.to_k.lora_B.weight": ([4608, 32], "BF16"),
    "transformer.layers.0.attention.to_out.0.lora_A.weight": ([32, 4608], "BF16"),
    "transformer.layers.0.attention.to_out.0.lora_B.weight": ([4608, 32], "BF16"),
    "transformer.layers.0.attention.to_q.lora_A.weight": ([32, 4608], "BF16"),
    "transformer.layers.0.attention.to_q.lora_B.weight": ([4608, 32], "BF16"),
    "transformer.layers.0.attention.to_v.lora_A.weight": ([32, 4608], "BF16"),
    "transformer.layers.0.attention.to_v.lora_B.weight": ([4608, 32], "BF16"),
    "transformer.layers.0.feed_forward.w1.lora_A.weight": ([32, 4608], "BF16"),
    "transformer.layers.0.feed_forward.w1.lora_B.weight": ([12288, 32], "BF16"),
    "transformer.layers.0.feed_forward.w2.lora_A.weight": ([32, 12288], "BF16"),
    "transformer.layers.0.feed_forward.w2.lora_B.weight": ([4608, 32], "BF16"),
    "transformer.layers.0.feed_forward.w3.lora_A.weight": ([32, 4608], "BF16"),
    "transformer.layers.0.feed_forward.w3.lora_B.weight": ([12288, 32], "BF16"),
}
