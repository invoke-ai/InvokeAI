"""Key layout of ostris' TurboTime LoRA for Ideogram 4 (ai-toolkit, rank 128).

Captured on 2026-10-11 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`ostris/ideogram_4_turbotime_lora/ideogram_4_turbotime_v1.safetensors`.

Subsetting rule: `diffusion_model.layers` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

ai-toolkit writes the vendored transformer's own names behind ComfyUI's `diffusion_model.` prefix, with the
attention projections fused (`attention.qkv`, 13824 rows) -- the names the single-file and GGUF loaders load
strict, so the conversion is one-to-one. No `alpha` tensors: alpha = rank.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.layers.0.adaln_modulation.lora_A.weight": ([128, 512], "BF16"),
    "diffusion_model.layers.0.adaln_modulation.lora_B.weight": ([18432, 128], "BF16"),
    "diffusion_model.layers.0.attention.o.lora_A.weight": ([128, 4608], "BF16"),
    "diffusion_model.layers.0.attention.o.lora_B.weight": ([4608, 128], "BF16"),
    "diffusion_model.layers.0.attention.qkv.lora_A.weight": ([128, 4608], "BF16"),
    "diffusion_model.layers.0.attention.qkv.lora_B.weight": ([13824, 128], "BF16"),
    "diffusion_model.layers.0.feed_forward.w1.lora_A.weight": ([128, 4608], "BF16"),
    "diffusion_model.layers.0.feed_forward.w1.lora_B.weight": ([12288, 128], "BF16"),
    "diffusion_model.layers.0.feed_forward.w2.lora_A.weight": ([128, 12288], "BF16"),
    "diffusion_model.layers.0.feed_forward.w2.lora_B.weight": ([4608, 128], "BF16"),
    "diffusion_model.layers.0.feed_forward.w3.lora_A.weight": ([128, 4608], "BF16"),
    "diffusion_model.layers.0.feed_forward.w3.lora_B.weight": ([12288, 128], "BF16"),
}
