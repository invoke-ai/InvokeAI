"""A PDD few-step FLUX.3 LoRA as PEFT saved it, NoiseToNova/Flux-3-Klein-turbo-pdd-4_8-steps (rank 128).

Captured on 2026-10-31 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`NoiseToNova/Flux-3-Klein-turbo-pdd-4_8-steps/Flux-3-Klein-turbo-pdd-v0.1-8_4step-lora-r128.safetensors`.

Subsetting rule: `mode_blocks.image` index 0, `single_transformer_blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

Keys are diffusers' module paths without a prefix, with PEFT's adapter name in each (`lora_A.default.weight`),
F32 and with no `alpha`. The LoRA targets the image stream only (its `pdd_config.json` excludes `image_cond` and
`txt`): every block's attention and feed-forward, `x_embedders.image`, the image modulations and diffusers' own
`norm_out.linear`. Beside it, `proj_out.weight` is a bank of 128 per-step output heads, 128 x 128 x 3072, which PDD's
sampler fuses per jump: no parameter of the transformer has that shape.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "early_stream_modulations.image.linear.lora_A.default.weight": ([128, 3072], "F32"),
    "early_stream_modulations.image.linear.lora_B.default.weight": ([9216, 128], "F32"),
    "mode_blocks.image.0.attn.to_k.lora_A.default.weight": ([128, 3072], "F32"),
    "mode_blocks.image.0.attn.to_k.lora_B.default.weight": ([3072, 128], "F32"),
    "mode_blocks.image.0.attn.to_out.0.lora_A.default.weight": ([128, 3072], "F32"),
    "mode_blocks.image.0.attn.to_out.0.lora_B.default.weight": ([3072, 128], "F32"),
    "mode_blocks.image.0.attn.to_q.lora_A.default.weight": ([128, 3072], "F32"),
    "mode_blocks.image.0.attn.to_q.lora_B.default.weight": ([3072, 128], "F32"),
    "mode_blocks.image.0.attn.to_v.lora_A.default.weight": ([128, 3072], "F32"),
    "mode_blocks.image.0.attn.to_v.lora_B.default.weight": ([3072, 128], "F32"),
    "mode_blocks.image.0.ff.linear_in.lora_A.default.weight": ([128, 3072], "F32"),
    "mode_blocks.image.0.ff.linear_in.lora_B.default.weight": ([18432, 128], "F32"),
    "mode_blocks.image.0.ff.linear_out.lora_A.default.weight": ([128, 9216], "F32"),
    "mode_blocks.image.0.ff.linear_out.lora_B.default.weight": ([3072, 128], "F32"),
    "norm_out.linear.lora_A.default.weight": ([128, 3072], "F32"),
    "norm_out.linear.lora_B.default.weight": ([6144, 128], "F32"),
    "proj_out.weight": ([128, 128, 3072], "F32"),
    "single_stream_modulations.image.linear.lora_A.default.weight": ([128, 3072], "F32"),
    "single_stream_modulations.image.linear.lora_B.default.weight": ([9216, 128], "F32"),
    "single_transformer_blocks.0.attn.to_k.lora_A.default.weight": ([128, 3072], "F32"),
    "single_transformer_blocks.0.attn.to_k.lora_B.default.weight": ([3072, 128], "F32"),
    "single_transformer_blocks.0.attn.to_out.0.lora_A.default.weight": ([128, 3072], "F32"),
    "single_transformer_blocks.0.attn.to_out.0.lora_B.default.weight": ([3072, 128], "F32"),
    "single_transformer_blocks.0.attn.to_q.lora_A.default.weight": ([128, 3072], "F32"),
    "single_transformer_blocks.0.attn.to_q.lora_B.default.weight": ([3072, 128], "F32"),
    "single_transformer_blocks.0.attn.to_v.lora_A.default.weight": ([128, 3072], "F32"),
    "single_transformer_blocks.0.attn.to_v.lora_B.default.weight": ([3072, 128], "F32"),
    "single_transformer_blocks.0.ff.linear_in.lora_A.default.weight": ([128, 3072], "F32"),
    "single_transformer_blocks.0.ff.linear_in.lora_B.default.weight": ([18432, 128], "F32"),
    "single_transformer_blocks.0.ff.linear_out.lora_A.default.weight": ([128, 9216], "F32"),
    "single_transformer_blocks.0.ff.linear_out.lora_B.default.weight": ([3072, 128], "F32"),
    "x_embedders.image.lora_A.default.weight": ([128, 128], "F32"),
    "x_embedders.image.lora_B.default.weight": ([3072, 128], "F32"),
}
