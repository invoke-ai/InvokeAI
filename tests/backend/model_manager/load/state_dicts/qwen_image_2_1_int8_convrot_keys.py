"""Representative key layout of Comfy-Org's `int8_tensorwise` (+convrot) Qwen-Image-2.1 transformer.

Captured on 2026-10-10 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`Comfy-Org/Qwen-Image-2.1/diffusion_models/qwen_image_2.1_int8_convrot.safetensors`.

Subsetting rule: `transformer_blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

The names are diffusers' except one: the gated MLP's two input projections are fused into
`img_mlp.gate_up` ([24576, 4096], gate rows first), with a per-output-channel `weight_scale` and its own
`.comfy_quant` marker. diffusers' module has them as `gate_layer` and `proj`. The loader has to split the
codes, the scale and the marker together *before* the int8 helpers read the side channel; split after,
the marker names a module that does not exist and the layer loads unrotated. Rotation is along the input
dimension (`convrot_groupsize` 256), so splitting the output rows keeps it valid.

No `__metadata__` block: the scheme lives only in the per-tensor markers, as in the other Comfy-Org builds.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "img_in.weight": ([4096, 64], "BF16"),
    "modulation.1.weight": ([16384, 4096], "BF16"),
    "norm_out.linear.weight": ([4096, 4096], "BF16"),
    "proj_out.weight": ([64, 4096], "BF16"),
    "time_text_embed.timestep_embedder.linear_1.weight": ([4096, 256], "BF16"),
    "time_text_embed.timestep_embedder.linear_2.weight": ([4096, 4096], "BF16"),
    "transformer_blocks.0.attn.norm_k.weight": ([128], "BF16"),
    "transformer_blocks.0.attn.norm_q.weight": ([128], "BF16"),
    "transformer_blocks.0.attn.to_k.comfy_quant": ([72], "U8"),
    "transformer_blocks.0.attn.to_k.weight": ([4096, 4096], "I8"),
    "transformer_blocks.0.attn.to_k.weight_scale": ([4096, 1], "F32"),
    "transformer_blocks.0.attn.to_out.0.comfy_quant": ([72], "U8"),
    "transformer_blocks.0.attn.to_out.0.weight": ([4096, 4096], "I8"),
    "transformer_blocks.0.attn.to_out.0.weight_scale": ([4096, 1], "F32"),
    "transformer_blocks.0.attn.to_q.comfy_quant": ([72], "U8"),
    "transformer_blocks.0.attn.to_q.weight": ([4096, 4096], "I8"),
    "transformer_blocks.0.attn.to_q.weight_scale": ([4096, 1], "F32"),
    "transformer_blocks.0.attn.to_v.comfy_quant": ([72], "U8"),
    "transformer_blocks.0.attn.to_v.weight": ([4096, 4096], "I8"),
    "transformer_blocks.0.attn.to_v.weight_scale": ([4096, 1], "F32"),
    "transformer_blocks.0.img_mlp.gate_up.comfy_quant": ([72], "U8"),
    "transformer_blocks.0.img_mlp.gate_up.weight": ([24576, 4096], "I8"),
    "transformer_blocks.0.img_mlp.gate_up.weight_scale": ([24576, 1], "F32"),
    "transformer_blocks.0.img_mlp.out.comfy_quant": ([72], "U8"),
    "transformer_blocks.0.img_mlp.out.weight": ([4096, 12288], "I8"),
    "transformer_blocks.0.img_mlp.out.weight_scale": ([4096, 1], "F32"),
    "txt_in.in_layer.weight": ([4096, 4096], "BF16"),
    "txt_in.out_layer.weight": ([4096, 4096], "BF16"),
    "txt_in.text_norm.weight": ([4096], "BF16"),
}

# The per-layer `.comfy_quant` markers, decoded from the blobs themselves.
markers: dict[str, dict[str, object]] = {
    "transformer_blocks.0.attn.to_k": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "transformer_blocks.0.attn.to_out.0": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "transformer_blocks.0.attn.to_q": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "transformer_blocks.0.attn.to_v": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "transformer_blocks.0.img_mlp.gate_up": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "transformer_blocks.0.img_mlp.out": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
}
