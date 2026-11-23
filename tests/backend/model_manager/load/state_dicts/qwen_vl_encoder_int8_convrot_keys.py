"""Representative key layout of supermind's `int8_tensorwise` (+convrot) Qwen2.5-VL 7B encoder.

Captured on 2026-11-06 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`https://huggingface.co/supermind/int8_convrot_models/resolve/main/text_encoders/qwen_2.5_vl_7b_int8_convrot.safetensors`.

Subsetting rule: `model.layers` index 0, `visual.blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

The legacy ComfyUI names (`model.layers.*`, `visual.*`), which only reach the module tree through
`_remap_qwen_vl_checkpoint_keys`: the markers sit under the old names and have to follow it. The visual tower's
`mlp.down_proj` is bf16 with no marker because its input width, 3420, is not a multiple of the 256-wide rotation
group, so a rotated build has nothing to quantize there; the language model's projections are all int8."""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "lm_head.weight": ([152064, 3584], "BF16"),
    "model.embed_tokens.weight": ([152064, 3584], "BF16"),
    "model.layers.0.input_layernorm.weight": ([3584], "BF16"),
    "model.layers.0.mlp.down_proj.comfy_quant": ([72], "U8"),
    "model.layers.0.mlp.down_proj.weight": ([3584, 18944], "I8"),
    "model.layers.0.mlp.down_proj.weight_scale": ([3584, 1], "F32"),
    "model.layers.0.mlp.gate_proj.comfy_quant": ([72], "U8"),
    "model.layers.0.mlp.gate_proj.weight": ([18944, 3584], "I8"),
    "model.layers.0.mlp.gate_proj.weight_scale": ([18944, 1], "F32"),
    "model.layers.0.mlp.up_proj.comfy_quant": ([72], "U8"),
    "model.layers.0.mlp.up_proj.weight": ([18944, 3584], "I8"),
    "model.layers.0.mlp.up_proj.weight_scale": ([18944, 1], "F32"),
    "model.layers.0.post_attention_layernorm.weight": ([3584], "BF16"),
    "model.layers.0.self_attn.k_proj.bias": ([512], "BF16"),
    "model.layers.0.self_attn.k_proj.comfy_quant": ([72], "U8"),
    "model.layers.0.self_attn.k_proj.weight": ([512, 3584], "I8"),
    "model.layers.0.self_attn.k_proj.weight_scale": ([512, 1], "F32"),
    "model.layers.0.self_attn.o_proj.comfy_quant": ([72], "U8"),
    "model.layers.0.self_attn.o_proj.weight": ([3584, 3584], "I8"),
    "model.layers.0.self_attn.o_proj.weight_scale": ([3584, 1], "F32"),
    "model.layers.0.self_attn.q_proj.bias": ([3584], "BF16"),
    "model.layers.0.self_attn.q_proj.comfy_quant": ([72], "U8"),
    "model.layers.0.self_attn.q_proj.weight": ([3584, 3584], "I8"),
    "model.layers.0.self_attn.q_proj.weight_scale": ([3584, 1], "F32"),
    "model.layers.0.self_attn.v_proj.bias": ([512], "BF16"),
    "model.layers.0.self_attn.v_proj.comfy_quant": ([72], "U8"),
    "model.layers.0.self_attn.v_proj.weight": ([512, 3584], "I8"),
    "model.layers.0.self_attn.v_proj.weight_scale": ([512, 1], "F32"),
    "model.norm.weight": ([3584], "BF16"),
    "visual.blocks.0.attn.proj.bias": ([1280], "BF16"),
    "visual.blocks.0.attn.proj.comfy_quant": ([72], "U8"),
    "visual.blocks.0.attn.proj.weight": ([1280, 1280], "I8"),
    "visual.blocks.0.attn.proj.weight_scale": ([1280, 1], "F32"),
    "visual.blocks.0.attn.qkv.bias": ([3840], "BF16"),
    "visual.blocks.0.attn.qkv.comfy_quant": ([72], "U8"),
    "visual.blocks.0.attn.qkv.weight": ([3840, 1280], "I8"),
    "visual.blocks.0.attn.qkv.weight_scale": ([3840, 1], "F32"),
    "visual.blocks.0.mlp.down_proj.bias": ([1280], "BF16"),
    "visual.blocks.0.mlp.down_proj.weight": ([1280, 3420], "BF16"),
    "visual.blocks.0.mlp.gate_proj.bias": ([3420], "BF16"),
    "visual.blocks.0.mlp.gate_proj.comfy_quant": ([72], "U8"),
    "visual.blocks.0.mlp.gate_proj.weight": ([3420, 1280], "I8"),
    "visual.blocks.0.mlp.gate_proj.weight_scale": ([3420, 1], "F32"),
    "visual.blocks.0.mlp.up_proj.bias": ([3420], "BF16"),
    "visual.blocks.0.mlp.up_proj.comfy_quant": ([72], "U8"),
    "visual.blocks.0.mlp.up_proj.weight": ([3420, 1280], "I8"),
    "visual.blocks.0.mlp.up_proj.weight_scale": ([3420, 1], "F32"),
    "visual.blocks.0.norm1.weight": ([1280], "BF16"),
    "visual.blocks.0.norm2.weight": ([1280], "BF16"),
    "visual.merger.ln_q.weight": ([1280], "BF16"),
    "visual.merger.mlp.0.bias": ([5120], "BF16"),
    "visual.merger.mlp.0.weight": ([5120, 5120], "BF16"),
    "visual.merger.mlp.2.bias": ([3584], "BF16"),
    "visual.merger.mlp.2.weight": ([3584, 5120], "BF16"),
    "visual.patch_embed.proj.weight": ([1280, 3, 2, 14, 14], "BF16"),
}

# The per-layer `.comfy_quant` markers, decoded from the blobs themselves.
markers: dict[str, dict[str, object]] = {
    "model.layers.0.mlp.down_proj": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "model.layers.0.mlp.gate_proj": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "model.layers.0.mlp.up_proj": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "model.layers.0.self_attn.k_proj": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "model.layers.0.self_attn.o_proj": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "model.layers.0.self_attn.q_proj": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "model.layers.0.self_attn.v_proj": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "visual.blocks.0.attn.proj": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "visual.blocks.0.attn.qkv": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "visual.blocks.0.mlp.gate_proj": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "visual.blocks.0.mlp.up_proj": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
}
