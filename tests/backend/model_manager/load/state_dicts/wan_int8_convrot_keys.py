"""Representative key layout of the community `int8_tensorwise` Wan 2.2 T2V-14B high-noise build.

Captured on 2026-11-06 by range-reading the safetensors header (no tensor data) of
`FartingBackwards/wan2.2-t2v-14B-int8-convrot/wan2.2_t2v_high_noise_14B_int8_convrot.safetensors`.

Subsetting rule: `blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.
The full file has 40 blocks and 1895 tensors.

These are the native Wan names (`blocks.0.self_attn.q`, `ffn.0`, `text_embedding.0`), so the layout reaches the
module tree only through `_convert_wan_native_to_diffusers`. Only the self/cross attention projections and the two
FFN linears are int8; the `condition_embedder` linears (`text_embedding`, `time_embedding`, `time_projection`),
`patch_embedding` and `head` are dense F16.

The build declares its scheme twice: a `.comfy_quant` marker beside every quantized weight and an identical
`_quantization_metadata` header entry per layer. `convrot` and the group size are in both.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "blocks.0.cross_attn.k.bias": ([5120], "F16"),
    "blocks.0.cross_attn.k.comfy_quant": ([89], "U8"),
    "blocks.0.cross_attn.k.weight": ([5120, 5120], "I8"),
    "blocks.0.cross_attn.k.weight_scale": ([5120, 1], "F32"),
    "blocks.0.cross_attn.norm_k.weight": ([5120], "F16"),
    "blocks.0.cross_attn.norm_q.weight": ([5120], "F16"),
    "blocks.0.cross_attn.o.bias": ([5120], "F16"),
    "blocks.0.cross_attn.o.comfy_quant": ([89], "U8"),
    "blocks.0.cross_attn.o.weight": ([5120, 5120], "I8"),
    "blocks.0.cross_attn.o.weight_scale": ([5120, 1], "F32"),
    "blocks.0.cross_attn.q.bias": ([5120], "F16"),
    "blocks.0.cross_attn.q.comfy_quant": ([89], "U8"),
    "blocks.0.cross_attn.q.weight": ([5120, 5120], "I8"),
    "blocks.0.cross_attn.q.weight_scale": ([5120, 1], "F32"),
    "blocks.0.cross_attn.v.bias": ([5120], "F16"),
    "blocks.0.cross_attn.v.comfy_quant": ([89], "U8"),
    "blocks.0.cross_attn.v.weight": ([5120, 5120], "I8"),
    "blocks.0.cross_attn.v.weight_scale": ([5120, 1], "F32"),
    "blocks.0.ffn.0.bias": ([13824], "F16"),
    "blocks.0.ffn.0.comfy_quant": ([89], "U8"),
    "blocks.0.ffn.0.weight": ([13824, 5120], "I8"),
    "blocks.0.ffn.0.weight_scale": ([13824, 1], "F32"),
    "blocks.0.ffn.2.bias": ([5120], "F16"),
    "blocks.0.ffn.2.comfy_quant": ([89], "U8"),
    "blocks.0.ffn.2.weight": ([5120, 13824], "I8"),
    "blocks.0.ffn.2.weight_scale": ([5120, 1], "F32"),
    "blocks.0.modulation": ([1, 6, 5120], "F16"),
    "blocks.0.norm3.bias": ([5120], "F16"),
    "blocks.0.norm3.weight": ([5120], "F16"),
    "blocks.0.self_attn.k.bias": ([5120], "F16"),
    "blocks.0.self_attn.k.comfy_quant": ([89], "U8"),
    "blocks.0.self_attn.k.weight": ([5120, 5120], "I8"),
    "blocks.0.self_attn.k.weight_scale": ([5120, 1], "F32"),
    "blocks.0.self_attn.norm_k.weight": ([5120], "F16"),
    "blocks.0.self_attn.norm_q.weight": ([5120], "F16"),
    "blocks.0.self_attn.o.bias": ([5120], "F16"),
    "blocks.0.self_attn.o.comfy_quant": ([89], "U8"),
    "blocks.0.self_attn.o.weight": ([5120, 5120], "I8"),
    "blocks.0.self_attn.o.weight_scale": ([5120, 1], "F32"),
    "blocks.0.self_attn.q.bias": ([5120], "F16"),
    "blocks.0.self_attn.q.comfy_quant": ([89], "U8"),
    "blocks.0.self_attn.q.weight": ([5120, 5120], "I8"),
    "blocks.0.self_attn.q.weight_scale": ([5120, 1], "F32"),
    "blocks.0.self_attn.v.bias": ([5120], "F16"),
    "blocks.0.self_attn.v.comfy_quant": ([89], "U8"),
    "blocks.0.self_attn.v.weight": ([5120, 5120], "I8"),
    "blocks.0.self_attn.v.weight_scale": ([5120, 1], "F32"),
    "head.head.bias": ([64], "F16"),
    "head.head.weight": ([64, 5120], "F16"),
    "head.modulation": ([1, 2, 5120], "F16"),
    "patch_embedding.bias": ([5120], "F16"),
    "patch_embedding.weight": ([5120, 16, 1, 2, 2], "F16"),
    "text_embedding.0.bias": ([5120], "F16"),
    "text_embedding.0.weight": ([5120, 4096], "F16"),
    "text_embedding.2.bias": ([5120], "F16"),
    "text_embedding.2.weight": ([5120, 5120], "F16"),
    "time_embedding.0.bias": ([5120], "F16"),
    "time_embedding.0.weight": ([5120, 256], "F16"),
    "time_embedding.2.bias": ([5120], "F16"),
    "time_embedding.2.weight": ([5120, 5120], "F16"),
    "time_projection.1.bias": ([30720], "F16"),
    "time_projection.1.weight": ([30720, 5120], "F16"),
}

# The per-layer `.comfy_quant` markers, decoded from a blob read from the file (all ten blocks.0 layers share it).
MARKER: dict[str, object] = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256, "per_row": True}

markers: dict[str, dict[str, object]] = {
    "blocks.0.cross_attn.k": MARKER,
    "blocks.0.cross_attn.o": MARKER,
    "blocks.0.cross_attn.q": MARKER,
    "blocks.0.cross_attn.v": MARKER,
    "blocks.0.ffn.0": MARKER,
    "blocks.0.ffn.2": MARKER,
    "blocks.0.self_attn.k": MARKER,
    "blocks.0.self_attn.o": MARKER,
    "blocks.0.self_attn.q": MARKER,
    "blocks.0.self_attn.v": MARKER,
}

# The `_quantization_metadata` header entries for the same block.
header_layers: dict[str, dict[str, object]] = {
    "blocks.0.cross_attn.k": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.cross_attn.o": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.cross_attn.q": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.cross_attn.v": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.ffn.0": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.ffn.2": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.self_attn.k": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.self_attn.o": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.self_attn.q": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.self_attn.v": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
}
