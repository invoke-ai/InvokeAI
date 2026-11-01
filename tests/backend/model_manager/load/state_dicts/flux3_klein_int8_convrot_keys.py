"""Comfy-Org's `int8_convrot` build of FLUX.3 Image Klein, `flux3-image-klein-int8-convrot.safetensors`.

Captured on 2026-10-31 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/feef60b0aaf63a4773ace96636ebbcf43b881312/diffusion_models/flux3-image-klein-int8-convrot.safetensors`.

Subsetting rule: `content_mode_blocks.image` index 0, `content_mode_blocks.image_cond` index 0, `single_blocks` index 0, `txt_mode_blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

Every block linear of all 43 blocks is `int8_tensorwise` + convrot, with a `[N, 1]` float32 scale and a
`.comfy_quant` marker each, and each block's query, key and value are fused into one `qkv` (`[9216, 3072]`, rows
in that order), which diffusers keeps apart: the loader splits the codes and the scales three ways and gives each
part the marker. The modulations, embedders and output layers stay bf16. `<block>.comfy_attention.config` (33 of
them, on the image and the joint blocks) asks for ComfyUI's own int8 attention kernel and says nothing about the
weights; the loader drops it. Klein Base's build has the same layout without `ref_time_in`.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "content_mode_blocks.image.0.attn_out.comfy_quant": ([72], "U8"),
    "content_mode_blocks.image.0.attn_out.weight": ([3072, 3072], "I8"),
    "content_mode_blocks.image.0.attn_out.weight_scale": ([3072, 1], "F32"),
    "content_mode_blocks.image.0.comfy_attention.config": ([35], "U8"),
    "content_mode_blocks.image.0.mlp_in.comfy_quant": ([72], "U8"),
    "content_mode_blocks.image.0.mlp_in.weight": ([18432, 3072], "I8"),
    "content_mode_blocks.image.0.mlp_in.weight_scale": ([18432, 1], "F32"),
    "content_mode_blocks.image.0.mlp_out.comfy_quant": ([72], "U8"),
    "content_mode_blocks.image.0.mlp_out.weight": ([3072, 9216], "I8"),
    "content_mode_blocks.image.0.mlp_out.weight_scale": ([3072, 1], "F32"),
    "content_mode_blocks.image.0.norm.key_norm.scale": ([128], "BF16"),
    "content_mode_blocks.image.0.norm.query_norm.scale": ([128], "BF16"),
    "content_mode_blocks.image.0.qkv.comfy_quant": ([72], "U8"),
    "content_mode_blocks.image.0.qkv.weight": ([9216, 3072], "I8"),
    "content_mode_blocks.image.0.qkv.weight_scale": ([9216, 1], "F32"),
    "content_mode_blocks.image_cond.0.attn_out.comfy_quant": ([72], "U8"),
    "content_mode_blocks.image_cond.0.attn_out.weight": ([3072, 3072], "I8"),
    "content_mode_blocks.image_cond.0.attn_out.weight_scale": ([3072, 1], "F32"),
    "content_mode_blocks.image_cond.0.mlp_in.comfy_quant": ([72], "U8"),
    "content_mode_blocks.image_cond.0.mlp_in.weight": ([18432, 3072], "I8"),
    "content_mode_blocks.image_cond.0.mlp_in.weight_scale": ([18432, 1], "F32"),
    "content_mode_blocks.image_cond.0.mlp_out.comfy_quant": ([72], "U8"),
    "content_mode_blocks.image_cond.0.mlp_out.weight": ([3072, 9216], "I8"),
    "content_mode_blocks.image_cond.0.mlp_out.weight_scale": ([3072, 1], "F32"),
    "content_mode_blocks.image_cond.0.norm.key_norm.scale": ([128], "BF16"),
    "content_mode_blocks.image_cond.0.norm.query_norm.scale": ([128], "BF16"),
    "content_mode_blocks.image_cond.0.qkv.comfy_quant": ([72], "U8"),
    "content_mode_blocks.image_cond.0.qkv.weight": ([9216, 3072], "I8"),
    "content_mode_blocks.image_cond.0.qkv.weight_scale": ([9216, 1], "F32"),
    "early_stream_modulations.image.lin.weight": ([9216, 3072], "BF16"),
    "early_stream_modulations.image_cond.lin.weight": ([9216, 3072], "BF16"),
    "early_stream_modulations.txt.lin.weight": ([9216, 3072], "BF16"),
    "emb_in.image.weight": ([3072, 128], "BF16"),
    "emb_in.image_cond.weight": ([3072, 128], "BF16"),
    "final_layer.image.adaLN_modulation.1.weight": ([6144, 3072], "BF16"),
    "final_layer.image.linear.weight": ([128, 3072], "BF16"),
    "ref_time_in.in_layer.weight": ([3072, 256], "BF16"),
    "ref_time_in.out_layer.weight": ([3072, 3072], "BF16"),
    "single_blocks.0.attn_out.comfy_quant": ([72], "U8"),
    "single_blocks.0.attn_out.weight": ([3072, 3072], "I8"),
    "single_blocks.0.attn_out.weight_scale": ([3072, 1], "F32"),
    "single_blocks.0.comfy_attention.config": ([35], "U8"),
    "single_blocks.0.mlp_in.comfy_quant": ([72], "U8"),
    "single_blocks.0.mlp_in.weight": ([18432, 3072], "I8"),
    "single_blocks.0.mlp_in.weight_scale": ([18432, 1], "F32"),
    "single_blocks.0.mlp_out.comfy_quant": ([72], "U8"),
    "single_blocks.0.mlp_out.weight": ([3072, 9216], "I8"),
    "single_blocks.0.mlp_out.weight_scale": ([3072, 1], "F32"),
    "single_blocks.0.norm.key_norm.scale": ([128], "BF16"),
    "single_blocks.0.norm.query_norm.scale": ([128], "BF16"),
    "single_blocks.0.qkv.comfy_quant": ([72], "U8"),
    "single_blocks.0.qkv.weight": ([9216, 3072], "I8"),
    "single_blocks.0.qkv.weight_scale": ([9216, 1], "F32"),
    "single_stream_modulations.image.lin.weight": ([9216, 3072], "BF16"),
    "single_stream_modulations.image_cond.lin.weight": ([9216, 3072], "BF16"),
    "single_stream_modulations.txt.lin.weight": ([9216, 3072], "BF16"),
    "time_in.in_layer.weight": ([3072, 256], "BF16"),
    "time_in.out_layer.weight": ([3072, 3072], "BF16"),
    "txt_in.weight": ([3072, 20480], "BF16"),
    "txt_mode_blocks.0.attn_out.comfy_quant": ([72], "U8"),
    "txt_mode_blocks.0.attn_out.weight": ([3072, 3072], "I8"),
    "txt_mode_blocks.0.attn_out.weight_scale": ([3072, 1], "F32"),
    "txt_mode_blocks.0.mlp_in.comfy_quant": ([72], "U8"),
    "txt_mode_blocks.0.mlp_in.weight": ([18432, 3072], "I8"),
    "txt_mode_blocks.0.mlp_in.weight_scale": ([18432, 1], "F32"),
    "txt_mode_blocks.0.mlp_out.comfy_quant": ([72], "U8"),
    "txt_mode_blocks.0.mlp_out.weight": ([3072, 9216], "I8"),
    "txt_mode_blocks.0.mlp_out.weight_scale": ([3072, 1], "F32"),
    "txt_mode_blocks.0.norm.key_norm.scale": ([128], "BF16"),
    "txt_mode_blocks.0.norm.query_norm.scale": ([128], "BF16"),
    "txt_mode_blocks.0.qkv.comfy_quant": ([72], "U8"),
    "txt_mode_blocks.0.qkv.weight": ([9216, 3072], "I8"),
    "txt_mode_blocks.0.qkv.weight_scale": ([9216, 1], "F32"),
}

# The per-layer `.comfy_quant` markers, decoded from the blobs themselves.
markers: dict[str, dict[str, object]] = {
    "content_mode_blocks.image.0.attn_out": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "content_mode_blocks.image.0.mlp_in": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "content_mode_blocks.image.0.mlp_out": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "content_mode_blocks.image.0.qkv": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "content_mode_blocks.image_cond.0.attn_out": {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
    },
    "content_mode_blocks.image_cond.0.mlp_in": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "content_mode_blocks.image_cond.0.mlp_out": {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": 256,
    },
    "content_mode_blocks.image_cond.0.qkv": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "single_blocks.0.attn_out": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "single_blocks.0.mlp_in": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "single_blocks.0.mlp_out": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "single_blocks.0.qkv": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "txt_mode_blocks.0.attn_out": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "txt_mode_blocks.0.mlp_in": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "txt_mode_blocks.0.mlp_out": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "txt_mode_blocks.0.qkv": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
}

# The rest of the header's `__metadata__`, as the producer wrote it. A producer that names its scheme
# only here -- NVIDIA ModelOpt's `quantization` -- is told apart by these alone.
metadata: dict[str, str] = {
    "dtype": "bfloat16",
    "format": "pt",
    "model": "f3-image-small",
}
