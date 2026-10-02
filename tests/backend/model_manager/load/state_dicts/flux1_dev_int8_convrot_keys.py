"""Representative key layout of the `int8_tensorwise` FLUX.1 [dev] repack.

Captured from `AX1Y2JP/FLUX.1-dev-INT8-ConvRot` (`flux1-dev-int8-convrot.safetensors`),
a community repack -- neither BFL nor Comfy-Org publishes an int8 build of FLUX.1.

This is the scheme in its usual form, unlike the FLUX.2 Klein fixture beside it: the marker is
`{"format": "int8_tensorwise", "per_row": true, "convrot": true, "convrot_groupsize": 256}`,
so the weights were rotated along their input dimension before quantization and the scales are
per-output-channel `[out, 1]` rather than scalars.

What it quantizes is its own choice and worth reading: both attention `qkv` tensors but
*neither* `attn.proj`, both MLPs, and every modulation `lin`. `img_in`, `txt_in`, `time_in`,
`vector_in`, `guidance_in` and `final_layer` stay BF16.

No key conversion stands between these names and the module tree -- `FluxCheckpointModel`
builds the BFL `Flux` class directly and `qkv` stays one fused Linear -- which is what makes
this the simple case next to FLUX.2.

Subsetting rule as for the sibling fixtures: block 0 of each stack plus every non-block key.
Values are `(shape, dtype)`.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "double_blocks.0.img_attn.norm.key_norm.scale": ([128], "BF16"),
    "double_blocks.0.img_attn.norm.query_norm.scale": ([128], "BF16"),
    "double_blocks.0.img_attn.proj.bias": ([3072], "BF16"),
    "double_blocks.0.img_attn.proj.weight": ([3072, 3072], "BF16"),
    "double_blocks.0.img_attn.qkv.bias": ([9216], "BF16"),
    "double_blocks.0.img_attn.qkv.comfy_quant": ([89], "U8"),
    "double_blocks.0.img_attn.qkv.weight": ([9216, 3072], "I8"),
    "double_blocks.0.img_attn.qkv.weight_scale": ([9216, 1], "F32"),
    "double_blocks.0.img_mlp.0.bias": ([12288], "BF16"),
    "double_blocks.0.img_mlp.0.comfy_quant": ([89], "U8"),
    "double_blocks.0.img_mlp.0.weight": ([12288, 3072], "I8"),
    "double_blocks.0.img_mlp.0.weight_scale": ([12288, 1], "F32"),
    "double_blocks.0.img_mlp.2.bias": ([3072], "BF16"),
    "double_blocks.0.img_mlp.2.comfy_quant": ([89], "U8"),
    "double_blocks.0.img_mlp.2.weight": ([3072, 12288], "I8"),
    "double_blocks.0.img_mlp.2.weight_scale": ([3072, 1], "F32"),
    "double_blocks.0.img_mod.lin.bias": ([18432], "BF16"),
    "double_blocks.0.img_mod.lin.comfy_quant": ([89], "U8"),
    "double_blocks.0.img_mod.lin.weight": ([18432, 3072], "I8"),
    "double_blocks.0.img_mod.lin.weight_scale": ([18432, 1], "F32"),
    "double_blocks.0.txt_attn.norm.key_norm.scale": ([128], "BF16"),
    "double_blocks.0.txt_attn.norm.query_norm.scale": ([128], "BF16"),
    "double_blocks.0.txt_attn.proj.bias": ([3072], "BF16"),
    "double_blocks.0.txt_attn.proj.weight": ([3072, 3072], "BF16"),
    "double_blocks.0.txt_attn.qkv.bias": ([9216], "BF16"),
    "double_blocks.0.txt_attn.qkv.comfy_quant": ([89], "U8"),
    "double_blocks.0.txt_attn.qkv.weight": ([9216, 3072], "I8"),
    "double_blocks.0.txt_attn.qkv.weight_scale": ([9216, 1], "F32"),
    "double_blocks.0.txt_mlp.0.bias": ([12288], "BF16"),
    "double_blocks.0.txt_mlp.0.comfy_quant": ([89], "U8"),
    "double_blocks.0.txt_mlp.0.weight": ([12288, 3072], "I8"),
    "double_blocks.0.txt_mlp.0.weight_scale": ([12288, 1], "F32"),
    "double_blocks.0.txt_mlp.2.bias": ([3072], "BF16"),
    "double_blocks.0.txt_mlp.2.comfy_quant": ([89], "U8"),
    "double_blocks.0.txt_mlp.2.weight": ([3072, 12288], "I8"),
    "double_blocks.0.txt_mlp.2.weight_scale": ([3072, 1], "F32"),
    "double_blocks.0.txt_mod.lin.bias": ([18432], "BF16"),
    "double_blocks.0.txt_mod.lin.comfy_quant": ([89], "U8"),
    "double_blocks.0.txt_mod.lin.weight": ([18432, 3072], "I8"),
    "double_blocks.0.txt_mod.lin.weight_scale": ([18432, 1], "F32"),
    "final_layer.adaLN_modulation.1.bias": ([6144], "BF16"),
    "final_layer.adaLN_modulation.1.weight": ([6144, 3072], "BF16"),
    "final_layer.linear.bias": ([64], "BF16"),
    "final_layer.linear.weight": ([64, 3072], "BF16"),
    "guidance_in.in_layer.bias": ([3072], "BF16"),
    "guidance_in.in_layer.weight": ([3072, 256], "BF16"),
    "guidance_in.out_layer.bias": ([3072], "BF16"),
    "guidance_in.out_layer.weight": ([3072, 3072], "BF16"),
    "img_in.bias": ([3072], "BF16"),
    "img_in.weight": ([3072, 64], "BF16"),
    "single_blocks.0.linear1.bias": ([21504], "BF16"),
    "single_blocks.0.linear1.comfy_quant": ([89], "U8"),
    "single_blocks.0.linear1.weight": ([21504, 3072], "I8"),
    "single_blocks.0.linear1.weight_scale": ([21504, 1], "F32"),
    "single_blocks.0.linear2.bias": ([3072], "BF16"),
    "single_blocks.0.linear2.comfy_quant": ([89], "U8"),
    "single_blocks.0.linear2.weight": ([3072, 15360], "I8"),
    "single_blocks.0.linear2.weight_scale": ([3072, 1], "F32"),
    "single_blocks.0.modulation.lin.bias": ([9216], "BF16"),
    "single_blocks.0.modulation.lin.comfy_quant": ([89], "U8"),
    "single_blocks.0.modulation.lin.weight": ([9216, 3072], "I8"),
    "single_blocks.0.modulation.lin.weight_scale": ([9216, 1], "F32"),
    "single_blocks.0.norm.key_norm.scale": ([128], "BF16"),
    "single_blocks.0.norm.query_norm.scale": ([128], "BF16"),
    "time_in.in_layer.bias": ([3072], "BF16"),
    "time_in.in_layer.weight": ([3072, 256], "BF16"),
    "time_in.out_layer.bias": ([3072], "BF16"),
    "time_in.out_layer.weight": ([3072, 3072], "BF16"),
    "txt_in.bias": ([3072], "BF16"),
    "txt_in.weight": ([3072, 4096], "BF16"),
    "vector_in.in_layer.bias": ([3072], "BF16"),
    "vector_in.in_layer.weight": ([3072, 768], "BF16"),
    "vector_in.out_layer.bias": ([3072], "BF16"),
    "vector_in.out_layer.weight": ([3072, 3072], "BF16"),
}
