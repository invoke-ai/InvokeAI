"""Representative key layout of a *mixed* fp8 FLUX.2 checkpoint.

Captured from Comfy-Org's `flux2_dev_fp8mixed.safetensors`. Two properties make it worth having
next to the all-fp8 fixtures:

- It is *mixed*: only some Linears are quantized, and the rest stay bf16 in the same file. A
  metadata filter that is too broad silently deletes those bf16 weights instead of only the scale
  bookkeeping, which no all-fp8 fixture can catch.
- Every quantized Linear carries a calibrated `.input_scale` next to its `.weight_scale`. That key
  has to be stripped as well, or `load_state_dict(..., strict=True)` rejects it.

Same subsetting rule as the sibling fixtures: block 0 of each stack plus every non-block key.
Values are `(shape, dtype)`.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "double_blocks.0.img_attn.norm.key_norm.scale": ([128], "BF16"),
    "double_blocks.0.img_attn.norm.query_norm.scale": ([128], "BF16"),
    "double_blocks.0.img_attn.proj.weight": ([6144, 6144], "BF16"),
    "double_blocks.0.img_attn.qkv.weight": ([18432, 6144], "BF16"),
    "double_blocks.0.img_mlp.0.input_scale": ([], "F32"),
    "double_blocks.0.img_mlp.0.weight": ([36864, 6144], "F8_E4M3"),
    "double_blocks.0.img_mlp.0.weight_scale": ([], "F32"),
    "double_blocks.0.img_mlp.2.input_scale": ([], "F32"),
    "double_blocks.0.img_mlp.2.weight": ([6144, 18432], "F8_E4M3"),
    "double_blocks.0.img_mlp.2.weight_scale": ([], "F32"),
    "double_blocks.0.txt_attn.norm.key_norm.scale": ([128], "BF16"),
    "double_blocks.0.txt_attn.norm.query_norm.scale": ([128], "BF16"),
    "double_blocks.0.txt_attn.proj.weight": ([6144, 6144], "BF16"),
    "double_blocks.0.txt_attn.qkv.weight": ([18432, 6144], "BF16"),
    "double_blocks.0.txt_mlp.0.input_scale": ([], "F32"),
    "double_blocks.0.txt_mlp.0.weight": ([36864, 6144], "F8_E4M3"),
    "double_blocks.0.txt_mlp.0.weight_scale": ([], "F32"),
    "double_blocks.0.txt_mlp.2.input_scale": ([], "F32"),
    "double_blocks.0.txt_mlp.2.weight": ([6144, 18432], "F8_E4M3"),
    "double_blocks.0.txt_mlp.2.weight_scale": ([], "F32"),
    "double_stream_modulation_img.lin.weight": ([36864, 6144], "BF16"),
    "double_stream_modulation_txt.lin.weight": ([36864, 6144], "BF16"),
    "final_layer.adaLN_modulation.1.weight": ([12288, 6144], "BF16"),
    "final_layer.linear.weight": ([128, 6144], "BF16"),
    "guidance_in.in_layer.weight": ([6144, 256], "BF16"),
    "guidance_in.out_layer.weight": ([6144, 6144], "BF16"),
    "img_in.weight": ([6144, 128], "BF16"),
    "single_blocks.0.linear1.input_scale": ([], "F32"),
    "single_blocks.0.linear1.weight": ([55296, 6144], "F8_E4M3"),
    "single_blocks.0.linear1.weight_scale": ([], "F32"),
    "single_blocks.0.linear2.input_scale": ([], "F32"),
    "single_blocks.0.linear2.weight": ([6144, 24576], "F8_E4M3"),
    "single_blocks.0.linear2.weight_scale": ([], "F32"),
    "single_blocks.0.norm.key_norm.scale": ([128], "BF16"),
    "single_blocks.0.norm.query_norm.scale": ([128], "BF16"),
    "single_stream_modulation.lin.weight": ([18432, 6144], "BF16"),
    "time_in.in_layer.weight": ([6144, 256], "BF16"),
    "time_in.out_layer.weight": ([6144, 6144], "BF16"),
    "txt_in.weight": ([6144, 15360], "BF16"),
}
