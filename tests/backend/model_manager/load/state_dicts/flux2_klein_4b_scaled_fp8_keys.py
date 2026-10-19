"""Representative key layout of the official scaled-fp8 FLUX.2 Klein 4B checkpoint.

Captured from `black-forest-labs/FLUX.2-klein-4b-fp8`. Kept alongside
`flux2_transformer_fp8mixed_keys` because that one comes from FLUX.2 **dev**, whose quantizer left
the fused `qkv` alone -- so it cannot exercise the one case that makes FLUX.2 harder than FLUX.1.

Here `double_blocks.0.img_attn.qkv.weight` is a fused `[9216, 3072]` fp8 tensor carrying a single
**scalar** `weight_scale`. diffusers wants three separate projections, so the weight is chunked and
the scalar has to be *copied* to all three -- splitting it would be wrong, and leaving it on the
fused path is worse: `attach_fp8_scales` then matches nothing and the three weights stay quantized
but unscaled, off by 1/weight_scale with nothing logged.

The layer flags arrive through the safetensors header (`_quantization_metadata`), which names
layers in the BFL scheme -- so they need the same one-to-many rename as the scales.

Subsetting rule as for the sibling fixtures: block 0 of each stack plus every non-block key.
Values are `(shape, dtype)`.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "double_blocks.0.img_attn.norm.key_norm.scale": ([128], "BF16"),
    "double_blocks.0.img_attn.norm.query_norm.scale": ([128], "BF16"),
    "double_blocks.0.img_attn.proj.input_scale": ([], "F32"),
    "double_blocks.0.img_attn.proj.weight": ([3072, 3072], "F8_E4M3"),
    "double_blocks.0.img_attn.proj.weight_scale": ([], "F32"),
    "double_blocks.0.img_attn.qkv.input_scale": ([], "F32"),
    "double_blocks.0.img_attn.qkv.weight": ([9216, 3072], "F8_E4M3"),
    "double_blocks.0.img_attn.qkv.weight_scale": ([], "F32"),
    "double_blocks.0.img_mlp.0.input_scale": ([], "F32"),
    "double_blocks.0.img_mlp.0.weight": ([18432, 3072], "F8_E4M3"),
    "double_blocks.0.img_mlp.0.weight_scale": ([], "F32"),
    "double_blocks.0.img_mlp.2.input_scale": ([], "F32"),
    "double_blocks.0.img_mlp.2.weight": ([3072, 9216], "F8_E4M3"),
    "double_blocks.0.img_mlp.2.weight_scale": ([], "F32"),
    "double_blocks.0.txt_attn.norm.key_norm.scale": ([128], "BF16"),
    "double_blocks.0.txt_attn.norm.query_norm.scale": ([128], "BF16"),
    "double_blocks.0.txt_attn.proj.input_scale": ([], "F32"),
    "double_blocks.0.txt_attn.proj.weight": ([3072, 3072], "F8_E4M3"),
    "double_blocks.0.txt_attn.proj.weight_scale": ([], "F32"),
    "double_blocks.0.txt_attn.qkv.input_scale": ([], "F32"),
    "double_blocks.0.txt_attn.qkv.weight": ([9216, 3072], "F8_E4M3"),
    "double_blocks.0.txt_attn.qkv.weight_scale": ([], "F32"),
    "double_blocks.0.txt_mlp.0.input_scale": ([], "F32"),
    "double_blocks.0.txt_mlp.0.weight": ([18432, 3072], "F8_E4M3"),
    "double_blocks.0.txt_mlp.0.weight_scale": ([], "F32"),
    "double_blocks.0.txt_mlp.2.input_scale": ([], "F32"),
    "double_blocks.0.txt_mlp.2.weight": ([3072, 9216], "F8_E4M3"),
    "double_blocks.0.txt_mlp.2.weight_scale": ([], "F32"),
    "double_stream_modulation_img.lin.weight": ([18432, 3072], "BF16"),
    "double_stream_modulation_txt.lin.weight": ([18432, 3072], "BF16"),
    "final_layer.adaLN_modulation.1.weight": ([6144, 3072], "BF16"),
    "final_layer.linear.weight": ([128, 3072], "BF16"),
    "img_in.weight": ([3072, 128], "BF16"),
    "single_blocks.0.linear1.input_scale": ([], "F32"),
    "single_blocks.0.linear1.weight": ([27648, 3072], "F8_E4M3"),
    "single_blocks.0.linear1.weight_scale": ([], "F32"),
    "single_blocks.0.linear2.input_scale": ([], "F32"),
    "single_blocks.0.linear2.weight": ([3072, 12288], "F8_E4M3"),
    "single_blocks.0.linear2.weight_scale": ([], "F32"),
    "single_blocks.0.norm.key_norm.scale": ([128], "BF16"),
    "single_blocks.0.norm.query_norm.scale": ([128], "BF16"),
    "single_stream_modulation.lin.weight": ([9216, 3072], "BF16"),
    "time_in.in_layer.weight": ([3072, 256], "BF16"),
    "time_in.out_layer.weight": ([3072, 3072], "BF16"),
    "txt_in.weight": ([3072, 7680], "BF16"),
}
