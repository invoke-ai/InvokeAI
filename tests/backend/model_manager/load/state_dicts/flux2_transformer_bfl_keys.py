"""Representative BFL-format key layout of a FLUX.2 transformer single-file checkpoint.

Captured from `flux-2-klein-9b-kv.safetensors` (FLUX.2 Klein 9B, bf16). The full
checkpoint has 201 tensors (8 double blocks, 24 single blocks); this
fixture keeps every top-level key plus block 0 of the double/single stacks, which is enough
to exercise `convert_flux2_bfl_to_diffusers` (fused-QKV split, block renames, adaLN
scale/shift swap) and validate against a single-layer `Flux2Transformer2DModel`.

BFL layout uses `double_blocks.*`, `single_blocks.*`, `img_in`, `txt_in`, `time_in`,
`*_modulation.lin`, `final_layer.*`; diffusers expects `transformer_blocks.*`,
`single_transformer_blocks.*`, `x_embedder`, `context_embedder`, `time_guidance_embed.*`,
`proj_out`, `norm_out`.
"""

state_dict_keys: dict[str, list[int]] = {
    "double_blocks.0.img_attn.norm.key_norm.scale": [128],
    "double_blocks.0.img_attn.norm.query_norm.scale": [128],
    "double_blocks.0.img_attn.proj.weight": [4096, 4096],
    "double_blocks.0.img_attn.qkv.weight": [12288, 4096],
    "double_blocks.0.img_mlp.0.weight": [24576, 4096],
    "double_blocks.0.img_mlp.2.weight": [4096, 12288],
    "double_blocks.0.txt_attn.norm.key_norm.scale": [128],
    "double_blocks.0.txt_attn.norm.query_norm.scale": [128],
    "double_blocks.0.txt_attn.proj.weight": [4096, 4096],
    "double_blocks.0.txt_attn.qkv.weight": [12288, 4096],
    "double_blocks.0.txt_mlp.0.weight": [24576, 4096],
    "double_blocks.0.txt_mlp.2.weight": [4096, 12288],
    "double_stream_modulation_img.lin.weight": [24576, 4096],
    "double_stream_modulation_txt.lin.weight": [24576, 4096],
    "final_layer.adaLN_modulation.1.weight": [8192, 4096],
    "final_layer.linear.weight": [128, 4096],
    "img_in.weight": [4096, 128],
    "single_blocks.0.linear1.weight": [36864, 4096],
    "single_blocks.0.linear2.weight": [4096, 16384],
    "single_blocks.0.norm.key_norm.scale": [128],
    "single_blocks.0.norm.query_norm.scale": [128],
    "single_stream_modulation.lin.weight": [12288, 4096],
    "time_in.in_layer.weight": [4096, 256],
    "time_in.out_layer.weight": [4096, 4096],
    "txt_in.weight": [4096, 12288],
}
