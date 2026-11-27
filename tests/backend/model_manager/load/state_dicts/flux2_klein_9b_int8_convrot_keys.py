"""Representative key layout of the `int8_tensorwise` FLUX.2 Klein 9B repack.

Captured from `Winnougan/Klein9b-Distilled-Base-INT8-Convrot` (`flux-2-klein-9b-int8-convrot.safetensors`),
which is the community repack v7 installs -- Comfy-Org publishes no int8 build of FLUX.2.

Two things make this the interesting int8 fixture rather than a copy of the fp8 one:

- `double_blocks.0.img_attn.qkv.weight` is a fused **int8** tensor whose `weight_scale` is a
  scalar and whose `comfy_quant` marker is a byte blob. Both describe the whole fused tensor,
  so each of the three diffusers projections has to inherit them *unchanged* -- and the marker
  has to arrive at all, or `extract_int8_convrot_markers` claims nothing and the int8 codes are
  cast to bf16 as raw integers.
- The marker carries no `convrot` flag (`{"format": "int8_tensorwise"}` and nothing else), so
  this build is the unrotated half of the scheme. A loader that assumed rotation would
  derotate a weight that was never rotated, which loads cleanly and generates noise.

`img_in`, `txt_in`, `time_in`, `final_layer` and the three modulation `lin`s stay BF16 in this
repack -- the repacker's choice, not the loader's. `Flux2Transformer2DModel` declares only
`('pos_embed', 'norm')` as precision-sensitive, of which just `final_layer.adaLN_modulation.1`
(renamed to `norm_out.linear`) is a quantizable match, so a repack that also quantized the
embedders would keep them int8. That is safe here because nothing in this architecture reads
`weight.dtype` to decide what to cast its activations to -- the hazard those patterns exist for on
Z-Image and Ideogram 4.

Subsetting rule as for the sibling fixtures: block 0 of each stack plus every non-block key.
Values are `(shape, dtype)`.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "double_blocks.0.img_attn.norm.key_norm.scale": ([128], "BF16"),
    "double_blocks.0.img_attn.norm.query_norm.scale": ([128], "BF16"),
    "double_blocks.0.img_attn.proj.comfy_quant": ([29], "U8"),
    "double_blocks.0.img_attn.proj.weight": ([4096, 4096], "I8"),
    "double_blocks.0.img_attn.proj.weight_scale": ([], "F32"),
    "double_blocks.0.img_attn.qkv.comfy_quant": ([29], "U8"),
    "double_blocks.0.img_attn.qkv.weight": ([12288, 4096], "I8"),
    "double_blocks.0.img_attn.qkv.weight_scale": ([], "F32"),
    "double_blocks.0.img_mlp.0.comfy_quant": ([29], "U8"),
    "double_blocks.0.img_mlp.0.weight": ([24576, 4096], "I8"),
    "double_blocks.0.img_mlp.0.weight_scale": ([], "F32"),
    "double_blocks.0.img_mlp.2.comfy_quant": ([29], "U8"),
    "double_blocks.0.img_mlp.2.weight": ([4096, 12288], "I8"),
    "double_blocks.0.img_mlp.2.weight_scale": ([], "F32"),
    "double_blocks.0.txt_attn.norm.key_norm.scale": ([128], "BF16"),
    "double_blocks.0.txt_attn.norm.query_norm.scale": ([128], "BF16"),
    "double_blocks.0.txt_attn.proj.comfy_quant": ([29], "U8"),
    "double_blocks.0.txt_attn.proj.weight": ([4096, 4096], "I8"),
    "double_blocks.0.txt_attn.proj.weight_scale": ([], "F32"),
    "double_blocks.0.txt_attn.qkv.comfy_quant": ([29], "U8"),
    "double_blocks.0.txt_attn.qkv.weight": ([12288, 4096], "I8"),
    "double_blocks.0.txt_attn.qkv.weight_scale": ([], "F32"),
    "double_blocks.0.txt_mlp.0.comfy_quant": ([29], "U8"),
    "double_blocks.0.txt_mlp.0.weight": ([24576, 4096], "I8"),
    "double_blocks.0.txt_mlp.0.weight_scale": ([], "F32"),
    "double_blocks.0.txt_mlp.2.comfy_quant": ([29], "U8"),
    "double_blocks.0.txt_mlp.2.weight": ([4096, 12288], "I8"),
    "double_blocks.0.txt_mlp.2.weight_scale": ([], "F32"),
    "double_stream_modulation_img.lin.weight": ([24576, 4096], "BF16"),
    "double_stream_modulation_txt.lin.weight": ([24576, 4096], "BF16"),
    "final_layer.adaLN_modulation.1.weight": ([8192, 4096], "BF16"),
    "final_layer.linear.weight": ([128, 4096], "BF16"),
    "img_in.weight": ([4096, 128], "BF16"),
    "single_blocks.0.linear1.comfy_quant": ([29], "U8"),
    "single_blocks.0.linear1.weight": ([36864, 4096], "I8"),
    "single_blocks.0.linear1.weight_scale": ([], "F32"),
    "single_blocks.0.linear2.comfy_quant": ([29], "U8"),
    "single_blocks.0.linear2.weight": ([4096, 16384], "I8"),
    "single_blocks.0.linear2.weight_scale": ([], "F32"),
    "single_blocks.0.norm.key_norm.scale": ([128], "BF16"),
    "single_blocks.0.norm.query_norm.scale": ([128], "BF16"),
    "single_stream_modulation.lin.weight": ([12288, 4096], "BF16"),
    "time_in.in_layer.weight": ([4096, 256], "BF16"),
    "time_in.out_layer.weight": ([4096, 4096], "BF16"),
    "txt_in.weight": ([4096, 12288], "BF16"),
}
