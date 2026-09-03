"""Representative key layout of a scaled-fp8 Anima checkpoint.

Captured from `pachiiahri/anima-fp8-comfyui` (`anima-preview_tcfp8_mixed`). It is the only
checkpoint in this series that ships **both** hint transports at once: a safetensors-header
`_quantization_metadata` block *and* per-layer `.comfy_quant` marker tensors. It also carries
exactly one `full_precision_matrix_mult` layer -- every other captured checkpoint marks either
none or a large fraction, so the single-marked case was previously untested.

The header names its layers `net.`-prefixed, i.e. in the checkpoint's own scheme, while the scales
are read after `_strip_anima_bundle_prefix` has run. Reading the header without renaming matches
nothing and drops every flag silently.

Subsetting rule: block 24 (the one carrying the marked layer) plus every non-block key. Values are
`(shape, dtype)`; `layer_hints` is the header block, subset the same way.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "net.blocks.24.adaln_modulation_cross_attn.1.weight": ([256, 2048], "BF16"),
    "net.blocks.24.adaln_modulation_cross_attn.2.weight": ([6144, 256], "BF16"),
    "net.blocks.24.adaln_modulation_mlp.1.weight": ([256, 2048], "BF16"),
    "net.blocks.24.adaln_modulation_mlp.2.weight": ([6144, 256], "BF16"),
    "net.blocks.24.adaln_modulation_self_attn.1.weight": ([256, 2048], "BF16"),
    "net.blocks.24.adaln_modulation_self_attn.2.weight": ([6144, 256], "BF16"),
    "net.blocks.24.cross_attn.k_norm.weight": ([128], "BF16"),
    "net.blocks.24.cross_attn.k_proj.comfy_quant": ([64], "U8"),
    "net.blocks.24.cross_attn.k_proj.input_scale": ([], "F32"),
    "net.blocks.24.cross_attn.k_proj.weight": ([2048, 1024], "F8_E4M3"),
    "net.blocks.24.cross_attn.k_proj.weight_scale": ([], "F32"),
    "net.blocks.24.cross_attn.output_proj.comfy_quant": ([64], "U8"),
    "net.blocks.24.cross_attn.output_proj.input_scale": ([], "F32"),
    "net.blocks.24.cross_attn.output_proj.weight": ([2048, 2048], "F8_E4M3"),
    "net.blocks.24.cross_attn.output_proj.weight_scale": ([], "F32"),
    "net.blocks.24.cross_attn.q_norm.weight": ([128], "BF16"),
    "net.blocks.24.cross_attn.q_proj.comfy_quant": ([63], "U8"),
    "net.blocks.24.cross_attn.q_proj.weight": ([2048, 2048], "F8_E4M3"),
    "net.blocks.24.cross_attn.q_proj.weight_scale": ([], "F32"),
    "net.blocks.24.cross_attn.v_proj.comfy_quant": ([64], "U8"),
    "net.blocks.24.cross_attn.v_proj.input_scale": ([], "F32"),
    "net.blocks.24.cross_attn.v_proj.weight": ([2048, 1024], "F8_E4M3"),
    "net.blocks.24.cross_attn.v_proj.weight_scale": ([], "F32"),
    "net.blocks.24.mlp.layer1.comfy_quant": ([64], "U8"),
    "net.blocks.24.mlp.layer1.input_scale": ([], "F32"),
    "net.blocks.24.mlp.layer1.weight": ([8192, 2048], "F8_E4M3"),
    "net.blocks.24.mlp.layer1.weight_scale": ([], "F32"),
    "net.blocks.24.mlp.layer2.comfy_quant": ([64], "U8"),
    "net.blocks.24.mlp.layer2.input_scale": ([], "F32"),
    "net.blocks.24.mlp.layer2.weight": ([2048, 8192], "F8_E4M3"),
    "net.blocks.24.mlp.layer2.weight_scale": ([], "F32"),
    "net.blocks.24.self_attn.k_norm.weight": ([128], "BF16"),
    "net.blocks.24.self_attn.k_proj.comfy_quant": ([64], "U8"),
    "net.blocks.24.self_attn.k_proj.input_scale": ([], "F32"),
    "net.blocks.24.self_attn.k_proj.weight": ([2048, 2048], "F8_E4M3"),
    "net.blocks.24.self_attn.k_proj.weight_scale": ([], "F32"),
    "net.blocks.24.self_attn.output_proj.comfy_quant": ([64], "U8"),
    "net.blocks.24.self_attn.output_proj.input_scale": ([], "F32"),
    "net.blocks.24.self_attn.output_proj.weight": ([2048, 2048], "F8_E4M3"),
    "net.blocks.24.self_attn.output_proj.weight_scale": ([], "F32"),
    "net.blocks.24.self_attn.q_norm.weight": ([128], "BF16"),
    "net.blocks.24.self_attn.q_proj.comfy_quant": ([64], "U8"),
    "net.blocks.24.self_attn.q_proj.input_scale": ([], "F32"),
    "net.blocks.24.self_attn.q_proj.weight": ([2048, 2048], "F8_E4M3"),
    "net.blocks.24.self_attn.q_proj.weight_scale": ([], "F32"),
    "net.blocks.24.self_attn.v_proj.comfy_quant": ([64], "U8"),
    "net.blocks.24.self_attn.v_proj.input_scale": ([], "F32"),
    "net.blocks.24.self_attn.v_proj.weight": ([2048, 2048], "F8_E4M3"),
    "net.blocks.24.self_attn.v_proj.weight_scale": ([], "F32"),
    "net.final_layer.adaln_modulation.1.weight": ([256, 2048], "BF16"),
    "net.final_layer.adaln_modulation.2.weight": ([4096, 256], "BF16"),
    "net.final_layer.linear.weight": ([64, 2048], "BF16"),
    "net.llm_adapter.embed.weight": ([32128, 1024], "BF16"),
    "net.llm_adapter.norm.weight": ([1024], "BF16"),
    "net.llm_adapter.out_proj.bias": ([1024], "BF16"),
    "net.llm_adapter.out_proj.weight": ([1024, 1024], "BF16"),
    "net.t_embedder.1.linear_1.weight": ([2048, 2048], "BF16"),
    "net.t_embedder.1.linear_2.weight": ([6144, 2048], "BF16"),
    "net.t_embedding_norm.weight": ([2048], "BF16"),
    "net.x_embedder.proj.1.weight": ([2048, 68], "BF16"),
}

# The `_quantization_metadata` header block, layer names exactly as the producer wrote them.
layer_hints: dict[str, dict[str, object]] = {
    "net.blocks.24.cross_attn.k_proj": {"format": "float8_e4m3fn", "full_precision_matrix_mult": False},
    "net.blocks.24.cross_attn.output_proj": {"format": "float8_e4m3fn", "full_precision_matrix_mult": False},
    "net.blocks.24.cross_attn.q_proj": {"format": "float8_e4m3fn", "full_precision_matrix_mult": True},
    "net.blocks.24.cross_attn.v_proj": {"format": "float8_e4m3fn", "full_precision_matrix_mult": False},
    "net.blocks.24.mlp.layer1": {"format": "float8_e4m3fn", "full_precision_matrix_mult": False},
    "net.blocks.24.mlp.layer2": {"format": "float8_e4m3fn", "full_precision_matrix_mult": False},
    "net.blocks.24.self_attn.k_proj": {"format": "float8_e4m3fn", "full_precision_matrix_mult": False},
    "net.blocks.24.self_attn.output_proj": {"format": "float8_e4m3fn", "full_precision_matrix_mult": False},
    "net.blocks.24.self_attn.q_proj": {"format": "float8_e4m3fn", "full_precision_matrix_mult": False},
    "net.blocks.24.self_attn.v_proj": {"format": "float8_e4m3fn", "full_precision_matrix_mult": False},
}
