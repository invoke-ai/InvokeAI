"""Representative key layout of a *scaled* fp8 Z-Image transformer checkpoint.

Captured from `zImageTurboFP8Kijai_fp8ScaledE4m3fn.safetensors`. Unlike
`z_image_transformer_comfyui_keys`, which is a bf16 checkpoint, this one is ComfyUI "scaled fp8":
each quantized Linear carries an fp8 `<name>.weight` plus a scalar `<name>.scale_weight`, and the
file is marked with a stray `scaled_fp8` key.

Two things make it worth capturing. It spells the scale `.scale_weight` rather than
`.weight_scale` -- the spelling several loaders used to ignore -- and its scales are all *above*
one (1.5 to 7.6 in the real file), so dropping them leaves each weight at a different fraction of
its true magnitude rather than uniformly wrong.

Same subsetting rule as the sibling fixture: block 0 of each stack plus every non-block key.
Values are `(shape, dtype)`.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "cap_embedder.0.weight": ([2560], "F32"),
    "cap_embedder.1.bias": ([3840], "F32"),
    "cap_embedder.1.scale_weight": ([1], "F32"),
    "cap_embedder.1.weight": ([3840, 2560], "F8_E4M3"),
    "cap_pad_token": ([1, 3840], "F32"),
    "context_refiner.0.attention.k_norm.weight": ([128], "F32"),
    "context_refiner.0.attention.out.scale_weight": ([1], "F32"),
    "context_refiner.0.attention.out.weight": ([3840, 3840], "F8_E4M3"),
    "context_refiner.0.attention.q_norm.weight": ([128], "F32"),
    "context_refiner.0.attention.qkv.scale_weight": ([1], "F32"),
    "context_refiner.0.attention.qkv.weight": ([11520, 3840], "F8_E4M3"),
    "context_refiner.0.attention_norm1.weight": ([3840], "F32"),
    "context_refiner.0.attention_norm2.weight": ([3840], "F32"),
    "context_refiner.0.feed_forward.w1.scale_weight": ([1], "F32"),
    "context_refiner.0.feed_forward.w1.weight": ([10240, 3840], "F8_E4M3"),
    "context_refiner.0.feed_forward.w2.scale_weight": ([1], "F32"),
    "context_refiner.0.feed_forward.w2.weight": ([3840, 10240], "F8_E4M3"),
    "context_refiner.0.feed_forward.w3.scale_weight": ([1], "F32"),
    "context_refiner.0.feed_forward.w3.weight": ([10240, 3840], "F8_E4M3"),
    "context_refiner.0.ffn_norm1.weight": ([3840], "F32"),
    "context_refiner.0.ffn_norm2.weight": ([3840], "F32"),
    "final_layer.adaLN_modulation.1.bias": ([3840], "F32"),
    "final_layer.adaLN_modulation.1.scale_weight": ([1], "F32"),
    "final_layer.adaLN_modulation.1.weight": ([3840, 256], "F8_E4M3"),
    "final_layer.linear.bias": ([64], "F32"),
    "final_layer.linear.scale_weight": ([1], "F32"),
    "final_layer.linear.weight": ([64, 3840], "F8_E4M3"),
    "layers.0.adaLN_modulation.0.bias": ([15360], "F32"),
    "layers.0.adaLN_modulation.0.scale_weight": ([1], "F32"),
    "layers.0.adaLN_modulation.0.weight": ([15360, 256], "F8_E4M3"),
    "layers.0.attention.k_norm.weight": ([128], "F32"),
    "layers.0.attention.out.scale_weight": ([1], "F32"),
    "layers.0.attention.out.weight": ([3840, 3840], "F8_E4M3"),
    "layers.0.attention.q_norm.weight": ([128], "F32"),
    "layers.0.attention.qkv.scale_weight": ([1], "F32"),
    "layers.0.attention.qkv.weight": ([11520, 3840], "F8_E4M3"),
    "layers.0.attention_norm1.weight": ([3840], "F32"),
    "layers.0.attention_norm2.weight": ([3840], "F32"),
    "layers.0.feed_forward.w1.scale_weight": ([1], "F32"),
    "layers.0.feed_forward.w1.weight": ([10240, 3840], "F8_E4M3"),
    "layers.0.feed_forward.w2.scale_weight": ([1], "F32"),
    "layers.0.feed_forward.w2.weight": ([3840, 10240], "F8_E4M3"),
    "layers.0.feed_forward.w3.scale_weight": ([1], "F32"),
    "layers.0.feed_forward.w3.weight": ([10240, 3840], "F8_E4M3"),
    "layers.0.ffn_norm1.weight": ([3840], "F32"),
    "layers.0.ffn_norm2.weight": ([3840], "F32"),
    "noise_refiner.0.adaLN_modulation.0.bias": ([15360], "F32"),
    "noise_refiner.0.adaLN_modulation.0.scale_weight": ([1], "F32"),
    "noise_refiner.0.adaLN_modulation.0.weight": ([15360, 256], "F8_E4M3"),
    "noise_refiner.0.attention.k_norm.weight": ([128], "F32"),
    "noise_refiner.0.attention.out.scale_weight": ([1], "F32"),
    "noise_refiner.0.attention.out.weight": ([3840, 3840], "F8_E4M3"),
    "noise_refiner.0.attention.q_norm.weight": ([128], "F32"),
    "noise_refiner.0.attention.qkv.scale_weight": ([1], "F32"),
    "noise_refiner.0.attention.qkv.weight": ([11520, 3840], "F8_E4M3"),
    "noise_refiner.0.attention_norm1.weight": ([3840], "F32"),
    "noise_refiner.0.attention_norm2.weight": ([3840], "F32"),
    "noise_refiner.0.feed_forward.w1.scale_weight": ([1], "F32"),
    "noise_refiner.0.feed_forward.w1.weight": ([10240, 3840], "F8_E4M3"),
    "noise_refiner.0.feed_forward.w2.scale_weight": ([1], "F32"),
    "noise_refiner.0.feed_forward.w2.weight": ([3840, 10240], "F8_E4M3"),
    "noise_refiner.0.feed_forward.w3.scale_weight": ([1], "F32"),
    "noise_refiner.0.feed_forward.w3.weight": ([10240, 3840], "F8_E4M3"),
    "noise_refiner.0.ffn_norm1.weight": ([3840], "F32"),
    "noise_refiner.0.ffn_norm2.weight": ([3840], "F32"),
    "scaled_fp8": ([2], "F8_E4M3"),
    "t_embedder.mlp.0.bias": ([1024], "F32"),
    "t_embedder.mlp.0.scale_weight": ([1], "F32"),
    "t_embedder.mlp.0.weight": ([1024, 256], "F8_E4M3"),
    "t_embedder.mlp.2.bias": ([256], "F32"),
    "t_embedder.mlp.2.scale_weight": ([1], "F32"),
    "t_embedder.mlp.2.weight": ([256, 1024], "F8_E4M3"),
    "x_embedder.bias": ([3840], "F32"),
    "x_embedder.scale_weight": ([1], "F32"),
    "x_embedder.weight": ([3840, 64], "F8_E4M3"),
    "x_pad_token": ([1, 3840], "F32"),
}
