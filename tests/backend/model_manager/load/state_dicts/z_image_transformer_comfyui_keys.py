"""Representative ComfyUI key layout of a Z-Image transformer single-file checkpoint.

Captured from `zimageTurboBadmilk_v10.safetensors` (Z-Image Turbo) after stripping the
`model.diffusion_model.` prefix -- this is exactly the input `_convert_z_image_gguf_to_diffusers`
receives (the converter runs on both the checkpoint and GGUF paths after prefix stripping).
The full transformer has 453 keys (context_refiner / noise_refiner / layers stacks); this
fixture keeps block 0 of each stack plus all non-block keys, and adds a synthetic
`norm_final.weight` to exercise the skip branch.

Legacy layout uses fused `*.attention.qkv.*`, `*.attention.out.*`, `*.attention.q_norm/k_norm`,
`x_embedder.*`, `final_layer.*`, `x_pad_token`, `cap_pad_token`; diffusers expects split
`to_q/to_k/to_v`, `to_out.0`, `norm_q/norm_k`, `all_x_embedder.2-1.*`, `all_final_layer.2-1.*`.
"""

state_dict_keys: dict[str, list[int]] = {
    "cap_embedder.0.weight": [2560],
    "cap_embedder.1.bias": [3840],
    "cap_embedder.1.weight": [3840, 2560],
    "cap_pad_token": [1, 3840],
    "context_refiner.0.attention.k_norm.weight": [128],
    "context_refiner.0.attention.out.weight": [3840, 3840],
    "context_refiner.0.attention.q_norm.weight": [128],
    "context_refiner.0.attention.qkv.weight": [11520, 3840],
    "context_refiner.0.attention_norm1.weight": [3840],
    "context_refiner.0.attention_norm2.weight": [3840],
    "context_refiner.0.feed_forward.w1.weight": [10240, 3840],
    "context_refiner.0.feed_forward.w2.weight": [3840, 10240],
    "context_refiner.0.feed_forward.w3.weight": [10240, 3840],
    "context_refiner.0.ffn_norm1.weight": [3840],
    "context_refiner.0.ffn_norm2.weight": [3840],
    "final_layer.adaLN_modulation.1.bias": [3840],
    "final_layer.adaLN_modulation.1.weight": [3840, 256],
    "final_layer.linear.bias": [64],
    "final_layer.linear.weight": [64, 3840],
    "layers.0.adaLN_modulation.0.bias": [15360],
    "layers.0.adaLN_modulation.0.weight": [15360, 256],
    "layers.0.attention.k_norm.weight": [128],
    "layers.0.attention.out.weight": [3840, 3840],
    "layers.0.attention.q_norm.weight": [128],
    "layers.0.attention.qkv.weight": [11520, 3840],
    "layers.0.attention_norm1.weight": [3840],
    "layers.0.attention_norm2.weight": [3840],
    "layers.0.feed_forward.w1.weight": [10240, 3840],
    "layers.0.feed_forward.w2.weight": [3840, 10240],
    "layers.0.feed_forward.w3.weight": [10240, 3840],
    "layers.0.ffn_norm1.weight": [3840],
    "layers.0.ffn_norm2.weight": [3840],
    "noise_refiner.0.adaLN_modulation.0.bias": [15360],
    "noise_refiner.0.adaLN_modulation.0.weight": [15360, 256],
    "noise_refiner.0.attention.k_norm.weight": [128],
    "noise_refiner.0.attention.out.weight": [3840, 3840],
    "noise_refiner.0.attention.q_norm.weight": [128],
    "noise_refiner.0.attention.qkv.weight": [11520, 3840],
    "noise_refiner.0.attention_norm1.weight": [3840],
    "noise_refiner.0.attention_norm2.weight": [3840],
    "noise_refiner.0.feed_forward.w1.weight": [10240, 3840],
    "noise_refiner.0.feed_forward.w2.weight": [3840, 10240],
    "noise_refiner.0.feed_forward.w3.weight": [10240, 3840],
    "noise_refiner.0.ffn_norm1.weight": [3840],
    "noise_refiner.0.ffn_norm2.weight": [3840],
    "t_embedder.mlp.0.bias": [1024],
    "t_embedder.mlp.0.weight": [1024, 256],
    "t_embedder.mlp.2.bias": [256],
    "t_embedder.mlp.2.weight": [256, 1024],
    "x_embedder.bias": [3840],
    "x_embedder.weight": [3840, 64],
    "x_pad_token": [1, 3840],
    "norm_final.weight": [2304],
}
