"""Key layout of Comfy-Org's single-file InstantX Qwen-Image ControlNet-Union.

Captured on 2026-10-18 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`Comfy-Org/Qwen-Image-InstantX-ControlNets/split_files/controlnet/Qwen-Image-InstantX-ControlNet-Union.safetensors`.

Subsetting rule: `controlnet_blocks` index 0, `transformer_blocks` index 0, plus every key outside a stack.
Values are `(shape, dtype)`.

A different design from the Fun branch: a trimmed copy of the Qwen-Image transformer itself (`img_in`,
`txt_in` at Qwen2.5-VL's 3584 width, `txt_norm`, `time_text_embed`, `transformer_blocks.N`) with
`controlnet_x_embedder` for the control latent and one `controlnet_blocks.N` output projection per block --
the key family InstantX's FLUX ControlNets use. It has no `control_img_in`, which is what keeps it out of
the Qwen-Image Fun config.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "controlnet_blocks.0.bias": ([3072], "BF16"),
    "controlnet_blocks.0.weight": ([3072, 3072], "BF16"),
    "controlnet_x_embedder.bias": ([3072], "BF16"),
    "controlnet_x_embedder.weight": ([3072, 64], "BF16"),
    "img_in.bias": ([3072], "BF16"),
    "img_in.weight": ([3072, 64], "BF16"),
    "time_text_embed.timestep_embedder.linear_1.bias": ([3072], "BF16"),
    "time_text_embed.timestep_embedder.linear_1.weight": ([3072, 256], "BF16"),
    "time_text_embed.timestep_embedder.linear_2.bias": ([3072], "BF16"),
    "time_text_embed.timestep_embedder.linear_2.weight": ([3072, 3072], "BF16"),
    "transformer_blocks.0.attn.add_k_proj.bias": ([3072], "BF16"),
    "transformer_blocks.0.attn.add_k_proj.weight": ([3072, 3072], "BF16"),
    "transformer_blocks.0.attn.add_q_proj.bias": ([3072], "BF16"),
    "transformer_blocks.0.attn.add_q_proj.weight": ([3072, 3072], "BF16"),
    "transformer_blocks.0.attn.add_v_proj.bias": ([3072], "BF16"),
    "transformer_blocks.0.attn.add_v_proj.weight": ([3072, 3072], "BF16"),
    "transformer_blocks.0.attn.norm_added_k.weight": ([128], "BF16"),
    "transformer_blocks.0.attn.norm_added_q.weight": ([128], "BF16"),
    "transformer_blocks.0.attn.norm_k.weight": ([128], "BF16"),
    "transformer_blocks.0.attn.norm_q.weight": ([128], "BF16"),
    "transformer_blocks.0.attn.to_add_out.bias": ([3072], "BF16"),
    "transformer_blocks.0.attn.to_add_out.weight": ([3072, 3072], "BF16"),
    "transformer_blocks.0.attn.to_k.bias": ([3072], "BF16"),
    "transformer_blocks.0.attn.to_k.weight": ([3072, 3072], "BF16"),
    "transformer_blocks.0.attn.to_out.0.bias": ([3072], "BF16"),
    "transformer_blocks.0.attn.to_out.0.weight": ([3072, 3072], "BF16"),
    "transformer_blocks.0.attn.to_q.bias": ([3072], "BF16"),
    "transformer_blocks.0.attn.to_q.weight": ([3072, 3072], "BF16"),
    "transformer_blocks.0.attn.to_v.bias": ([3072], "BF16"),
    "transformer_blocks.0.attn.to_v.weight": ([3072, 3072], "BF16"),
    "transformer_blocks.0.img_mlp.net.0.proj.bias": ([12288], "BF16"),
    "transformer_blocks.0.img_mlp.net.0.proj.weight": ([12288, 3072], "BF16"),
    "transformer_blocks.0.img_mlp.net.2.bias": ([3072], "BF16"),
    "transformer_blocks.0.img_mlp.net.2.weight": ([3072, 12288], "BF16"),
    "transformer_blocks.0.img_mod.1.bias": ([18432], "BF16"),
    "transformer_blocks.0.img_mod.1.weight": ([18432, 3072], "BF16"),
    "transformer_blocks.0.txt_mlp.net.0.proj.bias": ([12288], "BF16"),
    "transformer_blocks.0.txt_mlp.net.0.proj.weight": ([12288, 3072], "BF16"),
    "transformer_blocks.0.txt_mlp.net.2.bias": ([3072], "BF16"),
    "transformer_blocks.0.txt_mlp.net.2.weight": ([3072, 12288], "BF16"),
    "transformer_blocks.0.txt_mod.1.bias": ([18432], "BF16"),
    "transformer_blocks.0.txt_mod.1.weight": ([18432, 3072], "BF16"),
    "txt_in.bias": ([3072], "BF16"),
    "txt_in.weight": ([3072, 3584], "BF16"),
    "txt_norm.weight": ([3584], "BF16"),
}
