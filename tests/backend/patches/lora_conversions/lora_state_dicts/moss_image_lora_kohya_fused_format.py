"""A MOSS-Image LoRA in kohya's names with a fused `proj.qkv`, ShootTheSound/Fizgig-MOSS-Image-SFT-Look-LoRA (F16).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`ShootTheSound/Fizgig-MOSS-Image-SFT-Look-LoRA/moss_image_sft_look_lora.safetensors`.

Subsetting rule: blocks [0, 41] (first, last, and one next to a missing block where the file has a gap) of the 42 it names, plus every key outside a block. Values are `(shape, dtype)`.

Like `moss_image_lora_kohya_format`, with the attention fused: `lora_unet_layers_N_proj_qkv` is one layer of
9216 rows. The file is an SVD extraction of SFT minus Base, so its rank follows the block, not the module: 128 in
blocks 0 to 20 and 896 in blocks 21 to 41 (the capture keeps blocks 0 and 41), and 512 in the text refiner.
`final_linear_video` is trained too, its `lora_up` 128 x 128. The fused `lora_up` is not
block-diagonal (checked on the values of the real file).
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "lora_unet_final_linear_video.alpha": ([], "F32"),
    "lora_unet_final_linear_video.lora_down.weight": ([128, 3072], "F16"),
    "lora_unet_final_linear_video.lora_up.weight": ([128, 128], "F16"),
    "lora_unet_layers_0_proj_ffn_down.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_ffn_down.lora_down.weight": ([128, 8192], "F16"),
    "lora_unet_layers_0_proj_ffn_down.lora_up.weight": ([3072, 128], "F16"),
    "lora_unet_layers_0_proj_ffn_up_gate.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_ffn_up_gate.lora_down.weight": ([128, 3072], "F16"),
    "lora_unet_layers_0_proj_ffn_up_gate.lora_up.weight": ([16384, 128], "F16"),
    "lora_unet_layers_0_proj_o.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_o.lora_down.weight": ([128, 3072], "F16"),
    "lora_unet_layers_0_proj_o.lora_up.weight": ([3072, 128], "F16"),
    "lora_unet_layers_0_proj_qkv.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_qkv.lora_down.weight": ([128, 3072], "F16"),
    "lora_unet_layers_0_proj_qkv.lora_up.weight": ([9216, 128], "F16"),
    "lora_unet_layers_41_proj_ffn_down.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_ffn_down.lora_down.weight": ([896, 8192], "F16"),
    "lora_unet_layers_41_proj_ffn_down.lora_up.weight": ([3072, 896], "F16"),
    "lora_unet_layers_41_proj_ffn_up_gate.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_ffn_up_gate.lora_down.weight": ([896, 3072], "F16"),
    "lora_unet_layers_41_proj_ffn_up_gate.lora_up.weight": ([16384, 896], "F16"),
    "lora_unet_layers_41_proj_o.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_o.lora_down.weight": ([896, 3072], "F16"),
    "lora_unet_layers_41_proj_o.lora_up.weight": ([3072, 896], "F16"),
    "lora_unet_layers_41_proj_qkv.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_qkv.lora_down.weight": ([896, 3072], "F16"),
    "lora_unet_layers_41_proj_qkv.lora_up.weight": ([9216, 896], "F16"),
    "lora_unet_text_embedder_blocks_0_ffn_down.alpha": ([], "F32"),
    "lora_unet_text_embedder_blocks_0_ffn_down.lora_down.weight": ([512, 8192], "F16"),
    "lora_unet_text_embedder_blocks_0_ffn_down.lora_up.weight": ([3072, 512], "F16"),
    "lora_unet_text_embedder_blocks_0_ffn_gate.alpha": ([], "F32"),
    "lora_unet_text_embedder_blocks_0_ffn_gate.lora_down.weight": ([512, 3072], "F16"),
    "lora_unet_text_embedder_blocks_0_ffn_gate.lora_up.weight": ([8192, 512], "F16"),
    "lora_unet_text_embedder_blocks_0_ffn_up.alpha": ([], "F32"),
    "lora_unet_text_embedder_blocks_0_ffn_up.lora_down.weight": ([512, 3072], "F16"),
    "lora_unet_text_embedder_blocks_0_ffn_up.lora_up.weight": ([8192, 512], "F16"),
    "lora_unet_text_embedder_blocks_0_proj.alpha": ([], "F32"),
    "lora_unet_text_embedder_blocks_0_proj.lora_down.weight": ([512, 3072], "F16"),
    "lora_unet_text_embedder_blocks_0_proj.lora_up.weight": ([3072, 512], "F16"),
    "lora_unet_text_embedder_blocks_0_qkv.alpha": ([], "F32"),
    "lora_unet_text_embedder_blocks_0_qkv.lora_down.weight": ([512, 3072], "F16"),
    "lora_unet_text_embedder_blocks_0_qkv.lora_up.weight": ([9216, 512], "F16"),
    "lora_unet_text_embedder_blocks_1_ffn_down.alpha": ([], "F32"),
    "lora_unet_text_embedder_blocks_1_ffn_down.lora_down.weight": ([512, 8192], "F16"),
    "lora_unet_text_embedder_blocks_1_ffn_down.lora_up.weight": ([3072, 512], "F16"),
    "lora_unet_text_embedder_blocks_1_ffn_gate.alpha": ([], "F32"),
    "lora_unet_text_embedder_blocks_1_ffn_gate.lora_down.weight": ([512, 3072], "F16"),
    "lora_unet_text_embedder_blocks_1_ffn_gate.lora_up.weight": ([8192, 512], "F16"),
    "lora_unet_text_embedder_blocks_1_ffn_up.alpha": ([], "F32"),
    "lora_unet_text_embedder_blocks_1_ffn_up.lora_down.weight": ([512, 3072], "F16"),
    "lora_unet_text_embedder_blocks_1_ffn_up.lora_up.weight": ([8192, 512], "F16"),
    "lora_unet_text_embedder_blocks_1_proj.alpha": ([], "F32"),
    "lora_unet_text_embedder_blocks_1_proj.lora_down.weight": ([512, 3072], "F16"),
    "lora_unet_text_embedder_blocks_1_proj.lora_up.weight": ([3072, 512], "F16"),
    "lora_unet_text_embedder_blocks_1_qkv.alpha": ([], "F32"),
    "lora_unet_text_embedder_blocks_1_qkv.lora_down.weight": ([512, 3072], "F16"),
    "lora_unet_text_embedder_blocks_1_qkv.lora_up.weight": ([9216, 512], "F16"),
}
