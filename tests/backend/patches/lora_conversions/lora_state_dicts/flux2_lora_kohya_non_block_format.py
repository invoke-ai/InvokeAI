"""A FLUX.2 Klein 4B LoRA in the kohya format with BFL module names, a distillation extract at rank 16
(`flux2-klein-4b_kohya-bfl_pjmixers-distill-extract-rank16.safetensors`, issue #10030).

Captured on 2026-11-08 with `scripts/capture_state_dict_fixture.py`,
which reads the header only.

Subsetting rule: block 0 of `double_blocks` and of `single_blocks`, plus every key outside a block stack. Values are
`(shape, dtype)`.

Besides the block layers the file trains every non-block layer of the model: the input projections (`img_in`,
`txt_in` with 7680 inputs, which is what marks Klein 4B), the time embedder (`time_in_*`), the last layer
(`final_layer_linear` and `final_layer_adaLN_modulation_1`) and the FLUX.2 shared modulations
(`double_stream_modulation_{img,txt}_lin`, `single_stream_modulation_lin`). Klein has no `vector_in`. Names are
BFL's module paths with every period turned into an underscore, so `time_in_in_layer` is `time_in.in_layer`.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "lora_unet_double_blocks_0_img_attn_proj.alpha": ([], "BF16"),
    "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_double_blocks_0_img_attn_qkv.alpha": ([], "BF16"),
    "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": ([9216, 16], "BF16"),
    "lora_unet_double_blocks_0_img_mlp_0.alpha": ([], "BF16"),
    "lora_unet_double_blocks_0_img_mlp_0.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_double_blocks_0_img_mlp_0.lora_up.weight": ([18432, 16], "BF16"),
    "lora_unet_double_blocks_0_img_mlp_2.alpha": ([], "BF16"),
    "lora_unet_double_blocks_0_img_mlp_2.lora_down.weight": ([16, 9216], "BF16"),
    "lora_unet_double_blocks_0_img_mlp_2.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_double_blocks_0_txt_attn_proj.alpha": ([], "BF16"),
    "lora_unet_double_blocks_0_txt_attn_proj.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_double_blocks_0_txt_attn_proj.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_double_blocks_0_txt_attn_qkv.alpha": ([], "BF16"),
    "lora_unet_double_blocks_0_txt_attn_qkv.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_double_blocks_0_txt_attn_qkv.lora_up.weight": ([9216, 16], "BF16"),
    "lora_unet_double_blocks_0_txt_mlp_0.alpha": ([], "BF16"),
    "lora_unet_double_blocks_0_txt_mlp_0.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_double_blocks_0_txt_mlp_0.lora_up.weight": ([18432, 16], "BF16"),
    "lora_unet_double_blocks_0_txt_mlp_2.alpha": ([], "BF16"),
    "lora_unet_double_blocks_0_txt_mlp_2.lora_down.weight": ([16, 9216], "BF16"),
    "lora_unet_double_blocks_0_txt_mlp_2.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_double_stream_modulation_img_lin.alpha": ([], "BF16"),
    "lora_unet_double_stream_modulation_img_lin.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_double_stream_modulation_img_lin.lora_up.weight": ([18432, 16], "BF16"),
    "lora_unet_double_stream_modulation_txt_lin.alpha": ([], "BF16"),
    "lora_unet_double_stream_modulation_txt_lin.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_double_stream_modulation_txt_lin.lora_up.weight": ([18432, 16], "BF16"),
    "lora_unet_final_layer_adaLN_modulation_1.alpha": ([], "BF16"),
    "lora_unet_final_layer_adaLN_modulation_1.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_final_layer_adaLN_modulation_1.lora_up.weight": ([6144, 16], "BF16"),
    "lora_unet_final_layer_linear.alpha": ([], "BF16"),
    "lora_unet_final_layer_linear.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_final_layer_linear.lora_up.weight": ([128, 16], "BF16"),
    "lora_unet_img_in.alpha": ([], "BF16"),
    "lora_unet_img_in.lora_down.weight": ([16, 128], "BF16"),
    "lora_unet_img_in.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_single_blocks_0_linear1.alpha": ([], "BF16"),
    "lora_unet_single_blocks_0_linear1.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_single_blocks_0_linear1.lora_up.weight": ([27648, 16], "BF16"),
    "lora_unet_single_blocks_0_linear2.alpha": ([], "BF16"),
    "lora_unet_single_blocks_0_linear2.lora_down.weight": ([16, 12288], "BF16"),
    "lora_unet_single_blocks_0_linear2.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_single_stream_modulation_lin.alpha": ([], "BF16"),
    "lora_unet_single_stream_modulation_lin.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_single_stream_modulation_lin.lora_up.weight": ([9216, 16], "BF16"),
    "lora_unet_time_in_in_layer.alpha": ([], "BF16"),
    "lora_unet_time_in_in_layer.lora_down.weight": ([16, 256], "BF16"),
    "lora_unet_time_in_in_layer.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_time_in_out_layer.alpha": ([], "BF16"),
    "lora_unet_time_in_out_layer.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_time_in_out_layer.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_txt_in.alpha": ([], "BF16"),
    "lora_unet_txt_in.lora_down.weight": ([16, 7680], "BF16"),
    "lora_unet_txt_in.lora_up.weight": ([3072, 16], "BF16"),
}
