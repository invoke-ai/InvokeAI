# Kohya-format Qwen Image LoRA state dict keys.
# Keys use the pattern: lora_unet_transformer_blocks_{N}_{sub_module}.{param}
# where sub_module uses underscores instead of dots.

state_dict_keys: dict[str, list[int]] = {
    # Block 0 - attention projections (LoKR format)
    "lora_unet_transformer_blocks_0_attn_to_k.lokr_w1": [3072, 16],
    "lora_unet_transformer_blocks_0_attn_to_k.lokr_w2": [16, 3072],
    "lora_unet_transformer_blocks_0_attn_to_k.alpha": [],
    "lora_unet_transformer_blocks_0_attn_to_q.lokr_w1": [3072, 16],
    "lora_unet_transformer_blocks_0_attn_to_q.lokr_w2": [16, 3072],
    "lora_unet_transformer_blocks_0_attn_to_q.alpha": [],
    "lora_unet_transformer_blocks_0_attn_to_v.lokr_w1": [3072, 16],
    "lora_unet_transformer_blocks_0_attn_to_v.lokr_w2": [16, 3072],
    "lora_unet_transformer_blocks_0_attn_to_v.alpha": [],
    "lora_unet_transformer_blocks_0_attn_to_out_0.lokr_w1": [3072, 16],
    "lora_unet_transformer_blocks_0_attn_to_out_0.lokr_w2": [16, 3072],
    "lora_unet_transformer_blocks_0_attn_to_out_0.alpha": [],
    # Block 0 - add projections (text stream)
    "lora_unet_transformer_blocks_0_attn_add_k_proj.lokr_w1": [3072, 16],
    "lora_unet_transformer_blocks_0_attn_add_k_proj.lokr_w2": [16, 3072],
    "lora_unet_transformer_blocks_0_attn_add_k_proj.alpha": [],
    # Block 0 - MLP
    "lora_unet_transformer_blocks_0_img_mlp_net_0_proj.lokr_w1": [12288, 16],
    "lora_unet_transformer_blocks_0_img_mlp_net_0_proj.lokr_w2": [16, 3072],
    "lora_unet_transformer_blocks_0_img_mlp_net_0_proj.alpha": [],
    "lora_unet_transformer_blocks_0_txt_mlp_net_2.lokr_w1": [3072, 16],
    "lora_unet_transformer_blocks_0_txt_mlp_net_2.lokr_w2": [16, 12288],
    "lora_unet_transformer_blocks_0_txt_mlp_net_2.alpha": [],
    # Block 1 - subset to keep test small
    "lora_unet_transformer_blocks_1_attn_to_k.lokr_w1": [3072, 16],
    "lora_unet_transformer_blocks_1_attn_to_k.lokr_w2": [16, 3072],
    "lora_unet_transformer_blocks_1_attn_to_k.alpha": [],
}
