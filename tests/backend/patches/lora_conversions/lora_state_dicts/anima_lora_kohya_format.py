# A sample state dict in the Kohya Anima LoRA format.
# These keys are based on Anima LoRAs targeting the Cosmos Predict2 DiT transformer.
# Keys follow the pattern: lora_unet_blocks_{N}_{component}.{suffix}
state_dict_keys: dict[str, list[int]] = {
    # Block 0 - cross attention
    "lora_unet_blocks_0_cross_attn_k_proj.lora_down.weight": [8, 2048],
    "lora_unet_blocks_0_cross_attn_k_proj.lora_up.weight": [2048, 8],
    "lora_unet_blocks_0_cross_attn_k_proj.alpha": [],
    "lora_unet_blocks_0_cross_attn_q_proj.lora_down.weight": [8, 2048],
    "lora_unet_blocks_0_cross_attn_q_proj.lora_up.weight": [2048, 8],
    "lora_unet_blocks_0_cross_attn_q_proj.alpha": [],
    "lora_unet_blocks_0_cross_attn_v_proj.lora_down.weight": [8, 2048],
    "lora_unet_blocks_0_cross_attn_v_proj.lora_up.weight": [2048, 8],
    "lora_unet_blocks_0_cross_attn_v_proj.alpha": [],
    "lora_unet_blocks_0_cross_attn_output_proj.lora_down.weight": [8, 2048],
    "lora_unet_blocks_0_cross_attn_output_proj.lora_up.weight": [2048, 8],
    "lora_unet_blocks_0_cross_attn_output_proj.alpha": [],
    # Block 0 - self attention
    "lora_unet_blocks_0_self_attn_k_proj.lora_down.weight": [8, 2048],
    "lora_unet_blocks_0_self_attn_k_proj.lora_up.weight": [2048, 8],
    "lora_unet_blocks_0_self_attn_k_proj.alpha": [],
    "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": [8, 2048],
    "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": [2048, 8],
    "lora_unet_blocks_0_self_attn_q_proj.alpha": [],
    "lora_unet_blocks_0_self_attn_v_proj.lora_down.weight": [8, 2048],
    "lora_unet_blocks_0_self_attn_v_proj.lora_up.weight": [2048, 8],
    "lora_unet_blocks_0_self_attn_v_proj.alpha": [],
    "lora_unet_blocks_0_self_attn_output_proj.lora_down.weight": [8, 2048],
    "lora_unet_blocks_0_self_attn_output_proj.lora_up.weight": [2048, 8],
    "lora_unet_blocks_0_self_attn_output_proj.alpha": [],
    # Block 0 - MLP
    "lora_unet_blocks_0_mlp_layer1.lora_down.weight": [8, 2048],
    "lora_unet_blocks_0_mlp_layer1.lora_up.weight": [8192, 8],
    "lora_unet_blocks_0_mlp_layer1.alpha": [],
    "lora_unet_blocks_0_mlp_layer2.lora_down.weight": [8, 8192],
    "lora_unet_blocks_0_mlp_layer2.lora_up.weight": [2048, 8],
    "lora_unet_blocks_0_mlp_layer2.alpha": [],
    # Block 0 - adaln modulation
    "lora_unet_blocks_0_adaln_modulation_cross_attn_1.lora_down.weight": [8, 2048],
    "lora_unet_blocks_0_adaln_modulation_cross_attn_1.lora_up.weight": [2048, 8],
    "lora_unet_blocks_0_adaln_modulation_cross_attn_1.alpha": [],
}
