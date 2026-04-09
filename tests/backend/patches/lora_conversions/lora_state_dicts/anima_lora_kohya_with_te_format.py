# A sample state dict in the Kohya Anima LoRA format with Qwen3 text encoder layers.
# Contains both lora_unet_ (transformer) and lora_te_ (Qwen3 encoder) keys.
state_dict_keys: dict[str, list[int]] = {
    # Transformer block 0 - cross attention
    "lora_unet_blocks_0_cross_attn_k_proj.lora_down.weight": [8, 2048],
    "lora_unet_blocks_0_cross_attn_k_proj.lora_up.weight": [2048, 8],
    "lora_unet_blocks_0_cross_attn_k_proj.alpha": [],
    "lora_unet_blocks_0_cross_attn_q_proj.lora_down.weight": [8, 2048],
    "lora_unet_blocks_0_cross_attn_q_proj.lora_up.weight": [2048, 8],
    "lora_unet_blocks_0_cross_attn_q_proj.alpha": [],
    # Qwen3 text encoder layer 0 - self attention
    "lora_te_layers_0_self_attn_q_proj.lora_down.weight": [8, 1024],
    "lora_te_layers_0_self_attn_q_proj.lora_up.weight": [1024, 8],
    "lora_te_layers_0_self_attn_q_proj.alpha": [],
    "lora_te_layers_0_self_attn_k_proj.lora_down.weight": [8, 1024],
    "lora_te_layers_0_self_attn_k_proj.lora_up.weight": [1024, 8],
    "lora_te_layers_0_self_attn_k_proj.alpha": [],
    "lora_te_layers_0_self_attn_v_proj.lora_down.weight": [8, 1024],
    "lora_te_layers_0_self_attn_v_proj.lora_up.weight": [1024, 8],
    "lora_te_layers_0_self_attn_v_proj.alpha": [],
    "lora_te_layers_0_self_attn_o_proj.lora_down.weight": [8, 1024],
    "lora_te_layers_0_self_attn_o_proj.lora_up.weight": [1024, 8],
    "lora_te_layers_0_self_attn_o_proj.alpha": [],
    # Qwen3 text encoder layer 0 - MLP
    "lora_te_layers_0_mlp_gate_proj.lora_down.weight": [8, 1024],
    "lora_te_layers_0_mlp_gate_proj.lora_up.weight": [2816, 8],
    "lora_te_layers_0_mlp_gate_proj.alpha": [],
    "lora_te_layers_0_mlp_down_proj.lora_down.weight": [8, 2816],
    "lora_te_layers_0_mlp_down_proj.lora_up.weight": [1024, 8],
    "lora_te_layers_0_mlp_down_proj.alpha": [],
    "lora_te_layers_0_mlp_up_proj.lora_down.weight": [8, 1024],
    "lora_te_layers_0_mlp_up_proj.lora_up.weight": [2816, 8],
    "lora_te_layers_0_mlp_up_proj.alpha": [],
}
