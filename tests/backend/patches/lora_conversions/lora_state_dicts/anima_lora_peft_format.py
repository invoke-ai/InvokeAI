# A sample state dict in the diffusers PEFT Anima LoRA format.
# Keys follow the pattern: diffusion_model.blocks.{N}.{component}.lora_{A|B}.weight
state_dict_keys: dict[str, list[int]] = {
    # Block 0 - cross attention
    "diffusion_model.blocks.0.cross_attn.k_proj.lora_A.weight": [8, 2048],
    "diffusion_model.blocks.0.cross_attn.k_proj.lora_B.weight": [2048, 8],
    "diffusion_model.blocks.0.cross_attn.q_proj.lora_A.weight": [8, 2048],
    "diffusion_model.blocks.0.cross_attn.q_proj.lora_B.weight": [2048, 8],
    "diffusion_model.blocks.0.cross_attn.v_proj.lora_A.weight": [8, 2048],
    "diffusion_model.blocks.0.cross_attn.v_proj.lora_B.weight": [2048, 8],
    # Block 0 - self attention
    "diffusion_model.blocks.0.self_attn.k_proj.lora_A.weight": [8, 2048],
    "diffusion_model.blocks.0.self_attn.k_proj.lora_B.weight": [2048, 8],
    "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight": [8, 2048],
    "diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight": [2048, 8],
    # Block 0 - MLP
    "diffusion_model.blocks.0.mlp.layer1.lora_A.weight": [8, 2048],
    "diffusion_model.blocks.0.mlp.layer1.lora_B.weight": [8192, 8],
}
