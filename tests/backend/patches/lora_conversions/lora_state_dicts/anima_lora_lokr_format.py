# A sample state dict in the LoKR Anima LoRA format (with DoRA).
# Some Anima LoRAs use LoKR weights (lokr_w1/lokr_w2) combined with DoRA (dora_scale).
# The dora_scale should be stripped from LoKR layers during conversion.
state_dict_keys: dict[str, list[int]] = {
    # Block 0 - cross attention with LoKR + DoRA
    "diffusion_model.blocks.0.cross_attn.k_proj.lokr_w1": [2048, 8],
    "diffusion_model.blocks.0.cross_attn.k_proj.lokr_w2": [8, 2048],
    "diffusion_model.blocks.0.cross_attn.k_proj.alpha": [],
    "diffusion_model.blocks.0.cross_attn.k_proj.dora_scale": [2048],
    "diffusion_model.blocks.0.cross_attn.q_proj.lokr_w1": [2048, 8],
    "diffusion_model.blocks.0.cross_attn.q_proj.lokr_w2": [8, 2048],
    "diffusion_model.blocks.0.cross_attn.q_proj.alpha": [],
    "diffusion_model.blocks.0.cross_attn.q_proj.dora_scale": [2048],
    # Block 0 - self attention with LoKR (no DoRA)
    "diffusion_model.blocks.0.self_attn.k_proj.lokr_w1": [2048, 8],
    "diffusion_model.blocks.0.self_attn.k_proj.lokr_w2": [8, 2048],
    "diffusion_model.blocks.0.self_attn.k_proj.alpha": [],
}
