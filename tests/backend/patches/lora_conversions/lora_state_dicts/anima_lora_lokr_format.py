# A sample state dict in the LoKR Anima LoRA format (with DoRA).
# Some Anima LoRAs use LoKR weights (lokr_w1/lokr_w2) combined with DoRA (dora_scale).
# Conversion keeps the dora_scale: each such layer is a weight-decomposed LoKR. The factors are those of the real
# modules (kron(w1, w2) is out x in): 2048 x 1024 for the cross-attention K, which reads the 1024-wide text context,
# and 2048 x 2048 for the others. A 1-D `dora_scale` is read per output row where that differs from the input size
# (the K here) and per input column on a square layer (the Q).
state_dict_keys: dict[str, list[int]] = {
    # Block 0 - cross attention with LoKR + DoRA
    "diffusion_model.blocks.0.cross_attn.k_proj.lokr_w1": [8, 8],
    "diffusion_model.blocks.0.cross_attn.k_proj.lokr_w2": [256, 128],
    "diffusion_model.blocks.0.cross_attn.k_proj.alpha": [],
    "diffusion_model.blocks.0.cross_attn.k_proj.dora_scale": [2048],
    "diffusion_model.blocks.0.cross_attn.q_proj.lokr_w1": [8, 8],
    "diffusion_model.blocks.0.cross_attn.q_proj.lokr_w2": [256, 256],
    "diffusion_model.blocks.0.cross_attn.q_proj.alpha": [],
    "diffusion_model.blocks.0.cross_attn.q_proj.dora_scale": [2048],
    # Block 0 - self attention with LoKR (no DoRA)
    "diffusion_model.blocks.0.self_attn.k_proj.lokr_w1": [8, 8],
    "diffusion_model.blocks.0.self_attn.k_proj.lokr_w2": [256, 256],
    "diffusion_model.blocks.0.self_attn.k_proj.alpha": [],
}
