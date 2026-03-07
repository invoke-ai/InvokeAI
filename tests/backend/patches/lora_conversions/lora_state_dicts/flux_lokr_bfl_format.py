# A sample state dict in the BFL LOKR format (FLUX.1 hidden_size=3072).
# These keys represent a LOKR model using BFL internal key names with 'diffusion_model.' prefix.
state_dict_keys = {
    "diffusion_model.double_blocks.0.img_attn.proj.lokr_w1": [32, 96],
    "diffusion_model.double_blocks.0.img_attn.proj.lokr_w2": [32, 32],
    "diffusion_model.double_blocks.0.img_attn.proj.alpha": [],
    "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1": [32, 96],
    "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2": [32, 288],
    "diffusion_model.double_blocks.0.img_attn.qkv.alpha": [],
    "diffusion_model.double_blocks.0.img_mlp.0.lokr_w1": [32, 96],
    "diffusion_model.double_blocks.0.img_mlp.0.lokr_w2": [32, 128],
    "diffusion_model.double_blocks.0.img_mlp.0.alpha": [],
    "diffusion_model.double_blocks.0.img_mlp.2.lokr_w1": [32, 128],
    "diffusion_model.double_blocks.0.img_mlp.2.lokr_w2": [32, 96],
    "diffusion_model.double_blocks.0.img_mlp.2.alpha": [],
    "diffusion_model.single_blocks.0.linear1.lokr_w1": [32, 128],
    "diffusion_model.single_blocks.0.linear1.lokr_w2": [32, 128],
    "diffusion_model.single_blocks.0.linear1.alpha": [],
    "diffusion_model.single_blocks.0.linear2.lokr_w1": [32, 64],
    "diffusion_model.single_blocks.0.linear2.lokr_w2": [32, 48],
    "diffusion_model.single_blocks.0.linear2.alpha": [],
}
