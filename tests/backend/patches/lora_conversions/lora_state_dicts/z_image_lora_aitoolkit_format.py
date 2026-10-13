"""Representative key layout of an ai-toolkit Z-Image LoRA.

Captured (header only, `scripts/capture_state_dict_fixture.py`) from
`tarn59/pixel_art_style_lora_z_image_turbo/pixel_art_style_z_image_turbo.safetensors` (metadata: `software=ai-toolkit`,
`ss_base_model_version=zimage`). The full adapter covers all 30 blocks (`attention.to_{q,k,v,out.0}`,
`feed_forward.w{1,2,3}`, `adaLN_modulation.0`) at rank 32 and width 3840; this fixture keeps the first and the last
block.

Z-Image keeps its blocks under `diffusion_model.layers.N`, the same container ERNIE-Image and Ideogram 4 use, so this
is the layout their probes must leave to Z-Image.
"""

state_dict_keys: dict[str, list[int]] = {
    "diffusion_model.layers.0.adaLN_modulation.0.lora_A.weight": [32, 256],
    "diffusion_model.layers.0.adaLN_modulation.0.lora_B.weight": [15360, 32],
    "diffusion_model.layers.0.attention.to_k.lora_A.weight": [32, 3840],
    "diffusion_model.layers.0.attention.to_k.lora_B.weight": [3840, 32],
    "diffusion_model.layers.0.attention.to_out.0.lora_A.weight": [32, 3840],
    "diffusion_model.layers.0.attention.to_out.0.lora_B.weight": [3840, 32],
    "diffusion_model.layers.0.attention.to_q.lora_A.weight": [32, 3840],
    "diffusion_model.layers.0.attention.to_q.lora_B.weight": [3840, 32],
    "diffusion_model.layers.0.attention.to_v.lora_A.weight": [32, 3840],
    "diffusion_model.layers.0.attention.to_v.lora_B.weight": [3840, 32],
    "diffusion_model.layers.0.feed_forward.w1.lora_A.weight": [32, 3840],
    "diffusion_model.layers.0.feed_forward.w1.lora_B.weight": [10240, 32],
    "diffusion_model.layers.0.feed_forward.w2.lora_A.weight": [32, 10240],
    "diffusion_model.layers.0.feed_forward.w2.lora_B.weight": [3840, 32],
    "diffusion_model.layers.0.feed_forward.w3.lora_A.weight": [32, 3840],
    "diffusion_model.layers.0.feed_forward.w3.lora_B.weight": [10240, 32],
    "diffusion_model.layers.29.adaLN_modulation.0.lora_A.weight": [32, 256],
    "diffusion_model.layers.29.adaLN_modulation.0.lora_B.weight": [15360, 32],
    "diffusion_model.layers.29.attention.to_k.lora_A.weight": [32, 3840],
    "diffusion_model.layers.29.attention.to_k.lora_B.weight": [3840, 32],
    "diffusion_model.layers.29.attention.to_out.0.lora_A.weight": [32, 3840],
    "diffusion_model.layers.29.attention.to_out.0.lora_B.weight": [3840, 32],
    "diffusion_model.layers.29.attention.to_q.lora_A.weight": [32, 3840],
    "diffusion_model.layers.29.attention.to_q.lora_B.weight": [3840, 32],
    "diffusion_model.layers.29.attention.to_v.lora_A.weight": [32, 3840],
    "diffusion_model.layers.29.attention.to_v.lora_B.weight": [3840, 32],
    "diffusion_model.layers.29.feed_forward.w1.lora_A.weight": [32, 3840],
    "diffusion_model.layers.29.feed_forward.w1.lora_B.weight": [10240, 32],
    "diffusion_model.layers.29.feed_forward.w2.lora_A.weight": [32, 10240],
    "diffusion_model.layers.29.feed_forward.w2.lora_B.weight": [3840, 32],
    "diffusion_model.layers.29.feed_forward.w3.lora_A.weight": [32, 3840],
    "diffusion_model.layers.29.feed_forward.w3.lora_B.weight": [10240, 32],
}
