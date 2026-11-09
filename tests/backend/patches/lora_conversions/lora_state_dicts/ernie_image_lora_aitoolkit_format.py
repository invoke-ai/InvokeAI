"""Representative key layout of an ai-toolkit ERNIE-Image LoRA.

Captured (header only, `scripts/capture_state_dict_fixture.py`) from
`svntax-dev/pixel_assets_general_ernie_v1/pixel_assets_general_ernie_v1_000002500.safetensors` (metadata:
`software=ai-toolkit 0.9.14`, `ss_base_model_version=ernie_image`). The full adapter has 252 modules (all 36 blocks x
`self_attention.to_{q,k,v,out.0}` + `mlp.{gate_proj,up_proj,linear_fc2}`) at rank 32 and no `.alpha` tensors; this
fixture keeps the first and the last block. Civitai's ERNIE trainer writes the same layout.

ai-toolkit saves ERNIE-Image LoRAs under ComfyUI's `diffusion_model.` prefix over diffusers' own module names, which
is also what every ERNIE-Image transformer build loads into -- so a change in either naming would surface here.
"""

state_dict_keys: dict[str, list[int]] = {
    "diffusion_model.layers.0.mlp.gate_proj.lora_A.weight": [32, 4096],
    "diffusion_model.layers.0.mlp.gate_proj.lora_B.weight": [12288, 32],
    "diffusion_model.layers.0.mlp.linear_fc2.lora_A.weight": [32, 12288],
    "diffusion_model.layers.0.mlp.linear_fc2.lora_B.weight": [4096, 32],
    "diffusion_model.layers.0.mlp.up_proj.lora_A.weight": [32, 4096],
    "diffusion_model.layers.0.mlp.up_proj.lora_B.weight": [12288, 32],
    "diffusion_model.layers.0.self_attention.to_k.lora_A.weight": [32, 4096],
    "diffusion_model.layers.0.self_attention.to_k.lora_B.weight": [4096, 32],
    "diffusion_model.layers.0.self_attention.to_out.0.lora_A.weight": [32, 4096],
    "diffusion_model.layers.0.self_attention.to_out.0.lora_B.weight": [4096, 32],
    "diffusion_model.layers.0.self_attention.to_q.lora_A.weight": [32, 4096],
    "diffusion_model.layers.0.self_attention.to_q.lora_B.weight": [4096, 32],
    "diffusion_model.layers.0.self_attention.to_v.lora_A.weight": [32, 4096],
    "diffusion_model.layers.0.self_attention.to_v.lora_B.weight": [4096, 32],
    "diffusion_model.layers.35.mlp.gate_proj.lora_A.weight": [32, 4096],
    "diffusion_model.layers.35.mlp.gate_proj.lora_B.weight": [12288, 32],
    "diffusion_model.layers.35.mlp.linear_fc2.lora_A.weight": [32, 12288],
    "diffusion_model.layers.35.mlp.linear_fc2.lora_B.weight": [4096, 32],
    "diffusion_model.layers.35.mlp.up_proj.lora_A.weight": [32, 4096],
    "diffusion_model.layers.35.mlp.up_proj.lora_B.weight": [12288, 32],
    "diffusion_model.layers.35.self_attention.to_k.lora_A.weight": [32, 4096],
    "diffusion_model.layers.35.self_attention.to_k.lora_B.weight": [4096, 32],
    "diffusion_model.layers.35.self_attention.to_out.0.lora_A.weight": [32, 4096],
    "diffusion_model.layers.35.self_attention.to_out.0.lora_B.weight": [4096, 32],
    "diffusion_model.layers.35.self_attention.to_q.lora_A.weight": [32, 4096],
    "diffusion_model.layers.35.self_attention.to_q.lora_B.weight": [4096, 32],
    "diffusion_model.layers.35.self_attention.to_v.lora_A.weight": [32, 4096],
    "diffusion_model.layers.35.self_attention.to_v.lora_B.weight": [4096, 32],
}
