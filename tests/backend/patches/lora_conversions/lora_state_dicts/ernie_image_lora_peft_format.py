"""Representative key layout of a PEFT-format ERNIE-Image LoRA with top-level modules.

Captured (header only, `scripts/capture_state_dict_fixture.py`) from
`e-n-v-y/ERNIE-Image-Turbo-LoRA/ernie_image_turbo_diff_lora_dim128.safetensors` (metadata: `lora_format=peft`,
`alpha_baked_into_lora_B=true`, `base=baidu/ERNIE-Image`, `tuned=baidu/ERNIE-Image-Turbo`): an extraction of the
Turbo distillation as a LoRA over the base model. The full adapter has 258 modules: all 36 blocks plus the
top-level `adaLN_modulation.1`, `final_linear`, `final_norm.linear`, `text_proj` and `time_embedding.linear_{1,2}`,
at per-module ranks (128, but 64 for `final_linear`). This fixture keeps the first and the last block and every
top-level module.

It uses diffusers' `transformer.` prefix, and is the only published layout that reaches the transformer's
top-level modules -- the ones a block-only fixture cannot show missing from the conversion.
"""

state_dict_keys: dict[str, list[int]] = {
    "transformer.adaLN_modulation.1.lora_A.weight": [128, 4096],
    "transformer.adaLN_modulation.1.lora_B.weight": [24576, 128],
    "transformer.final_linear.lora_A.weight": [64, 4096],
    "transformer.final_linear.lora_B.weight": [128, 64],
    "transformer.final_norm.linear.lora_A.weight": [128, 4096],
    "transformer.final_norm.linear.lora_B.weight": [8192, 128],
    "transformer.layers.0.mlp.gate_proj.lora_A.weight": [128, 4096],
    "transformer.layers.0.mlp.gate_proj.lora_B.weight": [12288, 128],
    "transformer.layers.0.mlp.linear_fc2.lora_A.weight": [128, 12288],
    "transformer.layers.0.mlp.linear_fc2.lora_B.weight": [4096, 128],
    "transformer.layers.0.mlp.up_proj.lora_A.weight": [128, 4096],
    "transformer.layers.0.mlp.up_proj.lora_B.weight": [12288, 128],
    "transformer.layers.0.self_attention.to_k.lora_A.weight": [128, 4096],
    "transformer.layers.0.self_attention.to_k.lora_B.weight": [4096, 128],
    "transformer.layers.0.self_attention.to_out.0.lora_A.weight": [128, 4096],
    "transformer.layers.0.self_attention.to_out.0.lora_B.weight": [4096, 128],
    "transformer.layers.0.self_attention.to_q.lora_A.weight": [128, 4096],
    "transformer.layers.0.self_attention.to_q.lora_B.weight": [4096, 128],
    "transformer.layers.0.self_attention.to_v.lora_A.weight": [128, 4096],
    "transformer.layers.0.self_attention.to_v.lora_B.weight": [4096, 128],
    "transformer.layers.35.mlp.gate_proj.lora_A.weight": [128, 4096],
    "transformer.layers.35.mlp.gate_proj.lora_B.weight": [12288, 128],
    "transformer.layers.35.mlp.linear_fc2.lora_A.weight": [128, 12288],
    "transformer.layers.35.mlp.linear_fc2.lora_B.weight": [4096, 128],
    "transformer.layers.35.mlp.up_proj.lora_A.weight": [128, 4096],
    "transformer.layers.35.mlp.up_proj.lora_B.weight": [12288, 128],
    "transformer.layers.35.self_attention.to_k.lora_A.weight": [128, 4096],
    "transformer.layers.35.self_attention.to_k.lora_B.weight": [4096, 128],
    "transformer.layers.35.self_attention.to_out.0.lora_A.weight": [128, 4096],
    "transformer.layers.35.self_attention.to_out.0.lora_B.weight": [4096, 128],
    "transformer.layers.35.self_attention.to_q.lora_A.weight": [128, 4096],
    "transformer.layers.35.self_attention.to_q.lora_B.weight": [4096, 128],
    "transformer.layers.35.self_attention.to_v.lora_A.weight": [128, 4096],
    "transformer.layers.35.self_attention.to_v.lora_B.weight": [4096, 128],
    "transformer.text_proj.lora_A.weight": [128, 3072],
    "transformer.text_proj.lora_B.weight": [4096, 128],
    "transformer.time_embedding.linear_1.lora_A.weight": [128, 4096],
    "transformer.time_embedding.linear_1.lora_B.weight": [4096, 128],
    "transformer.time_embedding.linear_2.lora_A.weight": [128, 4096],
    "transformer.time_embedding.linear_2.lora_B.weight": [4096, 128],
}
