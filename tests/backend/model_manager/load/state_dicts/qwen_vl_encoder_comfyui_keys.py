"""Representative key layout of a ComfyUI single-file Qwen2.5-VL encoder checkpoint.

Captured from `qwen_2.5_vl_7b_fp8_scaled.safetensors` (Qwen2.5-VL-7B, ComfyUI fp8_scaled).
The full checkpoint has 1446 tensors (32 visual blocks, 28 language layers); this
fixture keeps every top-level/structural key plus block 0 of each repeated stack, which is
enough to exercise the `visual.* -> model.visual.*` / `model.* -> model.language_model.*`
remap and the fp8 metadata stripping without shipping a ~1446-key dict.

Legacy ComfyUI layout uses `visual.*`, `model.*`, `lm_head.*` (transformers >=4.50 expects
`model.visual.*` and `model.language_model.*`). `scale_weight` / `scale_input` / `scaled_fp8`
are ComfyUI fp8 quantization metadata.
"""

state_dict_keys: dict[str, list[int]] = {
    "visual.blocks.0.attn.proj.bias": [1280],
    "visual.blocks.0.attn.proj.scale_input": [],
    "visual.blocks.0.attn.proj.scale_weight": [],
    "visual.blocks.0.attn.proj.weight": [1280, 1280],
    "visual.blocks.0.attn.qkv.bias": [3840],
    "visual.blocks.0.attn.qkv.scale_input": [],
    "visual.blocks.0.attn.qkv.scale_weight": [],
    "visual.blocks.0.attn.qkv.weight": [3840, 1280],
    "visual.blocks.0.mlp.down_proj.bias": [1280],
    "visual.blocks.0.mlp.down_proj.scale_input": [],
    "visual.blocks.0.mlp.down_proj.scale_weight": [],
    "visual.blocks.0.mlp.down_proj.weight": [1280, 3420],
    "visual.blocks.0.mlp.gate_proj.bias": [3420],
    "visual.blocks.0.mlp.gate_proj.scale_input": [],
    "visual.blocks.0.mlp.gate_proj.scale_weight": [],
    "visual.blocks.0.mlp.gate_proj.weight": [3420, 1280],
    "visual.blocks.0.mlp.up_proj.bias": [3420],
    "visual.blocks.0.mlp.up_proj.scale_input": [],
    "visual.blocks.0.mlp.up_proj.scale_weight": [],
    "visual.blocks.0.mlp.up_proj.weight": [3420, 1280],
    "visual.blocks.0.norm1.weight": [1280],
    "visual.blocks.0.norm2.weight": [1280],
    "visual.merger.ln_q.weight": [1280],
    "visual.merger.mlp.0.bias": [5120],
    "visual.merger.mlp.0.scale_input": [],
    "visual.merger.mlp.0.scale_weight": [],
    "visual.merger.mlp.0.weight": [5120, 5120],
    "visual.merger.mlp.2.bias": [3584],
    "visual.merger.mlp.2.scale_input": [],
    "visual.merger.mlp.2.scale_weight": [],
    "visual.merger.mlp.2.weight": [3584, 5120],
    "visual.patch_embed.proj.weight": [1280, 3, 2, 14, 14],
    "model.embed_tokens.weight": [152064, 3584],
    "model.layers.0.input_layernorm.weight": [3584],
    "model.layers.0.mlp.down_proj.scale_input": [],
    "model.layers.0.mlp.down_proj.scale_weight": [],
    "model.layers.0.mlp.down_proj.weight": [3584, 18944],
    "model.layers.0.mlp.gate_proj.scale_input": [],
    "model.layers.0.mlp.gate_proj.scale_weight": [],
    "model.layers.0.mlp.gate_proj.weight": [18944, 3584],
    "model.layers.0.mlp.up_proj.scale_input": [],
    "model.layers.0.mlp.up_proj.scale_weight": [],
    "model.layers.0.mlp.up_proj.weight": [18944, 3584],
    "model.layers.0.post_attention_layernorm.weight": [3584],
    "model.layers.0.self_attn.k_proj.bias": [512],
    "model.layers.0.self_attn.k_proj.scale_input": [],
    "model.layers.0.self_attn.k_proj.scale_weight": [],
    "model.layers.0.self_attn.k_proj.weight": [512, 3584],
    "model.layers.0.self_attn.o_proj.scale_input": [],
    "model.layers.0.self_attn.o_proj.scale_weight": [],
    "model.layers.0.self_attn.o_proj.weight": [3584, 3584],
    "model.layers.0.self_attn.q_proj.bias": [3584],
    "model.layers.0.self_attn.q_proj.scale_input": [],
    "model.layers.0.self_attn.q_proj.scale_weight": [],
    "model.layers.0.self_attn.q_proj.weight": [3584, 3584],
    "model.layers.0.self_attn.v_proj.bias": [512],
    "model.layers.0.self_attn.v_proj.scale_input": [],
    "model.layers.0.self_attn.v_proj.scale_weight": [],
    "model.layers.0.self_attn.v_proj.weight": [512, 3584],
    "model.norm.weight": [3584],
    "lm_head.weight": [152064, 3584],
    "scaled_fp8": [0],
}
