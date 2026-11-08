"""Representative key layout of the ComfyUI Qwen3-VL 4B encoder FLUX.3 and Krea-2 read (`qwen3vl_4b_bf16`).

Captured on 2026-10-31 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/feef60b0aaf63a4773ace96636ebbcf43b881312/text_encoders/qwen3vl_4b_bf16.safetensors`.

Subsetting rule: `model.language_model.layers` index 35, `model.visual.blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

The same outer layout as the FLUX.3 prompt upsampler's single files -- a language model under
`model.language_model.` beside a `model.visual.` tower -- but with Qwen3 decoder layers instead of Qwen3.5's Gated
DeltaNet, and no LM head (the 4B ties it to the embedding): neither the Qwen3.5 encoder config nor the Text LLM one
may claim it. Layer 35 is kept because the Qwen3-VL config looks for the last of its 36 layers.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "model.language_model.embed_tokens.weight": ([151936, 2560], "BF16"),
    "model.language_model.layers.35.input_layernorm.weight": ([2560], "BF16"),
    "model.language_model.layers.35.mlp.down_proj.weight": ([2560, 9728], "BF16"),
    "model.language_model.layers.35.mlp.gate_proj.weight": ([9728, 2560], "BF16"),
    "model.language_model.layers.35.mlp.up_proj.weight": ([9728, 2560], "BF16"),
    "model.language_model.layers.35.post_attention_layernorm.weight": ([2560], "BF16"),
    "model.language_model.layers.35.self_attn.k_norm.weight": ([128], "BF16"),
    "model.language_model.layers.35.self_attn.k_proj.weight": ([1024, 2560], "BF16"),
    "model.language_model.layers.35.self_attn.o_proj.weight": ([2560, 4096], "BF16"),
    "model.language_model.layers.35.self_attn.q_norm.weight": ([128], "BF16"),
    "model.language_model.layers.35.self_attn.q_proj.weight": ([4096, 2560], "BF16"),
    "model.language_model.layers.35.self_attn.v_proj.weight": ([1024, 2560], "BF16"),
    "model.language_model.norm.weight": ([2560], "BF16"),
    "model.visual.blocks.0.attn.proj.bias": ([1024], "BF16"),
    "model.visual.blocks.0.attn.proj.weight": ([1024, 1024], "BF16"),
    "model.visual.blocks.0.attn.qkv.bias": ([3072], "BF16"),
    "model.visual.blocks.0.attn.qkv.weight": ([3072, 1024], "BF16"),
    "model.visual.blocks.0.mlp.linear_fc1.bias": ([4096], "BF16"),
    "model.visual.blocks.0.mlp.linear_fc1.weight": ([4096, 1024], "BF16"),
    "model.visual.blocks.0.mlp.linear_fc2.bias": ([1024], "BF16"),
    "model.visual.blocks.0.mlp.linear_fc2.weight": ([1024, 4096], "BF16"),
    "model.visual.blocks.0.norm1.bias": ([1024], "BF16"),
    "model.visual.blocks.0.norm1.weight": ([1024], "BF16"),
    "model.visual.blocks.0.norm2.bias": ([1024], "BF16"),
    "model.visual.blocks.0.norm2.weight": ([1024], "BF16"),
    "model.visual.deepstack_merger_list.0.linear_fc1.bias": ([4096], "BF16"),
    "model.visual.deepstack_merger_list.0.linear_fc1.weight": ([4096, 4096], "BF16"),
    "model.visual.deepstack_merger_list.0.linear_fc2.bias": ([2560], "BF16"),
    "model.visual.deepstack_merger_list.0.linear_fc2.weight": ([2560, 4096], "BF16"),
    "model.visual.deepstack_merger_list.0.norm.bias": ([4096], "BF16"),
    "model.visual.deepstack_merger_list.0.norm.weight": ([4096], "BF16"),
    "model.visual.deepstack_merger_list.1.linear_fc1.bias": ([4096], "BF16"),
    "model.visual.deepstack_merger_list.1.linear_fc1.weight": ([4096, 4096], "BF16"),
    "model.visual.deepstack_merger_list.1.linear_fc2.bias": ([2560], "BF16"),
    "model.visual.deepstack_merger_list.1.linear_fc2.weight": ([2560, 4096], "BF16"),
    "model.visual.deepstack_merger_list.1.norm.bias": ([4096], "BF16"),
    "model.visual.deepstack_merger_list.1.norm.weight": ([4096], "BF16"),
    "model.visual.deepstack_merger_list.2.linear_fc1.bias": ([4096], "BF16"),
    "model.visual.deepstack_merger_list.2.linear_fc1.weight": ([4096, 4096], "BF16"),
    "model.visual.deepstack_merger_list.2.linear_fc2.bias": ([2560], "BF16"),
    "model.visual.deepstack_merger_list.2.linear_fc2.weight": ([2560, 4096], "BF16"),
    "model.visual.deepstack_merger_list.2.norm.bias": ([4096], "BF16"),
    "model.visual.deepstack_merger_list.2.norm.weight": ([4096], "BF16"),
    "model.visual.merger.linear_fc1.bias": ([4096], "BF16"),
    "model.visual.merger.linear_fc1.weight": ([4096, 4096], "BF16"),
    "model.visual.merger.linear_fc2.bias": ([2560], "BF16"),
    "model.visual.merger.linear_fc2.weight": ([2560, 4096], "BF16"),
    "model.visual.merger.norm.bias": ([1024], "BF16"),
    "model.visual.merger.norm.weight": ([1024], "BF16"),
    "model.visual.patch_embed.proj.bias": ([1024], "BF16"),
    "model.visual.patch_embed.proj.weight": ([1024, 3, 2, 16, 16], "BF16"),
    "model.visual.pos_embed.weight": ([2304, 1024], "BF16"),
}
