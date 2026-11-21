"""Key layout of ostris' Unconditional LoRA for Ideogram 4 (ai-toolkit, rank 16).

Captured on 2026-10-11 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`ostris/ideogram_4_unconditional_lora/ideogram_4_unconditional_lora_r16.safetensors`.

Subsetting rule: `diffusion_model.layers` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

Besides the blocks it trains every top-level projection (`input_proj`, `llm_cond_proj`, `adaln_proj`,
`t_embedding.mlp_{in,out}`, `final_layer.{linear,adaln_modulation}`). Its intended role -- standing in for the
unconditional transformer -- is not supported; as an ordinary LoRA it still has to identify and convert.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.adaln_proj.lora_A.weight": ([16, 4608], "BF16"),
    "diffusion_model.adaln_proj.lora_B.weight": ([512, 16], "BF16"),
    "diffusion_model.final_layer.adaln_modulation.lora_A.weight": ([16, 512], "BF16"),
    "diffusion_model.final_layer.adaln_modulation.lora_B.weight": ([4608, 16], "BF16"),
    "diffusion_model.final_layer.linear.lora_A.weight": ([16, 4608], "BF16"),
    "diffusion_model.final_layer.linear.lora_B.weight": ([128, 16], "BF16"),
    "diffusion_model.input_proj.lora_A.weight": ([16, 128], "BF16"),
    "diffusion_model.input_proj.lora_B.weight": ([4608, 16], "BF16"),
    "diffusion_model.layers.0.adaln_modulation.lora_A.weight": ([16, 512], "BF16"),
    "diffusion_model.layers.0.adaln_modulation.lora_B.weight": ([18432, 16], "BF16"),
    "diffusion_model.layers.0.attention.o.lora_A.weight": ([16, 4608], "BF16"),
    "diffusion_model.layers.0.attention.o.lora_B.weight": ([4608, 16], "BF16"),
    "diffusion_model.layers.0.attention.qkv.lora_A.weight": ([16, 4608], "BF16"),
    "diffusion_model.layers.0.attention.qkv.lora_B.weight": ([13824, 16], "BF16"),
    "diffusion_model.layers.0.feed_forward.w1.lora_A.weight": ([16, 4608], "BF16"),
    "diffusion_model.layers.0.feed_forward.w1.lora_B.weight": ([12288, 16], "BF16"),
    "diffusion_model.layers.0.feed_forward.w2.lora_A.weight": ([16, 12288], "BF16"),
    "diffusion_model.layers.0.feed_forward.w2.lora_B.weight": ([4608, 16], "BF16"),
    "diffusion_model.layers.0.feed_forward.w3.lora_A.weight": ([16, 4608], "BF16"),
    "diffusion_model.layers.0.feed_forward.w3.lora_B.weight": ([12288, 16], "BF16"),
    "diffusion_model.llm_cond_proj.lora_A.weight": ([16, 53248], "BF16"),
    "diffusion_model.llm_cond_proj.lora_B.weight": ([4608, 16], "BF16"),
    "diffusion_model.t_embedding.mlp_in.lora_A.weight": ([16, 4608], "BF16"),
    "diffusion_model.t_embedding.mlp_in.lora_B.weight": ([4608, 16], "BF16"),
    "diffusion_model.t_embedding.mlp_out.lora_A.weight": ([16, 4608], "BF16"),
    "diffusion_model.t_embedding.mlp_out.lora_B.weight": ([4608, 16], "BF16"),
}
