"""Key layout of tensorart's SD 3.5 Medium 4-step turbo LoRA (`tensorart/stable-diffusion-3.5-medium-turbo`, rank 32).

Captured on 2026-10-12 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`tensorart/stable-diffusion-3.5-medium-turbo/lora_sd3.5m_4steps.safetensors`.

Subsetting rule: `transformer.transformer_blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

Trains nearly every module: dual attention (`attn2`, blocks 0-12), both feed-forwards, both `norm1` modulations, and
the top-level `context_embedder`, `proj_out`, `time_text_embed.*` and `pos_embed.proj` -- a Conv2d, whose LoRA
pair is 4-D. Block 0 here has no `add_*_proj` layers.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "transformer.context_embedder.lora_A.weight": ([32, 4096], "F16"),
    "transformer.context_embedder.lora_B.weight": ([1536, 32], "F16"),
    "transformer.pos_embed.proj.lora_A.weight": ([32, 16, 2, 2], "F16"),
    "transformer.pos_embed.proj.lora_B.weight": ([1536, 32, 1, 1], "F16"),
    "transformer.proj_out.lora_A.weight": ([32, 1536], "F16"),
    "transformer.proj_out.lora_B.weight": ([64, 32], "F16"),
    "transformer.time_text_embed.text_embedder.linear_1.lora_A.weight": ([32, 2048], "F16"),
    "transformer.time_text_embed.text_embedder.linear_1.lora_B.weight": ([1536, 32], "F16"),
    "transformer.time_text_embed.text_embedder.linear_2.lora_A.weight": ([32, 1536], "F16"),
    "transformer.time_text_embed.text_embedder.linear_2.lora_B.weight": ([1536, 32], "F16"),
    "transformer.time_text_embed.timestep_embedder.linear_1.lora_A.weight": ([32, 256], "F16"),
    "transformer.time_text_embed.timestep_embedder.linear_1.lora_B.weight": ([1536, 32], "F16"),
    "transformer.time_text_embed.timestep_embedder.linear_2.lora_A.weight": ([32, 1536], "F16"),
    "transformer.time_text_embed.timestep_embedder.linear_2.lora_B.weight": ([1536, 32], "F16"),
    "transformer.transformer_blocks.0.attn.to_add_out.lora_A.weight": ([32, 1536], "F16"),
    "transformer.transformer_blocks.0.attn.to_add_out.lora_B.weight": ([1536, 32], "F16"),
    "transformer.transformer_blocks.0.attn.to_k.lora_A.weight": ([32, 1536], "F16"),
    "transformer.transformer_blocks.0.attn.to_k.lora_B.weight": ([1536, 32], "F16"),
    "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight": ([32, 1536], "F16"),
    "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight": ([1536, 32], "F16"),
    "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": ([32, 1536], "F16"),
    "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": ([1536, 32], "F16"),
    "transformer.transformer_blocks.0.attn.to_v.lora_A.weight": ([32, 1536], "F16"),
    "transformer.transformer_blocks.0.attn.to_v.lora_B.weight": ([1536, 32], "F16"),
    "transformer.transformer_blocks.0.attn2.to_k.lora_A.weight": ([32, 1536], "F16"),
    "transformer.transformer_blocks.0.attn2.to_k.lora_B.weight": ([1536, 32], "F16"),
    "transformer.transformer_blocks.0.attn2.to_out.0.lora_A.weight": ([32, 1536], "F16"),
    "transformer.transformer_blocks.0.attn2.to_out.0.lora_B.weight": ([1536, 32], "F16"),
    "transformer.transformer_blocks.0.attn2.to_q.lora_A.weight": ([32, 1536], "F16"),
    "transformer.transformer_blocks.0.attn2.to_q.lora_B.weight": ([1536, 32], "F16"),
    "transformer.transformer_blocks.0.attn2.to_v.lora_A.weight": ([32, 1536], "F16"),
    "transformer.transformer_blocks.0.attn2.to_v.lora_B.weight": ([1536, 32], "F16"),
    "transformer.transformer_blocks.0.ff.net.0.proj.lora_A.weight": ([32, 1536], "F16"),
    "transformer.transformer_blocks.0.ff.net.0.proj.lora_B.weight": ([6144, 32], "F16"),
    "transformer.transformer_blocks.0.ff.net.2.lora_A.weight": ([32, 6144], "F16"),
    "transformer.transformer_blocks.0.ff.net.2.lora_B.weight": ([1536, 32], "F16"),
    "transformer.transformer_blocks.0.ff_context.net.0.proj.lora_A.weight": ([32, 1536], "F16"),
    "transformer.transformer_blocks.0.ff_context.net.0.proj.lora_B.weight": ([6144, 32], "F16"),
    "transformer.transformer_blocks.0.ff_context.net.2.lora_A.weight": ([32, 6144], "F16"),
    "transformer.transformer_blocks.0.ff_context.net.2.lora_B.weight": ([1536, 32], "F16"),
    "transformer.transformer_blocks.0.norm1.linear.lora_A.weight": ([32, 1536], "F16"),
    "transformer.transformer_blocks.0.norm1.linear.lora_B.weight": ([13824, 32], "F16"),
    "transformer.transformer_blocks.0.norm1_context.linear.lora_A.weight": ([32, 1536], "F16"),
    "transformer.transformer_blocks.0.norm1_context.linear.lora_B.weight": ([9216, 32], "F16"),
}
