"""Key layout of a diffusers DreamBooth SD 3 LoRA that also trains both CLIP encoders (`linoyts/Yarn_art_SD3_LoRA`).

Captured on 2026-10-12 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`linoyts/Yarn_art_SD3_LoRA/pytorch_lora_weights.safetensors`.

Subsetting rule: `text_encoder.text_model.encoder.layers` index 0, `text_encoder_2.text_model.encoder.layers` index 0, `transformer.transformer_blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

Transformer attention behind `transformer.`, CLIP-L behind `text_encoder.` (768 wide) and CLIP-G behind
`text_encoder_2.` (1280 wide), both as `text_model.encoder.layers.N.self_attn.*`. The conversion keeps the encoders
under separate prefixes, since their module paths are identical.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "text_encoder.text_model.encoder.layers.0.self_attn.k_proj.lora_A.weight": ([32, 768], "F32"),
    "text_encoder.text_model.encoder.layers.0.self_attn.k_proj.lora_B.weight": ([768, 32], "F32"),
    "text_encoder.text_model.encoder.layers.0.self_attn.out_proj.lora_A.weight": ([32, 768], "F32"),
    "text_encoder.text_model.encoder.layers.0.self_attn.out_proj.lora_B.weight": ([768, 32], "F32"),
    "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight": ([32, 768], "F32"),
    "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.lora_B.weight": ([768, 32], "F32"),
    "text_encoder.text_model.encoder.layers.0.self_attn.v_proj.lora_A.weight": ([32, 768], "F32"),
    "text_encoder.text_model.encoder.layers.0.self_attn.v_proj.lora_B.weight": ([768, 32], "F32"),
    "text_encoder_2.text_model.encoder.layers.0.self_attn.k_proj.lora_A.weight": ([32, 1280], "F32"),
    "text_encoder_2.text_model.encoder.layers.0.self_attn.k_proj.lora_B.weight": ([1280, 32], "F32"),
    "text_encoder_2.text_model.encoder.layers.0.self_attn.out_proj.lora_A.weight": ([32, 1280], "F32"),
    "text_encoder_2.text_model.encoder.layers.0.self_attn.out_proj.lora_B.weight": ([1280, 32], "F32"),
    "text_encoder_2.text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight": ([32, 1280], "F32"),
    "text_encoder_2.text_model.encoder.layers.0.self_attn.q_proj.lora_B.weight": ([1280, 32], "F32"),
    "text_encoder_2.text_model.encoder.layers.0.self_attn.v_proj.lora_A.weight": ([32, 1280], "F32"),
    "text_encoder_2.text_model.encoder.layers.0.self_attn.v_proj.lora_B.weight": ([1280, 32], "F32"),
    "transformer.transformer_blocks.0.attn.to_k.lora_A.weight": ([32, 1536], "F32"),
    "transformer.transformer_blocks.0.attn.to_k.lora_B.weight": ([1536, 32], "F32"),
    "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight": ([32, 1536], "F32"),
    "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight": ([1536, 32], "F32"),
    "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": ([32, 1536], "F32"),
    "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": ([1536, 32], "F32"),
    "transformer.transformer_blocks.0.attn.to_v.lora_A.weight": ([32, 1536], "F32"),
    "transformer.transformer_blocks.0.attn.to_v.lora_B.weight": ([1536, 32], "F32"),
}
