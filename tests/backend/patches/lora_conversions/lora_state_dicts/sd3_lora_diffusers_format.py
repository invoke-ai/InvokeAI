"""Key layout of a diffusers DreamBooth SD 3.5 Medium LoRA (`AdamLucek/sd3.5M-jareklfoto-LoRA-DreamBooth`, rank 64).

Captured on 2026-10-12 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`AdamLucek/sd3.5M-jareklfoto-LoRA-DreamBooth/pytorch_lora_weights.safetensors`.

Subsetting rule: `transformer.transformer_blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

diffusers' own module names behind `transformer.`, attention only (`to_{q,k,v,out.0}`, `add_{q,k,v}_proj`,
`to_add_out`) at width 1536 -- the names Qwen-Image's LoRAs use too, so only the width tells the two apart.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "transformer.transformer_blocks.0.attn.add_k_proj.lora_A.weight": ([64, 1536], "BF16"),
    "transformer.transformer_blocks.0.attn.add_k_proj.lora_B.weight": ([1536, 64], "BF16"),
    "transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight": ([64, 1536], "BF16"),
    "transformer.transformer_blocks.0.attn.add_q_proj.lora_B.weight": ([1536, 64], "BF16"),
    "transformer.transformer_blocks.0.attn.add_v_proj.lora_A.weight": ([64, 1536], "BF16"),
    "transformer.transformer_blocks.0.attn.add_v_proj.lora_B.weight": ([1536, 64], "BF16"),
    "transformer.transformer_blocks.0.attn.to_add_out.lora_A.weight": ([64, 1536], "BF16"),
    "transformer.transformer_blocks.0.attn.to_add_out.lora_B.weight": ([1536, 64], "BF16"),
    "transformer.transformer_blocks.0.attn.to_k.lora_A.weight": ([64, 1536], "BF16"),
    "transformer.transformer_blocks.0.attn.to_k.lora_B.weight": ([1536, 64], "BF16"),
    "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight": ([64, 1536], "BF16"),
    "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight": ([1536, 64], "BF16"),
    "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": ([64, 1536], "BF16"),
    "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": ([1536, 64], "BF16"),
    "transformer.transformer_blocks.0.attn.to_v.lora_A.weight": ([64, 1536], "BF16"),
    "transformer.transformer_blocks.0.attn.to_v.lora_B.weight": ([1536, 64], "BF16"),
}
