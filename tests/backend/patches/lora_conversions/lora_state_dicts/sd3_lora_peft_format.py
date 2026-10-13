"""Key layout of a PEFT adapter over SD 3.5 Medium (`jieliu/SD3.5M-FlowGRPO-PickScore`, rank 32).

Captured on 2026-10-12 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`jieliu/SD3.5M-FlowGRPO-PickScore/adapter_model.safetensors`.

Subsetting rule: `base_model.model.transformer_blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

The diffusers names behind PEFT's `base_model.model.`, attention only, float32. The conversion strips exactly that
prefix; the probe tells it from Qwen-Image by width.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "base_model.model.transformer_blocks.0.attn.add_k_proj.lora_A.weight": ([32, 1536], "F32"),
    "base_model.model.transformer_blocks.0.attn.add_k_proj.lora_B.weight": ([1536, 32], "F32"),
    "base_model.model.transformer_blocks.0.attn.add_q_proj.lora_A.weight": ([32, 1536], "F32"),
    "base_model.model.transformer_blocks.0.attn.add_q_proj.lora_B.weight": ([1536, 32], "F32"),
    "base_model.model.transformer_blocks.0.attn.add_v_proj.lora_A.weight": ([32, 1536], "F32"),
    "base_model.model.transformer_blocks.0.attn.add_v_proj.lora_B.weight": ([1536, 32], "F32"),
    "base_model.model.transformer_blocks.0.attn.to_add_out.lora_A.weight": ([32, 1536], "F32"),
    "base_model.model.transformer_blocks.0.attn.to_add_out.lora_B.weight": ([1536, 32], "F32"),
    "base_model.model.transformer_blocks.0.attn.to_k.lora_A.weight": ([32, 1536], "F32"),
    "base_model.model.transformer_blocks.0.attn.to_k.lora_B.weight": ([1536, 32], "F32"),
    "base_model.model.transformer_blocks.0.attn.to_out.0.lora_A.weight": ([32, 1536], "F32"),
    "base_model.model.transformer_blocks.0.attn.to_out.0.lora_B.weight": ([1536, 32], "F32"),
    "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight": ([32, 1536], "F32"),
    "base_model.model.transformer_blocks.0.attn.to_q.lora_B.weight": ([1536, 32], "F32"),
    "base_model.model.transformer_blocks.0.attn.to_v.lora_A.weight": ([32, 1536], "F32"),
    "base_model.model.transformer_blocks.0.attn.to_v.lora_B.weight": ([1536, 32], "F32"),
}
