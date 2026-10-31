# Key names and shapes of LuciferTC/Anima-IP-Adapter `ip_adapter-Character_Reference-10.safetensors` (BF16, 728
# tensors), read from its safetensors header. Every block carries the same 26 tensors, so they are listed once and
# repeated over the 28 blocks. The bundled PEFT LoRA (rank 32) covers self_attn, cross_attn and mlp.
_PER_BLOCK: dict[str, list[int]] = {
    "blocks.{n}.ip_k_proj.weight": [2048, 768],
    "blocks.{n}.ip_k_proj.bias": [2048],
    "blocks.{n}.ip_v_proj.weight": [2048, 768],
    "blocks.{n}.ip_v_proj.bias": [2048],
    "blocks.{n}.adaln_ip.1.weight": [2048, 2048],
    "blocks.{n}.adaln_ip.1.bias": [2048],
    "lora.base_model.model.blocks.{n}.cross_attn.q_proj.lora_A.weight": [32, 2048],
    "lora.base_model.model.blocks.{n}.cross_attn.q_proj.lora_B.weight": [2048, 32],
    "lora.base_model.model.blocks.{n}.cross_attn.k_proj.lora_A.weight": [32, 1024],
    "lora.base_model.model.blocks.{n}.cross_attn.k_proj.lora_B.weight": [2048, 32],
    "lora.base_model.model.blocks.{n}.cross_attn.v_proj.lora_A.weight": [32, 1024],
    "lora.base_model.model.blocks.{n}.cross_attn.v_proj.lora_B.weight": [2048, 32],
    "lora.base_model.model.blocks.{n}.cross_attn.output_proj.lora_A.weight": [32, 2048],
    "lora.base_model.model.blocks.{n}.cross_attn.output_proj.lora_B.weight": [2048, 32],
    "lora.base_model.model.blocks.{n}.self_attn.q_proj.lora_A.weight": [32, 2048],
    "lora.base_model.model.blocks.{n}.self_attn.q_proj.lora_B.weight": [2048, 32],
    "lora.base_model.model.blocks.{n}.self_attn.k_proj.lora_A.weight": [32, 2048],
    "lora.base_model.model.blocks.{n}.self_attn.k_proj.lora_B.weight": [2048, 32],
    "lora.base_model.model.blocks.{n}.self_attn.v_proj.lora_A.weight": [32, 2048],
    "lora.base_model.model.blocks.{n}.self_attn.v_proj.lora_B.weight": [2048, 32],
    "lora.base_model.model.blocks.{n}.self_attn.output_proj.lora_A.weight": [32, 2048],
    "lora.base_model.model.blocks.{n}.self_attn.output_proj.lora_B.weight": [2048, 32],
    "lora.base_model.model.blocks.{n}.mlp.layer1.lora_A.weight": [32, 2048],
    "lora.base_model.model.blocks.{n}.mlp.layer1.lora_B.weight": [8192, 32],
    "lora.base_model.model.blocks.{n}.mlp.layer2.lora_A.weight": [32, 8192],
    "lora.base_model.model.blocks.{n}.mlp.layer2.lora_B.weight": [2048, 32],
}

NUM_BLOCKS = 28

state_dict_keys: dict[str, list[int]] = {
    key.format(n=n): shape for n in range(NUM_BLOCKS) for key, shape in _PER_BLOCK.items()
}

metadata: dict[str, str] = {
    "ip_inject_before_mlp": "False",
    "lora_alpha": "32",
    "format": "pt",
    "ip_norm_keys": "False",
    "lora_rank": "32",
}
