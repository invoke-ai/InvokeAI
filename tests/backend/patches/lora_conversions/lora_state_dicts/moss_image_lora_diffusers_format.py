"""A MOSS-Image LoRA as diffusers' `save_lora_weights` lays out a transformer adapter (rank 16, lora_alpha 8), synthesized.

No published file is captured here: diffusers' MOSS-Image LoRA trainer has none in this tree. The keys follow the
layout `save_lora_weights` writes for any transformer (`transformer.` + the module path + PEFT's `lora_A.weight`/
`lora_B.weight`, no `alpha` tensors) on the six block Linears the trainer targets, and the header holds the PEFT
config as diffusers saves it (`lora_adapter_metadata`, each field prefixed `transformer.`). Its lora_alpha of 8
against a rank of 16 makes PEFT's scale 0.5, which the `alpha` tensors this layout lacks cannot give: only the
header does (`peft_scale`).

Subsetting rule: `layers` indices 0 and 41, the first and last of the 42 blocks. Values are `(shape, dtype)`.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "transformer.layers.0.proj.q.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.layers.0.proj.q.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.layers.0.proj.k.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.layers.0.proj.k.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.layers.0.proj.v.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.layers.0.proj.v.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.layers.0.proj.o.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.layers.0.proj.o.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.layers.0.proj.ffn_up_gate.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.layers.0.proj.ffn_up_gate.lora_B.weight": ([16384, 16], "BF16"),
    "transformer.layers.0.proj.ffn_down.lora_A.weight": ([16, 8192], "BF16"),
    "transformer.layers.0.proj.ffn_down.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.layers.41.proj.q.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.layers.41.proj.q.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.layers.41.proj.k.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.layers.41.proj.k.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.layers.41.proj.v.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.layers.41.proj.v.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.layers.41.proj.o.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.layers.41.proj.o.lora_B.weight": ([3072, 16], "BF16"),
    "transformer.layers.41.proj.ffn_up_gate.lora_A.weight": ([16, 3072], "BF16"),
    "transformer.layers.41.proj.ffn_up_gate.lora_B.weight": ([16384, 16], "BF16"),
    "transformer.layers.41.proj.ffn_down.lora_A.weight": ([16, 8192], "BF16"),
    "transformer.layers.41.proj.ffn_down.lora_B.weight": ([3072, 16], "BF16"),
}

# The header's `__metadata__`, as diffusers saves the adapter's PEFT config.
metadata: dict[str, str] = {
    "lora_adapter_metadata": '{\n  "transformer.peft_type": "LORA",\n  "transformer.r": 16,\n  "transformer.lora_alpha": 8,\n  "transformer.lora_dropout": 0.0,\n  "transformer.target_modules": [\n    "proj.q",\n    "proj.k",\n    "proj.v",\n    "proj.o",\n    "proj.ffn_up_gate",\n    "proj.ffn_down"\n  ]\n}',
}
