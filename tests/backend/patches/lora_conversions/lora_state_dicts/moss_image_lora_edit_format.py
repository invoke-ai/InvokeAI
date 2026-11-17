"""A MOSS-Image edit LoRA, Comfy-Org/MOSS-Image `moss_image_edit_lora_rank_64_bf16`.

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`Comfy-Org/MOSS-Image/loras/moss_image_edit_lora_rank_64_bf16.safetensors`.

Subsetting rule: blocks [0, 41] (first, last, and one next to a missing block where the file has a gap) of the 42 it names, plus every key outside a block. Values are `(shape, dtype)`.

Keys are `layers.N.proj.*.lora_A`/`lora_B`, without the `.weight` of PEFT's names, and the header's
`adapter_config` names the format `moss-edit-lora` (shortened here to those two fields). The edit LoRAs
condition on reference images, which this build does not apply: they are not recognized.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "layers.0.proj.ffn_down.lora_A": ([64, 8192], "BF16"),
    "layers.0.proj.ffn_down.lora_B": ([3072, 64], "BF16"),
    "layers.0.proj.ffn_up_gate.lora_A": ([64, 3072], "BF16"),
    "layers.0.proj.ffn_up_gate.lora_B": ([16384, 64], "BF16"),
    "layers.0.proj.k.lora_A": ([64, 3072], "BF16"),
    "layers.0.proj.k.lora_B": ([3072, 64], "BF16"),
    "layers.0.proj.o.lora_A": ([64, 3072], "BF16"),
    "layers.0.proj.o.lora_B": ([3072, 64], "BF16"),
    "layers.0.proj.q.lora_A": ([64, 3072], "BF16"),
    "layers.0.proj.q.lora_B": ([3072, 64], "BF16"),
    "layers.0.proj.v.lora_A": ([64, 3072], "BF16"),
    "layers.0.proj.v.lora_B": ([3072, 64], "BF16"),
    "layers.41.proj.ffn_down.lora_A": ([64, 8192], "BF16"),
    "layers.41.proj.ffn_down.lora_B": ([3072, 64], "BF16"),
    "layers.41.proj.ffn_up_gate.lora_A": ([64, 3072], "BF16"),
    "layers.41.proj.ffn_up_gate.lora_B": ([16384, 64], "BF16"),
    "layers.41.proj.k.lora_A": ([64, 3072], "BF16"),
    "layers.41.proj.k.lora_B": ([3072, 64], "BF16"),
    "layers.41.proj.o.lora_A": ([64, 3072], "BF16"),
    "layers.41.proj.o.lora_B": ([3072, 64], "BF16"),
    "layers.41.proj.q.lora_A": ([64, 3072], "BF16"),
    "layers.41.proj.q.lora_B": ([3072, 64], "BF16"),
    "layers.41.proj.v.lora_A": ([64, 3072], "BF16"),
    "layers.41.proj.v.lora_B": ([3072, 64], "BF16"),
}

# The part of the header's `__metadata__` a loader reads or ignores on purpose, as the producer wrote it.
metadata: dict[str, str] = {
    "adapter_config": '{\n  "format": "moss-edit-lora",\n  "format_version": 1\n}',
}
