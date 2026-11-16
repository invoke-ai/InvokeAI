"""A MOSS-Image LoRA with bare module names, Angais/MOSS-Image-PixelArt-16x16 (rank 32, BF16).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`Angais/MOSS-Image-PixelArt-16x16/moss-image-pixelart-16x16.safetensors`.

Subsetting rule: blocks [0, 41] (first, last, and one next to a missing block where the file has a gap) of the 42 it names, plus every key outside a block. Values are `(shape, dtype)`.

Keys are the transformer's module paths without a prefix + `lora_A.weight`/`lora_B.weight`, no `alpha` tensors.
The header's `alpha` and `rank` (32 and 32) are not something a loader reads: the scale is alpha = rank, 1.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "layers.0.proj.ffn_down.lora_A.weight": ([32, 8192], "BF16"),
    "layers.0.proj.ffn_down.lora_B.weight": ([3072, 32], "BF16"),
    "layers.0.proj.ffn_up_gate.lora_A.weight": ([32, 3072], "BF16"),
    "layers.0.proj.ffn_up_gate.lora_B.weight": ([16384, 32], "BF16"),
    "layers.0.proj.k.lora_A.weight": ([32, 3072], "BF16"),
    "layers.0.proj.k.lora_B.weight": ([3072, 32], "BF16"),
    "layers.0.proj.o.lora_A.weight": ([32, 3072], "BF16"),
    "layers.0.proj.o.lora_B.weight": ([3072, 32], "BF16"),
    "layers.0.proj.q.lora_A.weight": ([32, 3072], "BF16"),
    "layers.0.proj.q.lora_B.weight": ([3072, 32], "BF16"),
    "layers.0.proj.v.lora_A.weight": ([32, 3072], "BF16"),
    "layers.0.proj.v.lora_B.weight": ([3072, 32], "BF16"),
    "layers.41.proj.ffn_down.lora_A.weight": ([32, 8192], "BF16"),
    "layers.41.proj.ffn_down.lora_B.weight": ([3072, 32], "BF16"),
    "layers.41.proj.ffn_up_gate.lora_A.weight": ([32, 3072], "BF16"),
    "layers.41.proj.ffn_up_gate.lora_B.weight": ([16384, 32], "BF16"),
    "layers.41.proj.k.lora_A.weight": ([32, 3072], "BF16"),
    "layers.41.proj.k.lora_B.weight": ([3072, 32], "BF16"),
    "layers.41.proj.o.lora_A.weight": ([32, 3072], "BF16"),
    "layers.41.proj.o.lora_B.weight": ([3072, 32], "BF16"),
    "layers.41.proj.q.lora_A.weight": ([32, 3072], "BF16"),
    "layers.41.proj.q.lora_B.weight": ([3072, 32], "BF16"),
    "layers.41.proj.v.lora_A.weight": ([32, 3072], "BF16"),
    "layers.41.proj.v.lora_B.weight": ([3072, 32], "BF16"),
}

# The part of the header's `__metadata__` a loader reads or ignores on purpose, as the producer wrote it.
metadata: dict[str, str] = {
    "alpha": "32",
    "rank": "32",
}
