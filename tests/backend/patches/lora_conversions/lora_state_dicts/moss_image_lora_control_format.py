"""A MOSS-Image control LoRA, Comfy-Org/MOSS-Image `moss_image_color_depth_control`.

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`Comfy-Org/MOSS-Image/loras/moss_image_color_depth_control.safetensors`.

Subsetting rule: blocks [0, 41] (first, last, and one next to a missing block where the file has a gap) of the 42 it names, plus every key outside a block. Values are `(shape, dtype)`.

Bare `lora_A.weight`/`lora_B.weight` on the blocks and the text refiner, and a replacement `video_embedder.weight`
of 3072 x 384 with its `bias`: the control adapter widens the input embedding from 128 to 384 channels, which
no layer applies. Its keys are not LoRA parameters.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "layers.0.proj.ffn_down.lora_A.weight": ([64, 8192], "BF16"),
    "layers.0.proj.ffn_down.lora_B.weight": ([3072, 64], "BF16"),
    "layers.0.proj.ffn_up_gate.lora_A.weight": ([64, 3072], "BF16"),
    "layers.0.proj.ffn_up_gate.lora_B.weight": ([16384, 64], "BF16"),
    "layers.0.proj.k.lora_A.weight": ([64, 3072], "BF16"),
    "layers.0.proj.k.lora_B.weight": ([3072, 64], "BF16"),
    "layers.0.proj.o.lora_A.weight": ([64, 3072], "BF16"),
    "layers.0.proj.o.lora_B.weight": ([3072, 64], "BF16"),
    "layers.0.proj.q.lora_A.weight": ([64, 3072], "BF16"),
    "layers.0.proj.q.lora_B.weight": ([3072, 64], "BF16"),
    "layers.0.proj.v.lora_A.weight": ([64, 3072], "BF16"),
    "layers.0.proj.v.lora_B.weight": ([3072, 64], "BF16"),
    "layers.41.proj.ffn_down.lora_A.weight": ([64, 8192], "BF16"),
    "layers.41.proj.ffn_down.lora_B.weight": ([3072, 64], "BF16"),
    "layers.41.proj.ffn_up_gate.lora_A.weight": ([64, 3072], "BF16"),
    "layers.41.proj.ffn_up_gate.lora_B.weight": ([16384, 64], "BF16"),
    "layers.41.proj.k.lora_A.weight": ([64, 3072], "BF16"),
    "layers.41.proj.k.lora_B.weight": ([3072, 64], "BF16"),
    "layers.41.proj.o.lora_A.weight": ([64, 3072], "BF16"),
    "layers.41.proj.o.lora_B.weight": ([3072, 64], "BF16"),
    "layers.41.proj.q.lora_A.weight": ([64, 3072], "BF16"),
    "layers.41.proj.q.lora_B.weight": ([3072, 64], "BF16"),
    "layers.41.proj.v.lora_A.weight": ([64, 3072], "BF16"),
    "layers.41.proj.v.lora_B.weight": ([3072, 64], "BF16"),
    "text_embedder.blocks.0.ffn_down.lora_A.weight": ([64, 8192], "BF16"),
    "text_embedder.blocks.0.ffn_down.lora_B.weight": ([3072, 64], "BF16"),
    "text_embedder.blocks.0.ffn_gate.lora_A.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.0.ffn_gate.lora_B.weight": ([8192, 64], "BF16"),
    "text_embedder.blocks.0.ffn_up.lora_A.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.0.ffn_up.lora_B.weight": ([8192, 64], "BF16"),
    "text_embedder.blocks.0.proj.lora_A.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.0.proj.lora_B.weight": ([3072, 64], "BF16"),
    "text_embedder.blocks.0.qkv.lora_A.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.0.qkv.lora_B.weight": ([9216, 64], "BF16"),
    "text_embedder.blocks.1.ffn_down.lora_A.weight": ([64, 8192], "BF16"),
    "text_embedder.blocks.1.ffn_down.lora_B.weight": ([3072, 64], "BF16"),
    "text_embedder.blocks.1.ffn_gate.lora_A.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.1.ffn_gate.lora_B.weight": ([8192, 64], "BF16"),
    "text_embedder.blocks.1.ffn_up.lora_A.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.1.ffn_up.lora_B.weight": ([8192, 64], "BF16"),
    "text_embedder.blocks.1.proj.lora_A.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.1.proj.lora_B.weight": ([3072, 64], "BF16"),
    "text_embedder.blocks.1.qkv.lora_A.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.1.qkv.lora_B.weight": ([9216, 64], "BF16"),
    "video_embedder.bias": ([3072], "BF16"),
    "video_embedder.weight": ([3072, 384], "BF16"),
}

# The part of the header's `__metadata__` a loader reads or ignores on purpose, as the producer wrote it.
metadata: dict[str, str] = {
    "alpha": "64.0",
    "rank": "64",
}
