"""A MOSS-Image LoRA in ComfyUI's form with a fused `proj.qkv` on 18 of the 42 blocks, DeVilDoNia/HaDeS-MOSS-Turbo (rank 2, BF16).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`DeVilDoNia/HaDeS-MOSS-Turbo/HaDeS-MOSS-Turbo-8steps-v1.0.safetensors`.

Subsetting rule: blocks [24, 41] (first, last, and one next to a missing block where the file has a gap) of the 18 it names, plus every key outside a block. Values are `(shape, dtype)`.

Like `moss_image_lora_comfy_fused_format`, for the blocks' attention and feed-forward only, with an F32 `alpha`
per layer. Blocks 0 to 23 are not trained, so the file holds blocks 24 to 41 and the capture keeps its first
and last. A layer a file does not name is left as it is.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.layers.24.proj.ffn_down.alpha": ([], "F32"),
    "diffusion_model.layers.24.proj.ffn_down.lora_down.weight": ([2, 8192], "BF16"),
    "diffusion_model.layers.24.proj.ffn_down.lora_up.weight": ([3072, 2], "BF16"),
    "diffusion_model.layers.24.proj.ffn_up_gate.alpha": ([], "F32"),
    "diffusion_model.layers.24.proj.ffn_up_gate.lora_down.weight": ([2, 3072], "BF16"),
    "diffusion_model.layers.24.proj.ffn_up_gate.lora_up.weight": ([16384, 2], "BF16"),
    "diffusion_model.layers.24.proj.o.alpha": ([], "F32"),
    "diffusion_model.layers.24.proj.o.lora_down.weight": ([2, 3072], "BF16"),
    "diffusion_model.layers.24.proj.o.lora_up.weight": ([3072, 2], "BF16"),
    "diffusion_model.layers.24.proj.qkv.alpha": ([], "F32"),
    "diffusion_model.layers.24.proj.qkv.lora_down.weight": ([2, 3072], "BF16"),
    "diffusion_model.layers.24.proj.qkv.lora_up.weight": ([9216, 2], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.ffn_down.lora_down.weight": ([2, 8192], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.lora_up.weight": ([3072, 2], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_down.weight": ([2, 3072], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_up.weight": ([16384, 2], "BF16"),
    "diffusion_model.layers.41.proj.o.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.o.lora_down.weight": ([2, 3072], "BF16"),
    "diffusion_model.layers.41.proj.o.lora_up.weight": ([3072, 2], "BF16"),
    "diffusion_model.layers.41.proj.qkv.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.qkv.lora_down.weight": ([2, 3072], "BF16"),
    "diffusion_model.layers.41.proj.qkv.lora_up.weight": ([9216, 2], "BF16"),
}

# The part of the header's `__metadata__` a loader reads or ignores on purpose, as the producer wrote it.
metadata: dict[str, str] = {
    "rank": "2",
}
