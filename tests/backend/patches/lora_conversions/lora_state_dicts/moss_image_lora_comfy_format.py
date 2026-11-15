"""A MOSS-Image LoRA in ComfyUI's form, lodestones/MOSS-Image-SFT-8-steps-dmd (an 8-step distillation, rank 128, F32).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`lodestones/MOSS-Image-SFT-8-steps-dmd/lora.safetensors`.

Subsetting rule: blocks [0, 41] (first, last, and one next to a missing block where the file has a gap) of the 42 it names, plus every key outside a block. Values are `(shape, dtype)`.

Keys are `diffusion_model.` + the module path + `lora_down.weight`/`lora_up.weight`, with an F32 `alpha` per layer
(128, its rank) and q, k and v split.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.layers.0.proj.ffn_down.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.ffn_down.lora_down.weight": ([128, 8192], "F32"),
    "diffusion_model.layers.0.proj.ffn_down.lora_up.weight": ([3072, 128], "F32"),
    "diffusion_model.layers.0.proj.ffn_up_gate.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_down.weight": ([128, 3072], "F32"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_up.weight": ([16384, 128], "F32"),
    "diffusion_model.layers.0.proj.k.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.k.lora_down.weight": ([128, 3072], "F32"),
    "diffusion_model.layers.0.proj.k.lora_up.weight": ([3072, 128], "F32"),
    "diffusion_model.layers.0.proj.o.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.o.lora_down.weight": ([128, 3072], "F32"),
    "diffusion_model.layers.0.proj.o.lora_up.weight": ([3072, 128], "F32"),
    "diffusion_model.layers.0.proj.q.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.q.lora_down.weight": ([128, 3072], "F32"),
    "diffusion_model.layers.0.proj.q.lora_up.weight": ([3072, 128], "F32"),
    "diffusion_model.layers.0.proj.v.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.v.lora_down.weight": ([128, 3072], "F32"),
    "diffusion_model.layers.0.proj.v.lora_up.weight": ([3072, 128], "F32"),
    "diffusion_model.layers.41.proj.ffn_down.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.ffn_down.lora_down.weight": ([128, 8192], "F32"),
    "diffusion_model.layers.41.proj.ffn_down.lora_up.weight": ([3072, 128], "F32"),
    "diffusion_model.layers.41.proj.ffn_up_gate.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_down.weight": ([128, 3072], "F32"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_up.weight": ([16384, 128], "F32"),
    "diffusion_model.layers.41.proj.k.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.k.lora_down.weight": ([128, 3072], "F32"),
    "diffusion_model.layers.41.proj.k.lora_up.weight": ([3072, 128], "F32"),
    "diffusion_model.layers.41.proj.o.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.o.lora_down.weight": ([128, 3072], "F32"),
    "diffusion_model.layers.41.proj.o.lora_up.weight": ([3072, 128], "F32"),
    "diffusion_model.layers.41.proj.q.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.q.lora_down.weight": ([128, 3072], "F32"),
    "diffusion_model.layers.41.proj.q.lora_up.weight": ([3072, 128], "F32"),
    "diffusion_model.layers.41.proj.v.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.v.lora_down.weight": ([128, 3072], "F32"),
    "diffusion_model.layers.41.proj.v.lora_up.weight": ([3072, 128], "F32"),
}

# The part of the header's `__metadata__` a loader reads or ignores on purpose, as the producer wrote it.
metadata: dict[str, str] = {
    "alpha": "128.0",
    "rank": "128",
}
