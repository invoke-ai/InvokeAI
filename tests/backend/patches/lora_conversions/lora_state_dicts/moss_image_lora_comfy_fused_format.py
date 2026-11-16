"""A MOSS-Image LoRA in ComfyUI's form with a fused `proj.qkv` that also trains the top modules, anyMODE/moss-image-12step-turbo-lora (rank 64, BF16).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`anyMODE/moss-image-12step-turbo-lora/12_step_turbo_moss_1024_distill_v8_r64.safetensors`.

Subsetting rule: blocks [0, 41] (first, last, and one next to a missing block where the file has a gap) of the 42 it names, plus every key outside a block. Values are `(shape, dtype)`.

Keys are `diffusion_model.` + the module path + `lora_down.weight`/`lora_up.weight` + an F32 `alpha`. The blocks'
attention is fused (`proj.qkv`, 9216 rows) and the file trains every Linear outside the blocks as well: the
whole text refiner, `time_embedding.0` and `.2`, `shared_block_modulation`, `video_head_modulation` and
`final_linear_video`. The header's `lora_alpha` of 128.0 is not read: each layer's `alpha` is.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.final_linear_video.alpha": ([], "F32"),
    "diffusion_model.final_linear_video.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.final_linear_video.lora_up.weight": ([128, 64], "BF16"),
    "diffusion_model.layers.0.proj.ffn_down.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.ffn_down.lora_down.weight": ([64, 8192], "BF16"),
    "diffusion_model.layers.0.proj.ffn_down.lora_up.weight": ([3072, 64], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_up.weight": ([16384, 64], "BF16"),
    "diffusion_model.layers.0.proj.o.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.o.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.layers.0.proj.o.lora_up.weight": ([3072, 64], "BF16"),
    "diffusion_model.layers.0.proj.qkv.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.qkv.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.layers.0.proj.qkv.lora_up.weight": ([9216, 64], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.ffn_down.lora_down.weight": ([64, 8192], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.lora_up.weight": ([3072, 64], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_up.weight": ([16384, 64], "BF16"),
    "diffusion_model.layers.41.proj.o.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.o.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.layers.41.proj.o.lora_up.weight": ([3072, 64], "BF16"),
    "diffusion_model.layers.41.proj.qkv.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.qkv.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.layers.41.proj.qkv.lora_up.weight": ([9216, 64], "BF16"),
    "diffusion_model.shared_block_modulation.alpha": ([], "F32"),
    "diffusion_model.shared_block_modulation.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.shared_block_modulation.lora_up.weight": ([18432, 64], "BF16"),
    "diffusion_model.text_embedder.blocks.0.ffn_down.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.0.ffn_down.lora_down.weight": ([64, 8192], "BF16"),
    "diffusion_model.text_embedder.blocks.0.ffn_down.lora_up.weight": ([3072, 64], "BF16"),
    "diffusion_model.text_embedder.blocks.0.ffn_gate.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.0.ffn_gate.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.text_embedder.blocks.0.ffn_gate.lora_up.weight": ([8192, 64], "BF16"),
    "diffusion_model.text_embedder.blocks.0.ffn_up.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.0.ffn_up.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.text_embedder.blocks.0.ffn_up.lora_up.weight": ([8192, 64], "BF16"),
    "diffusion_model.text_embedder.blocks.0.proj.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.0.proj.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.text_embedder.blocks.0.proj.lora_up.weight": ([3072, 64], "BF16"),
    "diffusion_model.text_embedder.blocks.0.qkv.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.0.qkv.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.text_embedder.blocks.0.qkv.lora_up.weight": ([9216, 64], "BF16"),
    "diffusion_model.text_embedder.blocks.1.ffn_down.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.1.ffn_down.lora_down.weight": ([64, 8192], "BF16"),
    "diffusion_model.text_embedder.blocks.1.ffn_down.lora_up.weight": ([3072, 64], "BF16"),
    "diffusion_model.text_embedder.blocks.1.ffn_gate.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.1.ffn_gate.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.text_embedder.blocks.1.ffn_gate.lora_up.weight": ([8192, 64], "BF16"),
    "diffusion_model.text_embedder.blocks.1.ffn_up.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.1.ffn_up.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.text_embedder.blocks.1.ffn_up.lora_up.weight": ([8192, 64], "BF16"),
    "diffusion_model.text_embedder.blocks.1.proj.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.1.proj.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.text_embedder.blocks.1.proj.lora_up.weight": ([3072, 64], "BF16"),
    "diffusion_model.text_embedder.blocks.1.qkv.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.1.qkv.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.text_embedder.blocks.1.qkv.lora_up.weight": ([9216, 64], "BF16"),
    "diffusion_model.text_embedder.proj.alpha": ([], "F32"),
    "diffusion_model.text_embedder.proj.lora_down.weight": ([64, 4096], "BF16"),
    "diffusion_model.text_embedder.proj.lora_up.weight": ([3072, 64], "BF16"),
    "diffusion_model.time_embedding.0.alpha": ([], "F32"),
    "diffusion_model.time_embedding.0.lora_down.weight": ([64, 256], "BF16"),
    "diffusion_model.time_embedding.0.lora_up.weight": ([3072, 64], "BF16"),
    "diffusion_model.time_embedding.2.alpha": ([], "F32"),
    "diffusion_model.time_embedding.2.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.time_embedding.2.lora_up.weight": ([3072, 64], "BF16"),
    "diffusion_model.video_head_modulation.alpha": ([], "F32"),
    "diffusion_model.video_head_modulation.lora_down.weight": ([64, 3072], "BF16"),
    "diffusion_model.video_head_modulation.lora_up.weight": ([6144, 64], "BF16"),
}

# The part of the header's `__metadata__` a loader reads or ignores on purpose, as the producer wrote it.
metadata: dict[str, str] = {
    "lora_alpha": "128.0",
}
