"""A MOSS-Image DoRA from ai-toolkit 0.13.33, a Civitai file (rank 32, BF16).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`mossloras/moss-image-lora-archive/2026-10-24/MOSS-Image/3018804_MOSS-Anime-Ultrasoft/3427316/moss-anime-ultrasoft.safetensors`.

Subsetting rule: blocks [0, 41] (first, last, and one next to a missing block where the file has a gap) of the 42 it names, plus every key outside a block. Values are `(shape, dtype)`.

q, k and v are split `lora_A`/`lora_B` layers with an `alpha` and no magnitude of their own; the one magnitude
of the three is `layers.N.proj.qkv.dora_scale`, 9216 x 1, left on the fused name and alone. `proj.o` and the
feed-forward carry `dora_scale` beside `lora_A`/`lora_B`.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.layers.0.proj.ffn_down.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.ffn_down.dora_scale": ([3072, 1], "BF16"),
    "diffusion_model.layers.0.proj.ffn_down.lora_A.weight": ([32, 8192], "BF16"),
    "diffusion_model.layers.0.proj.ffn_down.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.dora_scale": ([16384, 1], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_B.weight": ([16384, 32], "BF16"),
    "diffusion_model.layers.0.proj.k.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.k.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.layers.0.proj.k.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.layers.0.proj.o.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.o.dora_scale": ([3072, 1], "BF16"),
    "diffusion_model.layers.0.proj.o.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.layers.0.proj.o.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.layers.0.proj.q.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.q.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.layers.0.proj.q.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.layers.0.proj.qkv.dora_scale": ([9216, 1], "BF16"),
    "diffusion_model.layers.0.proj.v.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.v.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.layers.0.proj.v.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.dora_scale": ([3072, 1], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.lora_A.weight": ([32, 8192], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.dora_scale": ([16384, 1], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_B.weight": ([16384, 32], "BF16"),
    "diffusion_model.layers.41.proj.k.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.k.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.layers.41.proj.k.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.layers.41.proj.o.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.o.dora_scale": ([3072, 1], "BF16"),
    "diffusion_model.layers.41.proj.o.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.layers.41.proj.o.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.layers.41.proj.q.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.q.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.layers.41.proj.q.lora_B.weight": ([3072, 32], "BF16"),
    "diffusion_model.layers.41.proj.qkv.dora_scale": ([9216, 1], "BF16"),
    "diffusion_model.layers.41.proj.v.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.v.lora_A.weight": ([32, 3072], "BF16"),
    "diffusion_model.layers.41.proj.v.lora_B.weight": ([3072, 32], "BF16"),
}
