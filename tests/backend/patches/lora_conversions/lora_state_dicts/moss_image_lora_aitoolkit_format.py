"""A MOSS-Image LoRA as ai-toolkit 0.13.34 writes it with q, k and v split, rank 32 in F16 (a Civitai file).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`mossloras/moss-image-lora-archive/2026-10-21/MOSS-Image/3013030_Direct-Flash-Film---Point-and-Shoot-Flash-Snapshots---MOSS-K/3419791/directflashfilmconcept.safetensors`.

Subsetting rule: blocks [0, 41] (first, last, and one next to a missing block where the file has a gap) of the 42 it names, plus every key outside a block. Values are `(shape, dtype)`.

Keys are `diffusion_model.` + the transformer's own module path + `lora_A.weight`/`lora_B.weight`, with no
`alpha` (the scale is 1). The 504 tensors cover every block's attention and feed-forward and nothing else: no
text refiner, embedder or modulation.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.layers.0.proj.ffn_down.lora_A.weight": ([32, 8192], "F16"),
    "diffusion_model.layers.0.proj.ffn_down.lora_B.weight": ([3072, 32], "F16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_A.weight": ([32, 3072], "F16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_B.weight": ([16384, 32], "F16"),
    "diffusion_model.layers.0.proj.k.lora_A.weight": ([32, 3072], "F16"),
    "diffusion_model.layers.0.proj.k.lora_B.weight": ([3072, 32], "F16"),
    "diffusion_model.layers.0.proj.o.lora_A.weight": ([32, 3072], "F16"),
    "diffusion_model.layers.0.proj.o.lora_B.weight": ([3072, 32], "F16"),
    "diffusion_model.layers.0.proj.q.lora_A.weight": ([32, 3072], "F16"),
    "diffusion_model.layers.0.proj.q.lora_B.weight": ([3072, 32], "F16"),
    "diffusion_model.layers.0.proj.v.lora_A.weight": ([32, 3072], "F16"),
    "diffusion_model.layers.0.proj.v.lora_B.weight": ([3072, 32], "F16"),
    "diffusion_model.layers.41.proj.ffn_down.lora_A.weight": ([32, 8192], "F16"),
    "diffusion_model.layers.41.proj.ffn_down.lora_B.weight": ([3072, 32], "F16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_A.weight": ([32, 3072], "F16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_B.weight": ([16384, 32], "F16"),
    "diffusion_model.layers.41.proj.k.lora_A.weight": ([32, 3072], "F16"),
    "diffusion_model.layers.41.proj.k.lora_B.weight": ([3072, 32], "F16"),
    "diffusion_model.layers.41.proj.o.lora_A.weight": ([32, 3072], "F16"),
    "diffusion_model.layers.41.proj.o.lora_B.weight": ([3072, 32], "F16"),
    "diffusion_model.layers.41.proj.q.lora_A.weight": ([32, 3072], "F16"),
    "diffusion_model.layers.41.proj.q.lora_B.weight": ([3072, 32], "F16"),
    "diffusion_model.layers.41.proj.v.lora_A.weight": ([32, 3072], "F16"),
    "diffusion_model.layers.41.proj.v.lora_B.weight": ([3072, 32], "F16"),
}
