"""A MOSS-Image LoRA as ai-toolkit 0.13.34 writes it with a fused `proj.qkv`, rank 16 in BF16 (a Civitai file).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`mossloras/moss-image-lora-archive/2026-10-24/MOSS-Image/1396354_Digital-Dystopia/3427249/digital-dystopia-moss.safetensors`.

Subsetting rule: blocks [0, 21, 23, 41] (first, last, and one next to a missing block where the file has a gap) of the 41 it names, plus every key outside a block. Values are `(shape, dtype)`.

Like `moss_image_lora_aitoolkit_format`, but the attention is one `proj.qkv` layer of 9216 rows (q, k, v stacked
in that order), which the transformer holds as `proj.q`, `proj.k` and `proj.v`. Block 22, the layer MOSS-Image
skips on its negative branch, is not trained: 41 blocks, so the capture keeps blocks 21 and 23 beside the
gap.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.layers.0.proj.ffn_down.lora_A.weight": ([16, 8192], "BF16"),
    "diffusion_model.layers.0.proj.ffn_down.lora_B.weight": ([3072, 16], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_A.weight": ([16, 3072], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_B.weight": ([16384, 16], "BF16"),
    "diffusion_model.layers.0.proj.o.lora_A.weight": ([16, 3072], "BF16"),
    "diffusion_model.layers.0.proj.o.lora_B.weight": ([3072, 16], "BF16"),
    "diffusion_model.layers.0.proj.qkv.lora_A.weight": ([16, 3072], "BF16"),
    "diffusion_model.layers.0.proj.qkv.lora_B.weight": ([9216, 16], "BF16"),
    "diffusion_model.layers.21.proj.ffn_down.lora_A.weight": ([16, 8192], "BF16"),
    "diffusion_model.layers.21.proj.ffn_down.lora_B.weight": ([3072, 16], "BF16"),
    "diffusion_model.layers.21.proj.ffn_up_gate.lora_A.weight": ([16, 3072], "BF16"),
    "diffusion_model.layers.21.proj.ffn_up_gate.lora_B.weight": ([16384, 16], "BF16"),
    "diffusion_model.layers.21.proj.o.lora_A.weight": ([16, 3072], "BF16"),
    "diffusion_model.layers.21.proj.o.lora_B.weight": ([3072, 16], "BF16"),
    "diffusion_model.layers.21.proj.qkv.lora_A.weight": ([16, 3072], "BF16"),
    "diffusion_model.layers.21.proj.qkv.lora_B.weight": ([9216, 16], "BF16"),
    "diffusion_model.layers.23.proj.ffn_down.lora_A.weight": ([16, 8192], "BF16"),
    "diffusion_model.layers.23.proj.ffn_down.lora_B.weight": ([3072, 16], "BF16"),
    "diffusion_model.layers.23.proj.ffn_up_gate.lora_A.weight": ([16, 3072], "BF16"),
    "diffusion_model.layers.23.proj.ffn_up_gate.lora_B.weight": ([16384, 16], "BF16"),
    "diffusion_model.layers.23.proj.o.lora_A.weight": ([16, 3072], "BF16"),
    "diffusion_model.layers.23.proj.o.lora_B.weight": ([3072, 16], "BF16"),
    "diffusion_model.layers.23.proj.qkv.lora_A.weight": ([16, 3072], "BF16"),
    "diffusion_model.layers.23.proj.qkv.lora_B.weight": ([9216, 16], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.lora_A.weight": ([16, 8192], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.lora_B.weight": ([3072, 16], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_A.weight": ([16, 3072], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_B.weight": ([16384, 16], "BF16"),
    "diffusion_model.layers.41.proj.o.lora_A.weight": ([16, 3072], "BF16"),
    "diffusion_model.layers.41.proj.o.lora_B.weight": ([3072, 16], "BF16"),
    "diffusion_model.layers.41.proj.qkv.lora_A.weight": ([16, 3072], "BF16"),
    "diffusion_model.layers.41.proj.qkv.lora_B.weight": ([9216, 16], "BF16"),
}
