"""A MOSS-Image LoKr from ai-toolkit with an automatic factor, a Civitai file (BF16).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`mossloras/moss-image-lora-archive/2026-11-09/MOSS-Image/3043908_Bicep-Heart-Meme/3460156/moss-bicep-heart-meme-v1_000003407.safetensors`.

Subsetting rule: blocks [0, 21, 23, 41] (first, last, and one next to a missing block where the file has a gap) of the 41 it names, plus every key outside a block. Values are `(shape, dtype)`.

As `moss_image_lora_lokr_factor16_format`, with `lokr_w1` 96 x 48 and `lokr_w2` 96 x 64 on the fused `proj.qkv`:
96 rows of `lokr_w2` divide the 3072 rows of each of q, k and v into 32 whole blocks, so the split needs no cut
through a block.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.layers.0.proj.ffn_down.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.ffn_down.lokr_w1": ([48, 64], "BF16"),
    "diffusion_model.layers.0.proj.ffn_down.lokr_w2": ([64, 128], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lokr_w1": ([128, 48], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lokr_w2": ([128, 64], "BF16"),
    "diffusion_model.layers.0.proj.k.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.o.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.o.lokr_w1": ([48, 48], "BF16"),
    "diffusion_model.layers.0.proj.o.lokr_w2": ([64, 64], "BF16"),
    "diffusion_model.layers.0.proj.q.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.qkv.lokr_w1": ([96, 48], "BF16"),
    "diffusion_model.layers.0.proj.qkv.lokr_w2": ([96, 64], "BF16"),
    "diffusion_model.layers.0.proj.v.alpha": ([], "BF16"),
    "diffusion_model.layers.21.proj.ffn_down.alpha": ([], "BF16"),
    "diffusion_model.layers.21.proj.ffn_down.lokr_w1": ([48, 64], "BF16"),
    "diffusion_model.layers.21.proj.ffn_down.lokr_w2": ([64, 128], "BF16"),
    "diffusion_model.layers.21.proj.ffn_up_gate.alpha": ([], "BF16"),
    "diffusion_model.layers.21.proj.ffn_up_gate.lokr_w1": ([128, 48], "BF16"),
    "diffusion_model.layers.21.proj.ffn_up_gate.lokr_w2": ([128, 64], "BF16"),
    "diffusion_model.layers.21.proj.k.alpha": ([], "BF16"),
    "diffusion_model.layers.21.proj.o.alpha": ([], "BF16"),
    "diffusion_model.layers.21.proj.o.lokr_w1": ([48, 48], "BF16"),
    "diffusion_model.layers.21.proj.o.lokr_w2": ([64, 64], "BF16"),
    "diffusion_model.layers.21.proj.q.alpha": ([], "BF16"),
    "diffusion_model.layers.21.proj.qkv.lokr_w1": ([96, 48], "BF16"),
    "diffusion_model.layers.21.proj.qkv.lokr_w2": ([96, 64], "BF16"),
    "diffusion_model.layers.21.proj.v.alpha": ([], "BF16"),
    "diffusion_model.layers.23.proj.ffn_down.alpha": ([], "BF16"),
    "diffusion_model.layers.23.proj.ffn_down.lokr_w1": ([48, 64], "BF16"),
    "diffusion_model.layers.23.proj.ffn_down.lokr_w2": ([64, 128], "BF16"),
    "diffusion_model.layers.23.proj.ffn_up_gate.alpha": ([], "BF16"),
    "diffusion_model.layers.23.proj.ffn_up_gate.lokr_w1": ([128, 48], "BF16"),
    "diffusion_model.layers.23.proj.ffn_up_gate.lokr_w2": ([128, 64], "BF16"),
    "diffusion_model.layers.23.proj.k.alpha": ([], "BF16"),
    "diffusion_model.layers.23.proj.o.alpha": ([], "BF16"),
    "diffusion_model.layers.23.proj.o.lokr_w1": ([48, 48], "BF16"),
    "diffusion_model.layers.23.proj.o.lokr_w2": ([64, 64], "BF16"),
    "diffusion_model.layers.23.proj.q.alpha": ([], "BF16"),
    "diffusion_model.layers.23.proj.qkv.lokr_w1": ([96, 48], "BF16"),
    "diffusion_model.layers.23.proj.qkv.lokr_w2": ([96, 64], "BF16"),
    "diffusion_model.layers.23.proj.v.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.lokr_w1": ([48, 64], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.lokr_w2": ([64, 128], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lokr_w1": ([128, 48], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lokr_w2": ([128, 64], "BF16"),
    "diffusion_model.layers.41.proj.k.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.o.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.o.lokr_w1": ([48, 48], "BF16"),
    "diffusion_model.layers.41.proj.o.lokr_w2": ([64, 64], "BF16"),
    "diffusion_model.layers.41.proj.q.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.qkv.lokr_w1": ([96, 48], "BF16"),
    "diffusion_model.layers.41.proj.qkv.lokr_w2": ([96, 64], "BF16"),
    "diffusion_model.layers.41.proj.v.alpha": ([], "BF16"),
}
