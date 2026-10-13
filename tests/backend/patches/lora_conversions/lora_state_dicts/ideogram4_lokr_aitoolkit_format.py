"""Key layout of an ai-toolkit LoKR for Ideogram 4 (`Yofardev/kiki_ideogram4_v2`).

Captured on 2026-10-11 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`Yofardev/kiki_ideogram4_v2/kiki_ideogram4_v2.safetensors`.

Subsetting rule: `diffusion_model.layers` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

Same modules as the low-rank files, as Kronecker factors: `lokr_w1` [8, 8] times a full `lokr_w2`, e.g.
[1728, 576] for the fused `attention.qkv` (13824 x 4608). Neither factor is at an Ideogram 4 width, so identification
has to compute the patched shape from both.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.layers.0.adaln_modulation.alpha": ([], "BF16"),
    "diffusion_model.layers.0.adaln_modulation.lokr_w1": ([8, 8], "BF16"),
    "diffusion_model.layers.0.adaln_modulation.lokr_w2": ([2304, 64], "BF16"),
    "diffusion_model.layers.0.attention.o.alpha": ([], "BF16"),
    "diffusion_model.layers.0.attention.o.lokr_w1": ([8, 8], "BF16"),
    "diffusion_model.layers.0.attention.o.lokr_w2": ([576, 576], "BF16"),
    "diffusion_model.layers.0.attention.qkv.alpha": ([], "BF16"),
    "diffusion_model.layers.0.attention.qkv.lokr_w1": ([8, 8], "BF16"),
    "diffusion_model.layers.0.attention.qkv.lokr_w2": ([1728, 576], "BF16"),
    "diffusion_model.layers.0.feed_forward.w1.alpha": ([], "BF16"),
    "diffusion_model.layers.0.feed_forward.w1.lokr_w1": ([8, 8], "BF16"),
    "diffusion_model.layers.0.feed_forward.w1.lokr_w2": ([1536, 576], "BF16"),
    "diffusion_model.layers.0.feed_forward.w2.alpha": ([], "BF16"),
    "diffusion_model.layers.0.feed_forward.w2.lokr_w1": ([8, 8], "BF16"),
    "diffusion_model.layers.0.feed_forward.w2.lokr_w2": ([576, 1536], "BF16"),
    "diffusion_model.layers.0.feed_forward.w3.alpha": ([], "BF16"),
    "diffusion_model.layers.0.feed_forward.w3.lokr_w1": ([8, 8], "BF16"),
    "diffusion_model.layers.0.feed_forward.w3.lokr_w2": ([1536, 576], "BF16"),
}
