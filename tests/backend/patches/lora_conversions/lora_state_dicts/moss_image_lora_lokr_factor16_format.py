"""A MOSS-Image LoKr from ai-toolkit, aoleg/MOSS-Image-Kodachrome (factor 16, BF16).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`aoleg/MOSS-Image-Kodachrome/lokr-moss-kodachrome-v1.safetensors`.

Subsetting rule: blocks [0, 21, 23, 41] (first, last, and one next to a missing block where the file has a gap) of the 41 it names, plus every key outside a block. Values are `(shape, dtype)`.

LoKr sits on the fused `proj.qkv` alone (`lokr_w1` 16 x 16, `lokr_w2` 576 x 192: their Kronecker product is 9216
x 3072) and carries no `alpha` of its own; the attention's q, k and v keep a lone `alpha` each, left over from
the split layers. `proj.o` and the feed-forward are LoKr layers with an `alpha`. Block 22 is not trained. A third
of the fused rows is not a whole number of rows of `lokr_w2`'s 576: the split cuts through one.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.layers.0.proj.ffn_down.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.ffn_down.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.0.proj.ffn_down.lokr_w2": ([192, 512], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lokr_w2": ([1024, 192], "BF16"),
    "diffusion_model.layers.0.proj.k.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.o.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.o.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.0.proj.o.lokr_w2": ([192, 192], "BF16"),
    "diffusion_model.layers.0.proj.q.alpha": ([], "BF16"),
    "diffusion_model.layers.0.proj.qkv.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.0.proj.qkv.lokr_w2": ([576, 192], "BF16"),
    "diffusion_model.layers.0.proj.v.alpha": ([], "BF16"),
    "diffusion_model.layers.21.proj.ffn_down.alpha": ([], "BF16"),
    "diffusion_model.layers.21.proj.ffn_down.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.21.proj.ffn_down.lokr_w2": ([192, 512], "BF16"),
    "diffusion_model.layers.21.proj.ffn_up_gate.alpha": ([], "BF16"),
    "diffusion_model.layers.21.proj.ffn_up_gate.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.21.proj.ffn_up_gate.lokr_w2": ([1024, 192], "BF16"),
    "diffusion_model.layers.21.proj.k.alpha": ([], "BF16"),
    "diffusion_model.layers.21.proj.o.alpha": ([], "BF16"),
    "diffusion_model.layers.21.proj.o.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.21.proj.o.lokr_w2": ([192, 192], "BF16"),
    "diffusion_model.layers.21.proj.q.alpha": ([], "BF16"),
    "diffusion_model.layers.21.proj.qkv.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.21.proj.qkv.lokr_w2": ([576, 192], "BF16"),
    "diffusion_model.layers.21.proj.v.alpha": ([], "BF16"),
    "diffusion_model.layers.23.proj.ffn_down.alpha": ([], "BF16"),
    "diffusion_model.layers.23.proj.ffn_down.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.23.proj.ffn_down.lokr_w2": ([192, 512], "BF16"),
    "diffusion_model.layers.23.proj.ffn_up_gate.alpha": ([], "BF16"),
    "diffusion_model.layers.23.proj.ffn_up_gate.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.23.proj.ffn_up_gate.lokr_w2": ([1024, 192], "BF16"),
    "diffusion_model.layers.23.proj.k.alpha": ([], "BF16"),
    "diffusion_model.layers.23.proj.o.alpha": ([], "BF16"),
    "diffusion_model.layers.23.proj.o.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.23.proj.o.lokr_w2": ([192, 192], "BF16"),
    "diffusion_model.layers.23.proj.q.alpha": ([], "BF16"),
    "diffusion_model.layers.23.proj.qkv.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.23.proj.qkv.lokr_w2": ([576, 192], "BF16"),
    "diffusion_model.layers.23.proj.v.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.41.proj.ffn_down.lokr_w2": ([192, 512], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lokr_w2": ([1024, 192], "BF16"),
    "diffusion_model.layers.41.proj.k.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.o.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.o.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.41.proj.o.lokr_w2": ([192, 192], "BF16"),
    "diffusion_model.layers.41.proj.q.alpha": ([], "BF16"),
    "diffusion_model.layers.41.proj.qkv.lokr_w1": ([16, 16], "BF16"),
    "diffusion_model.layers.41.proj.qkv.lokr_w2": ([576, 192], "BF16"),
    "diffusion_model.layers.41.proj.v.alpha": ([], "BF16"),
}
