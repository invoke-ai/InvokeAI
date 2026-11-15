"""A MOSS-Image LoHa, comfyanonymous/MOSS_image_MAX_ULTRA_hires_lora_test (F16).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`comfyanonymous/MOSS_image_MAX_ULTRA_hires_lora_test/moss_hires_lora_test_v0.7.safetensors`.

Subsetting rule: blocks [0, 41] (first, last, and one next to a missing block where the file has a gap) of the 42 it names, plus every key outside a block. Values are `(shape, dtype)`.

Keys are `diffusion_model.` + the module path + `hada_w1_a`/`hada_w1_b`/`hada_w2_a`/`hada_w2_b` + `alpha`, on a
fused `proj.qkv` (rank 64, 9216-row `_a` factors), `proj.o` and the feed-forward.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.layers.0.proj.ffn_down.alpha": ([], "F16"),
    "diffusion_model.layers.0.proj.ffn_down.hada_w1_a": ([3072, 32], "F16"),
    "diffusion_model.layers.0.proj.ffn_down.hada_w1_b": ([32, 8192], "F16"),
    "diffusion_model.layers.0.proj.ffn_down.hada_w2_a": ([3072, 32], "F16"),
    "diffusion_model.layers.0.proj.ffn_down.hada_w2_b": ([32, 8192], "F16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.alpha": ([], "F16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.hada_w1_a": ([16384, 32], "F16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.hada_w1_b": ([32, 3072], "F16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.hada_w2_a": ([16384, 32], "F16"),
    "diffusion_model.layers.0.proj.ffn_up_gate.hada_w2_b": ([32, 3072], "F16"),
    "diffusion_model.layers.0.proj.o.alpha": ([], "F16"),
    "diffusion_model.layers.0.proj.o.hada_w1_a": ([3072, 16], "F16"),
    "diffusion_model.layers.0.proj.o.hada_w1_b": ([16, 3072], "F16"),
    "diffusion_model.layers.0.proj.o.hada_w2_a": ([3072, 16], "F16"),
    "diffusion_model.layers.0.proj.o.hada_w2_b": ([16, 3072], "F16"),
    "diffusion_model.layers.0.proj.qkv.alpha": ([], "F16"),
    "diffusion_model.layers.0.proj.qkv.hada_w1_a": ([9216, 64], "F16"),
    "diffusion_model.layers.0.proj.qkv.hada_w1_b": ([64, 3072], "F16"),
    "diffusion_model.layers.0.proj.qkv.hada_w2_a": ([9216, 64], "F16"),
    "diffusion_model.layers.0.proj.qkv.hada_w2_b": ([64, 3072], "F16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.alpha": ([], "F16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.hada_w1_a": ([16384, 16], "F16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.hada_w1_b": ([16, 3072], "F16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.hada_w2_a": ([16384, 16], "F16"),
    "diffusion_model.layers.41.proj.ffn_up_gate.hada_w2_b": ([16, 3072], "F16"),
    "diffusion_model.layers.41.proj.o.alpha": ([], "F16"),
    "diffusion_model.layers.41.proj.o.hada_w1_a": ([3072, 16], "F16"),
    "diffusion_model.layers.41.proj.o.hada_w1_b": ([16, 3072], "F16"),
    "diffusion_model.layers.41.proj.o.hada_w2_a": ([3072, 16], "F16"),
    "diffusion_model.layers.41.proj.o.hada_w2_b": ([16, 3072], "F16"),
    "diffusion_model.layers.41.proj.qkv.alpha": ([], "F16"),
    "diffusion_model.layers.41.proj.qkv.hada_w1_a": ([9216, 32], "F16"),
    "diffusion_model.layers.41.proj.qkv.hada_w1_b": ([32, 3072], "F16"),
    "diffusion_model.layers.41.proj.qkv.hada_w2_a": ([9216, 32], "F16"),
    "diffusion_model.layers.41.proj.qkv.hada_w2_b": ([32, 3072], "F16"),
}
