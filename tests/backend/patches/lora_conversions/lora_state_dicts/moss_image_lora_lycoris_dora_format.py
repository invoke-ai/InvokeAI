"""A MOSS-Image DoRA from LyCORIS in kohya's names, a Civitai file (rank 32, BF16).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`mossloras/moss-image-lora-archive/2026-10-29/MOSS-Image/3026108_Asou--麻生--style---cel-shaded-flat-anime-style---Asou-麻生-平涂动漫/3436797/asou_style_moss_v1.safetensors`.

Subsetting rule: blocks [0, 41] (first, last, and one next to a missing block where the file has a gap) of the 42 it names, plus every key outside a block. Values are `(shape, dtype)`.

Keys are `lora_unet_` + the flattened module path + `lora_down.weight`/`lora_up.weight` + `alpha` +
`dora_scale` (out_features x 1, the output axis), on q, k, v, `proj.o` and the feed-forward.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "lora_unet_layers_0_proj_ffn_down.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_ffn_down.dora_scale": ([3072, 1], "BF16"),
    "lora_unet_layers_0_proj_ffn_down.lora_down.weight": ([32, 8192], "BF16"),
    "lora_unet_layers_0_proj_ffn_down.lora_up.weight": ([3072, 32], "BF16"),
    "lora_unet_layers_0_proj_ffn_up_gate.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_ffn_up_gate.dora_scale": ([16384, 1], "BF16"),
    "lora_unet_layers_0_proj_ffn_up_gate.lora_down.weight": ([32, 3072], "BF16"),
    "lora_unet_layers_0_proj_ffn_up_gate.lora_up.weight": ([16384, 32], "BF16"),
    "lora_unet_layers_0_proj_k.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_k.dora_scale": ([3072, 1], "BF16"),
    "lora_unet_layers_0_proj_k.lora_down.weight": ([32, 3072], "BF16"),
    "lora_unet_layers_0_proj_k.lora_up.weight": ([3072, 32], "BF16"),
    "lora_unet_layers_0_proj_o.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_o.dora_scale": ([3072, 1], "BF16"),
    "lora_unet_layers_0_proj_o.lora_down.weight": ([32, 3072], "BF16"),
    "lora_unet_layers_0_proj_o.lora_up.weight": ([3072, 32], "BF16"),
    "lora_unet_layers_0_proj_q.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_q.dora_scale": ([3072, 1], "BF16"),
    "lora_unet_layers_0_proj_q.lora_down.weight": ([32, 3072], "BF16"),
    "lora_unet_layers_0_proj_q.lora_up.weight": ([3072, 32], "BF16"),
    "lora_unet_layers_0_proj_v.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_v.dora_scale": ([3072, 1], "BF16"),
    "lora_unet_layers_0_proj_v.lora_down.weight": ([32, 3072], "BF16"),
    "lora_unet_layers_0_proj_v.lora_up.weight": ([3072, 32], "BF16"),
    "lora_unet_layers_41_proj_ffn_down.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_ffn_down.dora_scale": ([3072, 1], "BF16"),
    "lora_unet_layers_41_proj_ffn_down.lora_down.weight": ([32, 8192], "BF16"),
    "lora_unet_layers_41_proj_ffn_down.lora_up.weight": ([3072, 32], "BF16"),
    "lora_unet_layers_41_proj_ffn_up_gate.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_ffn_up_gate.dora_scale": ([16384, 1], "BF16"),
    "lora_unet_layers_41_proj_ffn_up_gate.lora_down.weight": ([32, 3072], "BF16"),
    "lora_unet_layers_41_proj_ffn_up_gate.lora_up.weight": ([16384, 32], "BF16"),
    "lora_unet_layers_41_proj_k.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_k.dora_scale": ([3072, 1], "BF16"),
    "lora_unet_layers_41_proj_k.lora_down.weight": ([32, 3072], "BF16"),
    "lora_unet_layers_41_proj_k.lora_up.weight": ([3072, 32], "BF16"),
    "lora_unet_layers_41_proj_o.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_o.dora_scale": ([3072, 1], "BF16"),
    "lora_unet_layers_41_proj_o.lora_down.weight": ([32, 3072], "BF16"),
    "lora_unet_layers_41_proj_o.lora_up.weight": ([3072, 32], "BF16"),
    "lora_unet_layers_41_proj_q.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_q.dora_scale": ([3072, 1], "BF16"),
    "lora_unet_layers_41_proj_q.lora_down.weight": ([32, 3072], "BF16"),
    "lora_unet_layers_41_proj_q.lora_up.weight": ([3072, 32], "BF16"),
    "lora_unet_layers_41_proj_v.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_v.dora_scale": ([3072, 1], "BF16"),
    "lora_unet_layers_41_proj_v.lora_down.weight": ([32, 3072], "BF16"),
    "lora_unet_layers_41_proj_v.lora_up.weight": ([3072, 32], "BF16"),
}
