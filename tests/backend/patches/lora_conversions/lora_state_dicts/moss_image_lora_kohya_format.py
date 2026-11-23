"""A MOSS-Image LoRA in kohya's flattened names, Kandoo/MOSS-Image-Impressionism-LoRA (rank 16, BF16).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`Kandoo/MOSS-Image-Impressionism-LoRA/moss-image-impressionism-lora.safetensors`.

Subsetting rule: blocks [0, 41] (first, last, and one next to a missing block where the file has a gap) of the 42 it names, plus every key outside a block. Values are `(shape, dtype)`.

Keys are `lora_unet_` + the module path with its dots turned into underscores (`layers_0_proj_ffn_up_gate`) +
`lora_down.weight`/`lora_up.weight`, with an F32 `alpha` per layer. The name cannot be re-dotted on its own
(`ffn_up_gate` has underscores of its own), so a reader maps it through the table of modules.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "lora_unet_layers_0_proj_ffn_down.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_ffn_down.lora_down.weight": ([16, 8192], "BF16"),
    "lora_unet_layers_0_proj_ffn_down.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_layers_0_proj_ffn_up_gate.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_ffn_up_gate.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_layers_0_proj_ffn_up_gate.lora_up.weight": ([16384, 16], "BF16"),
    "lora_unet_layers_0_proj_k.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_k.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_layers_0_proj_k.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_layers_0_proj_o.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_o.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_layers_0_proj_o.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_layers_0_proj_q.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_q.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_layers_0_proj_q.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_layers_0_proj_v.alpha": ([], "F32"),
    "lora_unet_layers_0_proj_v.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_layers_0_proj_v.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_layers_41_proj_ffn_down.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_ffn_down.lora_down.weight": ([16, 8192], "BF16"),
    "lora_unet_layers_41_proj_ffn_down.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_layers_41_proj_ffn_up_gate.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_ffn_up_gate.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_layers_41_proj_ffn_up_gate.lora_up.weight": ([16384, 16], "BF16"),
    "lora_unet_layers_41_proj_k.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_k.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_layers_41_proj_k.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_layers_41_proj_o.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_o.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_layers_41_proj_o.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_layers_41_proj_q.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_q.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_layers_41_proj_q.lora_up.weight": ([3072, 16], "BF16"),
    "lora_unet_layers_41_proj_v.alpha": ([], "F32"),
    "lora_unet_layers_41_proj_v.lora_down.weight": ([16, 3072], "BF16"),
    "lora_unet_layers_41_proj_v.lora_up.weight": ([3072, 16], "BF16"),
}
