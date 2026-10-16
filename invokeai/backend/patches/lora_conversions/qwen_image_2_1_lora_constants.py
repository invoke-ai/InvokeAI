"""Which keys a Qwen-Image-2.1 LoRA may hold, shared by its identification and its conversion.

The published LoRAs target the module names diffusers uses, under a `diffusion_model.` (ComfyUI), `transformer.`
(diffusers/PEFT) or `lora_unet_` (Kohya, flattened) prefix -- with one exception: ComfyUI fuses each block's gated
MLP input into `img_mlp.gate_up`, which diffusers keeps as `gate_layer` (the first half of its output rows) and
`proj` (the second half), as the base checkpoint converter splits it.

Besides the Linears, a LoRA may carry full `diff`s on the RMSNorm weights (the official Turbo LoRA does). They are
merged into the weight whatever the transformer's format: a norm is never quantized, and a sidecar cannot patch it.
"""

import re
from typing import Any

QWEN_IMAGE_21_LORA_TRANSFORMER_PREFIX = "lora_transformer-"
# The norm diffs, kept apart so the denoise node can merge them while the Linears run as sidecars.
QWEN_IMAGE_21_LORA_NORM_PREFIX = "lora_transformer_norm-"

# Longest first, so `base_model.model.transformer.` wins over `base_model.model.`.
_PREFIXES = ("base_model.model.transformer.", "base_model.model.", "diffusion_model.", "transformer.")

# Each recognized parameter name and the value key the layer classes read. DoRA magnitudes come in two orientations
# (see `DoRALayer`): LyCORIS's `dora_scale` indexes the input dim, PEFT's and ai-toolkit's the output dim.
_PARAM_VALUE_KEYS = {
    "lora_A.weight": "lora_down.weight",
    "lora_B.weight": "lora_up.weight",
    "lora_down.weight": "lora_down.weight",
    "lora_up.weight": "lora_up.weight",
    "alpha": "alpha",
    "dora_scale": "dora_scale",
    "lora_magnitude_vector.weight": "dora_magnitude",
    # PEFT's own `get_peft_model_state_dict` writes it without the `.weight`.
    "lora_magnitude_vector": "dora_magnitude",
    "magnitude": "dora_magnitude",
    "diff": "diff",
    **{name: name for name in ("lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b", "lokr_t2")},
    **{name: name for name in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "hada_t1", "hada_t2")},
}

# Every Linear of the transformer, block-relative or top-level, plus ComfyUI's fused `gate_up`.
_BLOCK_MODULES = frozenset(
    {
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "img_mlp.gate_up",
        "img_mlp.gate_layer",
        "img_mlp.proj",
        "img_mlp.out",
    }
)
# The RMSNorm weights a full diff may target.
_BLOCK_NORMS = frozenset({"attn.norm_q", "attn.norm_k"})
_TOP_NORMS = frozenset({"txt_in.text_norm"})
_TOP_MODULES = frozenset(
    {
        "img_in",
        "proj_out",
        "norm_out.linear",
        "modulation.1",
        "txt_in.in_layer",
        "txt_in.out_layer",
        "time_text_embed.timestep_embedder.linear_1",
        "time_text_embed.timestep_embedder.linear_2",
    }
)
_BLOCK = re.compile(r"transformer_blocks\.(\d+)\.(.+)")
# Kohya flattens the dots, which makes a name ambiguous on its own (`to_out_0`), so it is mapped, not re-dotted.
_KOHYA_BLOCK = re.compile(r"transformer_blocks_(\d+)_(.+)")
_KOHYA_BLOCK_MODULES = {module.replace(".", "_"): module for module in _BLOCK_MODULES | _BLOCK_NORMS}
_KOHYA_TOP_MODULES = {module.replace(".", "_"): module for module in _TOP_MODULES | _TOP_NORMS}
_KOHYA_PREFIX = "lora_unet_"

GATE_UP = "img_mlp.gate_up"
# What a layer on the fused gate_up may hold to be split by its output rows: a low-rank update (its `up` rows),
# a PEFT/ai-toolkit DoRA (also its output-dim magnitude), or a full one. LoKR and LoHA mix the rows, and LyCORIS's
# input-dim `dora_scale` normalizes across both halves at once.
GATE_UP_SPLITTABLE = (
    frozenset({"lora_down.weight", "lora_up.weight"}),
    frozenset({"lora_down.weight", "lora_up.weight", "alpha"}),
    frozenset({"lora_down.weight", "lora_up.weight", "dora_magnitude"}),
    frozenset({"lora_down.weight", "lora_up.weight", "alpha", "dora_magnitude"}),
    frozenset({"diff"}),
)


def _module_path(name: str) -> str | None:
    """The diffusers module path a key's module part names, or None if this transformer has no such Linear."""
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.startswith(_KOHYA_PREFIX):
        flat = name[len(_KOHYA_PREFIX) :]
        if block := _KOHYA_BLOCK.fullmatch(flat):
            module = _KOHYA_BLOCK_MODULES.get(block.group(2))
            return f"transformer_blocks.{block.group(1)}.{module}" if module else None
        return _KOHYA_TOP_MODULES.get(flat)
    if block := _BLOCK.fullmatch(name):
        return name if block.group(2) in _BLOCK_MODULES | _BLOCK_NORMS else None
    return name if name in _TOP_MODULES | _TOP_NORMS else None


def is_norm(module: str) -> bool:
    """Whether a module path from `split_key` names an RMSNorm rather than a Linear."""
    if block := _BLOCK.fullmatch(module):
        return block.group(2) in _BLOCK_NORMS
    return module in _TOP_NORMS


def split_key(key: str) -> tuple[str, str] | None:
    """`(module path, value key)` for a key of a Qwen-Image-2.1 LoRA, or None for one that is not."""
    for param, value_key in _PARAM_VALUE_KEYS.items():
        if key.endswith("." + param):
            module = _module_path(key[: -len(param) - 1])
            return (module, value_key) if module is not None else None
    return None


def unsupported_keys(state_dict: dict[str | int, Any]) -> list[str]:
    """The keys that keep this state dict from being applied to Qwen-Image-2.1 as a whole.

    A key that names no module of this transformer (another model's, or a text encoder's), anything but a full
    `diff` on a norm, and a fused gate_up layer that cannot be split. Empty for a LoRA that applies in full.
    """
    unsupported: list[str] = []
    gate_up_layers: dict[str, set[str]] = {}
    for key in state_dict:
        if not isinstance(key, str) or (split := split_key(key)) is None:
            unsupported.append(str(key))
            continue
        module, value_key = split
        if is_norm(module) and value_key != "diff":
            unsupported.append(key)
        elif module.endswith(GATE_UP):
            gate_up_layers.setdefault(module, set()).add(value_key)
    unsupported += [module for module, values in gate_up_layers.items() if frozenset(values) not in GATE_UP_SPLITTABLE]
    return sorted(unsupported)
