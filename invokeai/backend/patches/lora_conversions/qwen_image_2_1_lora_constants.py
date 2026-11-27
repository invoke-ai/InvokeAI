"""Which keys a Qwen-Image-2.1 LoRA may hold, and the shapes their layers must have, shared by its identification
and its conversion.

The published LoRAs target the module names diffusers uses, under a `diffusion_model.` (ComfyUI), `transformer.`
(diffusers/PEFT) or `lora_unet_` (Kohya, flattened) prefix -- with one exception: ComfyUI fuses each block's gated
MLP input into `img_mlp.gate_up`, which diffusers keeps as `gate_layer` (the first half of its output rows) and
`proj` (the second half), as the base checkpoint converter splits it.

Besides the Linears, a LoRA may carry full `diff`s on the RMSNorm weights (the official Turbo LoRA does). They are
merged into the weight whatever the transformer's format: a norm is never quantized, and a sidecar cannot patch it.

Every layer must update a weight of exactly its module's shape. Another 4096-wide model's LoRA can share module names
(FLUX.2 Klein 9B's `proj_out` and `norm_out.linear`, 128 and 8192 rows) and would otherwise install, then fail at
generation on an impossible reshape or a mismatched output.
"""

import itertools
import re
from typing import Any

import torch

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

# The released transformer's dimensions (`transformer/config.json` of Qwen/Qwen-Image-2.1, also the class defaults):
# 32 attention heads of 128 make the hidden size, `mlp_ratio` 3 the MLP width, 64 latent channels in and out at
# `patch_size` 1, and the text encoder is 4096 wide (`context_in_dim`). `QwenImage21Transformer2DModel` embeds the
# timestep from 256 channels and modulates with 4 x hidden. The checkpoint's weights have exactly these shapes.
_HEAD_DIM = 128
_HIDDEN = 32 * _HEAD_DIM
_MLP = 3 * _HIDDEN
_LATENT_CHANNELS = 64
_CONTEXT = 4096
_TIMESTEP_CHANNELS = 256

# Every Linear of the transformer, block-relative or top-level, plus ComfyUI's fused `gate_up`, with its weight's
# (out_features, in_features).
_BLOCK_LINEARS = {
    "attn.to_q": (_HIDDEN, _HIDDEN),
    "attn.to_k": (_HIDDEN, _HIDDEN),
    "attn.to_v": (_HIDDEN, _HIDDEN),
    "attn.to_out.0": (_HIDDEN, _HIDDEN),
    "img_mlp.gate_up": (2 * _MLP, _HIDDEN),
    "img_mlp.gate_layer": (_MLP, _HIDDEN),
    "img_mlp.proj": (_MLP, _HIDDEN),
    "img_mlp.out": (_HIDDEN, _MLP),
}
_TOP_LINEARS = {
    "img_in": (_HIDDEN, _LATENT_CHANNELS),
    "proj_out": (_LATENT_CHANNELS, _HIDDEN),
    "norm_out.linear": (_HIDDEN, _HIDDEN),
    "modulation.1": (4 * _HIDDEN, _HIDDEN),
    "txt_in.in_layer": (_HIDDEN, _CONTEXT),
    "txt_in.out_layer": (_HIDDEN, _HIDDEN),
    "time_text_embed.timestep_embedder.linear_1": (_HIDDEN, _TIMESTEP_CHANNELS),
    "time_text_embed.timestep_embedder.linear_2": (_HIDDEN, _HIDDEN),
}
# The RMSNorm weights a full diff may target: each head's query and key, and the text encoder's output.
_BLOCK_NORMS = {"attn.norm_q": (_HEAD_DIM,), "attn.norm_k": (_HEAD_DIM,)}
_TOP_NORMS = {"txt_in.text_norm": (_CONTEXT,)}
_BLOCK_WEIGHTS = _BLOCK_LINEARS | _BLOCK_NORMS
_TOP_WEIGHTS = _TOP_LINEARS | _TOP_NORMS
# diffusers' `QwenImage21Transformer2DModel` default `num_layers`, which every released checkpoint has.
_NUM_BLOCKS = 32
# A block index as the transformer spells it, without leading zeros: `01` would be block 1 under a second name.
_BLOCK = re.compile(r"transformer_blocks\.(0|[1-9]\d?)\.(.+)")
# Kohya flattens the dots, which makes a name ambiguous on its own (`to_out_0`), so it is mapped, not re-dotted.
_KOHYA_BLOCK = re.compile(r"transformer_blocks_(0|[1-9]\d?)_(.+)")
_KOHYA_BLOCK_MODULES = {module.replace(".", "_"): module for module in _BLOCK_WEIGHTS}
_KOHYA_TOP_MODULES = {module.replace(".", "_"): module for module in _TOP_WEIGHTS}
_KOHYA_PREFIX = "lora_unet_"


def _layer_value_sets() -> frozenset[frozenset[str]]:
    """Every set of value keys that builds one layer reading all of them: nothing missing, nothing it would ignore.

    The kinds `any_lora_layer_from_state_dict` builds: low-rank, DoRA with either magnitude, LoKR with each operand
    whole or factored, LoHA, and a full diff, which has no rank for an alpha to scale. (A LoKR with both operands
    whole has none either and scales by 1 whatever its alpha; the alpha is accepted there, as it changes nothing.)
    LyCORIS writes Tucker cores (`lokr_t2`, `hada_t1`/`hada_t2`) only for convolution kernels, and this transformer
    is all Linears, so a layer with one is refused. No Linear here has a bias, so no bias key is recognized.
    """
    low_rank = frozenset({"lora_down.weight", "lora_up.weight"})
    lokr_w1 = (frozenset({"lokr_w1"}), frozenset({"lokr_w1_a", "lokr_w1_b"}))
    lokr_w2 = (frozenset({"lokr_w2"}), frozenset({"lokr_w2_a", "lokr_w2_b"}))
    ranked = [
        low_rank,
        low_rank | {"dora_scale"},
        low_rank | {"dora_magnitude"},
        *(w1 | w2 for w1, w2 in itertools.product(lokr_w1, lokr_w2)),
        frozenset({"hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"}),
    ]
    return frozenset([*ranked, *(layer | {"alpha"} for layer in ranked), frozenset({"diff"})])


_LAYERS = _layer_value_sets()

GATE_UP = "img_mlp.gate_up"
# The values of a layer on the fused gate_up that split along its output rows. `lora_down` reads the input both
# halves share and `alpha` is a scalar, so each half takes them whole.
GATE_UP_ROW_KEYS = frozenset({"lora_up.weight", "dora_magnitude", "diff"})
# What a gate_up layer may be to be split: low-rank, PEFT/ai-toolkit DoRA (an output-dim magnitude), or full. LoKR
# and LoHA mix the rows, and LyCORIS's input-dim `dora_scale` normalizes across both halves at once.
_GATE_UP_SPLITTABLE = frozenset(layer for layer in _LAYERS if layer <= GATE_UP_ROW_KEYS | {"lora_down.weight", "alpha"})


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
            exists = module is not None and int(block.group(1)) < _NUM_BLOCKS
            return f"transformer_blocks.{block.group(1)}.{module}" if exists else None
        return _KOHYA_TOP_MODULES.get(flat)
    if block := _BLOCK.fullmatch(name):
        exists = block.group(2) in _BLOCK_WEIGHTS and int(block.group(1)) < _NUM_BLOCKS
        return name if exists else None
    return name if name in _TOP_WEIGHTS else None


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


def _weight_shape(module: str) -> tuple[int, ...]:
    """The shape of the weight a module path from `split_key` names."""
    if block := _BLOCK.fullmatch(module):
        return _BLOCK_WEIGHTS[block.group(2)]
    return _TOP_WEIGHTS[module]


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def _product(a: Any, b: Any) -> tuple[int, ...] | None:
    """The shape of `a @ b`, or None unless both are matrices whose inner sizes agree."""
    a_shape, b_shape = _shape(a), _shape(b)
    if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[1] != b_shape[0]:
        return None
    return a_shape[0], b_shape[1]


def _update_shape(layer: dict[str, Any]) -> tuple[int, ...] | None:
    """The shape of the weight update one layer's values make, or None if they make none.

    A full diff is the update itself; a low-rank (or DoRA) layer's is up @ down; LoHA's the elementwise product of
    two low-rank products, which must agree; LoKR's the Kronecker product of two matrices, each given whole or as a
    low-rank product. Every factor is a matrix: a convolution-shaped one updates no Linear.
    """
    if "diff" in layer:
        return _shape(layer["diff"])
    if "lora_up.weight" in layer:
        return _product(layer["lora_up.weight"], layer["lora_down.weight"])
    if "hada_w1_a" in layer:
        w1 = _product(layer["hada_w1_a"], layer["hada_w1_b"])
        return w1 if w1 == _product(layer["hada_w2_a"], layer["hada_w2_b"]) else None
    w1 = _shape(layer["lokr_w1"]) if "lokr_w1" in layer else _product(layer["lokr_w1_a"], layer["lokr_w1_b"])
    w2 = _shape(layer["lokr_w2"]) if "lokr_w2" in layer else _product(layer["lokr_w2_a"], layer["lokr_w2_b"])
    if w1 is None or w2 is None or len(w1) != 2 or len(w2) != 2:
        return None
    return w1[0] * w2[0], w1[1] * w2[1]


def _magnitude_fits(layer: dict[str, Any], out_features: int, in_features: int) -> bool:
    """Whether a DoRA layer's magnitude, if it has one, has an entry per row or column of its weight in the
    orientation `DoRALayer` reads it in.

    PEFT's and ai-toolkit's magnitude is one per output row, `(out,)` or `(out, 1)`: rows, as the fused gate_up's
    split cuts it along its first dimension. LyCORIS's `dora_scale` is per output row only in the shapes
    `DoRALayer._dora_scale_is_out_dim` reads as such; any other is one per input column, `(1, in)` or `(in,)`.
    """
    if "dora_magnitude" in layer:
        shape = _shape(layer["dora_magnitude"])
        return len(shape) > 0 and shape[0] == out_features and all(size == 1 for size in shape[1:])
    if "dora_scale" in layer:
        # Imported here: `dora_layer` imports the model manager's loaders, which import the configs that import this.
        from invokeai.backend.patches.layers.dora_layer import DoRALayer

        scale = layer["dora_scale"]
        if isinstance(scale, torch.Tensor) and DoRALayer._dora_scale_is_out_dim(scale, out_features):
            return True
        return _shape(scale) in {(1, in_features), (in_features,)}
    return True


def _fits(module: str, layer: dict[str, Any]) -> bool:
    """Whether one layer's values update a weight of exactly its module's shape. For the fused gate_up, whose rows
    are twice the MLP width, that also makes the two halves it splits into whole."""
    shape = _weight_shape(module)
    if _update_shape(layer) != shape:
        return False
    return len(shape) == 1 or _magnitude_fits(layer, *shape)


def unsupported_keys(state_dict: dict[str | int, Any]) -> list[str]:
    """The keys that keep this state dict from being applied to Qwen-Image-2.1 as a whole; empty for a LoRA that
    applies in full.

    A key that names no module of this transformer (another model's, or a text encoder's), and every key that
    gives a value another key of its module gives too (two prefixes, or two spellings of one parameter). All keys
    of a module whose values are not exactly one layer (see `_layer_value_sets`): half a pair, an alpha or a
    magnitude alone, two layer kinds at once. All keys of a layer that does not update a weight of exactly its
    module's shape (see `_fits`). On a norm only a full `diff` applies. A fused gate_up must be a layer that splits by
    output rows, in a block that does not also name the gate_layer or proj it becomes.
    """
    unsupported: set[str] = set()
    modules: dict[str, dict[str, list[str]]] = {}
    for key in state_dict:
        if not isinstance(key, str) or (split := split_key(key)) is None:
            unsupported.add(str(key))
            continue
        module, value_key = split
        modules.setdefault(module, {}).setdefault(value_key, []).append(key)

    for module, values in modules.items():
        unsupported.update(key for spellings in values.values() if len(spellings) > 1 for key in spellings)
        layer = frozenset(values)
        if is_norm(module):
            applies = layer == {"diff"}
        elif module.endswith(GATE_UP):
            stem = module[: -len("gate_up")]
            applies = (
                layer in _GATE_UP_SPLITTABLE and f"{stem}gate_layer" not in modules and f"{stem}proj" not in modules
            )
        else:
            applies = layer in _LAYERS
        if applies:
            applies = _fits(module, {value_key: state_dict[spellings[0]] for value_key, spellings in values.items()})
        if not applies:
            unsupported.update(key for spellings in values.values() for key in spellings)
    return sorted(unsupported)
