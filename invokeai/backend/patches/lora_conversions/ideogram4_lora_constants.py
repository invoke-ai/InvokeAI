# Ideogram 4 LoRA prefix constants and key-shape detection helpers.
#
# Ideogram 4 LoRAs come in two namings, both measured on published files:
#
# - ai-toolkit (also what Civitai's Ideogram LoRAs are): the vendored transformer's own names behind ComfyUI's
#   prefix, with the attention projections fused --
#       diffusion_model.layers.0.attention.qkv.lora_A.weight        [r, 4608]  /  lora_B [13824, r]
#       diffusion_model.layers.0.attention.o.lora_A.weight
#       diffusion_model.layers.0.feed_forward.w1.lora_A.weight      (w1 / w2 / w3, SwiGLU)
#       diffusion_model.layers.0.adaln_modulation.lora_A.weight
#   plus LoKR files over the same modules (`lokr_w1` / `lokr_w2`), and, in ostris' Unconditional LoRA, top-level
#   `input_proj`, `llm_cond_proj`, `adaln_proj`, `t_embedding.mlp_{in,out}`, `final_layer.{linear,adaln_modulation}`.
# - diffusers' training script: `transformer.layers.N.attention.to_{q,k,v,out.0}` with the projections split, and
#   the feed-forward under the same names as above.
#
# The detection helpers are shared with ``configs/lora.py`` so the Ideogram 4 probe, the Z-Image probe's exclusion
# and the conversion agree. This file imports nothing from the config package, which keeps it free of cycles.

import re
from typing import Any

import torch

IDEOGRAM4_LORA_TRANSFORMER_PREFIX = "lora_transformer-"

# Hidden width, fused qkv rows, SwiGLU width and per-block modulation rows (4 x 4608) of the released transformer
# (`Ideogram4Config`). Z-Image keeps its blocks under the same `layers.N.attention` / `feed_forward.w1..3` names at
# width 3840 (qkv 11520, FFN 10240, modulation 15360), so a name alone cannot tell the two apart; a tensor at one of
# these widths can.
IDEOGRAM4_WIDTHS = frozenset({4608, 13824, 12288, 18432})

# The container prefixes the conversion strips, longest first so a nested prefix is not left with a tail.
# `conditional_transformer.` is how a LoRA over the pipeline's conditional branch is saved by diffusers.
IDEOGRAM4_LORA_KEY_PREFIXES = (
    "base_model.model.transformer.",
    "model.diffusion_model.",
    "conditional_transformer.",
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
)

_PREFIX_RE = r"^(?:" + "|".join(re.escape(p) for p in IDEOGRAM4_LORA_KEY_PREFIXES) + r")?"

_IDEOGRAM4_MODULE_RE = re.compile(
    _PREFIX_RE
    + r"(?:layers\.\d+\.(?:attention\.(?:qkv|o|to_q|to_k|to_v|to_out\.0)|feed_forward\.w[123]|adaln_modulation)"
    + r"|input_proj|llm_cond_proj|adaln_proj|t_embedding\.mlp_(?:in|out)|final_layer\.(?:linear|adaln_modulation))\."
)
_SPLIT_QKV_RE = re.compile(r"\.attention\.to_[qkv]$")
# Names Ideogram 4 never uses, but its neighbours under `layers.N` do.
_NON_IDEOGRAM4_ANTI_RES = (
    re.compile(_PREFIX_RE + r"layers\.\d+\.(?:self_attention|mlp)\."),  # ERNIE-Image
    re.compile(_PREFIX_RE + r"layers\.\d+\.(?:attention\.out|adaLN_modulation)\."),  # Z-Image
    re.compile(r"(?:^|\.)(?:context_refiner|noise_refiner|x_embedder|cap_embedder|t_embedder)\."),  # Z-Image
    re.compile(r"(?:^|\.|_)(?:double_blocks|single_blocks|single_transformer_blocks|transformer_blocks)[\._]\d+"),
    re.compile(r"(?:^|\.)blocks\.\d+\."),  # Wan / Anima / MiniMax H3
    re.compile(r"(?:^|_)lora_(?:unet|te\d?)_"),  # kohya-flattened keys
)

# Low-rank and LoKR parameter names the conversion builds layers from. LoHA has not been seen for this family, and
# DoRA is refused: the sidecar wrappers of the quantized builds (GGUF, int8, fp8, nf4) hand a patch only a
# placeholder weight, and DoRA needs the real one.
IDEOGRAM4_LORA_PARAM_NAMES = (
    "lora_A.weight",
    "lora_B.weight",
    "lora_down.weight",
    "lora_up.weight",
    "alpha",
    "lokr_w1",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_w2",
    "lokr_w2_a",
    "lokr_w2_b",
    "lokr_t2",
)


def strip_ideogram4_prefix(layer_path: str) -> str:
    """The layer path without the one container prefix in front of the module name."""
    for prefix in IDEOGRAM4_LORA_KEY_PREFIXES:
        if layer_path.startswith(prefix):
            return layer_path[len(prefix) :]
    return layer_path


def split_ideogram4_lora_key(key: str) -> tuple[str, str] | None:
    """Split a key into ``(layer path, parameter name)``, or ``None`` if its tail is not a supported parameter."""
    for param_name in IDEOGRAM4_LORA_PARAM_NAMES:
        if key.endswith("." + param_name):
            return key[: -len(param_name) - 1], param_name
    return None


def unsupported_ideogram4_lora_keys(str_keys: list[str]) -> list[str]:
    """Keys whose tail is not a parameter the conversion can build a layer from (DoRA, LoHA, biases, ...)."""
    return [k for k in str_keys if split_ideogram4_lora_key(k) is None]


def _factor_shape(values: dict[str, torch.Tensor], name: str) -> tuple[int, int] | None:
    """The (rows, cols) of a LoKR factor, whether stored whole (`lokr_w1`) or as a low-rank pair (`_a` @ `_b`)."""
    if name in values:
        return int(values[name].shape[0]), int(values[name].shape[1])
    if f"{name}_a" in values and f"{name}_b" in values:
        return int(values[f"{name}_a"].shape[0]), int(values[f"{name}_b"].shape[1])
    return None


def layer_delta_shape(values: dict[str, torch.Tensor]) -> tuple[int, int] | None:
    """The (out, in) shape of the weight a layer's tensors patch, from the tensors alone."""
    down = values.get("lora_A.weight", values.get("lora_down.weight"))
    up = values.get("lora_B.weight", values.get("lora_up.weight"))
    if down is not None and up is not None:
        return int(up.shape[0]), int(down.shape[1])
    w1, w2 = _factor_shape(values, "lokr_w1"), _factor_shape(values, "lokr_w2")
    if w1 is not None and w2 is not None:
        return w1[0] * w2[0], w1[1] * w2[1]
    return None


def group_ideogram4_lora_layers(state_dict: dict[str | int, Any]) -> dict[str, dict[str, torch.Tensor]]:
    """The tensors of every supported key, grouped by layer path as written in the file (prefix included)."""
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and (split := split_ideogram4_lora_key(key)) is not None:
            grouped.setdefault(split[0], {})[split[1]] = value
    return grouped


def has_ideogram4_lora_keys(state_dict: dict[str | int, Any]) -> bool:
    """True if the state dict names Ideogram 4 modules and patches at least one of them at an Ideogram 4 width."""
    str_keys = [k for k in state_dict if isinstance(k, str)]
    if not any(_IDEOGRAM4_MODULE_RE.search(k) for k in str_keys):
        return False
    if any(anti.search(k) for k in str_keys for anti in _NON_IDEOGRAM4_ANTI_RES):
        return False
    for layer_path, values in group_ideogram4_lora_layers(state_dict).items():
        if _IDEOGRAM4_MODULE_RE.search(layer_path + "."):
            shape = layer_delta_shape(values)
            if shape is not None and (shape[0] in IDEOGRAM4_WIDTHS or shape[1] in IDEOGRAM4_WIDTHS):
                return True
    return False


def ideogram4_lora_refusal(state_dict: dict[str | int, Any]) -> str | None:
    """Why an Ideogram 4 LoRA cannot be applied, or ``None`` if every tensor converts.

    The probe refuses the install with this message and the conversion raises it, so a file is never accepted and
    then half-applied -- layers the conversion could not place would otherwise be dropped with only a log warning.
    """
    unsupported = unsupported_ideogram4_lora_keys([k for k in state_dict if isinstance(k, str)])
    if unsupported:
        return (
            f"{len(unsupported)} tensor(s) are not low-rank or LoKR parameters, e.g. {unsupported[0]!r}. "
            "Supported are lora_A/lora_B, lora_down/lora_up and LoKR layers; LoHA and DoRA are not."
        )
    grouped = group_ideogram4_lora_layers(state_dict)
    unknown = [path for path in grouped if not _IDEOGRAM4_MODULE_RE.search(path + ".")]
    if unknown:
        return f"{len(unknown)} layer(s) target no Ideogram 4 transformer module, e.g. {unknown[0]!r}."
    # Half a low-rank pair, one Kronecker factor, or an `alpha` alone: nothing the conversion could build a layer from.
    incomplete = [path for path, values in grouped.items() if layer_delta_shape(values) is None]
    if incomplete:
        return (
            f"layer {incomplete[0]!r} is incomplete (has {sorted(grouped[incomplete[0]])}); the file is truncated or "
            "was edited."
        )
    # The split projections are fused into `attention.qkv`, which would then replace a fused layer of the same block.
    stripped = {strip_ideogram4_prefix(path) for path in grouped}
    collisions = sorted(
        path
        for path in stripped
        if path.endswith(".attention.qkv") and any(f"{path[: -len('qkv')]}to_{p}" in stripped for p in "qkv")
    )
    if collisions:
        return f"{collisions[0]!r} is trained both fused and as split q/k/v projections; only one can be applied."
    split_non_low_rank = [
        path
        for path, values in grouped.items()
        if _SPLIT_QKV_RE.search(path) and not ({"lora_A.weight", "lora_down.weight"} & values.keys())
    ]
    if split_non_low_rank:
        return (
            f"layer {split_non_low_rank[0]!r} is not low-rank. Split q/k/v projections are fused into the "
            "transformer's single `attention.qkv`, which only works for low-rank pairs."
        )
    return None
