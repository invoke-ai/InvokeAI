"""Wan 2.2 LoRA conversion utilities.

Wan LoRAs target the ``WanTransformer3DModel`` attention and FFN layers. We
normalise every supported source layout to the diffusers parameter-path naming
the loaded model uses at runtime (``blocks.<idx>.attn1.to_q``,
``blocks.<idx>.attn2.to_k``, ``blocks.<idx>.ffn.net.0.proj``, etc.).

Supported source layouts:

- **Diffusers PEFT**: ``[transformer.|base_model.model.transformer.]blocks.X.attn1.to_q.lora_A.weight``
- **Native PEFT** (ComfyUI / Wan-AI native naming, with diffusion_model or transformer prefix):
  ``diffusion_model.blocks.X.self_attn.q.lora_A.weight``
- **Kohya** in either naming: ``lora_unet_blocks_X_attn1_to_q.lora_down.weight``
  or ``lora_unet_blocks_X_self_attn_q.lora_down.weight``
"""

import re
from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.wan_lora_constants import (
    WAN_LORA_TRANSFORMER_PREFIX,
    has_wan_kohya_keys,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# Kohya layer-name regex: lora_unet_blocks_<idx>_<rest>
_KOHYA_KEY_REGEX = re.compile(r"lora_unet_blocks_(\d+)_(.*)")


# Kohya submodule name -> diffusers parameter-path tail.
#
# Longest-match-first ordering matters because some keys are prefixes of others
# (e.g. ``attn1_to_q`` vs ``attn1_to_out_0``). The lookup is exact (not prefix),
# so this is purely cosmetic, but kept consistent with QwenImage's convention.
_KOHYA_SUBMODULE_MAP: list[tuple[str, str]] = [
    # --- Diffusers naming ---
    # Self-attention (attn1)
    ("attn1_to_q", "attn1.to_q"),
    ("attn1_to_k", "attn1.to_k"),
    ("attn1_to_v", "attn1.to_v"),
    ("attn1_to_out_0", "attn1.to_out.0"),
    ("attn1_norm_q", "attn1.norm_q"),
    ("attn1_norm_k", "attn1.norm_k"),
    # Cross-attention (attn2)
    ("attn2_to_q", "attn2.to_q"),
    ("attn2_to_k", "attn2.to_k"),
    ("attn2_to_v", "attn2.to_v"),
    ("attn2_to_out_0", "attn2.to_out.0"),
    ("attn2_norm_q", "attn2.norm_q"),
    ("attn2_norm_k", "attn2.norm_k"),
    # FFN diffusers
    ("ffn_net_0_proj", "ffn.net.0.proj"),
    ("ffn_net_2", "ffn.net.2"),
    # --- Native naming (mapped onto diffusers paths) ---
    # self_attn -> attn1
    ("self_attn_q", "attn1.to_q"),
    ("self_attn_k", "attn1.to_k"),
    ("self_attn_v", "attn1.to_v"),
    ("self_attn_o", "attn1.to_out.0"),
    ("self_attn_norm_q", "attn1.norm_q"),
    ("self_attn_norm_k", "attn1.norm_k"),
    # cross_attn -> attn2
    ("cross_attn_q", "attn2.to_q"),
    ("cross_attn_k", "attn2.to_k"),
    ("cross_attn_v", "attn2.to_v"),
    ("cross_attn_o", "attn2.to_out.0"),
    ("cross_attn_norm_q", "attn2.norm_q"),
    ("cross_attn_norm_k", "attn2.norm_k"),
    # FFN native
    ("ffn_0", "ffn.net.0.proj"),
    ("ffn_2", "ffn.net.2"),
]


# Layer-path rules used for PEFT-style keys: applied as substring replacements
# to the *layer path* (everything between an optional prefix and the LoRA suffix).
# Order matters — see ``convert_wan_transformer_to_diffusers`` in diffusers for
# the equivalent state-dict-key rules. We use trailing-dot semantics so e.g.
# ``.q.`` matches ``self_attn.q.something`` but not ``norm_q``.
#
# Paths are augmented with a sentinel trailing ``.`` before applying these
# rules so that bare endings like ``blocks.0.self_attn.q`` get rewritten as
# ``blocks.0.attn1.to_q``.
_NATIVE_TO_DIFFUSERS_PATH_RULES: tuple[tuple[str, str], ...] = (
    ("cross_attn.", "attn2."),
    ("self_attn.", "attn1."),
    (".o.", ".to_out.0."),
    (".q.", ".to_q."),
    (".k.", ".to_k."),
    (".v.", ".to_v."),
    ("ffn.0.", "ffn.net.0.proj."),
    ("ffn.2.", "ffn.net.2."),
)

# Prefixes seen on PEFT-style Wan LoRA keys.
_PEFT_PREFIXES_TO_STRIP: tuple[str, ...] = (
    "base_model.model.transformer.",
    "transformer.",
    "diffusion_model.",
)


def lora_model_from_wan_state_dict(
    state_dict: Dict[str, torch.Tensor], alpha: float | None = None
) -> ModelPatchRaw:
    """Convert any supported Wan LoRA state dict into a ``ModelPatchRaw``.

    Detects Kohya vs PEFT layouts and dispatches accordingly. Layer paths in
    the returned patch use diffusers naming (``blocks.X.attn1.to_q``) prefixed
    with ``WAN_LORA_TRANSFORMER_PREFIX`` so the runtime ``LayerPatcher`` can
    match them against ``WanTransformer3DModel`` parameters.
    """
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    if has_wan_kohya_keys(str_keys):
        return _convert_kohya_format(state_dict, alpha)
    return _convert_peft_format(state_dict, alpha)


def _convert_kohya_format(state_dict: Dict[str, torch.Tensor], alpha: float | None) -> ModelPatchRaw:
    """Convert a Kohya-format Wan LoRA state dict.

    Keys look like ``lora_unet_blocks_<idx>_<submodule>.{lora_down,lora_up,alpha}.weight``.
    Unrecognised submodules are silently skipped (logged at conversion debug level
    by the layer factory if needed).
    """
    layers: dict[str, BaseLayerPatch] = {}
    grouped = _group_by_layer(state_dict)

    for kohya_layer, layer_dict in grouped.items():
        path = _kohya_layer_to_diffusers_path(kohya_layer)
        if path is None:
            continue
        values = _normalize_lora_param_names(layer_dict, alpha)
        layers[f"{WAN_LORA_TRANSFORMER_PREFIX}{path}"] = any_lora_layer_from_state_dict(values)

    return ModelPatchRaw(layers=layers)


def _convert_peft_format(state_dict: Dict[str, torch.Tensor], alpha: float | None) -> ModelPatchRaw:
    """Convert a Diffusers-PEFT or native-PEFT Wan LoRA state dict."""
    layers: dict[str, BaseLayerPatch] = {}
    grouped = _group_by_layer(state_dict)

    for raw_layer_key, layer_dict in grouped.items():
        stripped = _strip_peft_prefix(raw_layer_key)
        path = _native_layer_path_to_diffusers(stripped)
        if path is None:
            continue
        values = _normalize_lora_param_names(layer_dict, alpha)
        layers[f"{WAN_LORA_TRANSFORMER_PREFIX}{path}"] = any_lora_layer_from_state_dict(values)

    return ModelPatchRaw(layers=layers)


def _kohya_layer_to_diffusers_path(kohya_layer: str) -> str | None:
    """``lora_unet_blocks_0_self_attn_q`` -> ``blocks.0.attn1.to_q``."""
    m = _KOHYA_KEY_REGEX.match(kohya_layer)
    if not m:
        return None
    block_idx = m.group(1)
    sub = m.group(2)
    for kohya_sub, diffusers_sub in _KOHYA_SUBMODULE_MAP:
        if sub == kohya_sub:
            return f"blocks.{block_idx}.{diffusers_sub}"
    return None


def _strip_peft_prefix(layer_key: str) -> str:
    """Strip ``transformer.``, ``diffusion_model.``, ``base_model.model.transformer.`` if present."""
    for prefix in _PEFT_PREFIXES_TO_STRIP:
        if layer_key.startswith(prefix):
            return layer_key[len(prefix):]
    return layer_key


def _native_layer_path_to_diffusers(path: str) -> str | None:
    """Rewrite a stripped PEFT layer path to diffusers naming.

    No-op if the path is already in diffusers form (contains attn1./attn2./ffn.net.).
    Returns None only if the path can't be plausibly identified as Wan.
    """
    if not path.startswith("blocks."):
        return None

    if "attn1." in path or "attn2." in path or "ffn.net." in path:
        return path

    # Apply the native-to-diffusers replacements with a sentinel trailing dot
    # so rules like ``.q.`` fire on a bare-ending ``...self_attn.q``.
    augmented = path + "."
    for needle, replacement in _NATIVE_TO_DIFFUSERS_PATH_RULES:
        augmented = augmented.replace(needle, replacement)
    return augmented.rstrip(".")


def _normalize_lora_param_names(
    layer_dict: dict[str, torch.Tensor], alpha: float | None
) -> dict[str, torch.Tensor]:
    """Map PEFT-style ``lora_A``/``lora_B`` to ``lora_down``/``lora_up``.

    Kohya-style ``lora_down``/``lora_up`` pass through unchanged.
    """
    if "lora_A.weight" in layer_dict:
        values: dict[str, torch.Tensor] = {
            "lora_down.weight": layer_dict["lora_A.weight"],
            "lora_up.weight": layer_dict["lora_B.weight"],
        }
        if alpha is not None:
            values["alpha"] = torch.tensor(alpha)
        if "alpha" in layer_dict:
            values["alpha"] = layer_dict["alpha"]
        if "dora_scale" in layer_dict:
            values["dora_scale"] = layer_dict["dora_scale"]
        return values
    return layer_dict


def _group_by_layer(state_dict: Dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Group state-dict keys by their layer path (everything before the LoRA-suffix tail)."""
    grouped: dict[str, dict[str, torch.Tensor]] = {}

    known_suffixes = [
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_down.weight",
        ".lora_up.weight",
        ".dora_scale",
        ".alpha",
    ]

    for key in state_dict:
        if not isinstance(key, str):
            continue

        layer_name = None
        key_name = None
        for suffix in known_suffixes:
            if key.endswith(suffix):
                layer_name = key[: -len(suffix)]
                key_name = suffix[1:]  # drop leading dot
                break

        if layer_name is None:
            parts = key.rsplit(".", maxsplit=2)
            layer_name = parts[0]
            key_name = ".".join(parts[1:])

        grouped.setdefault(layer_name, {})[key_name] = state_dict[key]

    return grouped
