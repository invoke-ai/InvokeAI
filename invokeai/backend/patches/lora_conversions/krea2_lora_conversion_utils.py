"""Krea-2 LoRA conversion utilities.

Krea-2 uses a single-stream MMDiT (``Krea2Transformer2DModel``) with a Qwen3-VL text encoder.
Published LoRAs (e.g. krea/Krea-2-LoRA-*) are diffusers PEFT format: keys like
``transformer.<module>.lora_A.weight`` / ``lora_B.weight``. The distinctive Krea-2 module is the
``text_fusion`` stage, which we use to disambiguate from Qwen-Image / Z-Image LoRAs (which otherwise
share the ``transformer.transformer_blocks.`` prefix).

Two other key layouts are normalized onto that one before conversion: the native (ComfyUI) module naming,
and the kohya / LyCORIS layout that additionally flattens the module path into ``lora_unet_<path>``.
"""

import re
from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import (
    KREA2_LORA_QWEN3VL_PREFIX,
    KREA2_LORA_TRANSFORMER_PREFIX,
    split_kohya_krea2_key,
    unflatten_kohya_krea2_module_path,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# Module-name fragments unique to the Krea-2 transformer (text-fusion stage + timestep modulation proj).
# ``text_fusion`` is the diffusers name; ``txtfusion`` is the native (ComfyUI) name for the same stage.
KREA2_TRANSFORMER_SIGNATURE_KEYS = ("text_fusion", "time_mod_proj", "txtfusion")

# --- Native (ComfyUI) -> diffusers key mapping ---------------------------------------------------------------
# Native Krea-2 LoRAs (e.g. sliders) name the modules differently from InvokeAI's diffusers
# ``Krea2Transformer2DModel``. The two are a verified 1:1 correspondence (every native module maps onto a real
# Linear in the diffusers model), so we rewrite native keys to the diffusers layout before conversion:
#   diffusion_model.blocks.N.              -> transformer_blocks.N.   (top-level blocks only)
#   diffusion_model.txtfusion.             -> text_fusion.            (layerwise_blocks / refiner_blocks kept)
#   attn.wq/wk/wv                          -> attn.to_q/to_k/to_v
#   attn.wo                                -> attn.to_out.0
#   attn.gate                              -> attn.to_gate
#   mlp.gate/up/down                       -> ff.gate/up/down         (SwiGLU)
_NATIVE_KREA2_LEAF_RENAMES = {
    ".attn.wq.": ".attn.to_q.",
    ".attn.wk.": ".attn.to_k.",
    ".attn.wv.": ".attn.to_v.",
    ".attn.wo.": ".attn.to_out.0.",
    ".attn.gate.": ".attn.to_gate.",
    ".mlp.gate.": ".ff.gate.",
    ".mlp.up.": ".ff.up.",
    ".mlp.down.": ".ff.down.",
}
_NATIVE_KREA2_TOP_LEVEL_RENAMES = {
    "first": "img_in",
    "tmlp.0": "time_embed.linear_1",
    "tmlp.2": "time_embed.linear_2",
    "tproj.1": "time_mod_proj",
    "txtmlp.1": "txt_in.linear_1",
    "txtmlp.3": "txt_in.linear_2",
    "last.linear": "final_layer.linear",
}
# The main transformer's top-level `blocks.` (preceded by start-of-string or a dot) becomes
# `transformer_blocks.`. The text-fusion sub-blocks (`layerwise_blocks`/`refiner_blocks`) are NOT touched
# because their `blocks` is preceded by `_`, not by a dot.
_NATIVE_KREA2_BLOCKS_RE = re.compile(r"(^|\.)blocks\.")


def _replace_module_path(key: str, native: str, diffusers: str) -> str:
    return re.sub(rf"(^|\.){re.escape(native)}\.", rf"\1{diffusers}.", key)


def _looks_like_native_krea2_key(key: str) -> bool:
    if "txtfusion" in key:
        return True
    native_module_names = (*_NATIVE_KREA2_LEAF_RENAMES, *_NATIVE_KREA2_TOP_LEVEL_RENAMES)
    return any(_replace_module_path(key, native.strip("."), "__native__") != key for native in native_module_names)


def _looks_like_native_krea2_lora(str_keys: list[str]) -> bool:
    """True if the keys use the native (ComfyUI) Krea-2 naming rather than the diffusers PEFT naming."""
    return any(_looks_like_native_krea2_key(key) for key in str_keys)


def _native_krea2_key_to_diffusers(key: str) -> str:
    key = key.replace("txtfusion.", "text_fusion.")
    key = _NATIVE_KREA2_BLOCKS_RE.sub(r"\1transformer_blocks.", key)
    for native, diffusers in _NATIVE_KREA2_TOP_LEVEL_RENAMES.items():
        key = _replace_module_path(key, native, diffusers)
    for native, diffusers in _NATIVE_KREA2_LEAF_RENAMES.items():
        key = key.replace(native, diffusers)
    return key


def _maybe_convert_native_krea2_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Rewrite native (ComfyUI) Krea-2 LoRA keys to the diffusers layout, leaving diffusers keys untouched."""
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    if not _looks_like_native_krea2_lora(str_keys):
        return state_dict
    converted_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        # `.pt`/`.ckpt` sources can carry non-string keys. They are never native Krea-2 keys, but the
        # substring tests in `_looks_like_native_krea2_key` raise TypeError rather than returning False.
        is_native = isinstance(key, str) and _looks_like_native_krea2_key(key)
        converted_key = _native_krea2_key_to_diffusers(key) if is_native else key
        if converted_key in converted_state_dict:
            raise ValueError(
                f"Krea-2 LoRA has conflicting layers that normalize to the same target '{converted_key}'. "
                "This mixed layout is unsupported - refusing to silently drop one of the layers."
            )
        converted_state_dict[converted_key] = value
    return converted_state_dict


def _maybe_convert_kohya_krea2_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Rewrite kohya/LyCORIS flattened Krea-2 keys to the dotted native layout, leaving all others untouched."""
    # Whether a module can be converted is a property of the *module*, not of each key on its own. A LyCORIS
    # module (``lokr_w1``, ``hada_w1_a``, ``diff``) has to stay verbatim — see below — but LyCORIS also saves
    # a sibling ``.alpha`` (and ``.dora_scale`` for the weight-decomposed variants), whose suffix this
    # converter *does* recognize. Deciding per key rewrites that sibling alone and splits one module across
    # two layer groups: the rewritten ``.alpha`` ends up in a group by itself, and ``_get_lora_layer_values``
    # aborts the whole adapter on ``{'alpha'}`` (the ``.dora_scale`` variant dies in ``DoRALayer`` instead).
    # So collect each flattened module's suffixes first and convert only modules where *all* of them convert.
    suffixes_by_flat_path: dict[str, set[str]] = {}
    for key in state_dict:
        split = split_kohya_krea2_key(key)
        if split is not None:
            flat_path, _, weight_suffix = split
            suffixes_by_flat_path.setdefault(flat_path, set()).add(f".{weight_suffix}")
    fully_convertible_flat_paths = {
        flat_path
        for flat_path, suffixes in suffixes_by_flat_path.items()
        if all(suffix in _SUFFIX_TO_VALUE_KEY for suffix in suffixes)
    }

    converted_state_dict: Dict[str, torch.Tensor] = {}
    source_keys: dict[str, str] = {}
    for key, value in state_dict.items():
        converted_key = key
        split = split_kohya_krea2_key(key)
        if split is not None:
            flat_path, dot, weight_suffix = split
            module_path = unflatten_kohya_krea2_module_path(flat_path)
            # Only rewrite when ``_group_by_layer`` can split the suffix back off. Un-flattening introduces
            # dots into the module path, and the grouper's fallback for an unknown suffix is a blind
            # ``rsplit(".", 2)`` — on a dotted path that cuts *inside the module name*, fusing two modules
            # into one bogus layer that aborts the whole load. LyCORIS suffixes such as ``.lokr_w1`` or
            # ``.hada_w1_a`` hit exactly that. Flattened, they have no interior dot and group harmlessly,
            # so leaving them verbatim keeps them at the pre-existing warn-and-skip behaviour.
            if module_path is not None and flat_path in fully_convertible_flat_paths:
                converted_key = f"{module_path}{dot}{weight_suffix}"
        if converted_key in converted_state_dict:
            raise ValueError(
                f"Krea-2 LoRA has conflicting layers that normalize to the same target '{converted_key}' "
                f"(from '{source_keys[converted_key]}' and '{key}'). This mixed layout is unsupported - "
                "refusing to silently drop one of the layers."
            )
        converted_state_dict[converted_key] = value
        source_keys[converted_key] = str(key)
    return converted_state_dict


def is_state_dict_likely_krea2_lora(state_dict: dict[str | int, torch.Tensor]) -> bool:
    """Checks if the provided state dict is likely a Krea-2 LoRA.

    Requires the distinctive Krea-2 ``text_fusion`` / ``txtfusion`` / ``time_mod_proj`` modules so it does not
    false-match Qwen-Image or Z-Image LoRAs that also carry ``transformer.transformer_blocks.`` keys.
    """
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    has_krea2_module = any(any(sig in k for sig in KREA2_TRANSFORMER_SIGNATURE_KEYS) for k in str_keys)
    has_lora_suffix = any(
        k.endswith((".lora_A.weight", ".lora_B.weight", ".lora_down.weight", ".lora_up.weight")) for k in str_keys
    )
    return has_krea2_module and has_lora_suffix


def lora_model_from_krea2_state_dict(state_dict: Dict[str, torch.Tensor], alpha: float | None = None) -> ModelPatchRaw:
    """Convert a Krea-2 LoRA state dict (diffusers PEFT) to a ModelPatchRaw.

    Handles transformer layers and (if present) Qwen3-VL text encoder layers. ``alpha=None`` is treated
    as ``alpha=rank`` internally (the common diffusers default).
    """
    layers: dict[str, BaseLayerPatch] = {}
    # Normalize the kohya/LyCORIS flattened naming (``lora_unet_blocks_6_attn_wv``) to the dotted native layout,
    # then the native (ComfyUI) naming to the diffusers layout, so the rest of the converter is layout-agnostic.
    state_dict = _maybe_convert_kohya_krea2_state_dict(state_dict)
    state_dict = _maybe_convert_native_krea2_state_dict(state_dict)
    grouped_state_dict = _group_by_layer(state_dict)

    transformer_prefixes = (
        "base_model.model.transformer.",
        "transformer.",
        "diffusion_model.",
    )
    text_encoder_prefixes = (
        "base_model.model.text_encoder.",
        "text_encoder.",
    )

    for layer_key, layer_dict in grouped_state_dict.items():
        values = _get_lora_layer_values(layer_key, layer_dict, alpha)

        is_text_encoder = False
        clean_key = layer_key
        for prefix in text_encoder_prefixes:
            if layer_key.startswith(prefix):
                clean_key = layer_key[len(prefix) :]
                is_text_encoder = True
                break
        if not is_text_encoder:
            for prefix in transformer_prefixes:
                if layer_key.startswith(prefix):
                    clean_key = layer_key[len(prefix) :]
                    break

        if is_text_encoder:
            final_key = f"{KREA2_LORA_QWEN3VL_PREFIX}{clean_key}"
        else:
            final_key = f"{KREA2_LORA_TRANSFORMER_PREFIX}{clean_key}"

        # The `transformer.` and `diffusion_model.` aliases normalize to the same target key. If two source
        # layers collide here, silently overwriting one would drop weights based on dict ordering, so reject
        # the mixed-layout adapter explicitly instead.
        if final_key in layers:
            raise ValueError(
                f"Krea-2 LoRA has conflicting layers that normalize to the same target '{final_key}' "
                "(e.g. both a 'transformer.' and a 'diffusion_model.' alias for one logical layer). "
                "This mixed layout is unsupported - refusing to silently drop one of the layers."
            )
        layers[final_key] = any_lora_layer_from_state_dict(values)

    return ModelPatchRaw(layers=layers)


def _get_lora_layer_values(
    layer_key: str, layer_dict: dict[str, torch.Tensor], alpha: float | None
) -> dict[str, torch.Tensor]:
    """Convert PEFT (lora_A/lora_B) layer values to internal (lora_down/lora_up) format."""
    if "lora_A.weight" in layer_dict:
        if "lora_B.weight" not in layer_dict:
            raise ValueError(
                f"Malformed Krea-2 LoRA: layer '{layer_key}' has lora_A.weight but no matching lora_B.weight. "
                "The LoRA file is incomplete or corrupt."
            )
        values = {
            "lora_down.weight": layer_dict["lora_A.weight"],
            "lora_up.weight": layer_dict["lora_B.weight"],
        }
        for magnitude_key in ("dora_scale", "dora_magnitude"):
            if magnitude_key in layer_dict:
                values[magnitude_key] = layer_dict[magnitude_key]
        if "alpha" in layer_dict:
            values["alpha"] = layer_dict["alpha"]
        if alpha is not None:
            values["alpha"] = torch.tensor(alpha)
        return values
    return layer_dict


# Maps each recognized weight-key suffix to the canonical value-key used downstream.
#
# DoRA magnitudes come in two orientations that must not be mixed up (see ``DoRALayer``):
#   - ``.dora_scale`` (LyCORIS/kohya) indexes the *input* dim  -> value key ``dora_scale``
#   - ``.lora_magnitude_vector.weight`` (PEFT/diffusers) and ``.magnitude`` (ai-toolkit) index the *output*
#     dim -> value key ``dora_magnitude``
# Mapping them here lets a DoRA adapter (A/B + magnitude) load as a DoRALayer instead of being split into a
# bogus, unrecognized layer.
_SUFFIX_TO_VALUE_KEY = {
    ".lora_A.weight": "lora_A.weight",
    ".lora_B.weight": "lora_B.weight",
    ".lora_down.weight": "lora_down.weight",
    ".lora_up.weight": "lora_up.weight",
    ".dora_scale": "dora_scale",
    ".lora_magnitude_vector.weight": "dora_magnitude",
    ".magnitude": "dora_magnitude",
    ".alpha": "alpha",
}


def _group_by_layer(state_dict: Dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Groups state dict keys by layer path, splitting off the LoRA weight suffix."""
    layer_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key in state_dict:
        if not isinstance(key, str):
            continue
        layer_name = None
        key_name = None
        for suffix, value_key in _SUFFIX_TO_VALUE_KEY.items():
            if key.endswith(suffix):
                layer_name = key[: -len(suffix)]
                key_name = value_key
                break
        if layer_name is None:
            parts = key.rsplit(".", maxsplit=2)
            layer_name = parts[0]
            key_name = ".".join(parts[1:])
        layer_dict.setdefault(layer_name, {})[key_name] = state_dict[key]
    return layer_dict
