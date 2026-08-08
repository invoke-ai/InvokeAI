"""Krea-2 LoRA conversion utilities.

Krea-2 uses a single-stream MMDiT (``Krea2Transformer2DModel``) with a Qwen3-VL text encoder.
Published LoRAs (e.g. krea/Krea-2-LoRA-*) are diffusers PEFT format: keys like
``transformer.<module>.lora_A.weight`` / ``lora_B.weight``. The distinctive Krea-2 module is the
``text_fusion`` stage, which we use to disambiguate from Qwen-Image / Z-Image LoRAs (which otherwise
share the ``transformer.transformer_blocks.`` prefix).
"""

import re
from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import (
    KREA2_LORA_QWEN3VL_PREFIX,
    KREA2_LORA_TRANSFORMER_PREFIX,
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
        converted_key = _native_krea2_key_to_diffusers(key) if _looks_like_native_krea2_key(key) else key
        if converted_key in converted_state_dict:
            raise ValueError(
                f"Krea-2 LoRA has conflicting layers that normalize to the same target '{converted_key}'. "
                "This mixed layout is unsupported - refusing to silently drop one of the layers."
            )
        converted_state_dict[converted_key] = value
    return converted_state_dict


def is_state_dict_likely_krea2_lora(state_dict: dict[str | int, torch.Tensor]) -> bool:
    """Checks if the provided state dict is likely a Krea-2 LoRA.

    Requires the distinctive Krea-2 ``text_fusion`` / ``txtfusion`` / ``time_mod_proj`` modules so it does not
    false-match Qwen-Image or Z-Image LoRAs that also carry ``transformer.transformer_blocks.`` keys.
    """
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    has_krea2_module = any(any(sig in k for sig in KREA2_TRANSFORMER_SIGNATURE_KEYS) for k in str_keys)
    has_lora_suffix = any(
        k.endswith(
            (
                ".lora_A.weight",
                ".lora_B.weight",
                ".lora_down.weight",
                ".lora_up.weight",
                # LyCORIS LoKr (e.g. ai-toolkit Krea-2 adapters) stores Kronecker factors, not an A/B pair.
                ".lokr_w1",
                ".lokr_w2",
                ".lokr_w1_a",
                ".lokr_w1_b",
                ".lokr_w2_a",
                ".lokr_w2_b",
            )
        )
        for k in str_keys
    )
    return has_krea2_module and has_lora_suffix


def lora_model_from_krea2_state_dict(state_dict: Dict[str, torch.Tensor], alpha: float | None = None) -> ModelPatchRaw:
    """Convert a Krea-2 LoRA state dict (diffusers PEFT) to a ModelPatchRaw.

    Handles transformer layers and (if present) Qwen3-VL text encoder layers. ``alpha=None`` is treated
    as ``alpha=rank`` internally (the common diffusers default).
    """
    layers: dict[str, BaseLayerPatch] = {}
    # Normalize native (ComfyUI) naming to the diffusers layout so the rest of the converter is layout-agnostic.
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
        if "dora_scale" in layer_dict:
            values["dora_scale"] = layer_dict["dora_scale"]
        if "alpha" in layer_dict:
            values["alpha"] = layer_dict["alpha"]
        if alpha is not None:
            values["alpha"] = torch.tensor(alpha)
        return values
    return layer_dict


# Maps each recognized weight-key suffix to the canonical value-key used downstream. The PEFT/diffusers DoRA
# magnitude is published as ``<layer>.lora_magnitude_vector.weight``; it is the same thing InvokeAI stores as
# ``dora_scale``, so mapping it here lets a standard Diffusers DoRA adapter (A/B + magnitude) load as a
# DoRALayer instead of being split into a bogus, unrecognized layer.
#
# LyCORIS LoKr factors are passed through untouched: `any_lora_layer_from_state_dict` routes a values dict
# containing `lokr_w1` / `lokr_w1_a` to LoKRLayer, so they only need to survive _group_by_layer intact.
_SUFFIX_TO_VALUE_KEY = {
    ".lora_A.weight": "lora_A.weight",
    ".lora_B.weight": "lora_B.weight",
    ".lora_down.weight": "lora_down.weight",
    ".lora_up.weight": "lora_up.weight",
    ".dora_scale": "dora_scale",
    ".lora_magnitude_vector.weight": "dora_scale",
    ".alpha": "alpha",
    ".lokr_w1": "lokr_w1",
    ".lokr_w2": "lokr_w2",
    ".lokr_w1_a": "lokr_w1_a",
    ".lokr_w1_b": "lokr_w1_b",
    ".lokr_w2_a": "lokr_w2_a",
    ".lokr_w2_b": "lokr_w2_b",
    ".lokr_t2": "lokr_t2",
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
