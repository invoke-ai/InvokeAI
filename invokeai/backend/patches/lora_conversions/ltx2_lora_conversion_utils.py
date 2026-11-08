"""Convert an LTX-2 LoRA from its published layout to the runtime module paths.

LTX-2 LoRAs are published against the official Lightricks transformer naming
(``adaln_single``, ``patchify_proj``, ``av_ca_a2v_gate_adaln_single`` ...), while the model the
patch is applied to is a ``diffusers.LTX2VideoTransformer3DModel`` whose modules carry diffusers
names (``time_embed``, ``proj_in``, ``av_cross_attn_video_a2v_gate`` ...). Something has to
translate between them.

That translation is **not written out again here**. The transformer loader already computes it,
for exactly the same reason, when it renames the base checkpoint's keys: it runs diffusers'
``convert_ltx2_transformer_to_diffusers`` and then this repo's ``finish_transformer_keys``, which
completes the 2.5-only names the 2.0-era map in diffusers does not cover. A second, hand-written
map would be a copy that silently drifts the first time either side gains a layer -- and the
failure mode of drift is not a crash but a LoRA that resolves onto nothing and quietly does
nothing. So the same two functions are driven here over sentinel keys built from the LoRA's own
layer paths, which makes the LoRA land on precisely the modules the base weights landed on, by
construction.

Measured against the 2.5 distilled accelerator (``ltx-2.5-22b-distilled-lora-450``): 1660 layers,
all 1660 resolving to ``nn.Linear`` modules on the instantiated transformer, 26 of them renamed by
the pipeline. The file carries no ``.alpha`` tensors, so alpha == rank by convention and the patch
applies unscaled.
"""

from collections.abc import Iterable
from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.ltx2_lora_constants import (
    LTX2_LORA_TRANSFORMER_PREFIX,
    has_ltx2_lora_keys,
    has_non_ltx2_architecture_keys,
    has_unsupported_ltx2_lora_variant_keys,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# Stripped before the official -> diffusers map is applied; the map is written against bare
# official paths. Ordered longest-first so the nested PEFT prefix is not left with a tail.
_PEFT_PREFIXES_TO_STRIP = (
    "base_model.model.transformer.",
    # Comfy packages some releases under this; diffusers' own rename table strips it too, so the
    # probe admits it and this keeps the two in step.
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
)


def is_state_dict_likely_in_ltx2_format(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Heuristic: does this state dict look like an LTX-2 LoRA in the published layout?"""
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    return has_ltx2_lora_keys(str_keys) and not has_non_ltx2_architecture_keys(str_keys)


def lora_model_from_ltx2_state_dict(state_dict: Dict[str, torch.Tensor], alpha: float | None = None) -> ModelPatchRaw:
    """Convert a published-layout LTX-2 LoRA state dict into a ``ModelPatchRaw``.

    Layer paths in the returned patch use the loaded transformer's runtime (diffusers) naming,
    prefixed with ``LTX2_LORA_TRANSFORMER_PREFIX``.
    """
    # The probe rejects these at install time; this guard covers state dicts reaching the
    # converter by another route. The conversion emits ordinary low-rank pairs, which is not the
    # shape a LoKR/LoHA factorization or DoRA's per-row magnitudes need.
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    if has_unsupported_ltx2_lora_variant_keys(str_keys):
        raise ValueError(
            "LTX-2 LoRAs must be plain low-rank (lora_A/lora_B); LoKR/LoHA/DoRA variants are not supported."
        )

    grouped = {_strip_peft_prefix(path): values for path, values in _group_by_layer(state_dict).items()}
    key_map = _official_to_diffusers_paths(grouped.keys())

    layers: dict[str, BaseLayerPatch] = {}
    for layer_path, layer_dict in grouped.items():
        # A name the map does not rename is carried through as-is rather than dropped: the rename
        # table only covers the modules whose names changed, so most layers legitimately pass
        # through unchanged, and a layer this architecture does not have is indistinguishable from
        # one of those here. `LayerPatcher` is where that is caught -- it logs the layer key it
        # could not resolve and skips it, which is how every other family behaves.
        runtime_path = key_map.get(layer_path, layer_path)
        layers[f"{LTX2_LORA_TRANSFORMER_PREFIX}{runtime_path}"] = any_lora_layer_from_state_dict(
            _normalize_lora_param_names(layer_dict, alpha)
        )

    return ModelPatchRaw(layers=layers)


def _official_to_diffusers_paths(layer_paths: Iterable[str]) -> dict[str, str]:
    """Map official layer paths to the transformer's runtime module paths.

    Driven through the *same* two functions the transformer loader uses on the base checkpoint,
    over ``<path>.weight`` sentinels -- the converters rename by key, and both of them key their
    special cases off the ``.weight`` / ``.bias`` tail.
    """
    from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers

    from invokeai.backend.ltx2 import component_configs as cc

    sentinels = {f"{path}.weight": f"{path}.weight" for path in layer_paths}
    renamed = convert_ltx2_transformer_to_diffusers(dict(sentinels))
    renamed, _ = cc.finish_transformer_keys(renamed)

    # The converters return ``{new_key: original_key}`` because the values are the sentinels they
    # were handed; invert to get official -> runtime.
    return {official[: -len(".weight")]: runtime[: -len(".weight")] for runtime, official in renamed.items()}


def _strip_peft_prefix(layer_path: str) -> str:
    for prefix in _PEFT_PREFIXES_TO_STRIP:
        if layer_path.startswith(prefix):
            return layer_path[len(prefix) :]
    return layer_path


def _normalize_lora_param_names(layer_dict: dict[str, torch.Tensor], alpha: float | None) -> dict[str, torch.Tensor]:
    """Map PEFT-style ``lora_A``/``lora_B`` to ``lora_down``/``lora_up``.

    Kohya-style ``lora_down``/``lora_up`` pass through unchanged. A per-layer ``alpha`` tensor in
    the file wins over the caller-supplied default.
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
        return values
    return layer_dict


def _group_by_layer(state_dict: Dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Group state-dict keys by their layer path (everything before the LoRA-suffix tail)."""
    known_suffixes = (
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_down.weight",
        ".lora_up.weight",
        # PEFT's `lora_bias=True` emits these alongside the pair. They are a real published shape,
        # and the layer builder consumes them, so refusing here would reject a usable file.
        ".lora_B.bias",
        ".lora_up.bias",
        ".alpha",
    )

    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        for suffix in known_suffixes:
            if key.endswith(suffix):
                grouped.setdefault(key[: -len(suffix)], {})[suffix[1:]] = value
                break
        else:
            raise ValueError(
                f"LTX-2 LoRA key {key!r} does not end in a recognized LoRA suffix "
                f"({', '.join(known_suffixes)}). This file is not in a layout this conversion reads."
            )

    # A pair split across the file leaves a half-layer, which `any_lora_layer_from_state_dict` would
    # meet as a bare KeyError deep in the patch build -- after the transformer has loaded. Name it.
    for layer_path, values in grouped.items():
        has_down = {"lora_A.weight", "lora_down.weight"} & values.keys()
        has_up = {"lora_B.weight", "lora_up.weight"} & values.keys()
        if not (has_down and has_up):
            raise ValueError(
                f"LTX-2 LoRA layer {layer_path!r} is missing one half of its low-rank pair "
                f"(has {sorted(values)}). The file is truncated or was edited."
            )

    return grouped
