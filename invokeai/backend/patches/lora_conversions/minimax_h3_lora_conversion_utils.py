"""MiniMax H3 LoRA conversion utilities.

H3 LoRAs (e.g. larryvrh/MiniMax-H3-Turbo-Lora) target the original remote-code /
Comfy single-file transformer layout. We normalise them to the parameter paths the
vendored ``MiniMaxH3Transformer3DModel`` uses at runtime, mirroring the base-model
state-dict converter in
``invokeai.backend.model_manager.load.model_loaders.minimax_h3_state_dict_utils``:

- ``blocks.N.``                     -> ``transformer_blocks.N.``
- ``token_refiner.blocks.N.``       -> ``token_refiner.refiner_blocks.N.``
- ``attn.qkv_proj`` (fused)         -> fanned out to ``attn.to_q`` / ``to_k`` / ``to_v``:
  ``lora_up`` (``[3*dim, rank]``) row-splits into thirds; ``lora_down`` is shared.
  This is exact — the fused delta ``B @ A`` row-splits the same way ``W`` does.
- ``attn.out_proj``                 -> ``attn.to_out.0``
- ``mlp.fc1`` (fused SwiGLU)        -> ``ff.net.0.proj`` with the two halves of
  ``lora_up`` SWAPPED (``[gate; value]`` -> ``[value; gate]``), exactly like the base
  converter swaps the fused weight's rows. Without the swap every patched MLP would
  perturb ``silu(value) * gate`` instead of ``silu(gate) * value``.
- ``mlp.fc2``                       -> ``ff.net.2``
- ``adaln_proj.linear``             -> unchanged (module exists under the same path)
- ``final_layer.adaln_proj.linear`` -> ``norm_out.linear``

Note on AdaLN layers: on the *full* transformer these convert to ordinary weight
patches. The AdaLN-pruned transformer has no full-width AdaLN projections to patch —
those layers are instead re-injected at run time from a precomputed ``silu(t_emb)``
grid (see ``invokeai.backend.minimax_h3.pruned_adaln_lora``). The conversion here is
the same either way; the denoise invocation routes AdaLN layers per model variant.
"""

from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.minimax_h3_lora_constants import (
    MINIMAX_H3_LORA_TRANSFORMER_PREFIX,
    has_minimax_h3_lora_keys,
    has_non_minimax_h3_architecture_keys,
    has_unsupported_minimax_h3_lora_variant_keys,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# Layer paths of the AdaLN time-conditioning projections in the *converted* (runtime)
# naming, without the ModelPatchRaw prefix. Used by the denoise invocation to route
# these layers away from the LayerPatcher on AdaLN-pruned transformers.
MINIMAX_H3_ADALN_LINEAR_SUFFIX = ".adaln_proj.linear"
MINIMAX_H3_NORM_OUT_LINEAR_PATH = "norm_out.linear"


def is_minimax_h3_adaln_layer_path(layer_path: str) -> bool:
    """True for converted layer paths that target an AdaLN time-conditioning projection."""
    return layer_path.endswith(MINIMAX_H3_ADALN_LINEAR_SUFFIX) or layer_path == MINIMAX_H3_NORM_OUT_LINEAR_PATH


_PEFT_PREFIXES_TO_STRIP: tuple[str, ...] = (
    "base_model.model.transformer.",
    "transformer.",
    "diffusion_model.",
)

# Order matters: token_refiner.blocks. must be rewritten before the bare blocks. prefix.
_PREFIX_RENAMES: tuple[tuple[str, str], ...] = (
    ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
    ("blocks.", "transformer_blocks."),
    ("final_layer.adaln_proj.linear", MINIMAX_H3_NORM_OUT_LINEAR_PATH),
)

_SUBKEY_RENAMES: tuple[tuple[str, str], ...] = (
    (".attn.out_proj", ".attn.to_out.0"),
    (".mlp.fc1", ".ff.net.0.proj"),
    (".mlp.fc2", ".ff.net.2"),
)

_QKV_SUFFIX = ".attn.qkv_proj"
_SWIGLU_SUFFIX = ".ff.net.0.proj"


def is_state_dict_likely_in_minimax_h3_format(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Heuristic: does this state dict look like a MiniMax H3 LoRA in the native layout?"""
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    return has_minimax_h3_lora_keys(str_keys) and not has_non_minimax_h3_architecture_keys(str_keys)


def lora_model_from_minimax_h3_state_dict(
    state_dict: Dict[str, torch.Tensor], alpha: float | None = None
) -> ModelPatchRaw:
    """Convert a native-layout MiniMax H3 LoRA state dict into a ``ModelPatchRaw``.

    Layer paths in the returned patch use the vendored transformer's runtime naming
    (``transformer_blocks.N.attn.to_q`` etc.) prefixed with
    ``MINIMAX_H3_LORA_TRANSFORMER_PREFIX``.
    """
    # The probe rejects these at install time; this guard covers state dicts that reach the
    # converter through other paths. LoKR/LoHA factorizations cannot be row-split across the
    # fused qkv/SwiGLU tensors, and DoRA's per-output-row magnitudes would be silently
    # mis-applied by the half-swap below.
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    if has_unsupported_minimax_h3_lora_variant_keys(str_keys):
        raise ValueError(
            "MiniMax H3 LoRAs must be plain low-rank (lora_A/lora_B); LoKR/LoHA/DoRA variants are not "
            "supported on H3's fused transformer layers."
        )

    layers: dict[str, BaseLayerPatch] = {}

    for raw_layer_path, layer_dict in _group_by_layer(state_dict).items():
        layer_path = _rename_layer_path(_strip_peft_prefix(raw_layer_path))
        values = _normalize_lora_param_names(layer_dict, alpha)

        if layer_path.endswith(_QKV_SUFFIX):
            # Fused QKV: row-split lora_up into thirds; lora_down (which acts on the
            # *input*, shared by all three projections) is reused as-is.
            up = values["lora_up.weight"]
            if up.shape[0] % 3 != 0:
                raise ValueError(
                    f"Fused-QKV LoRA layer {raw_layer_path!r} has {up.shape[0]} output rows, not divisible by 3"
                )
            rows = up.shape[0] // 3
            stem = layer_path[: -len("qkv_proj")]
            for i, proj in enumerate(("to_q", "to_k", "to_v")):
                proj_values = dict(values)
                proj_values["lora_up.weight"] = up[i * rows : (i + 1) * rows]
                layers[f"{MINIMAX_H3_LORA_TRANSFORMER_PREFIX}{stem}{proj}"] = any_lora_layer_from_state_dict(
                    proj_values
                )
            continue

        if layer_path.endswith(_SWIGLU_SUFFIX):
            # Fused SwiGLU input projection: the checkpoint orders the halves
            # [gate; value], the vendored diffusers module [value; gate]. Swap the
            # output rows of lora_up to match, exactly like the base-model converter
            # swaps the fused weight (see minimax_h3_state_dict_utils).
            up = values["lora_up.weight"]
            if up.shape[0] % 2 != 0:
                raise ValueError(f"Fused-SwiGLU LoRA layer {raw_layer_path!r} has odd output row count {up.shape[0]}")
            half = up.shape[0] // 2
            values = dict(values)
            values["lora_up.weight"] = torch.cat([up[half:], up[:half]], dim=0)

        layers[f"{MINIMAX_H3_LORA_TRANSFORMER_PREFIX}{layer_path}"] = any_lora_layer_from_state_dict(values)

    return ModelPatchRaw(layers=layers)


def _strip_peft_prefix(layer_path: str) -> str:
    for prefix in _PEFT_PREFIXES_TO_STRIP:
        if layer_path.startswith(prefix):
            return layer_path[len(prefix) :]
    return layer_path


def _rename_layer_path(layer_path: str) -> str:
    for old, new in _PREFIX_RENAMES:
        if layer_path.startswith(old):
            layer_path = new + layer_path[len(old) :]
            break
    for old, new in _SUBKEY_RENAMES:
        if layer_path.endswith(old):
            layer_path = layer_path[: -len(old)] + new
            break
    return layer_path


def _normalize_lora_param_names(layer_dict: dict[str, torch.Tensor], alpha: float | None) -> dict[str, torch.Tensor]:
    """Map PEFT-style ``lora_A``/``lora_B`` to ``lora_down``/``lora_up``.

    Kohya-style ``lora_down``/``lora_up`` pass through unchanged. A per-layer
    ``alpha`` tensor in the file wins over the caller-supplied default.
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
