"""Convert an ERNIE-Image LoRA into a ``ModelPatchRaw`` for the ERNIE-Image transformer.

Published ERNIE-Image LoRAs already use the module names of diffusers' ``ErnieImageTransformer2DModel``, which
is also the layout every ERNIE-Image transformer build loads into: the Diffusers folder, the single-file
checkpoint and the community GGUFs all keep diffusers' key names (see ``load/model_loaders/ernie_image.py``).
So there is no name map to keep in step with a loader -- the conversion strips the one container prefix the
trainer wrote (``diffusion_model.`` for ai-toolkit and Comfy, ``transformer.`` for diffusers/PEFT exports,
``base_model.model.`` for PEFT adapters) and builds one layer per module.

ai-toolkit writes PEFT ``lora_A``/``lora_B`` pairs with no ``.alpha`` tensors (alpha == rank, so the patch
applies unscaled); a kohya-style ``alpha`` tensor in the file wins over that default.
"""

from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.ernie_image_lora_constants import (
    ERNIE_IMAGE_LORA_KEY_PREFIXES,
    ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX,
    ernie_image_lora_refusal,
    ernie_image_weight_shape,
    split_ernie_image_lora_key,
)
from invokeai.backend.patches.lora_conversions.flux3_lora_constants import update_shape
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw


def lora_model_from_ernie_image_state_dict(state_dict: Dict[str, torch.Tensor]) -> ModelPatchRaw:
    """Convert an ERNIE-Image LoRA state dict into a ``ModelPatchRaw``.

    Layer keys in the returned patch are the transformer's module paths, prefixed with
    ``ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX``. Raises ``ValueError`` for unsupported keys, a half pair, mixed layer
    types or a layer the factories cannot build. It does not check shapes: that is ``ernie_image_lora_defect``'s job
    at install.
    """
    refusal = ernie_image_lora_refusal([k for k in state_dict.keys() if isinstance(k, str)])
    if refusal is not None:
        raise ValueError(f"ERNIE-Image LoRA cannot be applied: {refusal}")

    layers: dict[str, BaseLayerPatch] = {}
    for layer_path, values in _group_by_layer(state_dict).items():
        layers[f"{ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX}{layer_path}"] = _build_layer(layer_path, values)
    return ModelPatchRaw(layers=layers)


def ernie_image_lora_defect(state_dict: Dict[str, torch.Tensor]) -> str | None:
    """Why this file cannot be applied, or ``None`` if every layer converts and fits its module.

    The install probe refuses with this message, so a file is never accepted and then fails (or is half-applied)
    at generation. It runs the conversion's own grouping and layer construction, then checks that each update has
    the shape of the weight it patches.

    The install probe hands this ``meta`` tensors (shapes only), so nothing here may read a value: layers are built
    without their ``alpha`` (the factory calls ``.item()`` on it) and alpha is checked by its element count instead.
    """
    refusal = ernie_image_lora_refusal([k for k in state_dict.keys() if isinstance(k, str)])
    if refusal is not None:
        return refusal
    try:
        for layer_path, values in _group_by_layer(state_dict).items():
            _build_layer(layer_path, values, read_alpha=False)
            _check_alpha(layer_path, values)
            _check_update_shape(layer_path, values)
    except ValueError as e:
        return str(e)
    return None


def _build_layer(layer_path: str, values: dict[str, torch.Tensor], *, read_alpha: bool = True) -> BaseLayerPatch:
    """Build one layer; the layer factories signal a missing or inconsistent tensor with assorted exceptions.

    ``read_alpha=False`` leaves the ``alpha`` tensor out of the build, which is the only place a factory reads a value.
    """
    kinds = {
        "lora": "lora_A.weight" in values or "lora_down.weight" in values,
        "hada": any(k.startswith("hada_") for k in values),
    }
    kinds["lokr"] = any(k.startswith("lokr_") for k in values)
    if sum(kinds.values()) > 1:
        raise ValueError(
            f"ERNIE-Image LoRA layer {layer_path!r} mixes layer types (has {sorted(values)}); "
            "only one of low-rank, LoHA or LoKR per module can be applied."
        )
    try:
        buildable = values if read_alpha else {k: v for k, v in values.items() if k != "alpha"}
        return any_lora_layer_from_state_dict(_normalize_lora_param_names(buildable))
    except (ValueError, KeyError, AssertionError, TypeError, RuntimeError) as e:
        raise ValueError(
            f"ERNIE-Image LoRA layer {layer_path!r} is incomplete or inconsistent "
            f"(has {sorted(values)}): {type(e).__name__} {e}"
        ) from None


def _check_alpha(layer_path: str, values: dict[str, torch.Tensor]) -> None:
    """Loading reads alpha with ``.item()``, which only a single value survives."""
    alpha = values.get("alpha")
    if alpha is not None and alpha.numel() != 1:
        raise ValueError(
            f"ERNIE-Image LoRA layer {layer_path!r} is incomplete or inconsistent: its alpha holds "
            f"{alpha.numel()} values, not one."
        )


def _tucker_update_shape(layer_path: str, values: dict[str, torch.Tensor]) -> tuple[int, ...] | None:
    """The update shape of a LoHA or LoKR layer with Tucker cores (``hada_t1``/``hada_t2``, ``lokr_t2``), or ``None``
    if the layer has none. Raises ``ValueError`` if the factors do not fit the einsum the layer applies them with.

    LyCORIS writes cores for convolution kernels. On a Linear weight the kernel axes must be 1x1 for the patch to
    reshape onto it.
    """
    is_loha = "hada_t1" in values  # the layer factory refuses a lone core, so by now both are there
    is_lokr = "lokr_t2" in values and "lokr_w2" not in values  # a full ``lokr_w2`` makes the layer ignore the core
    if not is_loha and not is_lokr:
        return None

    def refuse(why: str) -> ValueError:
        return ValueError(f"ERNIE-Image LoRA layer {layer_path!r} has inconsistent Tucker factors: {why}.")

    def core_product(core: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple[int, ...]:
        # einsum "i j k l, i p, j r -> p r k l" for LoKR, "i j k l, j r, i p -> p r k l" for LoHA: both are
        # core(i, j, k, l) contracted with a(i, p) and b(j, r).
        if core.ndim != 4 or a.ndim != 2 or b.ndim != 2:
            raise refuse(
                f"expected a 4-D core and two matrices, got {tuple(core.shape)}, {tuple(a.shape)}, {tuple(b.shape)}"
            )
        if a.shape[0] != core.shape[0] or b.shape[0] != core.shape[1]:
            raise refuse(f"core {tuple(core.shape)} does not fit factors {tuple(a.shape)} and {tuple(b.shape)}")
        if core.shape[2] * core.shape[3] != 1:
            raise refuse(f"a {core.shape[2]}x{core.shape[3]} kernel cannot patch a Linear weight")
        return a.shape[1], b.shape[1]

    if is_loha:
        first = core_product(values["hada_t1"], values["hada_w1_a"], values["hada_w1_b"])
        second = core_product(values["hada_t2"], values["hada_w2_a"], values["hada_w2_b"])
        if first != second:
            raise refuse(f"the two products have shapes {first} and {second}")
        return first

    w2 = core_product(values["lokr_t2"], values["lokr_w2_a"], values["lokr_w2_b"])
    # The Kronecker product itself is sized like any other LoKR: w1 times the w2 the core rebuilds.
    layer = {k: v for k, v in values.items() if k.startswith("lokr_w1")}
    layer["lokr_w2"] = torch.empty(w2, device="meta")
    return update_shape(layer)


def _check_update_shape(layer_path: str, values: dict[str, torch.Tensor]) -> None:
    weight_shape = ernie_image_weight_shape(layer_path)
    if weight_shape is None:
        return  # Placement was settled by the key check.
    shape = _tucker_update_shape(layer_path, values)
    if shape is None:
        shape = update_shape(_normalize_lora_param_names(values))
    if shape != weight_shape:
        raise ValueError(
            f"ERNIE-Image LoRA layer {layer_path!r} updates a weight of shape {shape}, "
            f"but the module's weight is {weight_shape}."
        )


def _strip_prefix(layer_path: str) -> str:
    for prefix in ERNIE_IMAGE_LORA_KEY_PREFIXES:
        if layer_path.startswith(prefix):
            return layer_path[len(prefix) :]
    return layer_path


def _group_by_layer(state_dict: Dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Group the tensors by module path, after stripping the container prefix."""
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        split = split_ernie_image_lora_key(key)
        assert split is not None  # unsupported tails were refused by the caller
        layer_path, param_name = split
        grouped.setdefault(_strip_prefix(layer_path), {})[param_name] = value

    # A low-rank pair split across the file would otherwise surface as a bare KeyError inside the layer
    # factory, after the transformer has been loaded for the generation.
    for layer_path, values in grouped.items():
        has_down = bool({"lora_A.weight", "lora_down.weight"} & values.keys())
        has_up = bool({"lora_B.weight", "lora_up.weight"} & values.keys())
        if has_down != has_up:
            raise ValueError(
                f"ERNIE-Image LoRA layer {layer_path!r} is missing one half of its low-rank pair "
                f"(has {sorted(values)}). The file is truncated or was edited."
            )
    return grouped


def _normalize_lora_param_names(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map PEFT ``lora_A``/``lora_B`` onto the ``lora_down``/``lora_up`` names the layer factory reads."""
    if "lora_A.weight" not in values:
        return values
    normalized = {k: v for k, v in values.items() if k not in ("lora_A.weight", "lora_B.weight")}
    normalized["lora_down.weight"] = values["lora_A.weight"]
    normalized["lora_up.weight"] = values["lora_B.weight"]
    return normalized
