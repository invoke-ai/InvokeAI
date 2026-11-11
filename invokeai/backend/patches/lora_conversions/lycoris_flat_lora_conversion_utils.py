"""Convert SimpleTuner's LyCORIS flat-key LoRAs (`lycoris_` prefix) into a ``ModelPatchRaw`` per family.

The layout is resolved back to each family's diffusers module names (`lycoris_flat_lora_constants`) and handed to
the same paths the dotted spellings take: FLUX.1 through `lora_layers_from_flux_diffusers_grouped_state_dict`,
whose diffusers-to-BFL mapping (fused QKV, swapped final modulation) it reuses in full; Wan through the kohya
converter, whose `lora_unet_blocks_<idx>_<tail>` names the flattened names already are; SDXL through the SD
converter's flattened `lora_unet_` keys. FLUX.2 Klein and Z-Image take diffusers module paths as their patch keys
directly, which is what their converters produce from the dotted spellings.

A file is refused rather than half-applied: any key that names no module of the claimed family raises, so a layer
can never be silently dropped and the LoRA quietly do less than it should.
"""

from collections.abc import Mapping

import torch

from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.flux_diffusers_lora_conversion_utils import (
    lora_layers_from_flux_diffusers_grouped_state_dict,
)
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.lora_conversions.lycoris_flat_lora_constants import (
    LycorisFlatFamily,
    group_lycoris_flat_layers,
    lycoris_flat_candidate_family,
    resolve_lycoris_flat_module,
)
from invokeai.backend.patches.lora_conversions.sd_lora_conversion_utils import lora_model_from_sd_state_dict
from invokeai.backend.patches.lora_conversions.wan_lora_conversion_utils import lora_model_from_wan_state_dict
from invokeai.backend.patches.lora_conversions.z_image_lora_constants import Z_IMAGE_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

_FAMILY_BASES: dict[LycorisFlatFamily, BaseModelType] = {
    "flux1": BaseModelType.Flux,
    "flux2": BaseModelType.Flux2,
    "sdxl": BaseModelType.StableDiffusionXL,
    "z_image": BaseModelType.ZImage,
    "wan": BaseModelType.Wan,
}


def lora_model_from_lycoris_flat_state_dict(
    state_dict: Mapping[str | int, torch.Tensor], base: BaseModelType | None = None
) -> ModelPatchRaw:
    """Convert a SimpleTuner LyCORIS flat-key state dict into a ``ModelPatchRaw``.

    ``base`` is the base the model was registered with; a disagreement with the keys raises, like the FLUX.3
    re-identification guard, instead of mapping layers onto the wrong model.
    """
    family = lycoris_flat_candidate_family(state_dict)
    if family is None:
        raise ValueError("LoRA model is not in a recognized SimpleTuner LyCORIS flat-key layout")
    if base is not None and base is not _FAMILY_BASES[family]:
        raise ValueError(
            f"LoRA model's keys are those of a {_FAMILY_BASES[family].value} LoRA, but it is registered as a "
            f"{base.value} LoRA by an earlier version of InvokeAI. Use Re-identify model in the Model Manager."
        )

    layers = group_lycoris_flat_layers(state_dict)
    assert layers is not None  # `lycoris_flat_family` grouped it
    resolved: dict[str, dict[str, torch.Tensor]] = {}
    for flat_name, values in layers.items():
        module = resolve_lycoris_flat_module(family, flat_name)
        if module is None:
            raise ValueError(
                f"LoRA layer 'lycoris_{flat_name}' targets no {_FAMILY_BASES[family].value} module; "
                "the file is truncated, edited, or not this family's."
            )
        resolved[module] = values

    return _FAMILY_BUILDERS[family](resolved)


def _flux1_builder(resolved: dict[str, dict[str, torch.Tensor]]) -> ModelPatchRaw:
    # The grouped state dict keys are bare diffusers module paths; the converter's prefix handling is a no-op on
    # them and its diffusers-to-BFL mapping applies as it does to dotted files.
    layers = lora_layers_from_flux_diffusers_grouped_state_dict(resolved, alpha=None)
    return ModelPatchRaw(layers=layers)


def _flux2_builder(resolved: dict[str, dict[str, torch.Tensor]]) -> ModelPatchRaw:
    # What `lora_model_from_flux2_diffusers_state_dict` does to dotted keys: the module path is the patch key.
    layers = {
        f"{FLUX_LORA_TRANSFORMER_PREFIX}{module}": any_lora_layer_from_state_dict(values)
        for module, values in resolved.items()
    }
    return ModelPatchRaw(layers=layers)


def _z_image_builder(resolved: dict[str, dict[str, torch.Tensor]]) -> ModelPatchRaw:
    layers = {
        f"{Z_IMAGE_LORA_TRANSFORMER_PREFIX}{module}": any_lora_layer_from_state_dict(values)
        for module, values in resolved.items()
    }
    return ModelPatchRaw(layers=layers)


def _wan_builder(resolved: dict[str, dict[str, torch.Tensor]]) -> ModelPatchRaw:
    # The flattened names are kohya's `lora_unet_blocks_<idx>_<tail>` spelling without the prefix; let the kohya
    # converter map them onto diffusers' `blocks.X.attn1.to_q` paths as it does for kohya files.
    state_dict: dict[str, torch.Tensor] = {
        f"lora_unet_{module.replace('.', '_')}.{param}": value
        for module, values in resolved.items()
        for param, value in values.items()
    }
    return lora_model_from_wan_state_dict(state_dict, alpha=None)


def _sdxl_builder(resolved: dict[str, dict[str, torch.Tensor]]) -> ModelPatchRaw:
    # SD/SDXL patches keep flattened keys behind `lora_unet_` (see `LayerPatcher.apply_smart_model_patch`); the
    # kohya spelling of these very names is what `convert_sdxl_keys_to_diffusers_format` leaves behind.
    state_dict: dict[str, torch.Tensor] = {
        f"lora_unet_{module.replace('.', '_')}.{param}": value
        for module, values in resolved.items()
        for param, value in values.items()
    }
    return lora_model_from_sd_state_dict(state_dict=state_dict)


_FAMILY_BUILDERS = {
    "flux1": _flux1_builder,
    "flux2": _flux2_builder,
    "sdxl": _sdxl_builder,
    "z_image": _z_image_builder,
    "wan": _wan_builder,
}
