from invokeai.backend.model_manager.model_on_disk import StateDict
from invokeai.backend.model_manager.omi.vendor.convert.lora import (
    convert_flux_lora as omi_flux,
)
from invokeai.backend.model_manager.omi.vendor.convert.lora import (
    convert_lora_util as lora_util,
)
from invokeai.backend.model_manager.omi.vendor.convert.lora import (
    convert_sdxl_lora as omi_sdxl,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType


def convert_from_omi(weights_sd: StateDict, base: BaseModelType):
    keyset = {
        BaseModelType.Flux: omi_flux.convert_flux_lora_key_sets(),
        BaseModelType.StableDiffusionXL: omi_sdxl.convert_sdxl_lora_key_sets(),
    }[base]
    source = "omi"
    target = "legacy_diffusers"
    return lora_util.__convert(weights_sd, keyset, source, target)  # type: ignore
