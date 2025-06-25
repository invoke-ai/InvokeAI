# import omi_model_standards.convert.lora.convert_lora_util as lora_util
# from omi_model_standards.convert.lora.convert_flux_lora import convert_flux_lora_key_sets
# from omi_model_standards.convert.lora.convert_sdxl_lora import convert_sdxl_lora_key_sets

from invokeai.backend.model_manager.model_on_disk import StateDict
from invokeai.backend.model_manager.taxonomy import BaseModelType


def convert_from_omi(weights_sd: StateDict, base: BaseModelType):
    raise NotImplementedError
    # keyset = {
    #     BaseModelType.Flux: convert_flux_lora_key_sets(),
    #     BaseModelType.StableDiffusionXL: convert_sdxl_lora_key_sets(),
    # }[base]
    # source = "omi"
    # target = "legacy_diffusers"
    # return lora_util.__convert(weights_sd, keyset, source, target)  # type: ignore
