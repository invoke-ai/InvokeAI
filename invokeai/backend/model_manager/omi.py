from invokeai.backend.model_manager.model_on_disk import StateDict
from invokeai.backend.model_manager.taxonomy import BaseModelType
from omi_model_standards.convert.lora.convert_sdxl_lora import convert_sdxl_lora_key_sets
from omi_model_standards.convert.lora.convert_flux_lora import convert_flux_lora_key_sets
from omi_model_standards.convert.lora.convert_sd_lora import convert_sd_lora_key_sets
from omi_model_standards.convert.lora.convert_sd3_lora import convert_sd3_lora_key_sets
import omi_model_standards.convert.lora.convert_lora_util as lora_util


def convert_to_omi(weights_sd: StateDict, base: BaseModelType):
    keyset = {
        BaseModelType.Flux: convert_flux_lora_key_sets(),
        BaseModelType.StableDiffusionXL: convert_sdxl_lora_key_sets(),
        BaseModelType.StableDiffusion1: convert_sd_lora_key_sets(),
        BaseModelType.StableDiffusion3: convert_sd3_lora_key_sets(),
    }[base]
    return lora_util.convert_to_omi(weights_sd, keyset)
