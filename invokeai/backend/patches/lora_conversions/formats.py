from invokeai.backend.model_manager.taxonomy import FluxLoRAFormat
from invokeai.backend.patches.lora_conversions.flux_control_lora_utils import is_state_dict_likely_flux_control
from invokeai.backend.patches.lora_conversions.flux_diffusers_lora_conversion_utils import (
    is_state_dict_likely_in_flux_diffusers_format,
)
from invokeai.backend.patches.lora_conversions.flux_kohya_lora_conversion_utils import (
    is_state_dict_likely_in_flux_kohya_format,
)
from invokeai.backend.patches.lora_conversions.flux_onetrainer_lora_conversion_utils import (
    is_state_dict_likely_in_flux_onetrainer_format,
)


def flux_format_from_state_dict(state_dict):
    if is_state_dict_likely_in_flux_kohya_format(state_dict):
        return FluxLoRAFormat.Kohya
    elif is_state_dict_likely_in_flux_onetrainer_format(state_dict):
        return FluxLoRAFormat.OneTrainer
    elif is_state_dict_likely_in_flux_diffusers_format(state_dict):
        return FluxLoRAFormat.Diffusers
    elif is_state_dict_likely_flux_control(state_dict):
        return FluxLoRAFormat.Control
    else:
        return None
