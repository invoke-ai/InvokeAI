import re
from typing import Any, Dict

from invokeai.backend.patches.lora_conversions.flux_kohya_lora_conversion_utils import (
    FLUX_KOHYA_CLIP_KEY_REGEX,
    FLUX_KOHYA_T5_KEY_REGEX,
)

# A regex pattern that matches all of the transformer keys in the OneTrainer FLUX LoRA format.
# The OneTrainer format uses a mix of the Kohya and Diffusers formats:
#   - The base model keys are in Diffusers format.
#   - Periods are replaced with underscores, to match Kohya.
#   - The LoRA key suffixes (e.g. .alpha, .lora_down.weight, .lora_up.weight) match Kohya.
# Example keys:
# - "lora_transformer_single_transformer_blocks_0_attn_to_k.alpha"
# - "lora_transformer_single_transformer_blocks_0_attn_to_k.dora_scale"
# - "lora_transformer_single_transformer_blocks_0_attn_to_k.lora_down.weight"
# - "lora_transformer_single_transformer_blocks_0_attn_to_k.lora_up.weight"
FLUX_ONETRAINER_TRANSFORMER_KEY_REGEX = (
    r"lora_transformer_(single_transformer_blocks|transformer_blocks)_(\d+)_(\w+)\.(.*)"
)


def is_state_dict_likely_in_flux_onetrainer_format(state_dict: Dict[str, Any]) -> bool:
    """Checks if the provided state dict is likely in the OneTrainer FLUX LoRA format.

    This is intended to be a high-precision detector, but it is not guaranteed to have perfect precision. (A
    perfect-precision detector would require checking all keys against a whitelist and verifying tensor shapes.)

    Note that OneTrainer matches the Kohya format for the CLIP and T5 models.
    """
    return all(
        re.match(FLUX_ONETRAINER_TRANSFORMER_KEY_REGEX, k)
        or re.match(FLUX_KOHYA_CLIP_KEY_REGEX, k)
        or re.match(FLUX_KOHYA_T5_KEY_REGEX, k)
        for k in state_dict.keys()
    )
