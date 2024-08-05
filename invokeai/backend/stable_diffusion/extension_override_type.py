from enum import Enum


class ExtensionOverrideType(Enum):
    STEP = "step"
    UNET_FORWARD = "unet_forward"
    COMBINE_NOISE_PREDS = "combine_noise_preds"
