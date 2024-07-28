from enum import Enum


class ExtensionOverrideType(Enum):
    STEP = "step"
    UNET_FORWARD = "unet_forward"  # predict_noise, run_model, ...?
    COMBINE_NOISE_PREDS = "combine_noise_preds"
