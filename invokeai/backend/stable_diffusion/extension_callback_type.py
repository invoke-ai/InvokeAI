from enum import Enum


class ExtensionCallbackType(Enum):
    SETUP = "setup"
    PRE_DENOISE_LOOP = "pre_denoise_loop"
    POST_DENOISE_LOOP = "post_denoise_loop"
    PRE_STEP = "pre_step"
    POST_STEP = "post_step"
    PRE_UNET = "pre_unet"
    POST_UNET = "post_unet"
    POST_COMBINE_NOISE_PREDS = "post_combine_noise_preds"
