from enum import Enum


class ExtensionCallbackType(Enum):
    SETUP = "setup"
    PRE_DENOISE_LOOP = "pre_denoise_loop"
    POST_DENOISE_LOOP = "post_denoise_loop"
    PRE_STEP = "pre_step"
    POST_STEP = "post_step"
    PRE_UNET_FORWARD = "pre_unet_forward"
    POST_UNET_FORWARD = "post_unet_forward"
    POST_COMBINE_NOISE_PREDS = "post_combine_noise_preds"
