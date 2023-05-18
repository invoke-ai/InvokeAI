import warnings

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as ImageType

import invokeai.backend.util.logging as logger
from invokeai.app.services.config import get_invokeai_config
config = get_invokeai_config()

class ESRGAN:
    def __init__(self, bg_tile_size=400) -> None:
        self.bg_tile_size = bg_tile_size

    def load_esrgan_bg_upsampler(self, denoise_str):
        if not torch.cuda.is_available():  # CPU or MPS on M1
            use_half_precision = False
        else:
            use_half_precision = True

        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact

        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        model_path = config.root_dir / "models/realesrgan/realesr-general-x4v3.pth"
        wdn_model_path = config.root_dir / "models/realesrgan/realesr-general-wdn-x4v3.pth"
        scale = 4

        bg_upsampler = RealESRGANer(
            scale=scale,
            model_path=[model_path, wdn_model_path],
            model=model,
            tile=self.bg_tile_size,
            dni_weight=[denoise_str, 1 - denoise_str],
            tile_pad=10,
            pre_pad=0,
            half=use_half_precision,
        )

        return bg_upsampler

    def process(
        self,
        image: ImageType,
        strength: float,
        seed: str = None,
        upsampler_scale: int = 2,
        denoise_str: float = 0.75,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            try:
                upsampler = self.load_esrgan_bg_upsampler(denoise_str)
            except Exception:
                import sys
                import traceback

                logger.error("Error loading Real-ESRGAN:")
                print(traceback.format_exc(), file=sys.stderr)

        if upsampler_scale == 0:
            logger.warning("Real-ESRGAN: Invalid scaling option. Image not upscaled.")
            return image

        if seed is not None:
            logger.info(
                f"Real-ESRGAN Upscaling seed:{seed}, scale:{upsampler_scale}x, tile:{self.bg_tile_size}, denoise:{denoise_str}"
            )
        # ESRGAN outputs images with partial transparency if given RGBA images; convert to RGB
        image = image.convert("RGB")

        # REALSRGAN expects a BGR np array; make array and flip channels
        bgr_image_array = np.array(image, dtype=np.uint8)[..., ::-1]

        output, _ = upsampler.enhance(
            bgr_image_array,
            outscale=upsampler_scale,
            alpha_upsampler="realesrgan",
        )

        # Flip the channels back to RGB
        res = Image.fromarray(output[..., ::-1])

        if strength < 1.0:
            # Resize the image to the new image if the sizes have changed
            if output.size != image.size:
                image = image.resize(res.size)
            res = Image.blend(image, res, strength)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        upsampler = None

        return res
