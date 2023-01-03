import torch
import warnings
import os
import sys
import numpy as np
from ldm.invoke.globals import Globals

from PIL import Image


class GFPGAN():
    def __init__(
            self,
            gfpgan_model_path='models/gfpgan/GFPGANv1.4.pth'
    ) -> None:

        if not os.path.isabs(gfpgan_model_path):
            gfpgan_model_path=os.path.abspath(os.path.join(Globals.root,gfpgan_model_path))
        self.model_path = gfpgan_model_path
        self.gfpgan_model_exists = os.path.isfile(self.model_path)

        if not self.gfpgan_model_exists:
            print('## NOT FOUND: GFPGAN model not found at ' + self.model_path)
            return None

    def model_exists(self):
        return os.path.isfile(self.model_path)

    def process(self, image, strength: float, seed: str = None):
        if seed is not None:
            print(f'>> GFPGAN - Restoring Faces for image seed:{seed}')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            cwd = os.getcwd()
            os.chdir(os.path.join(Globals.root,'models'))
            try:
                from gfpgan import GFPGANer
                self.gfpgan = GFPGANer(
                    model_path=self.model_path,
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                )
            except Exception:
                import traceback
                print('>> Error loading GFPGAN:', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
            os.chdir(cwd)

        if self.gfpgan is None:
            print(
                f'>> WARNING: GFPGAN not initialized.'
            )
            print(
                f'>> Download https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth to {self.model_path}'
            )

        image = image.convert('RGB')

        # GFPGAN expects a BGR np array; make array and flip channels
        bgr_image_array = np.array(image, dtype=np.uint8)[...,::-1]

        _, _, restored_img = self.gfpgan.enhance(
            bgr_image_array,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )

        # Flip the channels back to RGB
        res = Image.fromarray(restored_img[...,::-1])

        if strength < 1.0:
            # Resize the image to the new image if the sizes have changed
            if restored_img.size != image.size:
                image = image.resize(res.size)
            res = Image.blend(image, res, strength)


        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.gfpgan = None

        return res
