import sys
import traceback
import torch
from typing import types
from ...backend.restoration import Restoration
from ...backend.util import choose_torch_device, CPU_DEVICE, MPS_DEVICE

# This should be a real base class for postprocessing functions,
# but right now we just instantiate the existing gfpgan, esrgan
# and codeformer functions.
class RestorationServices:
    '''Face restoration and upscaling'''
    
    def __init__(self,args,logger:types.ModuleType):
        try:
            gfpgan, codeformer, esrgan = None, None, None
            if args.restore or args.esrgan:
                restoration = Restoration()
                if args.restore:
                    gfpgan, codeformer = restoration.load_face_restore_models(
                        args.gfpgan_model_path
                    )
                else:
                    logger.info("Face restoration disabled")
                    if args.esrgan:
                        esrgan = restoration.load_esrgan(args.esrgan_bg_tile)
                    else:
                        logger.info("Upscaling disabled")
            else:
                logger.info("Face restoration and upscaling disabled")
        except (ModuleNotFoundError, ImportError):
            print(traceback.format_exc(), file=sys.stderr)
            logger.info("You may need to install the ESRGAN and/or GFPGAN modules")
        self.device = torch.device(choose_torch_device())
        self.gfpgan = gfpgan
        self.codeformer = codeformer
        self.esrgan = esrgan
        self.logger = logger
        self.logger.info('Face restoration initialized')

    # note that this one method does gfpgan and codepath reconstruction, as well as
    # esrgan upscaling
    # TO DO: refactor into separate methods
    def upscale_and_reconstruct(
        self,
        image_list,
        facetool="gfpgan",
        upscale=None,
        upscale_denoise_str=0.75,
        strength=0.0,
        codeformer_fidelity=0.75,
        save_original=False,
        image_callback=None,
        prefix=None,
    ):
        results = []
        for r in image_list:
            image, seed = r
            try:
                if strength > 0:
                    if self.gfpgan is not None or self.codeformer is not None:
                        if facetool == "gfpgan":
                            if self.gfpgan is None:
                                self.logger.info(
                                    "GFPGAN not found. Face restoration is disabled."
                                )
                            else:
                                image = self.gfpgan.process(image, strength, seed)
                        if facetool == "codeformer":
                            if self.codeformer is None:
                                self.logger.info(
                                    "CodeFormer not found. Face restoration is disabled."
                                )
                            else:
                                cf_device = (
                                    CPU_DEVICE if self.device == MPS_DEVICE else self.device
                                )
                                image = self.codeformer.process(
                                    image=image,
                                    strength=strength,
                                    device=cf_device,
                                    seed=seed,
                                    fidelity=codeformer_fidelity,
                                )
                    else:
                        self.logger.info("Face Restoration is disabled.")
                if upscale is not None:
                    if self.esrgan is not None:
                        if len(upscale) < 2:
                            upscale.append(0.75)
                        image = self.esrgan.process(
                            image,
                            upscale[1],
                            seed,
                            int(upscale[0]),
                            denoise_str=upscale_denoise_str,
                        )
                    else:
                        self.logger.info("ESRGAN is disabled. Image not upscaled.")
            except Exception as e:
                self.logger.info(
                    f"Error running RealESRGAN or GFPGAN. Your image was not upscaled.\n{e}"
                )

            if image_callback is not None:
                image_callback(image, seed, upscaled=True, use_prefix=prefix)
            else:
                r[0] = image

            results.append([image, seed])

        return results
