import os
import sys
import torch
import traceback
from argparse import Namespace
from omegaconf import OmegaConf

import invokeai.version
from ...backend import Globals

def load_face_restoration(opt):
    try:
        gfpgan, codeformer, esrgan = None, None, None
        if opt.restore or opt.esrgan:
            from invokeai.backend.restoration import Restoration

            restoration = Restoration()
            if opt.restore:
                gfpgan, codeformer = restoration.load_face_restore_models(
                    opt.gfpgan_model_path
                )
            else:
                print(">> Face restoration disabled")
            if opt.esrgan:
                esrgan = restoration.load_esrgan(opt.esrgan_bg_tile)
            else:
                print(">> Upscaling disabled")
        else:
            print(">> Face restoration and upscaling disabled")
    except (ModuleNotFoundError, ImportError):
        print(traceback.format_exc(), file=sys.stderr)
        print(">> You may need to install the ESRGAN and/or GFPGAN modules")
    return gfpgan, codeformer, esrgan


