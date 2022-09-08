import torch
import warnings
import os
import sys
import numpy as np

from PIL import Image
from scripts.dream import create_argv_parser
from ldm.dream.devices import choose_torch_device
import subprocess

arg_parser = create_argv_parser()
opt = arg_parser.parse_args()

model_path = os.path.join(opt.gfpgan_dir, opt.gfpgan_model_path)
gfpgan_model_exists = os.path.isfile(model_path)

def run_gfpgan(image, strength, seed, upsampler_scale=4):
    print(f'>> GFPGAN - Restoring Faces for image seed:{seed}')
    gfpgan = None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        try:
            if not gfpgan_model_exists:
                raise Exception('GFPGAN model not found at path ' + model_path)

            sys.path.append(os.path.abspath(opt.gfpgan_dir))
            from gfpgan import GFPGANer

            bg_upsampler = _load_gfpgan_bg_upsampler(
                opt.gfpgan_bg_upsampler, upsampler_scale, opt.gfpgan_bg_tile
            )

            gfpgan = GFPGANer(
                model_path=model_path,
                upscale=upsampler_scale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=bg_upsampler,
            )
        except Exception:
            import traceback

            print('>> Error loading GFPGAN:', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    if gfpgan is None:
        print(
            f'>> GFPGAN not initialized. Their packages must be installed as siblings to the "stable-diffusion" folder, or set explicitly using the --gfpgan_dir option.'
        )
        return image

    image = image.convert('RGB')

    cropped_faces, restored_faces, restored_img = gfpgan.enhance(
        np.array(image, dtype=np.uint8),
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    res = Image.fromarray(restored_img)

    if strength < 1.0:
        # Resize the image to the new image if the sizes have changed
        if restored_img.size != image.size:
            image = image.resize(res.size)
        res = Image.blend(image, res, strength)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gfpgan = None

    return res


def _load_gfpgan_bg_upsampler(bg_upsampler, upsampler_scale, bg_tile=400):
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            warnings.warn(
                'The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                'If you really want to use it, please modify the corresponding codes.'
            )
            bg_upsampler = None
        else:
            model_path = {
                2: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                4: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            }

            if upsampler_scale not in model_path:
                return None

            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            if upsampler_scale == 4:
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
            if upsampler_scale == 2:
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                )

            bg_upsampler = RealESRGANer(
                scale=upsampler_scale,
                model_path=model_path[upsampler_scale],
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True,
            )  # need to set False in CPU mode
    else:
        bg_upsampler = None

    return bg_upsampler

def create_tmp_image(image, seed, outdir):
    from ldm.dream.pngwriter import PngWriter

    pngwriter = PngWriter(outdir)
    prefix = pngwriter.unique_prefix()

    name = f'tmp_{prefix}.{seed}.png'
    cwd = os.getcwd()
    img_path = os.path.join(cwd, outdir, name)

    image.save(img_path)
    image.close()

    return img_path

def real_esrgan_upscale(image, strength, upsampler_scale, seed, outdir=None):
    print(
        f'>> Real-ESRGAN Upscaling seed:{seed} : scale:{upsampler_scale}x'
    )

    # Fix the Real-ESRGAN lib that does not work with Apple Silicon
    # Switch to the local Real-ESRGAN binary
    if choose_torch_device() == 'mps':
        img_path = create_tmp_image(image, seed, outdir)

        cmd = [
            './realesrgan-ncnn-vulkan',
            '-i', str(img_path),
            '-o', str(img_path),
            '-s', str(upsampler_scale)
        ]

        # Apply realesrgan-x4plus when x4 is selected.
        # `realesrgan-x4plus` provides a slightly better rendering
        # than the default model.
        if upsampler_scale == 4:
            cmd.append('-n')
            cmd.append('realesrgan-x4plus')

        try:
            # Run ../realesrgan/realesrgan-ncnn-vulkan -i <input> -o <ouput> -s [2..4]
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                cwd="../realesrgan"
            ).stdout.decode('utf-8')

            res = Image.open(img_path).convert('RGBA')
            os.remove(img_path)
        except Exception:
            import traceback
            print('>> Error ESRGAN resize failed:', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=UserWarning)

            try:
                upsampler = _load_gfpgan_bg_upsampler(
                    opt.gfpgan_bg_upsampler, upsampler_scale, opt.gfpgan_bg_tile
                )
            except Exception:
                import traceback

                print('>> Error loading Real-ESRGAN:', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

        output, img_mode = upsampler.enhance(
            np.array(image, dtype=np.uint8),
            outscale=upsampler_scale,
            alpha_upsampler=opt.gfpgan_bg_upsampler,
        )

        res = Image.fromarray(output)

        if strength < 1.0:
            # Resize the image to the new image if the sizes have changed
            if output.size != image.size:
                image = image.resize(res.size)
            res = Image.blend(image, res, strength)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        upsampler = None

    return res
