#!/usr/bin/env python3
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)
# Before running stable-diffusion on an internet-isolated machine,
# run this script from one with internet connectivity. The
# two machines must share a common .cache directory.
from transformers import CLIPTokenizer, CLIPTextModel
import clip
from transformers import BertTokenizerFast, AutoFeatureExtractor
import sys
import transformers
import os
import warnings
import torch
import urllib.request
import zipfile
import traceback

transformers.logging.set_verbosity_error()

#---------------------------------------------
# this will preload the Bert tokenizer fles
def download_bert():
    print('Installing bert tokenizer (ignore deprecation errors)...', end='')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        print('...success')
        sys.stdout.flush()

#---------------------------------------------
# this will download requirements for Kornia
def download_kornia():
    print('Installing Kornia requirements...', end='')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        import kornia
    print('...success')

#---------------------------------------------
def download_clip():
    version = 'openai/clip-vit-large-patch14'
    sys.stdout.flush()
    print('Loading CLIP model...',end='')
    tokenizer = CLIPTokenizer.from_pretrained(version)
    transformer = CLIPTextModel.from_pretrained(version)
    print('...success')

#---------------------------------------------
def download_gfpgan():
    print('Installing models from RealESRGAN and facexlib...',end='')
    try:
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        )

        FaceRestoreHelper(1, det_model='retinaface_resnet50')
        print('...success')
    except Exception:
        print('Error loading ESRGAN:')
        print(traceback.format_exc())

    print('Loading models from GFPGAN')
    import urllib.request
    for model in (
            [
                'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                'src/gfpgan/experiments/pretrained_models/GFPGANv1.4.pth'
            ],
            [
                'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
                './gfpgan/weights/detection_Resnet50_Final.pth'
            ],
            [
                'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth',
                './gfpgan/weights/parsing_parsenet.pth'
            ],
    ):
        model_url,model_dest  = model
        try:
            if not os.path.exists(model_dest):
                print(f'Downloading gfpgan model file {model_url}...',end='')
                os.makedirs(os.path.dirname(model_dest), exist_ok=True)
                urllib.request.urlretrieve(model_url,model_dest)
                print('...success')
        except Exception:
            print('Error loading GFPGAN:')
            print(traceback.format_exc())

#---------------------------------------------
def download_codeformer():
    print('Installing CodeFormer model file...',end='')
    try:
            model_url  = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
            model_dest = 'ldm/invoke/restoration/codeformer/weights/codeformer.pth'
            if not os.path.exists(model_dest):
                print('Downloading codeformer model file...')
                os.makedirs(os.path.dirname(model_dest), exist_ok=True)
                urllib.request.urlretrieve(model_url,model_dest)
    except Exception:
        print('Error loading CodeFormer:')
        print(traceback.format_exc())
    print('...success')
    
#---------------------------------------------
def download_clipseg():
    print('Installing clipseg model for text-based masking...',end='')
    try:
        model_url  = 'https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download'
        model_dest = 'src/clipseg/clipseg_weights.zip'
        weights_dir = 'src/clipseg/weights'
        if not os.path.exists(weights_dir):
            os.makedirs(os.path.dirname(model_dest), exist_ok=True)
            urllib.request.urlretrieve(model_url,model_dest)
            with zipfile.ZipFile(model_dest,'r') as zip:
                zip.extractall('src/clipseg')
                os.rename('src/clipseg/clipseg_weights','src/clipseg/weights')
            os.remove(model_dest)
            from clipseg_models.clipseg import CLIPDensePredT
            model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, )
            model.eval()
            model.load_state_dict(
                torch.load(
                    'src/clipseg/weights/rd64-uni-refined.pth',
                    map_location=torch.device('cpu')
                    ),
                strict=False,
            )
    except Exception:
        print('Error installing clipseg model:')
        print(traceback.format_exc())
    print('...success')

#-------------------------------------
def download_safety_checker():
    print('Installing safety model for NSFW content detection...',end='')
    try:
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    except ModuleNotFoundError:
        print('Error installing safety checker model:')
        print(traceback.format_exc())
        return
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
    print('...success')
    
#-------------------------------------
if __name__ == '__main__':
    download_bert()
    download_kornia()
    download_clip()
    download_gfpgan()
    download_codeformer()
    download_clipseg()
    download_safety_checker()

    
