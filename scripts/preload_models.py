#!/usr/bin/env python3
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)
# Before running stable-diffusion on an internet-isolated machine,
# run this script from one with internet connectivity. The
# two machines must share a common .cache directory.
#
# Coauthor: Kevin Turner http://github.com/keturn
#
print('Loading Python libraries...\n')
import argparse
import sys
import os
import warnings
from urllib import request
from tqdm import tqdm
from omegaconf import OmegaConf
from huggingface_hub import HfFolder, hf_hub_url
from pathlib import Path
from getpass_asterisk import getpass_asterisk
import traceback
import requests
import clip
import transformers
import torch
transformers.logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')
#warnings.simplefilter('ignore')
#warnings.filterwarnings('ignore',category=DeprecationWarning)
#warnings.filterwarnings('ignore',category=UserWarning)

# deferred loading so that help message can be printed quickly
def load_libs():
    pass

#--------------------------globals--
Model_dir = './models/ldm/stable-diffusion-v1/'
Default_config_file = './configs/models.yaml'
SD_Configs = './configs/stable-diffusion'
Datasets = {
    'stable-diffusion-1.5':  {
        'description': 'The newest Stable Diffusion version 1.5 weight file (4.27 GB)',
        'repo_id': 'runwayml/stable-diffusion-v1-5',
        'config': 'v1-inference.yaml',
        'file': 'v1-5-pruned-emaonly.ckpt',
        'recommended': True,
        'width': 512,
        'height': 512,
    },
    'inpainting-1.5': {
        'description': 'RunwayML SD 1.5 model optimized for inpainting (4.27 GB)',
        'repo_id': 'runwayml/stable-diffusion-inpainting',
        'config': 'v1-inpainting-inference.yaml',
        'file': 'sd-v1-5-inpainting.ckpt',
        'recommended': True,
        'width': 512,
        'height': 512,
    },
    'stable-diffusion-1.4': {
        'description': 'The original Stable Diffusion version 1.4 weight file (4.27 GB)',
        'repo_id': 'CompVis/stable-diffusion-v-1-4-original',
        'config': 'v1-inference.yaml',
        'file': 'sd-v1-4.ckpt',
        'recommended': False,
        'width': 512,
        'height': 512,
    },
    'waifu-diffusion-1.3': {
        'description': 'Stable Diffusion 1.4 fine tuned on anime-styled images (4.27)',
        'repo_id': 'hakurei/waifu-diffusion-v1-3',
        'config': 'v1-inference.yaml',
        'file': 'model-epoch09-float32.ckpt',
        'recommended': False,
        'width': 512,
        'height': 512,
    },
    'ft-mse-improved-autoencoder-840000': {
        'description': 'StabilityAI improved autoencoder fine-tuned for human faces (recommended; 335 MB)',
        'repo_id': 'stabilityai/sd-vae-ft-mse-original',
        'config': 'VAE',
        'file': 'vae-ft-mse-840000-ema-pruned.ckpt',
        'recommended': True,
        'width': 512,
        'height': 512,
    },
}
Config_preamble = '''# This file describes the alternative machine learning models
# available to InvokeAI script.
#
# To add a new model, follow the examples below. Each
# model requires a model config file, a weights file,
# and the width and height of the images it
# was trained on.
'''

#---------------------------------------------
def introduction():
    print(
        '''Welcome to InvokeAI. This script will help download the Stable Diffusion weight files
and other large models that are needed for text to image generation. At any point you may interrupt
this program and resume later.\n'''
    )

#--------------------------------------------
def postscript():
    print(
        '''\n** Model Installation Successful **\nYou're all set! You may now launch InvokeAI using one of these two commands:
Web version: 

    python scripts/invoke.py --web  (connect to http://localhost:9090)

Command-line version:

   python scripts/invoke.py

Have fun!
'''
)

#---------------------------------------------
def yes_or_no(prompt:str, default_yes=True):
    default = "y" if default_yes else 'n'
    response = input(f'{prompt} [{default}] ') or default
    if default_yes:
        return response[0] not in ('n','N')
    else:
        return response[0] in ('y','Y')

#---------------------------------------------
def user_wants_to_download_weights()->str:
    '''
    Returns one of "skip", "recommended" or "customized"
    '''
    print('''You can download and configure the weights files manually or let this
script do it for you. Manual installation is described at:

https://github.com/invoke-ai/InvokeAI/blob/main/docs/installation/INSTALLING_MODELS.md

You may download the recommended models (about 10GB total), select a customized set, or
completely skip this step.
'''
    )
    selection = None
    while selection is None:
        choice = input('Download <r>ecommended models, <c>ustomize the list, or <s>kip this step? [r]: ')
        if choice.startswith(('r','R')) or len(choice)==0:
            selection = 'recommended'
        elif choice.startswith(('c','C')):
            selection = 'customized'
        elif choice.startswith(('s','S')):
            selection = 'skip'
    return selection

#---------------------------------------------
def select_datasets(action:str):
    done = False
    while not done:
        datasets = dict()
        dflt = None   # the first model selected will be the default; TODO let user change
        counter = 1

        if action == 'customized':
            print('''
Choose the weight file(s) you wish to download. Before downloading you 
will be given the option to view and change your selections.
'''
        )
            for ds in Datasets.keys():
                recommended = '(recommended)' if Datasets[ds]['recommended'] else ''
                print(f'[{counter}] {ds}:\n    {Datasets[ds]["description"]} {recommended}')
                if yes_or_no('    Download?',default_yes=Datasets[ds]['recommended']):
                    datasets[ds]=counter
                    counter += 1
        else:
            for ds in Datasets.keys():
                if Datasets[ds]['recommended']:
                    datasets[ds]=counter
                    counter += 1
                
        print('The following weight files will be downloaded:')
        for ds in datasets:
            dflt = '*' if dflt is None else ''
            print(f'   [{datasets[ds]}] {ds}{dflt}')
        print("*default")
        ok_to_download = yes_or_no('Ok to download?')
        if not ok_to_download:
            if yes_or_no('Change your selection?'):
                action = 'customized'
                pass
            else:
                done = True
        else:
            done = True
    return datasets if ok_to_download else None


#-------------------------------Authenticate against Hugging Face
def authenticate():
    print('''
To download the Stable Diffusion weight files from the official Hugging Face 
repository, you need to read and accept the CreativeML Responsible AI license.

This involves a few easy steps.

1. If you have not already done so, create an account on Hugging Face's web site
   using the "Sign Up" button:

   https://huggingface.co/join

   You will need to verify your email address as part of the HuggingFace
   registration process.

2. Log into your Hugging Face account:

    https://huggingface.co/login

3. Accept the license terms located here:

   https://huggingface.co/runwayml/stable-diffusion-v1-5

   and here:

   https://huggingface.co/runwayml/stable-diffusion-inpainting

    (Yes, you have to accept two slightly different license agreements)
'''
    )
    input('Press <enter> when you are ready to continue:')
    print('(Fetching Hugging Face token from cache...',end='')
    access_token = HfFolder.get_token()
    if access_token is not None:
        print('found')
    
    if access_token is None:
        print('not found')
        print('''
4. Thank you! The last step is to enter your HuggingFace access token so that
   this script is authorized to initiate the download. Go to the access tokens
   page of your Hugging Face account and create a token by clicking the 
   "New token" button:

   https://huggingface.co/settings/tokens

   (You can enter anything you like in the token creation field marked "Name". 
   "Role" should be "read").

   Now copy the token to your clipboard and paste it here: '''
        )
        access_token = getpass_asterisk.getpass_asterisk()
    return access_token

#---------------------------------------------
# look for legacy model.ckpt in models directory and offer to
# normalize its name
def migrate_models_ckpt():
    if not os.path.exists(os.path.join(Model_dir,'model.ckpt')):
        return
    new_name = Datasets['stable-diffusion-1.4']['file']
    print('You seem to have the Stable Diffusion v4.1 "model.ckpt" already installed.')
    rename = yes_or_no(f'Ok to rename it to "{new_name}" for future reference?')
    if rename:
        print(f'model.ckpt => {new_name}')
        os.rename(os.path.join(Model_dir,'model.ckpt'),os.path.join(Model_dir,new_name))
            
#---------------------------------------------
def download_weight_datasets(models:dict, access_token:str):
    migrate_models_ckpt()
    successful = dict()
    for mod in models.keys():
        repo_id = Datasets[mod]['repo_id']
        filename = Datasets[mod]['file']
        success = download_with_resume(
            repo_id=repo_id,
            model_name=filename,
            access_token=access_token
        )
        if success:
            successful[mod] = True
    if len(successful) < len(models):
        print(f'\n\n** There were errors downloading one or more files. **')
        print('Please double-check your license agreements, and your access token.')
        HfFolder.delete_token()
        print('Press any key to try again. Type ^C to quit.\n')
        input()
        return None

    HfFolder.save_token(access_token)
    keys = ', '.join(successful.keys())
    print(f'Successfully installed {keys}') 
    return successful
    
#---------------------------------------------
def download_with_resume(repo_id:str, model_name:str, access_token:str)->bool:
    model_dest = os.path.join(Model_dir, model_name)
    os.makedirs(os.path.dirname(model_dest), exist_ok=True)
    url = hf_hub_url(repo_id, model_name)

    header = {"Authorization": f'Bearer {access_token}'}
    open_mode = 'wb'
    exist_size = 0
    
    if os.path.exists(model_dest):
        exist_size = os.path.getsize(model_dest)
        header['Range'] = f'bytes={exist_size}-'
        open_mode = 'ab'

    resp = requests.get(url, headers=header, stream=True)
    total = int(resp.headers.get('content-length', 0))
    
    if resp.status_code==416:  # "range not satisfiable", which means nothing to return
        print(f'* {model_name}: complete file found. Skipping.')
        return True
    elif resp.status_code != 200:
        print(f'** An error occurred during downloading {model_name}: {resp.reason}')
    elif exist_size > 0:
        print(f'* {model_name}: partial file found. Resuming...')
    else:
        print(f'* {model_name}: Downloading...')

    try:
        if total < 2000:
            print(f'*** ERROR DOWNLOADING {model_name}: {resp.text}')
            return False

        with open(model_dest, open_mode) as file, tqdm(
                desc=model_name,
                initial=exist_size,
                total=total+exist_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1000,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except Exception as e:
        print(f'An error occurred while downloading {model_name}: {str(e)}')
        return False
    return True
                             
#---------------------------------------------
def update_config_file(successfully_downloaded:dict,opt:dict):
    Config_file = opt.config_file or Default_config_file
    
    yaml = new_config_file_contents(successfully_downloaded,Config_file)

    try:
        if os.path.exists(Config_file):
            print(f'* {Config_file} exists. Renaming to {Config_file}.orig')
            os.rename(Config_file,f'{Config_file}.orig')
        tmpfile = os.path.join(os.path.dirname(Config_file),'new_config.tmp')
        with open(tmpfile, 'w') as outfile:
            outfile.write(Config_preamble)
            outfile.write(yaml)
        os.rename(tmpfile,Config_file)

    except Exception as e:
        print(f'**Error creating config file {Config_file}: {str(e)} **')
        return

    print(f'Successfully created new configuration file {Config_file}')

    
#---------------------------------------------    
def new_config_file_contents(successfully_downloaded:dict, Config_file:str)->str:
    if os.path.exists(Config_file):
        conf = OmegaConf.load(Config_file)
    else:
        conf = OmegaConf.create()

    # find the VAE file, if there is one
    vae = None
    default_selected = False
    
    for model in successfully_downloaded:
        if Datasets[model]['config'] == 'VAE':
            vae = Datasets[model]['file']
    
    for model in successfully_downloaded:
        if Datasets[model]['config'] == 'VAE': # skip VAE entries
            continue
        stanza = conf[model] if model in conf else { }
        
        stanza['description'] = Datasets[model]['description']
        stanza['weights'] = os.path.join(Model_dir,Datasets[model]['file'])
        stanza['config'] =os.path.join(SD_Configs, Datasets[model]['config'])
        stanza['width'] = Datasets[model]['width']
        stanza['height'] = Datasets[model]['height']
        stanza.pop('default',None)  # this will be set later
        if vae:
            stanza['vae'] = os.path.join(Model_dir,vae)
        # BUG - the first stanza is always the default. User should select.
        if not default_selected:
            stanza['default'] = True
            default_selected = True
        conf[model] = stanza
    return OmegaConf.to_yaml(conf)
    
#---------------------------------------------
# this will preload the Bert tokenizer fles
def download_bert():
    print('Installing bert tokenizer (ignore deprecation errors)...', end='')
    sys.stdout.flush()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        from transformers import BertTokenizerFast, AutoFeatureExtractor
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        print('...success')

#---------------------------------------------
# this will download requirements for Kornia
def download_kornia():
    print('Installing Kornia requirements (ignore deprecation errors)...', end='')
    sys.stdout.flush()
    import kornia
    print('...success')

#---------------------------------------------
def download_clip():
    print('Loading CLIP model...',end='')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        from transformers import CLIPTokenizer, CLIPTextModel
    sys.stdout.flush()
    version = 'openai/clip-vit-large-patch14'
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
                request.urlretrieve(model_url,model_dest)
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
                request.urlretrieve(model_url,model_dest)
    except Exception:
        print('Error loading CodeFormer:')
        print(traceback.format_exc())
    print('...success')
    
#---------------------------------------------
def download_clipseg():
    print('Installing clipseg model for text-based masking...',end='')
    import zipfile
    try:
        model_url  = 'https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download'
        model_dest = 'src/clipseg/clipseg_weights.zip'
        weights_dir = 'src/clipseg/weights'
        if not os.path.exists(weights_dir):
            os.makedirs(os.path.dirname(model_dest), exist_ok=True)
        if not os.path.exists('src/clipseg/weights/rd64-uni-refined.pth'):
            request.urlretrieve(model_url,model_dest)
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
        from transformers import AutoFeatureExtractor
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
    parser = argparse.ArgumentParser(description='InvokeAI model downloader')
    parser.add_argument('--interactive',
                        dest='interactive',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='run in interactive mode (default)')
    parser.add_argument('--config_file',
                        '-c',
                        dest='config_file',
                        type=str,
                        default='./configs/models.yaml',
                        help='path to configuration file to create')
    opt = parser.parse_args()
    load_libs()
    
    try:
        if opt.interactive:
            introduction()
            print('** WEIGHT SELECTION **')
            choice = user_wants_to_download_weights()
            if choice != 'skip':
                models = select_datasets(choice)
                if models is None:
                    if yes_or_no('Quit?',default_yes=False):
                        sys.exit(0)

                done = False
                while not done:
                    print('** LICENSE AGREEMENT FOR WEIGHT FILES **')
                    access_token = authenticate()
                    print('\n** DOWNLOADING WEIGHTS **')
                    successfully_downloaded = download_weight_datasets(models, access_token)
                    done = successfully_downloaded is not None
                update_config_file(successfully_downloaded,opt)

        print('\n** DOWNLOADING SUPPORT MODELS **')
        download_bert()
        download_kornia()
        download_clip()
        download_gfpgan()
        download_codeformer()
        download_clipseg()
        download_safety_checker()
        postscript()
    except KeyboardInterrupt:
        print('\nGoodbye! Come back soon.')
    except Exception as e:
        print(f'\nA problem occurred during download.\nThe error was: "{str(e)}"')


    
