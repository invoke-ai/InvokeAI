#!/usr/bin/env python
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
import io
import re
import warnings
import shutil
from urllib import request
from tqdm import tqdm
from omegaconf import OmegaConf
from huggingface_hub import HfFolder, hf_hub_url, login as hf_hub_login
from pathlib import Path
from typing import Union
from getpass_asterisk import getpass_asterisk
from transformers import CLIPTokenizer, CLIPTextModel
from ldm.invoke.globals import Globals
from ldm.invoke.readline import generic_completer

import traceback
import requests
import transformers
import warnings
warnings.filterwarnings('ignore')
import torch
transformers.logging.set_verbosity_error()

#--------------------------globals-----------------------
Model_dir = 'models'
Weights_dir = 'ldm/stable-diffusion-v1/'
Dataset_path = './configs/INITIAL_MODELS.yaml'
Default_config_file = './configs/models.yaml'
SD_Configs = './configs/stable-diffusion'

assert os.path.exists(Dataset_path),"The configs directory cannot be found. Please run this script from within the invokeai runtime directory."

Datasets = OmegaConf.load(Dataset_path)
completer = generic_completer(['yes','no'])

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
def postscript(errors: None):
    if not any(errors):
        message='''
** Model Installation Successful **

You're all set!

If you installed using one of the automated installation scripts,
execute 'invoke.sh' (Linux/macOS) or 'invoke.bat' (Windows) to
start InvokeAI.

If you installed manually, activate the 'invokeai' environment
(e.g. 'conda activate invokeai'), then run one of the following
commands to start InvokeAI.

Web UI:
    python scripts/invoke.py --web # (connect to http://localhost:9090)
Command-line interface:
   python scripts/invoke.py

Have fun!
'''

    else:
        message=f"\n** There were errors during installation. It is possible some of the models were not fully downloaded.\n"
        for err in errors:
            message += f"\t - {err}\n"
        message += "Please check the logs above and correct any issues."

    print(message)

#---------------------------------------------
def yes_or_no(prompt:str, default_yes=True):
    completer.set_options(['yes','no'])
    completer.complete_extensions(None)  # turn off path-completion mode
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

https://invoke-ai.github.io/InvokeAI/installation/020_INSTALL_MANUAL/

You may download the recommended models (about 10GB total), select a customized set, or
completely skip this step.
'''
    )
    completer.set_options(['recommended','customized','skip'])
    completer.complete_extensions(None)  # turn off path-completion mode
    selection = None
    while selection is None:
        choice = input('Download <r>ecommended models, <a>ll models, <c>ustomized list, or <s>kip this step? [r]: ')
        if choice.startswith(('r','R')) or len(choice)==0:
            selection = 'recommended'
        elif choice.startswith(('c','C')):
            selection = 'customized'
        elif choice.startswith(('a','A')):
            selection = 'all'
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

#---------------------------------------------
def recommended_datasets()->dict:
    datasets = dict()
    for ds in Datasets.keys():
        if Datasets[ds]['recommended']:
            datasets[ds]=True
    return datasets

#---------------------------------------------
def all_datasets()->dict:
    datasets = dict()
    for ds in Datasets.keys():
        datasets[ds]=True
    return datasets

#---------------------------------------------
def HfLogin(access_token) -> str:
    """
    Helper for logging in to Huggingface
    The stdout capture is needed to hide the irrelevant "git credential helper" warning
    """

    capture = io.StringIO()
    sys.stdout = capture
    try:
        hf_hub_login(token = access_token, add_to_git_credential=False)
        sys.stdout = sys.__stdout__
    except Exception as exc:
        sys.stdout = sys.__stdout__
        print(exc)
        raise exc

#-------------------------------Authenticate against Hugging Face
def authenticate(yes_to_all=False):
    print('** LICENSE AGREEMENT FOR WEIGHT FILES **')
    print("=" * shutil.get_terminal_size()[0])
    print('''
By downloading the Stable Diffusion weight files from the official Hugging Face
repository, you agree to have read and accepted the CreativeML Responsible AI License.
The license terms are located here:

   https://huggingface.co/spaces/CompVis/stable-diffusion-license

''')
    print("=" * shutil.get_terminal_size()[0])

    if not yes_to_all:
        accepted = False
        while not accepted:
            accepted = yes_or_no('Accept the above License terms?')
            if not accepted:
                print('Please accept the License or Ctrl+C to exit.')
            else:
                print('Thank you!')
    else:
        print("The program was started with a '--yes' flag, which indicates user's acceptance of the above License terms.")

    # Authenticate to Huggingface using environment variables.
    # If successful, authentication will persist for either interactive or non-interactive use.
    # Default env var expected by HuggingFace is HUGGING_FACE_HUB_TOKEN.
    print("=" * shutil.get_terminal_size()[0])
    print('Authenticating to Huggingface')
    hf_envvars = [ "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN" ]
    if not (access_token := HfFolder.get_token()):
        print(f"Huggingface token not found in cache.")

        for ev in hf_envvars:
            if (access_token := os.getenv(ev)):
                print(f"Token was found in the {ev} environment variable.... Logging in.")
                try:
                    HfLogin(access_token)
                    continue
                except ValueError:
                    print(f"Login failed due to invalid token found in {ev}")
            else:
                print(f"Token was not found in the environment variable {ev}.")
    else:
        print(f"Huggingface token found in cache.")
        try:
            HfLogin(access_token)
        except ValueError:
            print(f"Login failed due to invalid token found in cache")

    if not yes_to_all:
        print('''
You may optionally enter your Huggingface token now. InvokeAI *will* work without it, but some functionality may be limited.
See https://invoke-ai.github.io/InvokeAI/features/CONCEPTS/#using-a-hugging-face-concept for more information.

Visit https://huggingface.co/settings/tokens to generate a token. (Sign up for an account if needed).

Paste the token below using Ctrl-Shift-V (macOS/Linux) or right-click (Windows), and/or 'Enter' to continue.
You may re-run the configuration script again in the future if you do not wish to set the token right now.
        ''')
        again = True
        while again:
            try:
                access_token = getpass_asterisk.getpass_asterisk(prompt="HF Token ⮞ ")
                HfLogin(access_token)
                access_token = HfFolder.get_token()
                again = False
            except ValueError:
                again = yes_or_no('Failed to log in to Huggingface. Would you like to try again?')
                if not again:
                    print('\nRe-run the configuration script whenever you wish to set the token.')
                    print('...Continuing...')
            except EOFError:
                # this happens if the user pressed Enter on the prompt without any input; assume this means they don't want to input a token
                # safety net needed against accidental "Enter"?
                print("None provided - continuing")
                again = False

    elif access_token is None:
        print()
        print("HuggingFace login did not succeed. Some functionality may be limited; see https://invoke-ai.github.io/InvokeAI/features/CONCEPTS/#using-a-hugging-face-concept for more information")
        print()
        print(f"Re-run the configuration script without '--yes' to set the HuggingFace token interactively, or use one of the environment variables: {', '.join(hf_envvars)}")

    print("=" * shutil.get_terminal_size()[0])

    return access_token

#---------------------------------------------
# look for legacy model.ckpt in models directory and offer to
# normalize its name
def migrate_models_ckpt():
    model_path = os.path.join(Globals.root,Model_dir,Weights_dir)
    if not os.path.exists(os.path.join(model_path,'model.ckpt')):
        return
    new_name = Datasets['stable-diffusion-1.4']['file']
    print('You seem to have the Stable Diffusion v4.1 "model.ckpt" already installed.')
    rename = yes_or_no(f'Ok to rename it to "{new_name}" for future reference?')
    if rename:
        print(f'model.ckpt => {new_name}')
        os.replace(os.path.join(model_path,'model.ckpt'),os.path.join(model_path,new_name))

#---------------------------------------------
def download_weight_datasets(models:dict, access_token:str):
    migrate_models_ckpt()
    successful = dict()
    for mod in models.keys():
        repo_id = Datasets[mod]['repo_id']
        filename = Datasets[mod]['file']
        dest = os.path.join(Globals.root,Model_dir,Weights_dir)
        success = hf_download_with_resume(
            repo_id=repo_id,
            model_dir=dest,
            model_name=filename,
            access_token=access_token
        )
        if success:
            successful[mod] = True
    if len(successful) < len(models):
        print(f'\n\n** There were errors downloading one or more files. **')
        print('Press any key to try again. Type ^C to quit.\n')
        input()
        return None

    keys = ', '.join(successful.keys())
    print(f'Successfully installed {keys}')
    return successful

#---------------------------------------------
def hf_download_with_resume(repo_id:str, model_dir:str, model_name:str, access_token:str=None)->bool:
    model_dest = os.path.join(model_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    url = hf_hub_url(repo_id, model_name)

    header = {"Authorization": f'Bearer {access_token}'} if access_token else {}
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
def download_with_progress_bar(model_url:str, model_dest:str, label:str='the'):
    try:
        print(f'Installing {label} model file {model_url}...',end='',file=sys.stderr)
        if not os.path.exists(model_dest):
            os.makedirs(os.path.dirname(model_dest), exist_ok=True)
            print('',file=sys.stderr)
            request.urlretrieve(model_url,model_dest,ProgressBar(os.path.basename(model_dest)))
            print('...downloaded successfully', file=sys.stderr)
        else:
            print('...exists', file=sys.stderr)
    except Exception:
        print('...download failed')
        print(f'Error downloading {label} model')
        print(traceback.format_exc())


#---------------------------------------------
def update_config_file(successfully_downloaded:dict,opt:dict):
    config_file = opt.config_file or Default_config_file
    config_file = os.path.normpath(os.path.join(Globals.root,config_file))

    yaml = new_config_file_contents(successfully_downloaded,config_file)

    try:
        if os.path.exists(config_file):
            print(f'** {config_file} exists. Renaming to {config_file}.orig')
            os.replace(config_file,f'{config_file}.orig')
        tmpfile = os.path.join(os.path.dirname(config_file),'new_config.tmp')
        with open(tmpfile, 'w') as outfile:
            outfile.write(Config_preamble)
            outfile.write(yaml)
        os.replace(tmpfile,config_file)

    except Exception as e:
        print(f'**Error creating config file {config_file}: {str(e)} **')
        return

    print(f'Successfully created new configuration file {config_file}')


#---------------------------------------------
def new_config_file_contents(successfully_downloaded:dict, config_file:str)->str:
    if os.path.exists(config_file):
        conf = OmegaConf.load(config_file)
    else:
        conf = OmegaConf.create()

    # find the VAE file, if there is one
    vaes = {}
    default_selected = False

    for model in successfully_downloaded:
        a = Datasets[model]['config'].split('/')
        if a[0] != 'VAE':
            continue
        vae_target = a[1] if len(a)>1 else 'default'
        vaes[vae_target] = Datasets[model]['file']

    for model in successfully_downloaded:
        if Datasets[model]['config'].startswith('VAE'): # skip VAE entries
            continue
        stanza = conf[model] if model in conf else { }

        stanza['description'] = Datasets[model]['description']
        stanza['weights'] = os.path.join(Model_dir,Weights_dir,Datasets[model]['file'])
        stanza['config'] = os.path.normpath(os.path.join(SD_Configs, Datasets[model]['config']))
        stanza['width'] = Datasets[model]['width']
        stanza['height'] = Datasets[model]['height']
        stanza.pop('default',None)  # this will be set later
        if vaes:
            for target in vaes:
                if re.search(target, model, flags=re.IGNORECASE):
                    stanza['vae'] = os.path.normpath(os.path.join(Model_dir,Weights_dir,vaes[target]))
                else:
                    stanza['vae'] = os.path.normpath(os.path.join(Model_dir,Weights_dir,vaes['default']))
        # BUG - the first stanza is always the default. User should select.
        if not default_selected:
            stanza['default'] = True
            default_selected = True
        conf[model] = stanza
    return OmegaConf.to_yaml(conf)

#---------------------------------------------
# this will preload the Bert tokenizer fles
def download_bert():
    print('Installing bert tokenizer (ignore deprecation errors)...', end='',file=sys.stderr)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        from transformers import BertTokenizerFast, AutoFeatureExtractor
        download_from_hf(BertTokenizerFast,'bert-base-uncased')
        print('...success',file=sys.stderr)

#---------------------------------------------
def download_from_hf(model_class:object, model_name:str):
    print('',file=sys.stderr)  # to prevent tqdm from overwriting
    return model_class.from_pretrained(model_name,
                                       cache_dir=os.path.join(Globals.root,Model_dir,model_name),
                                       resume_download=True
    )

#---------------------------------------------
def download_clip():
    print('Installing CLIP model (ignore deprecation errors)...',file=sys.stderr)
    version = 'openai/clip-vit-large-patch14'
    print('Tokenizer...',file=sys.stderr, end='')
    download_from_hf(CLIPTokenizer,version)
    print('Text model...',file=sys.stderr, end='')
    download_from_hf(CLIPTextModel,version)
    print('...success',file=sys.stderr)

#---------------------------------------------
def download_realesrgan():
    print('Installing models from RealESRGAN...',file=sys.stderr)
    model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    model_dest = os.path.join(Globals.root,'models/realesrgan/realesr-general-x4v3.pth')
    download_with_progress_bar(model_url, model_dest, 'RealESRGAN')

def download_gfpgan():
    print('Installing GFPGAN models...',file=sys.stderr)
    for model in (
            [
                'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                './models/gfpgan/GFPGANv1.4.pth'
            ],
            [
                'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
                './models/gfpgan/weights/detection_Resnet50_Final.pth'
            ],
            [
                'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth',
                './models/gfpgan/weights/parsing_parsenet.pth'
            ],
    ):
        model_url,model_dest  = model[0],os.path.join(Globals.root,model[1])
        download_with_progress_bar(model_url, model_dest, 'GFPGAN weights')

#---------------------------------------------
def download_codeformer():
    print('Installing CodeFormer model file...',file=sys.stderr)
    model_url  = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
    model_dest = os.path.join(Globals.root,'models/codeformer/codeformer.pth')
    download_with_progress_bar(model_url, model_dest, 'CodeFormer')

#---------------------------------------------
def download_clipseg():
    print('Installing clipseg model for text-based masking...',end='', file=sys.stderr)
    import zipfile
    try:
        model_url = 'https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download'
        model_dest = os.path.join(Globals.root,'models/clipseg/clipseg_weights')
        weights_zip = 'models/clipseg/weights.zip'

        if not os.path.exists(model_dest):
            os.makedirs(os.path.dirname(model_dest), exist_ok=True)
        if not os.path.exists(f'{model_dest}/rd64-uni-refined.pth'):
            dest = os.path.join(Globals.root,weights_zip)
            request.urlretrieve(model_url,dest)
            with zipfile.ZipFile(dest,'r') as zip:
                zip.extractall(os.path.join(Globals.root,'models/clipseg'))
            os.remove(dest)

            from vendored.clipseg.models.clipseg import CLIPDensePredT
            model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, )
            model.eval()
            model.load_state_dict(
                torch.load(
                    os.path.join(Globals.root,'models/clipseg/clipseg_weights/rd64-uni-refined.pth'),
                    map_location=torch.device('cpu')
                    ),
                strict=False,
            )
    except Exception:
        print('Error installing clipseg model:')
        print(traceback.format_exc())
    print('...success',file=sys.stderr)

#-------------------------------------
def download_safety_checker():
    print('Installing model for NSFW content detection...',file=sys.stderr)
    try:
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        from transformers import AutoFeatureExtractor
    except ModuleNotFoundError:
        print('Error installing NSFW checker model:')
        print(traceback.format_exc())
        return
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    print('AutoFeatureExtractor...', end='',file=sys.stderr)
    download_from_hf(AutoFeatureExtractor,safety_model_id)
    print('StableDiffusionSafetyChecker...', end='',file=sys.stderr)
    download_from_hf(StableDiffusionSafetyChecker,safety_model_id)
    print('...success',file=sys.stderr)

#-------------------------------------
def download_weights(opt:dict) -> Union[str, None]:

    if opt.yes_to_all:
        models = recommended_datasets()
        access_token = authenticate(opt.yes_to_all)
        if len(models)>0:
            successfully_downloaded = download_weight_datasets(models, access_token)
            update_config_file(successfully_downloaded,opt)
            return

    else:
        choice = user_wants_to_download_weights()

    if choice == 'recommended':
        models = recommended_datasets()
    elif choice == 'all':
        models = all_datasets()
    elif choice == 'customized':
        models = select_datasets(choice)
        if models is None and yes_or_no('Quit?',default_yes=False):
                sys.exit(0)
    else:  # 'skip'
        return


    access_token = authenticate()

    print('\n** DOWNLOADING WEIGHTS **')
    successfully_downloaded = download_weight_datasets(models, access_token)

    update_config_file(successfully_downloaded,opt)
    if len(successfully_downloaded) < len(models):
        return "some of the model weights downloads were not successful"

#-------------------------------------
def get_root(root:str=None)->str:
    if root:
        return root
    elif os.environ.get('INVOKEAI_ROOT'):
        return os.environ.get('INVOKEAI_ROOT')
    else:
        return Globals.root

#-------------------------------------
def select_root(root:str, yes_to_all:bool=False):
    default = root or os.path.expanduser('~/invokeai')
    if (yes_to_all):
        return default
    completer.set_default_dir(default)
    completer.complete_extensions(())
    completer.set_line(default)
    directory = input(f"Select a directory in which to install InvokeAI's models and configuration files [{default}]: ").strip(' \\')
    return directory or default

#-------------------------------------
def select_outputs(root:str,yes_to_all:bool=False):
    default = os.path.normpath(os.path.join(root,'outputs'))
    if (yes_to_all):
        return default
    completer.set_default_dir(os.path.expanduser('~'))
    completer.complete_extensions(())
    completer.set_line(default)
    directory = input(f'Select the default directory for image outputs [{default}]: ').strip(' \\')
    return directory or default

#-------------------------------------
def initialize_rootdir(root:str,yes_to_all:bool=False):
    assert os.path.exists('./configs'),'Run this script from within the InvokeAI source code directory, "InvokeAI" or the runtime directory "invokeai".'

    print(f'** INITIALIZING INVOKEAI RUNTIME DIRECTORY **')
    root_selected = False
    while not root_selected:
        outputs = select_outputs(root,yes_to_all)
        outputs = outputs if os.path.isabs(outputs) else os.path.abspath(os.path.join(Globals.root,outputs))

        print(f'\nInvokeAI image outputs will be placed into "{outputs}".')
        if not yes_to_all:
            root_selected = yes_or_no('Accept this location?')
        else:
            root_selected = True

    print(f'\nYou may change the chosen output directory at any time by editing the --outdir options in "{Globals.initfile}",')
    print(f'You may also change the runtime directory by setting the environment variable INVOKEAI_ROOT.\n')

    enable_safety_checker = True
    if not yes_to_all:
        print('The NSFW (not safe for work) checker blurs out images that potentially contain sexual imagery.')
        print('It can be selectively enabled at run time with --nsfw_checker, and disabled with --no-nsfw_checker.')
        print('The following option will set whether the checker is enabled by default. Like other options, you can')
        print(f'change this setting later by editing the file {Globals.initfile}.')
        print(f"This is NOT recommended for systems with less than 6G VRAM because of the checker's memory requirements.")
        enable_safety_checker = yes_or_no('Enable the NSFW checker by default?',enable_safety_checker)

    safety_checker = '--nsfw_checker' if enable_safety_checker else '--no-nsfw_checker'

    for name in ('models','configs','embeddings'):
        os.makedirs(os.path.join(root,name), exist_ok=True)
    for src in (['configs']):
        dest = os.path.join(root,src)
        if not os.path.samefile(src,dest):
            shutil.copytree(src,dest,dirs_exist_ok=True)
        os.makedirs(outputs, exist_ok=True)

    init_file = os.path.join(Globals.root,Globals.initfile)

    print(f'Creating the initialization file at "{init_file}".\n')
    with open(init_file,'w') as f:
        f.write(f'''# InvokeAI initialization file
# This is the InvokeAI initialization file, which contains command-line default values.
# Feel free to edit. If anything goes wrong, you can re-initialize this file by deleting
# or renaming it and then running configure_invokeai.py again.

# the --outdir option controls the default location of image files.
--outdir="{outputs}"

# generation arguments
{safety_checker}

# You may place other  frequently-used startup commands here, one or more per line.
# Examples:
# --web --host=0.0.0.0
# --steps=20
# -Ak_euler_a -C10.0
#
''')

#-------------------------------------
class ProgressBar():
    def __init__(self,model_name='file'):
        self.pbar = None
        self.name = model_name

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=tqdm(desc=self.name,
                           initial=0,
                           unit='iB',
                           unit_scale=True,
                           unit_divisor=1000,
                           total=total_size)
        self.pbar.update(block_size)

#-------------------------------------
def main():
    parser = argparse.ArgumentParser(description='InvokeAI model downloader')
    parser.add_argument('--interactive',
                        dest='interactive',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='run in interactive mode (default) - DEPRECATED')
    parser.add_argument('--skip-sd-weights',
                        dest='skip_sd_weights',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='skip downloading the large Stable Diffusion weight files')
    parser.add_argument('--yes','-y',
                        dest='yes_to_all',
                        action='store_true',
                        help='answer "yes" to all prompts')
    parser.add_argument('--config_file',
                        '-c',
                        dest='config_file',
                        type=str,
                        default='./configs/models.yaml',
                        help='path to configuration file to create')
    parser.add_argument('--root_dir',
                        dest='root',
                        type=str,
                        default=None,
                        help='path to root of install directory')
    opt = parser.parse_args()


    # setting a global here
    Globals.root = os.path.expanduser(get_root(opt.root) or '')

    try:
        introduction()

        # We check for to see if the runtime directory is correctly initialized.
        if Globals.root == '' \
           or not os.path.exists(os.path.join(Globals.root,'invokeai.init')):
            initialize_rootdir(Globals.root,opt.yes_to_all)

        # Optimistically try to download all required assets. If any errors occur, add them and proceed anyway.
        errors=set()

        if not opt.interactive:
            print("WARNING: The --(no)-interactive argument is deprecated and will be removed. Use --skip-sd-weights.")
            opt.skip_sd_weights=True
        if opt.skip_sd_weights:
            print('** SKIPPING DIFFUSION WEIGHTS DOWNLOAD PER USER REQUEST **')
        else:
            print('** DOWNLOADING DIFFUSION WEIGHTS **')
            errors.add(download_weights(opt))
        print('\n** DOWNLOADING SUPPORT MODELS **')
        download_bert()
        download_clip()
        download_realesrgan()
        download_gfpgan()
        download_codeformer()
        download_clipseg()
        download_safety_checker()
        postscript(errors=errors)
    except KeyboardInterrupt:
        print('\nGoodbye! Come back soon.')
    except Exception as e:
        print(f'\nA problem occurred during initialization.\nThe error was: "{str(e)}"')
        print(traceback.format_exc())

#-------------------------------------
if __name__ == '__main__':
    main()
