'''
ldm.invoke.globals defines a small number of global variables that would
otherwise have to be passed through long and complex call chains.

It defines a Namespace object named "Globals" that contains
the attributes:

  - root           - the root directory under which "models" and "outputs" can be found
  - initfile       - path to the initialization file
  - try_patchmatch - option to globally disable loading of 'patchmatch' module
  - always_use_cpu - force use of CPU even if GPU is available
'''

import os
import os.path as osp
from argparse import Namespace
from pathlib import Path
from typing import Union

Globals = Namespace()

# Where to look for the initialization file and other key components
Globals.initfile = 'invokeai.init'
Globals.models_file = 'models.yaml'
Globals.models_dir = 'models'
Globals.config_dir = 'configs'
Globals.autoscan_dir = 'weights'
Globals.converted_ckpts_dir = 'converted_ckpts'

# Set the default root directory. This can be overwritten by explicitly
# passing the `--root <directory>` argument on the command line.
# logic is:
# 1) use INVOKEAI_ROOT environment variable (no check for this being a valid directory)
# 2) use VIRTUAL_ENV environment variable, with a check for initfile being there
# 3) use ~/invokeai

if os.environ.get('INVOKEAI_ROOT'):
    Globals.root = osp.abspath(os.environ.get('INVOKEAI_ROOT'))
elif os.environ.get('VIRTUAL_ENV') and Path(os.environ.get('VIRTUAL_ENV'),'..',Globals.initfile).exists():
    Globals.root = osp.abspath(osp.join(os.environ.get('VIRTUAL_ENV'), '..'))
else:
    Globals.root = osp.abspath(osp.expanduser('~/invokeai'))

# Try loading patchmatch
Globals.try_patchmatch = True

# Use CPU even if GPU is available (main use case is for debugging MPS issues)
Globals.always_use_cpu = False

# Whether the internet is reachable for dynamic downloads
# The CLI will test connectivity at startup time.
Globals.internet_available = True

# Whether to disable xformers
Globals.disable_xformers = False

# Low-memory tradeoff for guidance calculations.
Globals.sequential_guidance = False

# whether we are forcing full precision
Globals.full_precision = False

# whether we should convert ckpt files into diffusers models on the fly
Globals.ckpt_convert = False

# logging tokenization everywhere
Globals.log_tokenization = False

def global_config_file()->Path:
    return Path(Globals.root, Globals.config_dir, Globals.models_file)

def global_config_dir()->Path:
    return Path(Globals.root, Globals.config_dir)

def global_models_dir()->Path:
    return Path(Globals.root, Globals.models_dir)

def global_autoscan_dir()->Path:
    return Path(Globals.root, Globals.autoscan_dir)

def global_converted_ckpts_dir()->Path:
    return Path(global_models_dir(), Globals.converted_ckpts_dir)

def global_set_root(root_dir:Union[str,Path]):
    Globals.root = root_dir

def global_cache_dir(subdir:Union[str,Path]='')->Path:
    '''
    Returns Path to the model cache directory. If a subdirectory
    is provided, it will be appended to the end of the path, allowing
    for Hugging Face-style conventions. Currently, Hugging Face has
    moved all models into the "hub" subfolder, so for any pretrained
    HF model, use:
         global_cache_dir('hub')

    The legacy location for transformers used to be global_cache_dir('transformers')
    and global_cache_dir('diffusers') for diffusers.
    '''
    home: str = os.getenv('HF_HOME')

    if home is None:
        home = os.getenv('XDG_CACHE_HOME')

        if home is not None:
            # Set `home` to $XDG_CACHE_HOME/huggingface, which is the default location mentioned in HuggingFace Hub Client Library.
            # See: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables#xdgcachehome
            home += os.sep + 'huggingface'

    if home is not None:
        return Path(home,subdir)
    else:
        return Path(Globals.root,'models',subdir)
