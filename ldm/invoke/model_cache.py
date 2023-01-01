'''
Manage a cache of Stable Diffusion model files for fast switching.
They are moved between GPU and CPU as necessary. If CPU memory falls
below a preset minimum, the least recently used model will be
cleared and loaded from disk when next needed.
'''

import torch
import os
import io
import time
import gc
import hashlib
import psutil
import sys
import transformers
import traceback
import textwrap
import contextlib
from typing import Union
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from ldm.util import instantiate_from_config, ask_user
from ldm.invoke.globals import Globals
from picklescan.scanner import scan_file_path
from pathlib import Path

DEFAULT_MAX_MODELS=2

class ModelCache(object):
    def __init__(self, config:OmegaConf, device_type:str, precision:str, max_loaded_models=DEFAULT_MAX_MODELS):
        '''
        Initialize with the path to the models.yaml config file,
        the torch device type, and precision. The optional
        min_avail_mem argument specifies how much unused system
        (CPU) memory to preserve. The cache of models in RAM will
        grow until this value is approached. Default is 2G.
        '''
        # prevent nasty-looking CLIP log message
        transformers.logging.set_verbosity_error()
        self.config = config
        self.precision = precision
        self.device = torch.device(device_type)
        self.max_loaded_models = max_loaded_models
        self.models = {}
        self.stack = []  # this is an LRU FIFO
        self.current_model = None

    def valid_model(self, model_name:str)->bool:
        '''
        Given a model name, returns True if it is a valid
        identifier.
        '''
        return model_name in self.config

    def get_model(self, model_name:str):
        '''
        Given a model named identified in models.yaml, return
        the model object. If in RAM will load into GPU VRAM.
        If on disk, will load from there.
        '''
        if not self.valid_model(model_name):
            print(f'** "{model_name}" is not a known model name. Please check your models.yaml file')
            return self.current_model

        if self.current_model != model_name:
            if model_name not in self.models: # make room for a new one
                self._make_cache_room()
            self.offload_model(self.current_model)

        if model_name in self.models:
            requested_model = self.models[model_name]['model']
            print(f'>> Retrieving model {model_name} from system RAM cache')
            self.models[model_name]['model'] = self._model_from_cpu(requested_model)
            width = self.models[model_name]['width']
            height = self.models[model_name]['height']
            hash = self.models[model_name]['hash']

        else: # we're about to load a new model, so potentially offload the least recently used one
            try:
                requested_model, width, height, hash = self._load_model(model_name)
                self.models[model_name] = {
                    'model': requested_model,
                    'width': width,
                    'height': height,
                    'hash': hash,
                }

            except Exception as e:
                print(f'** model {model_name} could not be loaded: {str(e)}')
                print(traceback.format_exc())
                assert self.current_model,'** FATAL: no current model to restore to'
                print(f'** restoring {self.current_model}')
                self.get_model(self.current_model)
                return

        self.current_model = model_name
        self._push_newest_model(model_name)
        return {
            'model':requested_model,
            'width':width,
            'height':height,
            'hash': hash
        }

    def default_model(self) -> str:
        '''
        Returns the name of the default model, or None
        if none is defined.
        '''
        for model_name in self.config:
            if self.config[model_name].get('default'):
                return model_name

    def set_default_model(self,model_name:str) -> None:
        '''
        Set the default model. The change will not take
        effect until you call model_cache.commit()
        '''
        assert model_name in self.models,f"unknown model '{model_name}'"

        config = self.config
        for model in config:
            config[model].pop('default',None)
        config[model_name]['default'] = True

    def list_models(self) -> dict:
        '''
        Return a dict of models in the format:
        { model_name1: {'status': ('active'|'cached'|'not loaded'),
                        'description': description,
                       },
          model_name2: { etc }
        '''
        models = {}
        for name in self.config:
            description = self.config[name].description if 'description' in self.config[name] else '<no description>'
            weights = self.config[name].weights if 'weights' in self.config[name] else '<no weights>'
            config = self.config[name].config if 'config' in self.config[name] else '<no config>'
            width = self.config[name].width if 'width' in self.config[name] else 512
            height = self.config[name].height if 'height' in self.config[name] else 512
            default = self.config[name].default if 'default' in self.config[name] else False
            vae = self.config[name].vae if 'vae' in self.config[name] else '<no vae>'

            if self.current_model == name:
                status = 'active'
            elif name in self.models:
                status = 'cached'
            else:
                status = 'not loaded'

            models[name]={
                'status' : status,
                'description' : description,
                'weights': weights,
                'config': config,
                'width': width,
                'height': height,
                'vae': vae,
                'default': default
            }
        return models

    def print_models(self) -> None:
        '''
        Print a table of models, their descriptions, and load status
        '''
        models = self.list_models()
        for name in models:
            line = f'{name:25s} {models[name]["status"]:>10s}  {models[name]["description"]}'
            if models[name]['status'] == 'active':
                line = f'\033[1m{line}\033[0m'
            print(line)

    def del_model(self, model_name:str) -> None:
        '''
        Delete the named model.
        '''
        omega = self.config
        del omega[model_name]
        if model_name in self.stack:
            self.stack.remove(model_name)

    def add_model(self, model_name:str, model_attributes:dict, clobber=False) -> None:
        '''
        Update the named model with a dictionary of attributes. Will fail with an
        assertion error if the name already exists. Pass clobber=True to overwrite.
        On a successful update, the config will be changed in memory and the
        method will return True. Will fail with an assertion error if provided
        attributes are incorrect or the model name is missing.
        '''
        omega = self.config
        for field in ('description','weights','height','width','config'):
            assert field in model_attributes, f'required field {field} is missing'
        assert (clobber or model_name not in omega), f'attempt to overwrite existing model definition "{model_name}"'

        config = omega[model_name] if model_name in omega else {}
        for field in model_attributes:
            if field == 'weights':
                field.replace('\\', '/')
            config[field] = model_attributes[field]

        omega[model_name] = config
        if clobber:
            self._invalidate_cached_model(model_name)

    def _load_model(self, model_name:str):
        """Load and initialize the model from configuration variables passed at object creation time"""
        if model_name not in self.config:
            print(f'"{model_name}" is not a known model name. Please check your models.yaml file')

        mconfig = self.config[model_name]
        config = mconfig.config
        weights = mconfig.weights
        vae = mconfig.get('vae')
        width = mconfig.width
        height = mconfig.height

        if not os.path.isabs(weights):
            weights = os.path.normpath(os.path.join(Globals.root,weights))
        # scan model
        self.scan_model(model_name, weights)

        print(f'>> Loading {model_name} from {weights}')

        # for usage statistics
        if self._has_cuda():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        tic = time.time()

        # this does the work
        if not os.path.isabs(config):
            config = os.path.join(Globals.root,config)
        omega_config = OmegaConf.load(config)
        with open(weights,'rb') as f:
            weight_bytes = f.read()
        model_hash  = self._cached_sha256(weights,weight_bytes)
        sd = torch.load(io.BytesIO(weight_bytes), map_location='cpu')
        del weight_bytes
        # merged models from auto11 merge board are flat for some reason
        if 'state_dict' in sd:
            sd = sd['state_dict']

        print(f'   | Forcing garbage collection prior to loading new model')
        gc.collect()
        model = instantiate_from_config(omega_config.model)
        model.load_state_dict(sd, strict=False)

        if self.precision == 'float16':
            print('   | Using faster float16 precision')
            model.to(torch.float16)
        else:
            print('   | Using more accurate float32 precision')

        # look and load a matching vae file. Code borrowed from AUTOMATIC1111 modules/sd_models.py
        if vae:
            if not os.path.isabs(vae):
                vae = os.path.normpath(os.path.join(Globals.root,vae))
            if os.path.exists(vae):
                print(f'   | Loading VAE weights from: {vae}')
                vae_ckpt = torch.load(vae, map_location="cpu")
                vae_dict = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss"}
                model.first_stage_model.load_state_dict(vae_dict, strict=False)
            else:
                print(f'   | VAE file {vae} not found. Skipping.')

        model.to(self.device)
        # model.to doesn't change the cond_stage_model.device used to move the tokenizer output, so set it here
        model.cond_stage_model.device = self.device

        model.eval()

        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                module._orig_padding_mode = module.padding_mode

        # usage statistics
        toc = time.time()
        print(f'>> Model loaded in', '%4.2fs' % (toc - tic))

        if self._has_cuda():
            print(
                '>> Max VRAM used to load the model:',
                '%4.2fG' % (torch.cuda.max_memory_allocated() / 1e9),
                '\n>> Current VRAM usage:'
                '%4.2fG' % (torch.cuda.memory_allocated() / 1e9),
            )

        return model, width, height, model_hash

    def offload_model(self, model_name:str) -> None:
        '''
        Offload the indicated model to CPU. Will call
        _make_cache_room() to free space if needed.
        '''
        if model_name not in self.models:
            return

        print(f'>> Offloading {model_name} to CPU')
        model = self.models[model_name]['model']
        self.models[model_name]['model'] = self._model_to_cpu(model)

        gc.collect()
        if self._has_cuda():
            torch.cuda.empty_cache()

    def scan_model(self, model_name, checkpoint):
        # scan model
        print(f'>> Scanning Model: {model_name}')
        scan_result = scan_file_path(checkpoint)
        if scan_result.infected_files != 0:
            if scan_result.infected_files == 1:
                print(f'\n### Issues Found In Model: {scan_result.issues_count}')
                print('### WARNING: The model you are trying to load seems to be infected.')
                print('### For your safety, InvokeAI will not load this model.')
                print('### Please use checkpoints from trusted sources.')
                print("### Exiting InvokeAI")
                sys.exit()
            else:
                print('\n### WARNING: InvokeAI was unable to scan the model you are using.')
                model_safe_check_fail = ask_user('Do you want to to continue loading the model?', ['y', 'n'])
                if model_safe_check_fail.lower() != 'y':
                    print("### Exiting InvokeAI")
                    sys.exit()
        else:
            print('>> Model Scanned. OK!!')
    
    def search_models(self, search_folder):

        print(f'>> Finding Models In: {search_folder}')
        models_folder = Path(search_folder).glob('**/*.ckpt')

        files = [x for x in models_folder if x.is_file()]

        found_models = []
        for file in files:
            found_models.append({
                'name': file.stem,
                'location': str(file.resolve()).replace('\\', '/')
            })

        return search_folder, found_models

    def _make_cache_room(self) -> None:
        num_loaded_models = len(self.models)
        if num_loaded_models >= self.max_loaded_models:
            least_recent_model = self._pop_oldest_model()
            print(f'>> Cache limit (max={self.max_loaded_models}) reached. Purging {least_recent_model}')
            if least_recent_model is not None:
                del self.models[least_recent_model]
                gc.collect()

    def print_vram_usage(self) -> None:
        if self._has_cuda:
            print('>> Current VRAM usage: ','%4.2fG' % (torch.cuda.memory_allocated() / 1e9))

    def commit(self,config_file_path:str) -> None:
        '''
        Write current configuration out to the indicated file.
        '''
        yaml_str = OmegaConf.to_yaml(self.config)
        if not os.path.isabs(config_file_path):
            config_file_path = os.path.normpath(os.path.join(Globals.root,opt.conf))
        tmpfile = os.path.join(os.path.dirname(config_file_path),'new_config.tmp')
        with open(tmpfile, 'w') as outfile:
            outfile.write(self.preamble())
            outfile.write(yaml_str)
        os.replace(tmpfile,config_file_path)

    def preamble(self) -> str:
        '''
        Returns the preamble for the config file.
        '''
        return textwrap.dedent('''\
            # This file describes the alternative machine learning models
            # available to InvokeAI script.
            #
            # To add a new model, follow the examples below. Each
            # model requires a model config file, a weights file,
            # and the width and height of the images it
            # was trained on.
        ''')

    def _invalidate_cached_model(self,model_name:str) -> None:
        self.offload_model(model_name)
        if model_name in self.stack:
            self.stack.remove(model_name)
        self.models.pop(model_name,None)

    def _model_to_cpu(self,model):
        if self.device != 'cpu':
            model.cond_stage_model.device = 'cpu'
            model.first_stage_model.to('cpu')
            model.cond_stage_model.to('cpu')
            model.model.to('cpu')
            return model.to('cpu')
        else:
            return model

    def _model_from_cpu(self,model):
        if self.device != 'cpu':
            model.to(self.device)
            model.first_stage_model.to(self.device)
            model.cond_stage_model.to(self.device)
            model.cond_stage_model.device = self.device
        return model

    def _pop_oldest_model(self):
        '''
        Remove the first element of the FIFO, which ought
        to be the least recently accessed model. Do not
        pop the last one, because it is in active use!
        '''
        return self.stack.pop(0)

    def _push_newest_model(self,model_name:str) -> None:
        '''
        Maintain a simple FIFO. First element is always the
        least recent, and last element is always the most recent.
        '''
        with contextlib.suppress(ValueError):
            self.stack.remove(model_name)
        self.stack.append(model_name)

    def _has_cuda(self) -> bool:
        return self.device.type == 'cuda'

    def _cached_sha256(self,path,data) -> Union[str, bytes]:
        dirname    = os.path.dirname(path)
        basename   = os.path.basename(path)
        base, _    = os.path.splitext(basename)
        hashpath   = os.path.join(dirname,base+'.sha256')

        if os.path.exists(hashpath) and os.path.getmtime(path) <= os.path.getmtime(hashpath):
            with open(hashpath) as f:
                hash = f.read()
            return hash

        print(f'>> Calculating sha256 hash of weights file')
        tic = time.time()
        sha = hashlib.sha256()
        sha.update(data)
        hash = sha.hexdigest()
        toc = time.time()
        print(f'>> sha256 = {hash}','(%4.2fs)' % (toc - tic))

        with open(hashpath,'w') as f:
            f.write(hash)
        return hash
