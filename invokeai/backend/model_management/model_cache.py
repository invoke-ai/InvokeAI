"""
Manage a cache of Stable Diffusion model files for fast switching.
They are moved between GPU and CPU as necessary. If the cache
grows larger than a preset maximum, then the least recently used
model will be cleared and (re)loaded from disk when next needed.
"""

import contextlib
import hashlib
import gc
import time
import os
import psutil

import safetensors
import safetensors.torch
import torch
import transformers
import warnings

from pathlib import Path
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    SchedulerMixin,
    logging as diffusers_logging,
)
from transformers import(
    CLIPTokenizer,
    CLIPFeatureExtractor,
    CLIPTextModel,
    logging as transformers_logging,
)    
from huggingface_hub import scan_cache_dir
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from picklescan.scanner import scan_file_path
from typing import Sequence, Union

from invokeai.backend.globals import Globals, global_cache_dir
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
    )
from ..stable_diffusion import (
    StableDiffusionGeneratorPipeline,
)
from ..stable_diffusion.offloading import ModelGroup, FullyLoadedModelGroup
from ..util import CUDA_DEVICE, ask_user, download_with_resume

MAX_MODELS_CACHED = 4
        
class ModelCache(object):
    def __init__(
            self,
            max_models_cached: int=MAX_MODELS_CACHED,
            execution_device: torch.device=torch.device('cuda'),
            precision: torch.dtype=torch.float16,
            sequential_offload: bool=False,
    ):
        self.model_group: ModelGroup=FullyLoadedModelGroup(execution_device)
        self.models: dict = dict()
        self.stack: Sequence = list()
        self.sequential_offload: bool=sequential_offload
        self.precision: torch.dtype=precision
        self.max_models_cached: int=max_models_cached
        self.device: torch.device=execution_device

    def get_model(
            self,
            repo_id_or_path: Union[str,Path],
            model_class: type=StableDiffusionGeneratorPipeline,
            subfolder: Path=None,
            revision: str=None,
            )->Union[
                AutoencoderKL,
                CLIPTokenizer,
                CLIPFeatureExtractor,
                CLIPTextModel,
                UNet2DConditionModel,
                StableDiffusionSafetyChecker,
                StableDiffusionGeneratorPipeline,
            ]:
        '''
        Load and return a HuggingFace model, with RAM caching.
        :param repo_id_or_path: either the HuggingFace repo_id or a Path to a local model
        :param subfolder: name of a subfolder in which the model can be found, e.g. "vae"
        :param revision: model revision
        :param model_class: class of model to return
        '''
        key = self._model_key(repo_id_or_path,model_class,revision,subfolder) # internal unique identifier for the model
        if key in self.models: # cached - move to bottom of stack
            previous_key = self._current_model_key
            with contextlib.suppress(ValueError):
                self.stack.remove(key)
                self.stack.append(key)
            if previous_key != key:
                if hasattr(self.current_model,'to'):
                    print(f'DEBUG: loading {key} into GPU')
                    self.model_group.offload_current()
                    self.model_group.load(self.models[key])

        else:  # not cached -load
            self._make_cache_room()
            self.model_group.offload_current()
            print(f'DEBUG: loading {key} from disk/net')
            model = self._load_model_from_storage(
                repo_id_or_path=repo_id_or_path,
                subfolder=subfolder,
                revision=revision,
                model_class=model_class
            )
            if hasattr(model,'to'):
                self.model_group.install(model) # register with the model group
            self.stack.append(key)          # add to LRU cache
            self.models[key]=model          # keep copy of model in dict
        return self.models[key]

    @staticmethod
    def _model_key(path,model_class,revision,subfolder)->str:
        return ':'.join([str(path),str(model_class),str(revision),str(subfolder)])

    def _make_cache_room(self):
        models_in_ram = len(self.models)
        while models_in_ram >= self.max_models_cached:
            if least_recently_used_key := self.stack.pop(0):
                print(f'DEBUG: maximum cache size reached: cache_size={models_in_ram}; unloading model {least_recently_used_key}')
                self.model_group.uninstall(self.models[least_recently_used_key])
                del self.models[least_recently_used_key]
            models_in_ram = len(self.models)
        gc.collect()

    @property
    def current_model(self)->Union[
                AutoencoderKL,
                CLIPTokenizer,
                CLIPFeatureExtractor,
                CLIPTextModel,
                UNet2DConditionModel,
                StableDiffusionSafetyChecker,
                StableDiffusionGeneratorPipeline,
            ]:
        '''
        Returns current model.
        '''
        return self.models[self._current_model_key]

    @property
    def _current_model_key(self)->str:
        '''
        Returns key of currently loaded model.
        '''
        return self.stack[-1]

    def _load_model_from_storage(
            self,
            repo_id_or_path: Union[str,Path],
            subfolder: Path=None,
            revision: str=None,
            model_class: type=StableDiffusionGeneratorPipeline,
            )->Union[
                AutoencoderKL,
                CLIPTokenizer,
                CLIPFeatureExtractor,
                CLIPTextModel,
                UNet2DConditionModel,
                StableDiffusionSafetyChecker,
                StableDiffusionGeneratorPipeline,
            ]:
        '''
        Load and return a HuggingFace model.
        :param repo_id_or_path: either the HuggingFace repo_id or a Path to a local model
        :param subfolder: name of a subfolder in which the model can be found, e.g. "vae"
        :param revision: model revision
        :param model_class: class of model to return
        '''
        # silence transformer and diffuser warnings
        with SilenceWarnings():
            model = model_class.from_pretrained(
                repo_id_or_path,
                revision=revision,
                subfolder=subfolder or '.',
                cache_dir=global_cache_dir('hub'),
            )
        if self.sequential_offload and isinstance(model,StableDiffusionGeneratorPipeline):
            model.enable_offload_submodels(self.device)
        elif hasattr(model,'to'):
            model.to(self.device)
        return model

class SilenceWarnings(object):
    def __init__(self):
        self.transformers_verbosity = transformers_logging.get_verbosity()
        self.diffusers_verbosity = diffusers_logging.get_verbosity()
        
    def __enter__(self):
        transformers_logging.set_verbosity_error()
        diffusers_logging.set_verbosity_error()
        warnings.simplefilter('ignore')

    def __exit__(self,type,value,traceback):
        transformers_logging.set_verbosity(self.transformers_verbosity)
        diffusers_logging.set_verbosity(self.diffusers_verbosity)
        warnings.simplefilter('default')
