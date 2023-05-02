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

from enum import Enum
from pathlib import Path
from pydantic import BaseModel
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    SchedulerMixin,
    logging as diffusers_logging,
)
from huggingface_hub import list_repo_refs,HfApi
from transformers import(
    CLIPTokenizer,
    CLIPFeatureExtractor,
    CLIPTextModel,
    logging as transformers_logging,
)    
from huggingface_hub import scan_cache_dir
from picklescan.scanner import scan_file_path
from typing import Sequence, Union

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
    )
from . import load_pipeline_from_original_stable_diffusion_ckpt
from ..globals import Globals, global_cache_dir
from ..stable_diffusion import (
    StableDiffusionGeneratorPipeline,
)
from ..stable_diffusion.offloading import ModelGroup, FullyLoadedModelGroup
from ..util import CUDA_DEVICE, ask_user, download_with_resume

MAX_MODELS_CACHED = 4

# This is the mapping from the stable diffusion submodel dict key to the class
class SDModelType(Enum):
    diffusion_pipeline=StableDiffusionGeneratorPipeline # whole thing
    vae=AutoencoderKL                                   # parts
    text_encoder=CLIPTextModel
    tokenizer=CLIPTokenizer
    unet=UNet2DConditionModel
    scheduler=SchedulerMixin
    safety_checker=StableDiffusionSafetyChecker
    feature_extractor=CLIPFeatureExtractor
        
# The list of model classes we know how to fetch, for typechecking
ModelClass = Union[tuple([x.value for x in SDModelType])]

# Legacy information needed to load a legacy checkpoint file
class LegacyInfo(BaseModel):
    config_file: Path
    vae_file: Path

class ModelCache(object):
    def __init__(
            self,
            max_models_cached: int=MAX_MODELS_CACHED,
            execution_device: torch.device=torch.device('cuda'),
            precision: torch.dtype=torch.float16,
            sequential_offload: bool=False,
            sha_chunksize: int = 16777216,
    ):
        '''
        :param max_models_cached: Maximum number of models to cache in CPU RAM [4]
        :param execution_device: Torch device to load active model into [torch.device('cuda')]
        :param precision: Precision for loaded models [torch.float16]
        :param sequential_offload: Conserve VRAM by loading and unloading each stage of the pipeline sequentially
        :param sha_chunksize: Chunksize to use when calculating sha256 model hash
        '''
        self.model_group: ModelGroup=FullyLoadedModelGroup(execution_device)
        self.models: dict = dict()
        self.stack: Sequence = list()
        self.sequential_offload: bool=sequential_offload
        self.precision: torch.dtype=precision
        self.max_models_cached: int=max_models_cached
        self.device: torch.device=execution_device
        self.sha_chunksize=sha_chunksize

    def get_submodel(
            self,
            repo_id_or_path: Union[str,Path],
            submodel: SDModelType=SDModelType.vae,
            subfolder: Path=None,
            revision: str=None,
            legacy_info: LegacyInfo=None,
    )->ModelClass:
        '''
        Load and return a HuggingFace model, with RAM caching.
        :param repo_id_or_path: either the HuggingFace repo_id or a Path to a local model
        :param submodel: an SDModelType enum indicating the model part to return, e.g. SDModelType.vae
        :param subfolder: name of a subfolder in which the model can be found, e.g. "vae"
        :param revision: model revision name
        :param legacy_info: a LegacyInfo object containing additional info needed to load a legacy ckpt
        '''
        parent_model = self.get_model(
            repo_id_or_path=repo_id_or_path,
            subfolder=subfolder,
            revision=revision,
        )
        return getattr(parent_model, submodel.name)

    def get_model(
            self,
            repo_id_or_path: Union[str,Path],
            model_type: SDModelType=SDModelType.diffusion_pipeline,
            subfolder: Path=None,
            revision: str=None,
            legacy_info: LegacyInfo=None,
            )->ModelClass:
        '''
        Load and return a HuggingFace model, with RAM caching.
        :param repo_id_or_path: either the HuggingFace repo_id or a Path to a local model
        :param subfolder: name of a subfolder in which the model can be found, e.g. "vae"
        :param revision: model revision
        :param model_class: class of model to return
        :param legacy_info: a LegacyInfo object containing additional info needed to load a legacy ckpt
        '''
        key = self._model_key( # internal unique identifier for the model
            repo_id_or_path,
            model_type.value,
            revision,
            subfolder
        ) 
        if key in self.models: # cached - move to bottom of stack
            previous_key = self._current_model_key
            with contextlib.suppress(ValueError):
                self.stack.remove(key)
                self.stack.append(key)
            if previous_key != key:
                if hasattr(self.current_model,'to'):
                    print(f'  | loading {key} into GPU')
                    self.model_group.offload_current()
                    self.model_group.load(self.models[key])
        else:  # not cached -load
            self._make_cache_room()
            self.model_group.offload_current()
            print(f'  | loading model {key} from disk/net')
            model = self._load_model_from_storage(
                repo_id_or_path=repo_id_or_path,
                model_class=model_type.value,
                subfolder=subfolder,
                revision=revision,
                legacy_info=legacy_info,
            )
            if hasattr(model,'to'):
                self.model_group.install(model) # register with the model group
            self.stack.append(key)          # add to LRU cache
            self.models[key]=model          # keep copy of model in dict
        return self.models[key]

    @staticmethod
    def model_hash(repo_id_or_path: Union[str,Path],
                   revision: str=None)->str:
        '''
        Given the HF repo id or path to a model on disk, returns a unique
        hash. Works for legacy checkpoint files, HF models on disk, and HF repo IDs
        :param repo_id_or_path: repo_id string or Path to model file/directory on disk.
        :param revision: optional revision string (if fetching a HF repo_id)
        '''
        if self.is_legacy_ckpt(repo_id_or_path):
            return self._legacy_model_hash(repo_id_or_path)
        elif Path(repo_id_or_path).is_dir():
            return self._local_model_hash(repo_id_or_path)
        else:
            return self._hf_commit_hash(repo_id_or_path,revision)

    @staticmethod
    def _model_key(path,model_class,revision,subfolder)->str:
        return ':'.join([str(path),model_class.__name__,str(revision or ''),str(subfolder or '')])

    def _make_cache_room(self):
        models_in_ram = len(self.models)
        while models_in_ram >= self.max_models_cached:
            if least_recently_used_key := self.stack.pop(0):
                print(f'  | maximum cache size reached: cache_size={models_in_ram}; unloading model {least_recently_used_key}')
                self.model_group.uninstall(self.models[least_recently_used_key])
                del self.models[least_recently_used_key]
            models_in_ram = len(self.models)
        gc.collect()

    @property
    def current_model(self)->ModelClass:
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
            model_class: ModelClass=StableDiffusionGeneratorPipeline,
            legacy_info: LegacyInfo=None,
            )->ModelClass:
        '''
        Load and return a HuggingFace model.
        :param repo_id_or_path: either the HuggingFace repo_id or a Path to a local model
        :param subfolder: name of a subfolder in which the model can be found, e.g. "vae"
        :param revision: model revision
        :param model_class: class of model to return, defaults to StableDiffusionGeneratorPIpeline
        :param legacy_info: a LegacyInfo object containing additional info needed to load a legacy ckpt
        '''
        # silence transformer and diffuser warnings
        with SilenceWarnings():
            if self.is_legacy_ckpt(repo_id_or_path):
                model = self._load_ckpt_from_storage(repo_id_or_path, legacy_info)
            else:
                model = self._load_diffusers_from_storage(
                    repo_id_or_path,
                    subfolder,
                    revision,
                    model_class,
                )
        if self.sequential_offload and isinstance(model,StableDiffusionGeneratorPipeline):
            model.enable_offload_submodels(self.device)
        elif hasattr(model,'to'):
            model.to(self.device)
        return model

    def _load_diffusers_from_storage(
            self,
            repo_id_or_path: Union[str,Path],
            subfolder: Path=None,
            revision: str=None,
            model_class: ModelClass=StableDiffusionGeneratorPipeline,
    )->ModelClass:
        '''
        Load and return a HuggingFace model using from_pretrained().
        :param repo_id_or_path: either the HuggingFace repo_id or a Path to a local model
        :param subfolder: name of a subfolder in which the model can be found, e.g. "vae"
        :param revision: model revision
        :param model_class: class of model to return, defaults to StableDiffusionGeneratorPIpeline
        '''
        return model_class.from_pretrained(
            repo_id_or_path,
            revision=revision,
            subfolder=subfolder or '.',
            cache_dir=global_cache_dir('hub'),
        )

    @classmethod
    def is_legacy_ckpt(cls, repo_id_or_path: Union[str,Path])->bool:
        '''
        Return true if the indicated path is a legacy checkpoint
        :param repo_id_or_path: either the HuggingFace repo_id or a Path to a local model
        '''
        path = Path(repo_id_or_path)
        return path.is_file() and path.suffix in [".ckpt",".safetensors"]

    def _load_ckpt_from_storage(self,
                                ckpt_path: Union[str,Path],
                                legacy_info:LegacyInfo)->StableDiffusionGeneratorPipeline:
        '''
        Load a legacy checkpoint, convert it, and return a StableDiffusionGeneratorPipeline.
        :param ckpt_path: string or Path pointing to the weights file (.ckpt or .safetensors)
        :param legacy_info: LegacyInfo object containing paths to legacy config file and alternate vae if required
        '''
        assert legacy_info is not None
        pipeline = load_pipeline_from_original_stable_diffusion_ckpt(
            checkpoint_path=ckpt_path,
            original_config_file=legacy_info.config_file,
            vae_path=legacy_info.vae_file,
            return_generator_pipeline=True,
            precision=self.precision,
        )
        return pipeline

    def _legacy_model_hash(self, checkpoint_path: Union[str,Path])->str:
        sha = hashlib.sha256()
        path = Path(checkpoint_path)
        assert path.is_file()

        hashpath = path.parent / f"{path.name}.sha256"
        if hashpath.exists() and path.stat().st_mtime <= hashpath.stat().st_mtime:
            with open(hashpath) as f:
                hash = f.read()
            return hash
        
        print(f'  | computing hash of model {path.name}')
        with open(path, "rb") as f:
            while chunk := f.read(self.sha_chunksize):
                sha.update(chunk)
        hash = sha.hexdigest()
                
        with open(hashpath, "w") as f:
            f.write(hash)
        return hash
        
    def _local_model_hash(self, model_path: Union[str,Path])->str:
        sha = hashlib.sha256()
        path = Path(model_path)
        
        hashpath = path / "checksum.sha256"
        if hashpath.exists() and path.stat().st_mtime <= hashpath.stat().st_mtime:
            with open(hashpath) as f:
                hash = f.read()
            return hash
        
        print(f'  | computing hash of model {path.name}')
        for file in list(path.rglob("*.ckpt")) \
            + list(path.rglob("*.safetensors")) \
            + list(path.rglob("*.pth")):
            with open(file, "rb") as f:
                while chunk := f.read(self.sha_chunksize):
                    sha.update(chunk)
        hash = sha.hexdigest()
        with open(hashpath, "w") as f:
            f.write(hash)
        return hash
    
    def _hf_commit_hash(self, repo_id: str, revision: str='main')->str:
        api = HfApi()
        info = api.list_repo_refs(
            repo_id=repo_id,
            repo_type='model',
        )
        desired_revisions = [branch for branch in info.branches if branch.name==revision]
        if not desired_revisions:
            raise KeyError(f"Revision '{revision}' not found in {repo_id}")
        return desired_revisions[0].target_commit

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
