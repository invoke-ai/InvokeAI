"""
Manage a RAM cache of diffusion/transformer models for fast switching.
They are moved between GPU VRAM and CPU RAM as necessary. If the cache
grows larger than a preset maximum, then the least recently used
model will be cleared and (re)loaded from disk when next needed.

The cache returns context manager generators designed to load the
model into the GPU within the context, and unload outside the
context. Use like this:

   cache = ModelCache(max_models_cached=6)
   with cache.get_model('runwayml/stable-diffusion-1-5') as SD1,
          cache.get_model('stabilityai/stable-diffusion-2') as SD2:
       do_something_in_GPU(SD1,SD2)


"""

import contextlib
import gc
import hashlib
import warnings
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Sequence, Union, Tuple, types

import torch
import safetensors.torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, SchedulerMixin, UNet2DConditionModel
from diffusers import logging as diffusers_logging
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from huggingface_hub import HfApi
from picklescan.scanner import scan_file_path
from pydantic import BaseModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from transformers import logging as transformers_logging

import invokeai.backend.util.logging as logger
from ..globals import global_cache_dir
from ..stable_diffusion import StableDiffusionGeneratorPipeline

MAX_MODELS = 4

# This is the mapping from the stable diffusion submodel dict key to the class
class SDModelType(Enum):
    diffusion_pipeline=StableDiffusionGeneratorPipeline # whole thing
    diffusers=StableDiffusionGeneratorPipeline          # same thing
    vae=AutoencoderKL                                   # diffusers parts
    text_encoder=CLIPTextModel
    tokenizer=CLIPTokenizer
    unet=UNet2DConditionModel
    scheduler=SchedulerMixin
    safety_checker=StableDiffusionSafetyChecker
    feature_extractor=CLIPFeatureExtractor
    # These are all loaded as dicts of tensors
    lora=dict
    textual_inversion=dict
    ckpt=dict

class ModelStatus(Enum):
    unknown='unknown'
    not_loaded='not loaded'
    in_ram='cached'
    in_vram='in gpu'
    active='locked in gpu'
        
# The list of model classes we know how to fetch, for typechecking
ModelClass = Union[tuple([x.value for x in SDModelType])]
DiffusionClasses = (StableDiffusionGeneratorPipeline, AutoencoderKL, SchedulerMixin, UNet2DConditionModel)

# Legacy information needed to load a legacy checkpoint file
class LegacyInfo(BaseModel):
    config_file: Path
    vae_file: Path = None

class UnsafeModelException(Exception):
    "Raised when a legacy model file fails the picklescan test"
    pass

class UnscannableModelException(Exception):
    "Raised when picklescan is unable to scan a legacy model file"
    pass

class ModelLocker(object):
    "Forward declaration"
    pass

class ModelCache(object):
    def __init__(
            self,
            max_models: int=MAX_MODELS,
            execution_device: torch.device=torch.device('cuda'),
            storage_device: torch.device=torch.device('cpu'),
            precision: torch.dtype=torch.float16,
            sequential_offload: bool=False,
            lazy_offloading: bool=True,
            sha_chunksize: int = 16777216,
            logger: types.ModuleType = logger
    ):
        '''
        :param max_models: Maximum number of models to cache in CPU RAM [4]
        :param execution_device: Torch device to load active model into [torch.device('cuda')]
        :param storage_device: Torch device to save inactive model in [torch.device('cpu')]
        :param precision: Precision for loaded models [torch.float16]
        :param lazy_offloading: Keep model in VRAM until another model needs to be loaded
        :param sequential_offload: Conserve VRAM by loading and unloading each stage of the pipeline sequentially
        :param sha_chunksize: Chunksize to use when calculating sha256 model hash
        '''
        self.models: dict = dict()
        self.stack: Sequence = list()
        self.lazy_offloading = lazy_offloading
        self.sequential_offload: bool=sequential_offload
        self.precision: torch.dtype=precision
        self.max_models: int=max_models
        self.execution_device: torch.device=execution_device
        self.storage_device: torch.device=storage_device
        self.sha_chunksize=sha_chunksize
        self.logger = logger
        self.loaded_models: set = set()   # set of model keys loaded in GPU
        self.locked_models: Counter = Counter()   # set of model keys locked in GPU

    def get_model(
            self,
            repo_id_or_path: Union[str,Path],
            model_type: SDModelType=SDModelType.diffusion_pipeline,
            subfolder: Path=None,
            submodel: SDModelType=None,
            revision: str=None,
            legacy_info: LegacyInfo=None,
            attach_model_part: Tuple[SDModelType, str] = (None,None),
            gpu_load: bool=True,
            )->ModelLocker:  # ?? what does it return
        '''
        Load and return a HuggingFace model wrapped in a context manager generator, with RAM caching.
        Use like this:

              cache = ModelCache()
              with cache.get_model('stabilityai/stable-diffusion-2') as SD2:
                   do_something_with_the_model(SD2)

        You can fetch an individual part of a diffusers model by passing the submodel
        argument:

              vae_context = cache.get_model(
                                        'stabilityai/sd-stable-diffusion-2', 
                                         submodel=SDModelType.vae
                                         )

        Vice versa, you can load and attach an external submodel to a diffusers model 
        before returning it by passing the attach_submodel argument. This only works with
        diffusers models:

              pipeline_context = cache.get_model(
                                      'runwayml/stable-diffusion-v1-5',
                                      attach_model_part=(SDModelType.vae,'stabilityai/sd-vae-ft-mse')
                                      )

        The model will be locked into GPU VRAM for the duration of the context.
        :param repo_id_or_path: either the HuggingFace repo_id or a Path to a local model
        :param subfolder: name of a subfolder in which the model can be found, e.g. "vae"
        :param submodel: an SDModelType enum indicating the model part to return, e.g. SDModelType.vae
        :param attach_model_part: load and attach a diffusers model component. Pass a tuple of format (SDModelType,repo_id)
        :param revision: model revision
        :param model_class: class of model to return
        :param gpu_load: load the model into GPU [default True]
        :param legacy_info: a LegacyInfo object containing additional info needed to load a legacy ckpt
        '''
        key = self._model_key( # internal unique identifier for the model
            repo_id_or_path,
            model_type.value,
            revision,
            subfolder
        ) 
        if key in self.models: # cached - move to bottom of stack
            with contextlib.suppress(ValueError):
                self.stack.remove(key)
                self.stack.append(key)
            model = self.models[key]
        else:  # not cached -load
            self._make_cache_room()
            model = self._load_model_from_storage(
                repo_id_or_path=repo_id_or_path,
                model_class=model_type.value,
                subfolder=subfolder,
                revision=revision,
                legacy_info=legacy_info,
            )
            if model_type==SDModelType.diffusion_pipeline and attach_model_part[0]:
                self.attach_part(model,*attach_model_part)
            self.stack.append(key)          # add to LRU cache
            self.models[key]=model          # keep copy of model in dict
            
        if submodel:
            model = getattr(model, submodel.name)

        return self.ModelLocker(self, key, model, gpu_load)

    def uncache_model(self, key: str):
        '''Remove corresponding model from the cache'''
        if key is not None and key in self.models:
            with contextlib.suppress(ValueError):
                del self.models[key]
                del self.locked_models[key]
                self.stack.remove(key)
                self.loaded_models.remove(key)

    class ModelLocker(object):
        def __init__(self, cache, key, model, gpu_load):
            self.gpu_load = gpu_load
            self.cache = cache
            self.key = key
            # This will keep a copy of the model in RAM until the locker
            # is garbage collected. Needs testing!
            self.model = model

        def __enter__(self)->ModelClass:
            cache = self.cache
            key = self.key
            model = self.model
            if self.gpu_load and hasattr(model,'to'):
                cache.loaded_models.add(key)
                cache.locked_models[key] += 1
                if cache.lazy_offloading:
                   cache._offload_unlocked_models()
                cache.logger.debug(f'Loading {key} into {cache.execution_device}')
                model.to(cache.execution_device)  # move into GPU
                cache._print_cuda_stats()
            else:
                # in the event that the caller wants the model in RAM, we
                # move it into CPU if it is in GPU and not locked
                if hasattr(model,'to') and (key in cache.loaded_models
                                            and cache.locked_models[key] == 0):
                    model.to(cache.storage_device)
                    cache.loaded_models.remove(key)
            return model

        def __exit__(self, type, value, traceback):
            key = self.key
            cache = self.cache
            cache.locked_models[key] -= 1
            if not cache.lazy_offloading:
                cache._offload_unlocked_models()
                cache._print_cuda_stats()

    def attach_part(self,
                     diffusers_model: StableDiffusionPipeline,
                     part_type: SDModelType,
                     part_id: str
                     ):
        '''
        Attach a diffusers model part to a diffusers model. This can be
        used to replace the VAE, tokenizer, textencoder, unet, etc.
        :param diffuser_model: The diffusers model to attach the part to.
        :param part_type: An SD ModelType indicating the part
        :param part_id: A HF repo_id for the part
        '''
        part_key = part_type.name
        part_class = part_type.value
        part = self._load_diffusers_from_storage(
            part_id,
            model_class=part_class,
        )
        part.to(diffusers_model.device)
        setattr(diffusers_model,part_key,part)
        self.logger.debug(f'Attached {part_key} {part_id}')

    def status(self,
               repo_id_or_path: Union[str,Path],
               model_type: SDModelType=SDModelType.diffusion_pipeline,
               revision: str=None,
               subfolder: Path=None,
               )->ModelStatus:
        key = self._model_key(
            repo_id_or_path,
            model_type.value,
            revision,
            subfolder)
        if key not in self.models:
            return ModelStatus.not_loaded
        if key in self.loaded_models:
            if self.locked_models[key] > 0:
                return ModelStatus.active
            else:
                return ModelStatus.in_vram
        else:
            return ModelStatus.in_ram

    def model_hash(self,
                   repo_id_or_path: Union[str,Path],
                   revision: str="main")->str:
        '''
        Given the HF repo id or path to a model on disk, returns a unique
        hash. Works for legacy checkpoint files, HF models on disk, and HF repo IDs
        :param repo_id_or_path: repo_id string or Path to model file/directory on disk.
        :param revision: optional revision string (if fetching a HF repo_id)
        '''
        revision = revision or "main"
        if self.is_legacy_ckpt(repo_id_or_path):
            return self._legacy_model_hash(repo_id_or_path)
        elif Path(repo_id_or_path).is_dir():
            return self._local_model_hash(repo_id_or_path)
        else:
            return self._hf_commit_hash(repo_id_or_path,revision)

    def cache_size(self)->int:
        "Return the current number of models cached."
        return len(self.models)

    @classmethod
    def is_legacy_ckpt(cls, repo_id_or_path: Union[str,Path])->bool:
        '''
        Return true if the indicated path is a legacy checkpoint
        :param repo_id_or_path: either the HuggingFace repo_id or a Path to a local model
        '''
        path = Path(repo_id_or_path)
        return path.suffix in [".ckpt",".safetensors",".pt"]

    @classmethod
    def scan_model(cls, model_name, checkpoint):
        """
        Apply picklescanner to the indicated checkpoint and issue a warning
        and option to exit if an infected file is identified.
        """
        # scan model
        logger.debug(f"Scanning Model: {model_name}")
        scan_result = scan_file_path(checkpoint)
        if scan_result.infected_files != 0:
            if scan_result.infected_files == 1:
                raise UnsafeModelException("The legacy model you are trying to load may contain malware. Aborting.")
            else:
                raise UnscannableModelException("InvokeAI was unable to scan the legacy model you requested. Aborting")
        else:
            logger.debug("Model scanned ok")
            
    @staticmethod
    def _model_key(path,model_class,revision,subfolder)->str:
        return ':'.join([str(path),model_class.__name__,str(revision or ''),str(subfolder or '')])

    def _has_cuda(self)->bool:
        return self.execution_device.type == 'cuda'

    def _print_cuda_stats(self):
        vram = "%4.2fG" % (torch.cuda.memory_allocated() / 1e9)
        loaded_models = len(self.loaded_models)
        locked_models = len([x for x in self.locked_models if self.locked_models[x]>0])
        logger.debug(f"Current VRAM usage: {vram}; locked_models/loaded_models = {locked_models}/{loaded_models}")

    def _make_cache_room(self):
        models_in_ram = len(self.models)
        while models_in_ram >= self.max_models:
            if least_recently_used_key := self.stack.pop(0):
                logger.debug(f'Maximum cache size reached: cache_size={models_in_ram}; unloading model {least_recently_used_key}')
                del self.models[least_recently_used_key]
            models_in_ram = len(self.models)
        gc.collect()

    def _offload_unlocked_models(self):
        to_offload = set()
        for key in self.loaded_models:
            if key not in self.locked_models or self.locked_models[key] == 0:
                self.logger.debug(f'Offloading {key} from {self.execution_device} into {self.storage_device}')
                to_offload.add(key)
        for key in to_offload:
            self.models[key].to(self.storage_device)
            self.loaded_models.remove(key)

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
            model.enable_offload_submodels(self.execution_device)
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
        self.logger.info(f'Loading model {repo_id_or_path}')
        revisions = [revision] if revision \
            else ['fp16','main'] if self.precision==torch.float16 \
                 else ['main']
        extra_args = {'torch_dtype': self.precision,
                      'safety_checker': None}\
                      if model_class in DiffusionClasses\
                         else {}
        
        for rev in revisions:
            try:
                model =  model_class.from_pretrained(
                    repo_id_or_path,
                    revision=rev,
                    subfolder=subfolder or '.',
                    cache_dir=global_cache_dir('hub'),
                    **extra_args,
                )
                self.logger.debug(f'Found revision {rev}')
                break
            except OSError:
                pass
        return model

    def _load_ckpt_from_storage(self,
                                ckpt_path: Union[str,Path],
                                legacy_info:LegacyInfo)->StableDiffusionGeneratorPipeline:
        '''
        Load a legacy checkpoint, convert it, and return a StableDiffusionGeneratorPipeline.
        :param ckpt_path: string or Path pointing to the weights file (.ckpt or .safetensors)
        :param legacy_info: LegacyInfo object containing paths to legacy config file and alternate vae if required
        '''
        if legacy_info is None or legacy_info.config_file is None:
            if Path(ckpt_path).suffix == '.safetensors':
                return safetensors.torch.load_file(ckpt_path)
            else:
                return torch.load(ckpt_path)
        else:
            # deferred loading to avoid circular import errors
            from .convert_ckpt_to_diffusers import load_pipeline_from_original_stable_diffusion_ckpt
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
        assert path.is_file(),f"File {checkpoint_path} not found"

        hashpath = path.parent / f"{path.name}.sha256"
        if hashpath.exists() and path.stat().st_mtime <= hashpath.stat().st_mtime:
            with open(hashpath) as f:
                hash = f.read()
            return hash
        
        logger.debug(f'computing hash of model {path.name}')
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
        
        logger.debug(f'computing hash of model {path.name}')
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
