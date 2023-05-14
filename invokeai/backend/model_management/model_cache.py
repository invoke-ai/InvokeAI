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
from typing import Dict, Sequence, Union, Tuple, types, Optional

import torch
import safetensors.torch
    
from diffusers import DiffusionPipeline, StableDiffusionPipeline, AutoencoderKL, SchedulerMixin, UNet2DConditionModel, ConfigMixin
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

# Maximum size of the cache, in gigs
# Default is roughly enough to hold three fp16 diffusers models in RAM simultaneously
DEFAULT_MAX_CACHE_SIZE = 6.0

# actual size of a gig
GIG = 1073741824 

# This is the mapping from the stable diffusion submodel dict key to the class
class LoraType(dict):
    pass
class TIType(dict):
    pass

class SDModelType(str, Enum):
    Diffusers="diffusers"          # whole pipeline
    Vae="vae"                      # diffusers parts
    TextEncoder="text_encoder"
    Tokenizer="tokenizer"
    UNet="unet"
    Scheduler="scheduler"
    SafetyChecker="safety_checker"
    FeatureExtractor="feature_extractor"
    # These are all loaded as dicts of tensors, and we
    # distinguish them by class
    Lora="lora"
    TextualInversion="textual_inversion"

# TODO:
class EmptyScheduler(SchedulerMixin, ConfigMixin):
    pass

MODEL_CLASSES = {
    SDModelType.Diffusers: StableDiffusionGeneratorPipeline,
    SDModelType.Vae: AutoencoderKL,
    SDModelType.TextEncoder: CLIPTextModel, # TODO: t5
    SDModelType.Tokenizer: CLIPTokenizer, # TODO: t5
    SDModelType.UNet: UNet2DConditionModel,
    SDModelType.Scheduler: EmptyScheduler,
    SDModelType.SafetyChecker: StableDiffusionSafetyChecker,
    SDModelType.FeatureExtractor: CLIPFeatureExtractor,

    SDModelType.Lora: LoraType,
    SDModelType.TextualInversion: TIType,
}

class ModelStatus(Enum):
    unknown='unknown'
    not_loaded='not loaded'
    in_ram='cached'
    in_vram='in gpu'
    active='locked in gpu'

# This is used to guesstimate the size of a model before we load it.
# After loading, we will know it exactly.
# Sizes are in Gigs, estimated for float16; double for float32
SIZE_GUESSTIMATE = {
    SDModelType.Diffusers: 2.2,
    SDModelType.Vae: 0.35,
    SDModelType.TextEncoder: 0.5,
    SDModelType.Tokenizer: 0.001,
    SDModelType.UNet: 3.4,
    SDModelType.Scheduler: 0.001,
    SDModelType.SafetyChecker: 1.2,
    SDModelType.FeatureExtractor: 0.001,
    SDModelType.Lora: 0.1,
    SDModelType.TextualInversion: 0.001,
}
        
# The list of model classes we know how to fetch, for typechecking
ModelClass = Union[tuple([x for x in MODEL_CLASSES.values()])]
DiffusionClasses = (StableDiffusionGeneratorPipeline, AutoencoderKL, EmptyScheduler, UNet2DConditionModel, CLIPTextModel)

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
        max_cache_size: float=DEFAULT_MAX_CACHE_SIZE,
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
        self.current_cache_size: int=0
        self.max_cache_size: int=max_cache_size
        self.execution_device: torch.device=execution_device
        self.storage_device: torch.device=storage_device
        self.sha_chunksize=sha_chunksize
        self.logger = logger
        self.loaded_models: set = set()   # set of model keys loaded in GPU
        self.locked_models: Counter = Counter()   # set of model keys locked in GPU
        self.model_sizes: Dict[str,int] = dict()

    def get_model(
        self,
        repo_id_or_path: Union[str, Path],
        model_type: SDModelType = SDModelType.Diffusers,
        subfolder: Path = None,
        submodel: SDModelType = None,
        revision: str = None,
        attach_model_part: Tuple[SDModelType, str] = (None, None),
        gpu_load: bool = True,
    ) -> ModelLocker:  # ?? what does it return
        '''
        Load and return a HuggingFace model wrapped in a context manager generator, with RAM caching.
        Use like this:

              cache = ModelCache()
              with cache.get_model('stabilityai/stable-diffusion-2') as model:
                   do_something_with_the_model(model)

        While in context, model will be locked into GPU. If you want to do something
        with the model while it is in RAM, just use the context's `model` attribute:

              context = cache.get_model('stabilityai/stable-diffusion-2')
              context.model.device
              # device(type='cpu')

              with context as model:
                 model.device
              # device(type='cuda')

        You can fetch an individual part of a diffusers model by passing the submodel
        argument:

              vae_context = cache.get_model(
                                        'stabilityai/sd-stable-diffusion-2', 
                                         submodel=SDModelType.Vae
                                         )

        This is equivalent to:

              vae_context = cache.get_model(
                                        'stabilityai/sd-stable-diffusion-2', 
                                        model_type = SDModelType.Vae,
                                        subfolder='vae'
                                         )

        Vice versa, you can load and attach an external submodel to a diffusers model 
        before returning it by passing the attach_submodel argument. This only works with
        diffusers models:

              pipeline_context = cache.get_model(
                                      'runwayml/stable-diffusion-v1-5',
                                      attach_model_part=(SDModelType.Vae,'stabilityai/sd-vae-ft-mse')
                                      )

        The model will be locked into GPU VRAM for the duration of the context.
        :param repo_id_or_path: either the HuggingFace repo_id or a Path to a local model
        :param model_type: An SDModelType enum indicating the type of the (parent) model
        :param subfolder: name of a subfolder in which the model can be found, e.g. "vae"
        :param submodel: an SDModelType enum indicating the model part to return, e.g. SDModelType.Vae
        :param attach_model_part: load and attach a diffusers model component. Pass a tuple of format (SDModelType,repo_id)
        :param revision: model revision
        :param gpu_load: load the model into GPU [default True]
        '''
        key = self._model_key( # internal unique identifier for the model
            repo_id_or_path,
            revision,
            subfolder,
            model_type,
        )

        # optimization: if caller is asking to load a submodel of a diffusers pipeline, then
        # check whether it is already cached in RAM and return it instead of loading from disk again
        if subfolder and not submodel:
            possible_parent_key = self._model_key(
                repo_id_or_path,
                revision,
                None,
                SDModelType.Diffusers
            )
            if possible_parent_key in self.models:
                key = possible_parent_key
                submodel = model_type

        # Look for the model in the cache RAM
        if key in self.models: # cached - move to bottom of stack (most recently used)
            with contextlib.suppress(ValueError):
                self.stack.remove(key)
                self.stack.append(key)
            model = self.models[key]

        else:  # not cached -load
            self.logger.info(f'Loading model {repo_id_or_path}, type {model_type}')

            # this will remove older cached models until
            # there is sufficient room to load the requested model
            self._make_cache_room(key, model_type)

            # clean memory to make MemoryUsage() more accurate
            gc.collect()
            model = self._load_model_from_storage(
                repo_id_or_path=repo_id_or_path,
                model_type=model_type,
                subfolder=subfolder,
                revision=revision,
            )
            
            if mem_used := self.calc_model_size(model):
                logger.debug(f'CPU RAM used for load: {(mem_used/GIG):.2f} GB')
                self.model_sizes[key] = mem_used      # remember size of this model for cache cleansing
                self.current_cache_size += mem_used   # increment size of the cache
            
            # this is a bit of legacy work needed to support the old-style "load this diffuser with custom VAE"
            if model_type == SDModelType.Diffusers and attach_model_part[0]:
                self.attach_part(model, *attach_model_part)
                
            self.stack.append(key)          # add to LRU cache
            self.models[key] = model          # keep copy of model in dict
            
        if submodel:
            model = getattr(model, submodel)

        return self.ModelLocker(self, key, model, gpu_load)

    def uncache_model(self, key: str):
        '''Remove corresponding model from the cache'''
        if key is not None and key in self.models:
            self.models.pop(key, None)
            self.locked_models.pop(key, None)
            self.loaded_models.discard(key)
            with contextlib.suppress(ValueError):
                self.stack.remove(key)

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
            
            # NOTE that the model has to have the to() method in order for this
            # code to move it into GPU!
            if self.gpu_load and hasattr(model,'to'):
                cache.loaded_models.add(key)
                cache.locked_models[key] += 1
                
                if cache.lazy_offloading:
                   cache._offload_unlocked_models()
                   
                if  model.device != cache.execution_device:
                    cache.logger.debug(f'Moving {key} into {cache.execution_device}')
                    with VRAMUsage() as mem:
                        model.to(cache.execution_device)  # move into GPU
                    cache.logger.debug(f'GPU VRAM used for load: {(mem.vram_used/GIG):.2f} GB')
                    cache.model_sizes[key] = mem.vram_used   # more accurate size
                    
                cache.logger.debug(f'Locking {key} in {cache.execution_device}')                
                cache._print_cuda_stats()
                
            else:
                # in the event that the caller wants the model in RAM, we
                # move it into CPU if it is in GPU and not locked
                if hasattr(model, 'to') and (key in cache.loaded_models
                                            and cache.locked_models[key] == 0):
                    model.to(cache.storage_device)
                    cache.loaded_models.remove(key)
            return model

        def __exit__(self, type, value, traceback):
            if not hasattr(self.model, 'to'):
                return

            key = self.key
            cache = self.cache
            cache.locked_models[key] -= 1
            if not cache.lazy_offloading:
                cache._offload_unlocked_models()
                cache._print_cuda_stats()

    def attach_part(
        self,
        diffusers_model: StableDiffusionPipeline,
        part_type: SDModelType,
        part_id: str,
    ):
        '''
        Attach a diffusers model part to a diffusers model. This can be
        used to replace the VAE, tokenizer, textencoder, unet, etc.
        :param diffuser_model: The diffusers model to attach the part to.
        :param part_type: An SD ModelType indicating the part
        :param part_id: A HF repo_id for the part
        '''
        part = self._load_diffusers_from_storage(
            part_id,
            model_class=MODEL_CLASSES[part_type],
        )
        part.to(diffusers_model.device)
        setattr(diffusers_model, part_type, part)
        self.logger.debug(f'Attached {part_type} {part_id}')

    def status(
        self,
        repo_id_or_path: Union[str, Path],
        model_type: SDModelType = SDModelType.Diffusers,
        revision: str = None,
        subfolder: Path = None,
    ) -> ModelStatus:
        key = self._model_key(
            repo_id_or_path,
            revision,
            subfolder,
            model_type,
        )
        if key not in self.models:
            return ModelStatus.not_loaded
        if key in self.loaded_models:
            if self.locked_models[key] > 0:
                return ModelStatus.active
            else:
                return ModelStatus.in_vram
        else:
            return ModelStatus.in_ram

    def model_hash(
        self,
        repo_id_or_path: Union[str, Path],
        revision: str = "main",
    ) -> str:
        '''
        Given the HF repo id or path to a model on disk, returns a unique
        hash. Works for legacy checkpoint files, HF models on disk, and HF repo IDs
        :param repo_id_or_path: repo_id string or Path to model file/directory on disk.
        :param revision: optional revision string (if fetching a HF repo_id)
        '''
        revision = revision or "main"
        if Path(repo_id_or_path).is_dir():
            return self._local_model_hash(repo_id_or_path)
        else:
            return self._hf_commit_hash(repo_id_or_path,revision)

    def cache_size(self) -> float:
        "Return the current size of the cache, in GB"
        return self.current_cache_size / GIG

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
    def _model_key(path, revision, subfolder, model_class) -> str:
        return ':'.join([
            str(path),
            str(revision or ''),
            str(subfolder or ''),
            model_class,
        ])

    def _has_cuda(self) -> bool:
        return self.execution_device.type == 'cuda'

    def _print_cuda_stats(self):
        vram = "%4.2fG" % (torch.cuda.memory_allocated() / GIG)
        ram = "%4.2fG" % (self.current_cache_size / GIG)
        loaded_models = len(self.loaded_models)
        locked_models = len([x for x in self.locked_models if self.locked_models[x]>0])
        logger.debug(f"Current VRAM/RAM usage: {vram}/{ram}; locked_models/loaded_models = {locked_models}/{loaded_models}")

    def _make_cache_room(self, key, model_type):
        # calculate how much memory this model will require
        multiplier = 2 if self.precision==torch.float32 else 1
        bytes_needed = int(self.model_sizes.get(key,0) or SIZE_GUESSTIMATE.get(model_type,0.5)*GIG*multiplier)
        maximum_size = self.max_cache_size * GIG  # stored in GB, convert to bytes
        current_size = self.current_cache_size

        adjective = 'guesstimated' if key not in self.model_sizes else 'known from previous load'
        logger.debug(f'{(bytes_needed/GIG):.2f} GB needed to load this model ({adjective})')
        while current_size+bytes_needed > maximum_size:
            if least_recently_used_key := self.stack.pop(0):
                model_size = self.model_sizes.get(least_recently_used_key,0) 
                logger.debug(f'Max cache size exceeded: cache_size={(current_size/GIG):.2f} GB, need an additional {(bytes_needed/GIG):.2f} GB')
                logger.debug(f'Unloading model {least_recently_used_key} to free {(model_size/GIG):.2f} GB')
                self.uncache_model(least_recently_used_key)
                current_size -= model_size
        self.current_cache_size = current_size
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
        repo_id_or_path: Union[str, Path],
        subfolder: Optional[Path] = None,
        revision: Optional[str] = None,
        model_type: SDModelType = SDModelType.Diffusers,
    ) -> ModelClass:
        '''
        Load and return a HuggingFace model.
        :param repo_id_or_path: either the HuggingFace repo_id or a Path to a local model
        :param subfolder: name of a subfolder in which the model can be found, e.g. "vae"
        :param revision: model revision
        :param model_type: type of model to return, defaults to SDModelType.Diffusers
        '''
        # silence transformer and diffuser warnings
        with SilenceWarnings():
            if model_type==SDModelType.Lora:
                model = self._load_lora_from_storage(repo_id_or_path)
            elif model_type==SDModelType.TextualInversion:
                model = self._load_ti_from_storage(repo_id_or_path)
            else:
                model = self._load_diffusers_from_storage(
                    repo_id_or_path,
                    subfolder,
                    revision,
                    model_type,
                )
                if self.sequential_offload and isinstance(model, StableDiffusionGeneratorPipeline):
                    model.enable_offload_submodels(self.execution_device)
        return model

    def _load_diffusers_from_storage(
        self,
        repo_id_or_path: Union[str, Path],
        subfolder: Optional[Path] = None,
        revision: Optional[str] = None,
        model_type: ModelClass = StableDiffusionGeneratorPipeline,
    ) -> ModelClass:
        '''
        Load and return a HuggingFace model using from_pretrained().
        :param repo_id_or_path: either the HuggingFace repo_id or a Path to a local model
        :param subfolder: name of a subfolder in which the model can be found, e.g. "vae"
        :param revision: model revision
        :param model_class: class of model to return, defaults to StableDiffusionGeneratorPIpeline
        '''

        model_class = MODEL_CLASSES[model_type]

        if revision is not None:
            revisions = [revision]
        elif self.precision == torch.float16:
            revisions = ['fp16', 'main']
        else:
            revisions = ['main']

        extra_args = dict()
        if model_class in DiffusionClasses:
            extra_args.update(
                torch_dtype=self.precision,
            )
        if model_class == StableDiffusionGeneratorPipeline:
            extra_args.update(
                safety_checker=None,
            )

        for rev in revisions:
            try:
                model = model_class.from_pretrained(
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

    def _load_lora_from_storage(self, lora_path: Path) -> LoraType:
        assert False, "_load_lora_from_storage() is not yet implemented"

    def _load_ti_from_storage(self, lora_path: Path) -> TIType:
        assert False, "_load_ti_from_storage() is not yet implemented"

    def _legacy_model_hash(self, checkpoint_path: Union[str, Path]) -> str:
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
        
    def _local_model_hash(self, model_path: Union[str, Path]) -> str:
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
    
    def _hf_commit_hash(self, repo_id: str, revision: str='main') -> str:
        api = HfApi()
        info = api.list_repo_refs(
            repo_id=repo_id,
            repo_type='model',
        )
        desired_revisions = [branch for branch in info.branches if branch.name==revision]
        if not desired_revisions:
            raise KeyError(f"Revision '{revision}' not found in {repo_id}")
        return desired_revisions[0].target_commit

    @staticmethod
    def calc_model_size(model) -> int:
        if isinstance(model,DiffusionPipeline):
            return ModelCache._calc_pipeline(model)
        elif isinstance(model,torch.nn.Module):
            return ModelCache._calc_model(model)
        else:
            return None

    @staticmethod
    def _calc_pipeline(pipeline) -> int:
        res = 0
        for submodel_key in pipeline.components.keys():
            submodel = getattr(pipeline, submodel_key)
            if submodel is not None and isinstance(submodel, torch.nn.Module):
                res += ModelCache._calc_model(submodel)
        return res
    
    @staticmethod
    def _calc_model(model) -> int:
        mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
        mem = mem_params + mem_bufs # in bytes
        return mem

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

class VRAMUsage(object):
    def __init__(self):
        self.vram = None
        self.vram_used = 0
        
    def __enter__(self):
        self.vram = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *args):
        self.vram_used = torch.cuda.memory_allocated() - self.vram
