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
import os
import sys
import hashlib
import warnings
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import Dict, Sequence, Union, Tuple, types, Optional, List, Type, Any

import torch
import safetensors.torch
    
from diffusers import DiffusionPipeline, SchedulerMixin, ConfigMixin
from diffusers import logging as diffusers_logging
from huggingface_hub import HfApi, scan_cache_dir
from picklescan.scanner import scan_file_path
from pydantic import BaseModel
from transformers import logging as transformers_logging

import invokeai.backend.util.logging as logger
from ..globals import global_cache_dir


def get_model_path(repo_id_or_path: str):
    if os.path.exists(repo_id_or_path):
        return repo_id_or_path

    cache = scan_cache_dir(global_cache_dir("hub"))
    for repo in cache.repos:
        if repo.repo_id != repo_id_or_path:
            continue
        for rev in repo.revisions:
            if "main" in rev.refs:
                return rev.snapshot_path
    raise Exception(f"{repo_id_or_path} - not found")

def calc_model_size_by_fs(
    repo_id_or_path: str,
    subfolder: Optional[str] = None,
    variant: Optional[str] = None
):
    model_path = get_model_path(repo_id_or_path)
    if subfolder is not None:
        model_path = os.path.join(model_path, subfolder)

    all_files = os.listdir(model_path)
    all_files = [f for f in all_files if os.path.isfile(os.path.join(model_path, f))]

    fp16_files = set([f for f in all_files if ".fp16." in f or ".fp16-" in f])
    bit8_files = set([f for f in all_files if ".8bit." in f or ".8bit-" in f])
    other_files = set(all_files) - fp16_files - bit8_files

    if variant is None:
        files = other_files
    elif variant == "fp16":
        files = fp16_files
    elif variant == "8bit":
        files = bit8_files
    else:
        raise NotImplementedError(f"Unknown variant: {variant}")

    # try read from index if exists
    index_postfix = ".index.json"
    if variant is not None:
        index_postfix = f".index.{variant}.json"

    for file in files:
        if not file.endswith(index_postfix):
            continue
        try:
            with open(os.path.join(model_path, index_file), "r") as f:
                index_data = json.loads(f.read())
            return int(index_data["metadata"]["total_size"])
        except:
            pass

    # calculate files size if there is no index file
    formats = [
        (".safetensors",), # safetensors
        (".bin",), # torch
        (".onnx", ".pb"), # onnx
        (".msgpack",), # flax
        (".ckpt",), # tf
        (".h5",), # tf2
    ]

    for file_format in formats:
        model_files = [f for f in files if f.endswith(file_format)]
        if len(model_files) == 0:
            continue

        model_size = 0
        for model_file in model_files:
            file_stats = os.stat(os.path.join(model_path, model_file))
            model_size += file_stats.st_size
        return model_size
    
    #raise NotImplementedError(f"Unknown model structure! Files: {all_files}")
    return 0 # scheduler/feature_extractor/tokenizer - models without loading to gpu


def calc_model_size_by_data(model) -> int:
    if isinstance(model, DiffusionPipeline):
        return _calc_pipeline_by_data(model)
    elif isinstance(model, torch.nn.Module):
        return _calc_model_by_data(model)
    else:
        return 0


def _calc_pipeline_by_data(pipeline) -> int:
    res = 0
    for submodel_key in pipeline.components.keys():
        submodel = getattr(pipeline, submodel_key)
        if submodel is not None and isinstance(submodel, torch.nn.Module):
            res += _calc_model_by_data(submodel)
    return res
    

def _calc_model_by_data(model) -> int:
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    return mem




class SDModelType(str, Enum):
    Diffusers = "diffusers"
    Classifier = "classifier"
    UNet = "unet"
    TextEncoder = "text_encoder"
    Tokenizer = "tokenizer"
    Vae = "vae"
    Scheduler = "scheduler"


class ModelInfoBase:
    #model_path: str
    #model_type: SDModelType

    def __init__(self, repo_id_or_path: str, model_type: SDModelType):
        self.repo_id_or_path = repo_id_or_path # TODO: or use allways path?
        self.model_path = get_model_path(repo_id_or_path)
        self.model_type = model_type

    def _definition_to_type(self, subtypes: List[str]) -> Type:
        if len(subtypes) < 2:
            raise Exception("Invalid subfolder definition!")
        if subtypes[0] in ["diffusers", "transformers"]:
            res_type = sys.modules[subtypes[0]]
            subtypes = subtypes[1:]

        else:
            res_type = sys.modules["diffusers"]
            res_type = getattr(res_type, "pipelines")


        for subtype in subtypes:
            res_type = getattr(res_type, subtype)
        return res_type


class DiffusersModelInfo(ModelInfoBase):
    #child_types: Dict[str, Type]
    #child_sizes: Dict[str, int]

    def __init__(self, repo_id_or_path: str, model_type: SDModelType):
        assert model_type == SDModelType.Diffusers
        super().__init__(repo_id_or_path, model_type)

        self.child_types: Dict[str, Type] = dict()
        self.child_sizes: Dict[str, int] = dict()

        try:
            config_data = DiffusionPipeline.load_config(repo_id_or_path)
            #config_data = json.loads(os.path.join(self.model_path, "model_index.json"))
        except:
            raise Exception("Invalid diffusers model! (model_index.json not found or invalid)")

        config_data.pop("_ignore_files", None)

        # retrieve all folder_names that contain relevant files
        child_components = [k for k, v in config_data.items() if isinstance(v, list)]

        for child_name in child_components:
            child_type = self._definition_to_type(config_data[child_name])
            self.child_types[child_name] = child_type
            self.child_sizes[child_name] = calc_model_size_by_fs(repo_id_or_path, subfolder=child_name)


    def get_size(self, child_type: Optional[SDModelType] = None):
        if child_type is None:
            return sum(self.child_sizes.values())
        else:
            return self.child_sizes[child_type]


    def get_model(
        self,
        child_type: Optional[SDModelType] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        # return pipeline in different function to pass more arguments
        if child_type is None:
            raise Exception("Child model type can't be null on diffusers model")
        if child_type not in self.child_types:
            return None # TODO: or raise

        # TODO:
        for variant in ["fp16", "main", None]:
            try:
                model = self.child_types[child_type].from_pretrained(
                    self.repo_id_or_path,
                    subfolder=child_type.value,
                    cache_dir=global_cache_dir('hub'),
                    torch_dtype=torch_dtype,
                    variant=variant,
                )
                break
            except Exception as e:
                print("====ERR LOAD====")
                print(f"{variant}: {e}")

        # calc more accurate size
        self.child_sizes[child_type] = calc_model_size_by_data(model)
        return model


    def get_pipeline(self, **kwrags):
        return DiffusionPipeline.from_pretrained(
            self.repo_id_or_path,
            **kwargs,
        )


class EmptyConfigLoader(ConfigMixin):

    @classmethod
    def load_config(cls, *args, **kwargs):
        cls.config_name = kwargs.pop("config_name")
        return super().load_config(*args, **kwargs)


class ClassifierModelInfo(ModelInfoBase):
    #child_types: Dict[str, Type]
    #child_sizes: Dict[str, int]

    def __init__(self, repo_id_or_path: str, model_type: SDModelType):
        assert model_type == SDModelType.Classifier
        super().__init__(repo_id_or_path, model_type)

        self.child_types: Dict[str, Type] = dict()
        self.child_sizes: Dict[str, int] = dict()

        try:
            main_config = EmptyConfigLoader.load_config(repo_id_or_path, config_name="config.json")
            #main_config = json.loads(os.path.join(self.model_path, "config.json"))
        except:
            raise Exception("Invalid classifier model! (config.json not found or invalid)")

        self._load_tokenizer(main_config)
        self._load_text_encoder(main_config)
        self._load_feature_extractor(main_config)


    def _load_tokenizer(self, main_config: dict):
        try:
            tokenizer_config = EmptyConfigLoader.load_config(repo_id_or_path, config_name="tokenizer_config.json")
            #tokenizer_config = json.loads(os.path.join(self.model_path, "tokenizer_config.json"))
        except:
            raise Exception("Invalid classifier model! (Failed to load tokenizer_config.json)")

        if "tokenizer_class" in tokenizer_config:
            tokenizer_class_name = tokenizer_config["tokenizer_class"]
        elif "model_type" in main_config:
            tokenizer_class_name = transformers.models.auto.tokenization_auto.TOKENIZER_MAPPING_NAMES[main_config["model_type"]]
        else:
            raise Exception("Invalid classifier model! (Failed to detect tokenizer type)")

        self.child_types[SDModelType.Tokenizer] = self._definition_to_type(["transformers", tokenizer_class_name])
        self.child_sizes[SDModelType.Tokenizer] = 0


    def _load_text_encoder(self, main_config: dict):
        if "architectures" in main_config and len(main_config["architectures"]) > 0:
            text_encoder_class_name = main_config["architectures"][0]
        elif "model_type" in main_config:
            text_encoder_class_name = transformers.models.auto.modeling_auto.MODEL_FOR_PRETRAINING_MAPPING_NAMES[main_config["model_type"]]
        else:
            raise Exception("Invalid classifier model! (Failed to detect text_encoder type)")

        self.child_types[SDModelType.TextEncoder] = self._definition_to_type(["transformers", text_encoder_class_name])
        self.child_sizes[SDModelType.TextEncoder] = calc_model_size_by_fs(repo_id_or_path)


    def _load_feature_extractor(self, main_config: dict):
        self.child_sizes[SDModelType.FeatureExtractor] = 0
        try:
            feature_extractor_config = EmptyConfigLoader.load_config(repo_id_or_path, config_name="preprocessor_config.json")
        except:
            return # feature extractor not passed with t5

        try:
            feature_extractor_class_name = feature_extractor_config["feature_extractor_type"]
            self.child_types[SDModelType.FeatureExtractor] = self._definition_to_type(["transformers", feature_extractor_class_name])
        except:
            raise Exception("Invalid classifier model! (Unknown feature_extrator type)")


    def get_size(self, child_type: Optional[SDModelType] = None):
        if child_type is None:
            return sum(self.child_sizes.values())
        else:
            return self.child_sizes[child_type]


    def get_model(
        self,
        child_type: Optional[SDModelType] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        if child_type is None:
            raise Exception("Child model type can't be null on classififer model")
        if child_type not in self.child_types:
            return None # TODO: or raise
        
        model = self.child_types[child_type].from_pretrained(
            self.repo_id_or_path,
            subfolder=child_type.value,
            cache_dir=global_cache_dir('hub'),
            torch_dtype=torch_dtype,
        )
        # calc more accurate size
        self.child_sizes[child_type] = calc_model_size_by_data(model)
        return model



class VaeModelInfo(ModelInfoBase):
    #vae_class: Type
    #model_size: int

    def __init__(self, repo_id_or_path: str, model_type: SDModelType):
        assert model_type == SDModelType.Vae
        super().__init__(repo_id_or_path, model_type)

        try:
            config = EmptyConfigLoader.load_config(repo_id_or_path, config_name="config.json")
            #config = json.loads(os.path.join(self.model_path, "config.json"))
        except:
            raise Exception("Invalid vae model! (config.json not found or invalid)")

        try:
            vae_class_name = config.get("_class_name", "AutoencoderKL")
            self.vae_class = self._definition_to_type(["diffusers", vae_class_name])
            self.model_size = calc_model_size_by_fs(repo_id_or_path)
        except:
            raise Exception("Invalid vae model! (Unkown vae type)")

    def get_size(self, child_type: Optional[SDModelType] = None):
        if child_type is not None:
            raise Exception("There is no child models in vae model")
        return self.model_size

    def get_model(
        self,
        child_type: Optional[SDModelType] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        if child_type is not None:
            raise Exception("There is no child models in vae model")

        model = self.vae_type.from_pretrained(
            self.repo_id_or_path,
            cache_dir=global_cache_dir('hub'),
            torch_dtype=torch_dtype,
        )
        # calc more accurate size
        self.model_size = calc_model_size_by_data(model)
        return model


MODEL_TYPES = {
    SDModelType.Diffusers: DiffusersModelInfo,
    SDModelType.Classifier: ClassifierModelInfo,
    SDModelType.Vae: VaeModelInfo,
}


# Maximum size of the cache, in gigs
# Default is roughly enough to hold three fp16 diffusers models in RAM simultaneously
DEFAULT_MAX_CACHE_SIZE = 6.0

# actual size of a gig
GIG = 1073741824

# TODO:
class EmptyScheduler(SchedulerMixin, ConfigMixin):
    pass

class ModelLocker(object):
    "Forward declaration"
    pass

class ModelCache(object):
    "Forward declaration"
    pass

class _CacheRecord:
    model: Any
    size: int
    _locks: int
    _cache: ModelCache

    def __init__(self, cache, model: Any, size: int):
        self._cache = cache
        self.model = model
        self.size = size
        self._locks = 0

    def lock(self):
        self._locks += 1

    def unlock(self):
        self._locks -= 1
        assert self._locks >= 0

    @property
    def locked(self):
        return self._locks > 0

    @property
    def loaded(self):
        if self.model is not None and hasattr(self.model, "device"):
            return self.model.device != self._cache.storage_device
        else:
            return False
    

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
        max_cache_size = 9999
        execution_device = torch.device('cuda')

        self.models: Dict[str, _CacheRecord] = dict()
        self.model_infos: Dict[str, ModelInfoBase] = dict()
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

    def get_key(
        self,
        model_path: str,
        model_type: SDModelType,
        revision: Optional[str] = None,
        submodel_type: Optional[SDModelType] = None,
    ):
        revision = revision or "main"

        key = f"{model_path}:{model_type}:{revision}"
        if submodel_type:
            key += f":{submodel_type}"
        return key

    #def get_model(
    #    self,
    #    repo_id_or_path: Union[str, Path],
    #    model_type: SDModelType = SDModelType.Diffusers,
    #    subfolder: Path = None,
    #    submodel: SDModelType = None,
    #    revision: str = None,
    #    attach_model_part: Tuple[SDModelType, str] = (None, None),
    #    gpu_load: bool = True,
    #) -> ModelLocker:  # ?? what does it return
    def _get_model_info(
        self,
        model_path: str,
        model_type: SDModelType,
        revision: str,
    ):
        model_info_key = self.get_key(
            model_path=model_path,
            model_type=model_type,
            revision=revision,
            submodel_type=None,
        )

        if model_info_key not in self.model_infos:
            if model_type not in MODEL_TYPES:
                raise Exception(f"Unknown/unsupported model type: {model_type}")

            self.model_infos[model_info_key] = MODEL_TYPES[model_type](
                model_path,
                model_type,
            )

        return self.model_infos[model_info_key]

    def get_model(
        self,
        repo_id_or_path: Union[str, Path],
        model_type: SDModelType = SDModelType.Diffusers,
        submodel: SDModelType = None,
        revision: str = None,
        gpu_load: bool = True,
    ) -> Any:

        model_path = get_model_path(repo_id_or_path)
        model_info = self._get_model_info(
            model_path=model_path,
            model_type=model_type,
            revision=revision,
        )

        key = self.get_key(
            model_path=model_path,
            model_type=model_type,
            revision=revision,
            submodel_type=submodel,
        )

        if key not in self.models:
            self.logger.info(f'Loading model {repo_id_or_path}, type {model_type}:{submodel}')

            # this will remove older cached models until
            # there is sufficient room to load the requested model
            self._make_cache_room(model_info.get_size(submodel))

            # clean memory to make MemoryUsage() more accurate
            gc.collect()
            model_obj = model_info.get_model(submodel, torch_dtype=self.precision)
            if mem_used := model_info.get_size(submodel):
                logger.debug(f'CPU RAM used for load: {(mem_used/GIG):.2f} GB')
                self.current_cache_size += mem_used   # increment size of the cache

            self.models[key] = _CacheRecord(self, model_obj, mem_used)

        with suppress(Exception):
            self.stack.remove(key)
        self.stack.append(key)

        return self.ModelLocker(self, key, self.models[key].model, gpu_load)

    def uncache_model(self, key: str):
        '''Remove corresponding model from the cache'''
        self.models.pop(key, None)
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

        def __enter__(self) -> Any:
            if not hasattr(self.model, 'to'):
                return self.model

            cache_entry = self.cache.models[self.key]

            # NOTE that the model has to have the to() method in order for this
            # code to move it into GPU!
            if self.gpu_load:
                cache_entry.lock()
                
                if self.cache.lazy_offloading:
                   self.cache._offload_unlocked_models()
                   
                if self.model.device != self.cache.execution_device:
                    self.cache.logger.debug(f'Moving {self.key} into {self.cache.execution_device}')
                    with VRAMUsage() as mem:
                        self.model.to(self.cache.execution_device)  # move into GPU
                    self.cache.logger.debug(f'GPU VRAM used for load: {(mem.vram_used/GIG):.2f} GB')
                    
                self.cache.logger.debug(f'Locking {self.key} in {self.cache.execution_device}')                
                self.cache._print_cuda_stats()
            
            # TODO: not fully understand
            # in the event that the caller wants the model in RAM, we
            # move it into CPU if it is in GPU and not locked
            elif cache_entry.loaded and not cache_entry.locked:
                self.model.to(self.cache.storage_device)

            return self.model

        def __exit__(self, type, value, traceback):
            if not hasattr(self.model, 'to'):
                return

            self.cache.models[self.key].unlock()
            if not self.cache.lazy_offloading:
                self.cache._offload_unlocked_models()
                self.cache._print_cuda_stats()

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

    def _has_cuda(self) -> bool:
        return self.execution_device.type == 'cuda'

    def _print_cuda_stats(self):
        vram = "%4.2fG" % (torch.cuda.memory_allocated() / GIG)
        ram = "%4.2fG" % (self.current_cache_size / GIG)

        loaded_models = 0
        locked_models = 0
        for cache_entry in self.models.values():
            if cache_entry.loaded:
                loaded_models += 1
            if cache_entry.locked:
                locked_models += 1

        logger.debug(f"Current VRAM/RAM usage: {vram}/{ram}; locked_models/loaded_models = {locked_models}/{loaded_models}")

    def _make_cache_room(self, model_size):
        # calculate how much memory this model will require
        #multiplier = 2 if self.precision==torch.float32 else 1
        bytes_needed = model_size
        maximum_size = self.max_cache_size * GIG  # stored in GB, convert to bytes
        current_size = self.current_cache_size

        if current_size + bytes_needed > maximum_size:
            logger.debug(f'Max cache size exceeded: {(current_size/GIG):.2f}/{self.max_cache_size:.2f} GB, need an additional {(bytes_needed/GIG):.2f} GB')

        pos = 0
        while current_size + bytes_needed > maximum_size and current_size > 0 and len(self.stack) > 0 and pos < len(self.stack):
            model_key = self.stack[pos]
            cache_entry = self.models[model_key]
            if not cache_entry.locked:
                logger.debug(f'Unloading model {model_key} to free {(model_size/GIG):.2f} GB (-{(cache_entry.size/GIG):.2f} GB)')
                self.uncache_model(model_key) # del self.stack[pos]
                current_size -= cache_entry.size
            else:
                pos += 1

        self.current_cache_size = current_size
        gc.collect()

    def _offload_unlocked_models(self):
        for key in self.models.keys():
            cache_entry = self.models[key]
            if not cache_entry.locked and cache_entry.loaded:
                self.logger.debug(f'Offloading {key} from {self.execution_device} into {self.storage_device}')
                cache_entry.model.to(self.storage_device)
        
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
