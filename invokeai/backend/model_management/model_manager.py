"""This module manages the InvokeAI `models.yaml` file, mapping
symbolic diffusers model names to the paths and repo_ids used
by the underlying `from_pretrained()` call.

For fetching models, use manager.get_model('symbolic name'). This will
return a ModelInfo object that contains the following attributes:
   
   * context -- a context manager Generator that loads and locks the 
                model into GPU VRAM and returns the model for use. 
                See below for usage.
   * name -- symbolic name of the model
   * type -- SubModelType of the model
   * hash -- unique hash for the model
   * location -- path or repo_id of the model
   * revision -- revision of the model if coming from a repo id,
                e.g. 'fp16'
   * precision -- torch precision of the model

Typical usage:

   from invokeai.backend import ModelManager

   manager = ModelManager(
                 config='./configs/models.yaml',
                 max_cache_size=8
             ) # gigabytes

   model_info = manager.get_model('stable-diffusion-1.5', SubModelType.Diffusers)
   with model_info.context as my_model:
      my_model.latents_from_embeddings(...)

The manager uses the underlying ModelCache class to keep
frequently-used models in RAM and move them into GPU as needed for
generation operations. The optional `max_cache_size` argument
indicates the maximum size the cache can grow to, in gigabytes. The
underlying ModelCache object can be accessed using the manager's "cache"
attribute.

Because the model manager can return multiple different types of
models, you may wish to add additional type checking on the class
of model returned. To do this, provide the option `model_type`
parameter:

    model_info = manager.get_model(
                      'clip-tokenizer',
                       model_type=SubModelType.Tokenizer
                      )

This will raise an InvalidModelError if the format defined in the
config file doesn't match the requested model type.

MODELS.YAML

The general format of a models.yaml section is:

 type-of-model/name-of-model:
     path: /path/to/local/file/or/directory
     description: a description
     format: folder|ckpt|safetensors|pt
     base: SD-1|SD-2
     subfolder: subfolder-name

The type of model is given in the stanza key, and is one of
{diffusers, ckpt, vae, text_encoder, tokenizer, unet, scheduler,
safety_checker, feature_extractor, lora, textual_inversion,
controlnet}, and correspond to items in the SubModelType enum defined
in model_cache.py

The format indicates whether the model is organized as a folder with
model subdirectories, or is contained in a single checkpoint or
safetensors file.

One, but not both, of repo_id and path are provided. repo_id is the
HuggingFace repository ID of the model, and path points to the file or
directory on disk.

If subfolder is provided, then the model exists in a subdirectory of
the main model. These are usually named after the model type, such as
"unet".

This example summarizes the two ways of getting a non-diffuser model:

 text_encoder/clip-test-1:
   format: folder
   path: /path/to/folder
   description: Returns standalone CLIPTextModel

 text_encoder/clip-test-2:
   format: folder
   repo_id: /path/to/folder
   subfolder: text_encoder
   description: Returns the text_encoder in the subfolder of the diffusers model (just the encoder in RAM)

SUBMODELS:

It is also possible to fetch an isolated submodel from a diffusers
model. Use the `submodel` parameter to select which part:

 vae = manager.get_model('stable-diffusion-1.5',submodel=SubModelType.Vae)
 with vae.context as my_vae:
    print(type(my_vae))
    # "AutoencoderKL"

DIRECTORY_SCANNING:

Loras, textual_inversion and controlnet models are usually not listed
explicitly in models.yaml, but are added to the in-memory data
structure at initialization time by scanning the models directory. The
in-memory data structure can be resynchronized by calling
`manager.scan_models_directory`.

DISAMBIGUATION:

You may wish to use the same name for a related family of models. To
do this, disambiguate the stanza key with the model and and format
separated by "/". Example:

 tokenizer/clip-large:
   format: tokenizer
   path: /path/to/folder
   description: Returns standalone tokenizer

 text_encoder/clip-large:
   format: text_encoder
   path: /path/to/folder
   description: Returns standalone text encoder

You can now use the `model_type` argument to indicate which model you
want:

 tokenizer = mgr.get('clip-large',model_type=SubModelType.Tokenizer)
 encoder = mgr.get('clip-large',model_type=SubModelType.TextEncoder)

OTHER FUNCTIONS:

Other methods provided by ModelManager support importing, editing,
converting and deleting models.

IMPORTANT CHANGES AND LIMITATIONS SINCE 2.3:

1. Only local paths are supported. Repo_ids are no longer accepted. This
simplifies the logic.

2. VAEs can't be swapped in and out at load time. They must be baked
into the model when downloaded or converted.

"""
from __future__ import annotations

import os
import hashlib
import textwrap
from dataclasses import dataclass
from packaging import version
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, types
from shutil import rmtree

import torch
from huggingface_hub import scan_cache_dir
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from pydantic import BaseModel

import invokeai.backend.util.logging as logger
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.util import CUDA_DEVICE, download_with_resume
from .model_cache import ModelCache, ModelLocker
from .models import BaseModelType, ModelType, SubModelType, ModelError, MODEL_CLASSES

# We are only starting to number the config file with release 3.
# The config file version doesn't have to start at release version, but it will help
# reduce confusion.
CONFIG_FILE_VERSION='3.0.0'

@dataclass
class ModelInfo():
    context: ModelLocker
    name: str
    base_model: BaseModelType
    type: ModelType
    hash: str
    location: Union[Path, str]
    precision: torch.dtype
    revision: str = None
    _cache: ModelCache = None

    def __enter__(self):
        return self.context.__enter__()

    def __exit__(self,*args, **kwargs):
        self.context.__exit__(*args, **kwargs)

class InvalidModelError(Exception):
    "Raised when an invalid model is requested"
    pass

MAX_CACHE_SIZE = 6.0  # GB


# layout of the models directory:
# models
# ├── sd-1
# │   ├── controlnet
# │   ├── lora
# │   ├── pipeline
# │   └── textual_inversion
# ├── sd-2
# │   ├── controlnet
# │   ├── lora
# │   ├── pipeline
# │   └── textual_inversion
# └── core
#     ├── face_reconstruction
#     │   ├── codeformer
#     │   └── gfpgan
#     ├── sd-conversion
#     │   ├── clip-vit-large-patch14 - tokenizer, text_encoder subdirs
#     │   ├── stable-diffusion-2 - tokenizer, text_encoder subdirs
#     │   └── stable-diffusion-safety-checker
#     └── upscaling
#         └─── esrgan



class ConfigMeta(BaseModel):
    version: str

class ModelManager(object):
    """
    High-level interface to model management.
    """

    logger: types.ModuleType = logger

    def __init__(
        self,
        config: Union[Path, DictConfig, str],
        device_type: torch.device = CUDA_DEVICE,
        precision: torch.dtype = torch.float16,
        max_cache_size=MAX_CACHE_SIZE,
        sequential_offload=False,
        logger: types.ModuleType = logger,
    ):
        """
        Initialize with the path to the models.yaml config file. 
        Optional parameters are the torch device type, precision, max_models,
        and sequential_offload boolean. Note that the default device
        type and precision are set up for a CUDA system running at half precision.
        """

        self.config_path = None
        if isinstance(config, (str, Path)):
            self.config_path = Path(config)
            config = OmegaConf.load(self.config_path)

        elif not isinstance(config, DictConfig):
            raise ValueError('config argument must be an OmegaConf object, a Path or a string')

        self.config_meta = ConfigMeta(**config.pop("__metadata__"))
        # TODO: metadata not found
        # TODO: version check

        self.models = dict()
        for model_key, model_config in config.items():
            model_name, base_model, model_type = self.parse_key(model_key)
            model_class = MODEL_CLASSES[base_model][model_type]
            # alias for config file
            model_config["model_format"] = model_config.pop("format")
            self.models[model_key] = model_class.create_config(**model_config)

        # check config version number and update on disk/RAM if necessary
        self.globals = InvokeAIAppConfig.get_config()
        self.logger = logger
        self.cache = ModelCache(
            max_cache_size=max_cache_size,
            execution_device = device_type,
            precision = precision,
            sequential_offload = sequential_offload,
            logger = logger,
        )
        self.cache_keys = dict()

        # add controlnet, lora and textual_inversion models from disk
        self.scan_models_directory()

    def model_exists(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
    ) -> bool:
        """
        Given a model name, returns True if it is a valid
        identifier.
        """
        model_key = self.create_key(model_name, base_model, model_type)
        return model_key in self.models

    @classmethod
    def create_key(
        cls,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
    ) -> str:
        return f"{base_model}/{model_type}/{model_name}"

    def parse_key(self, model_key: str) -> Tuple[str, BaseModelType, ModelType]:
        base_model_str, model_type_str, model_name = model_key.split('/', 2)
        try:
            model_type = ModelType(model_type_str)
        except:
            raise Exception(f"Unknown model type: {model_type_str}")

        try:
            base_model = BaseModelType(base_model_str)
        except:
            raise Exception(f"Unknown base model: {base_model_str}")

        return (model_name, base_model, model_type)

    def get_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel_type: Optional[SubModelType] = None
    ):
        """Given a model named identified in models.yaml, return
        an ModelInfo object describing it.
        :param model_name: symbolic name of the model in models.yaml
        :param model_type: ModelType enum indicating the type of model to return
        :param submode_typel: an ModelType enum indicating the portion of 
               the model to retrieve (e.g. ModelType.Vae)

        If not provided, the model_type will be read from the `format` field
        of the corresponding stanza. If provided, the model_type will be used
        to disambiguate stanzas in the configuration file. The default is to
        assume a diffusers pipeline. The behavior is illustrated here:

        [models.yaml]
        diffusers/test1:
           repo_id: foo/bar
           description: Typical diffusers pipeline

        lora/test1:
           repo_id: /tmp/loras/test1.safetensors
           description: Typical lora file

        test1_pipeline = mgr.get_model('test1')
        # returns a StableDiffusionGeneratorPipeline

        test1_vae1 = mgr.get_model('test1', submodel=ModelType.Vae)
        # returns the VAE part of a diffusers model as an AutoencoderKL

        test1_vae2 = mgr.get_model('test1', model_type=ModelType.Diffusers, submodel=ModelType.Vae)
        # does the same thing as the previous  statement. Note that model_type
        # is for the parent model, and submodel is for the part

        test1_lora = mgr.get_model('test1', model_type=ModelType.Lora)
        # returns a LoRA embed (as a 'dict' of tensors)

        test1_encoder = mgr.get_modelI('test1', model_type=ModelType.TextEncoder)
        # raises an InvalidModelError

        """
        model_class = MODEL_CLASSES[base_model][model_type]
        model_key = self.create_key(model_name, base_model, model_type)

        # if model not found try to find it (maybe file just pasted)
        if model_key not in self.models:
            # TODO: find by mask or try rescan?
            path_mask = f"/models/{base_model}/{model_type}/{model_name}*"
            if False: # model_path = next(find_by_mask(path_mask)):
                model_path = None # TODO:
                model_config = model_class.probe_config(model_path)
                self.models[model_key] = model_config
            else:
                raise Exception(f"Model not found - {model_key}")

        # if it known model check that target path exists (if manualy deleted)
        else:
            # logic repeated twice(in rescan too) any way to optimize?
            if not os.path.exists(self.models[model_key].path):
                if model_class.save_to_config:
                    self.models[model_key].error = ModelError.NotFound
                    raise Exception(f"Files for model \"{model_key}\" not found")

                else:
                    self.models.pop(model_key, None)
                    raise Exception(f"Model not found - {model_key}")

            # reset model errors?



        model_config = self.models[model_key]

        # /models/{base_model}/{model_type}/{name}.ckpt or .safentesors
        # /models/{base_model}/{model_type}/{name}/
        model_path = model_config.path

        # vae/movq override
        # TODO: 
        if submodel_type is not None and hasattr(model_config, submodel_type):
            override_path = getattr(model_config, submodel_type)
            if override_path:
                model_path = override_path
                model_type = submodel_type
                submodel_type = None
                model_class = MODEL_CLASSES[base_model][model_type]

        # TODO: path
        # TODO: is it accurate to use path as id
        dst_convert_path = self.globals.models_dir / ".cache" / hashlib.md5(model_path.encode()).hexdigest()
        model_path = model_class.convert_if_required(
            base_model=base_model,
            model_path=model_path,
            output_path=dst_convert_path,
            config=model_config,
        )

        model_context = self.cache.get_model(
            model_path=model_path,
            model_class=model_class,
            base_model=base_model,
            model_type=model_type,
            submodel=submodel_type,
        )

        if model_key not in self.cache_keys:
            self.cache_keys[model_key] = set()
        self.cache_keys[model_key].add(model_context.key)

        model_hash = "<NO_HASH>" # TODO:
            
        return ModelInfo(
            context = model_context,
            name = model_name,
            base_model = base_model,
            type = submodel_type or model_type,
            hash = model_hash,
            location = model_path, # TODO:
            precision = self.cache.precision,
            _cache = self.cache,
        )

    def model_info(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
    ) -> dict:
        """
        Given a model name returns the OmegaConf (dict-like) object describing it.
        """
        model_key = self.create_key(model_name, base_model, model_type)
        if model_key in self.models:
            return self.models[model_key].dict(exclude_defaults=True)
        else:
            return None # TODO: None or empty dict on not found

    def model_names(self) -> List[Tuple[str, BaseModelType, ModelType]]:
        """
        Return a list of (str, BaseModelType, ModelType) corresponding to all models 
        known to the configuration.
        """
        return [(self.parse_key(x)) for x in self.models.keys()]

    def list_models(
        self,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None,
    ) -> list[dict]:
        """
        Return a list of models.

        Please use model_manager.models() to get all the model names,
        model_manager.model_info('model-name') to get the stanza for the model
        named 'model-name', and model_manager.config to get the full OmegaConf
        object derived from models.yaml
        """

        models = []
        for model_key in sorted(self.models, key=str.casefold):
            model_config = self.models[model_key]

            cur_model_name, cur_base_model, cur_model_type = self.parse_key(model_key)
            if base_model is not None and cur_base_model != base_model:
                continue
            if model_type is not None and cur_model_type != model_type:
                continue

            model_dict = dict(
                **model_config.dict(exclude_defaults=True),
                # OpenAPIModelInfoBase
                name=cur_model_name,
                base_model=cur_base_model,
                type=cur_model_type,
            )

            models.append(model_dict)

        return models

    def print_models(self) -> None:
        """
        Print a table of models, their descriptions
        """
        # TODO: redo
        for model_type, model_dict in self.list_models().items():
            for model_name, model_info in model_dict.items():
                line = f'{model_info["name"]:25s} {model_info["type"]:10s} {model_info["description"]}'
                print(line)

    # TODO: test when ui implemented
    def del_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
    ):
        """
        Delete the named model.
        """
        raise Exception("TODO: del_model") # TODO: redo
        model_key = self.create_key(model_name, base_model, model_type)
        model_cfg = self.models.pop(model_key, None)

        if model_cfg is None:
            self.logger.error(
                f"Unknown model {model_key}"
            )
            return

        # note: it not garantie to release memory(model can has other references)
        cache_ids = self.cache_keys.pop(model_key, [])
        for cache_id in cache_ids:
            self.cache.uncache_model(cache_id)

        # if model inside invoke models folder - delete files
        if model_cfg.path.startswith("models/") or model_cfg.path.startswith("models\\"):
            model_path = self.globals.root_dir / model_cfg.path
            if model_path.isdir():
                shutil.rmtree(str(model_path))
            else:
                model_path.unlink()

    # TODO: test when ui implemented
    def add_model(
        self,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        model_attributes: dict,
        clobber: bool = False,
    ) -> None:
        """
        Update the named model with a dictionary of attributes. Will fail with an
        assertion error if the name already exists. Pass clobber=True to overwrite.
        On a successful update, the config will be changed in memory and the
        method will return True. Will fail with an assertion error if provided
        attributes are incorrect or the model name is missing.
        """

        model_class = MODEL_CLASSES[base_model][model_type]
        model_config = model_class.create_config(**model_attributes)
        model_key = self.create_key(model_name, base_model, model_type)

        assert (
            clobber or model_key not in self.models
        ), f'attempt to overwrite existing model definition "{model_key}"'

        self.models[model_key] = model_config
            
        if clobber and model_key in self.cache_keys:
            # note: it not garantie to release memory(model can has other references)
            cache_ids = self.cache_keys.pop(model_key, [])
            for cache_id in cache_ids:
                self.cache.uncache_model(cache_id)

    def search_models(self, search_folder):
        self.logger.info(f"Finding Models In: {search_folder}")
        models_folder_ckpt = Path(search_folder).glob("**/*.ckpt")
        models_folder_safetensors = Path(search_folder).glob("**/*.safetensors")

        ckpt_files = [x for x in models_folder_ckpt if x.is_file()]
        safetensor_files = [x for x in models_folder_safetensors if x.is_file()]

        files = ckpt_files + safetensor_files

        found_models = []
        for file in files:
            location = str(file.resolve()).replace("\\", "/")
            if (
                "model.safetensors" not in location
                and "diffusion_pytorch_model.safetensors" not in location
            ):
                found_models.append({"name": file.stem, "location": location})

        return search_folder, found_models

    def commit(self, conf_file: Path=None) -> None:
        """
        Write current configuration out to the indicated file.
        """
        data_to_save = dict()
        data_to_save["__metadata__"] = self.config_meta.dict()

        for model_key, model_config in self.models.items():
            model_name, base_model, model_type = self.parse_key(model_key)
            model_class = MODEL_CLASSES[base_model][model_type]
            if model_class.save_to_config:
                # TODO: or exclude_unset better fits here?
                data_to_save[model_key] = model_config.dict(exclude_defaults=True, exclude={"error"})
                # alias for config file
                data_to_save[model_key]["format"] = data_to_save[model_key].pop("model_format")

        yaml_str = OmegaConf.to_yaml(data_to_save)
        config_file_path = conf_file or self.config_path
        assert config_file_path is not None,'no config file path to write to'
        config_file_path = self.globals.root_dir / config_file_path
        tmpfile = os.path.join(os.path.dirname(config_file_path), "new_config.tmp")
        with open(tmpfile, "w", encoding="utf-8") as outfile:
            outfile.write(self.preamble())
            outfile.write(yaml_str)
        os.replace(tmpfile, config_file_path)

    def preamble(self) -> str:
        """
        Returns the preamble for the config file.
        """
        return textwrap.dedent(
            """\
            # This file describes the alternative machine learning models
            # available to InvokeAI script.
            #
            # To add a new model, follow the examples below. Each
            # model requires a model config file, a weights file,
            # and the width and height of the images it
            # was trained on.
        """
        )

    def scan_models_directory(self):
        loaded_files = set()
        new_models_found = False

        for model_key, model_config in list(self.models.items()):
            model_name, base_model, model_type = self.parse_key(model_key)
            model_path = str(self.globals.root / model_config.path)
            if not os.path.exists(model_path):
                model_class = MODEL_CLASSES[base_model][model_type]
                if model_class.save_to_config:
                    model_config.error = ModelError.NotFound
                else:
                    self.models.pop(model_key, None)
            else:
                loaded_files.add(model_path)

        for base_model in BaseModelType:
            for model_type in ModelType:
                model_class = MODEL_CLASSES[base_model][model_type]
                models_dir = os.path.join(self.globals.models_path, base_model, model_type)

                if not os.path.exists(models_dir):
                    continue # TODO: or create all folders?
                
                for entry_name in os.listdir(models_dir):
                    model_path = os.path.join(models_dir, entry_name)
                    if model_path not in loaded_files: # TODO: check
                        model_name = Path(model_path).stem
                        model_key = self.create_key(model_name, base_model, model_type)

                        if model_key in self.models:
                            raise Exception(f"Model with key {model_key} added twice")

                        model_config: ModelConfigBase = model_class.probe_config(model_path)
                        self.models[model_key] = model_config
                        new_models_found = True

        if new_models_found:
            self.commit()
