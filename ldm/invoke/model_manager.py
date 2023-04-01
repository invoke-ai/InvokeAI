"""
Manage a cache of Stable Diffusion model files for fast switching.
They are moved between GPU and CPU as necessary. If CPU memory falls
below a preset minimum, the least recently used model will be
cleared and loaded from disk when next needed.
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import os
import re
import sys
import textwrap
import time
import warnings
from enum import Enum
from pathlib import Path
from shutil import move, rmtree
from typing import Any, Callable, Optional, Union, List

import safetensors
import safetensors.torch
import torch
import transformers
from diffusers import AutoencoderKL
from diffusers import logging as dlogging
from huggingface_hub import scan_cache_dir
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from picklescan.scanner import scan_file_path

from ldm.invoke.devices import CPU_DEVICE
from ldm.invoke.generator.diffusers_pipeline import StableDiffusionGeneratorPipeline
from ldm.invoke.globals import Globals, global_cache_dir
from ldm.util import ask_user, download_with_resume, instantiate_from_config, url_attachment_name


class SDLegacyType(Enum):
    V1 = 1
    V1_INPAINT = 2
    V2 = 3
    V2_e = 4
    V2_v = 5
    UNKNOWN = 99

DEFAULT_MAX_MODELS = 2

class ModelManager(object):
    def __init__(
        self,
        config: OmegaConf,
        device_type: torch.device = CPU_DEVICE,
        precision: str = "float16",
        max_loaded_models=DEFAULT_MAX_MODELS,
        sequential_offload=False,
    ):
        """
        Initialize with the path to the models.yaml config file,
        the torch device type, and precision. The optional
        min_avail_mem argument specifies how much unused system
        (CPU) memory to preserve. The cache of models in RAM will
        grow until this value is approached. Default is 2G.
        """
        # prevent nasty-looking CLIP log message
        transformers.logging.set_verbosity_error()
        self.config = config
        self.precision = precision
        self.device = torch.device(device_type)
        self.max_loaded_models = max_loaded_models
        self.models = {}
        self.stack = []  # this is an LRU FIFO
        self.current_model = None
        self.sequential_offload = sequential_offload

    def valid_model(self, model_name: str) -> bool:
        """
        Given a model name, returns True if it is a valid
        identifier.
        """
        return model_name in self.config

    def get_model(self, model_name: str):
        """
        Given a model named identified in models.yaml, return
        the model object. If in RAM will load into GPU VRAM.
        If on disk, will load from there.
        """
        if not self.valid_model(model_name):
            print(
                f'** "{model_name}" is not a known model name. Please check your models.yaml file'
            )
            return self.current_model

        if self.current_model != model_name:
            if model_name not in self.models:  # make room for a new one
                self._make_cache_room()
            self.offload_model(self.current_model)

        if model_name in self.models:
            requested_model = self.models[model_name]["model"]
            print(f">> Retrieving model {model_name} from system RAM cache")
            self.models[model_name]["model"] = self._model_from_cpu(requested_model)
        else:
            requested_model, width, height, hash = self._load_model(model_name)
            self.models[model_name] = {
                "model": requested_model,
                "width": width,
                "height": height,
                "hash": hash,
            }

        self.current_model = model_name
        self._push_newest_model(model_name)
        return self.models[model_name]
    
    def default_model(self) -> str | None:
        """
        Returns the name of the default model, or None
        if none is defined.
        """
        for model_name in self.config:
            if self.config[model_name].get("default"):
                return model_name
        return list(self.config.keys())[0]  # first one

    def set_default_model(self, model_name: str) -> None:
        """
        Set the default model. The change will not take
        effect until you call model_manager.commit()
        """
        assert model_name in self.model_names(), f"unknown model '{model_name}'"

        config = self.config
        for model in config:
            config[model].pop("default", None)
        config[model_name]["default"] = True

    def model_info(self, model_name: str) -> dict:
        """
        Given a model name returns the OmegaConf (dict-like) object describing it.
        """
        if model_name not in self.config:
            return None
        return self.config[model_name]

    def model_names(self) -> list[str]:
        """
        Return a list consisting of all the names of models defined in models.yaml
        """
        return list(self.config.keys())

    def is_legacy(self, model_name: str) -> bool:
        """
        Return true if this is a legacy (.ckpt) model
        """
        # if we are converting legacy files automatically, then
        # there are no legacy ckpts!
        info = self.model_info(model_name)
        if Globals.ckpt_convert or info.format=='diffusers' or self.is_v2_config(info.config):
            return False
        if "weights" in info and info["weights"].endswith((".ckpt", ".safetensors")):
            return True
        return False

    def list_models(self) -> dict:
        """
        Return a dict of models in the format:
        { model_name1: {'status': ('active'|'cached'|'not loaded'),
                        'description': description,
                        'format': ('ckpt'|'diffusers'|'vae'),
                       },
          model_name2: { etc }
        Please use model_manager.models() to get all the model names,
        model_manager.model_info('model-name') to get the stanza for the model
        named 'model-name', and model_manager.config to get the full OmegaConf
        object derived from models.yaml
        """
        models = {}
        for name in sorted(self.config, key=str.casefold):
            stanza = self.config[name]

            # don't include VAEs in listing (legacy style)
            if "config" in stanza and "/VAE/" in stanza["config"]:
                continue

            models[name] = dict()
            format = stanza.get("format", "ckpt")  # Determine Format

            # Common Attribs
            description = stanza.get("description", None)
            if self.current_model == name:
                status = "active"
            elif name in self.models:
                status = "cached"
            else:
                status = "not loaded"
            models[name].update(
                description=description,
                format=format,
                status=status,
            )

            # Checkpoint Config Parse
            if format == "ckpt":
                models[name].update(
                    config=str(stanza.get("config", None)),
                    weights=str(stanza.get("weights", None)),
                    vae=str(stanza.get("vae", None)),
                    width=str(stanza.get("width", 512)),
                    height=str(stanza.get("height", 512)),
                )

            # Diffusers Config Parse
            if vae := stanza.get("vae", None):
                if isinstance(vae, DictConfig):
                    vae = dict(
                        repo_id=str(vae.get("repo_id", None)),
                        path=str(vae.get("path", None)),
                        subfolder=str(vae.get("subfolder", None)),
                    )

            if format == "diffusers":
                models[name].update(
                    vae=vae,
                    repo_id=str(stanza.get("repo_id", None)),
                    path=str(stanza.get("path", None)),
                )

        return models

    def print_models(self) -> None:
        """
        Print a table of models, their descriptions, and load status
        """
        models = self.list_models()
        for name in models:
            if models[name]["format"] == "vae":
                continue
            line = f'{name:25s} {models[name]["status"]:>10s}  {models[name]["format"]:10s} {models[name]["description"]}'
            if models[name]["status"] == "active":
                line = f"\033[1m{line}\033[0m"
            print(line)

    def del_model(self, model_name: str, delete_files: bool = False) -> None:
        """
        Delete the named model.
        """
        omega = self.config
        if model_name not in omega:
            print(f"** Unknown model {model_name}")
            return
        # save these for use in deletion later
        conf = omega[model_name]
        repo_id = conf.get("repo_id", None)
        path = self._abs_path(conf.get("path", None))
        weights = self._abs_path(conf.get("weights", None))

        del omega[model_name]
        if model_name in self.stack:
            self.stack.remove(model_name)
        if delete_files:
            if weights:
                print(f"** Deleting file {weights}")
                Path(weights).unlink(missing_ok=True)
            elif path:
                print(f"** Deleting directory {path}")
                rmtree(path, ignore_errors=True)
            elif repo_id:
                print(f"** Deleting the cached model directory for {repo_id}")
                self._delete_model_from_cache(repo_id)

    def add_model(
        self, model_name: str, model_attributes: dict, clobber: bool = False
    ) -> None:
        """
        Update the named model with a dictionary of attributes. Will fail with an
        assertion error if the name already exists. Pass clobber=True to overwrite.
        On a successful update, the config will be changed in memory and the
        method will return True. Will fail with an assertion error if provided
        attributes are incorrect or the model name is missing.
        """
        omega = self.config
        assert "format" in model_attributes, 'missing required field "format"'
        if model_attributes["format"] == "diffusers":
            assert (
                "description" in model_attributes
            ), 'required field "description" is missing'
            assert (
                "path" in model_attributes or "repo_id" in model_attributes
            ), 'model must have either the "path" or "repo_id" fields defined'
        else:
            for field in ("description", "weights", "height", "width", "config"):
                assert field in model_attributes, f"required field {field} is missing"

        assert (
            clobber or model_name not in omega
        ), f'attempt to overwrite existing model definition "{model_name}"'

        omega[model_name] = model_attributes

        if "weights" in omega[model_name]:
            omega[model_name]["weights"].replace("\\", "/")

        if clobber:
            self._invalidate_cached_model(model_name)

    def _load_model(self, model_name: str):
        """Load and initialize the model from configuration variables passed at object creation time"""
        if model_name not in self.config:
            print(
                f'"{model_name}" is not a known model name. Please check your models.yaml file'
            )
            return

        mconfig = self.config[model_name]

        # for usage statistics
        if self._has_cuda():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        tic = time.time()

        # this does the work
        model_format = mconfig.get("format", "ckpt")
        if model_format == "ckpt":
            weights = mconfig.weights
            print(f">> Loading {model_name} from {weights}")
            model, width, height, model_hash = self._load_ckpt_model(
                model_name, mconfig
            )
        elif model_format == "diffusers":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model, width, height, model_hash = self._load_diffusers_model(mconfig)
        else:
            raise NotImplementedError(
                f"Unknown model format {model_name}: {model_format}"
            )

        # usage statistics
        toc = time.time()
        print(">> Model loaded in", "%4.2fs" % (toc - tic))
        if self._has_cuda():
            print(
                ">> Max VRAM used to load the model:",
                "%4.2fG" % (torch.cuda.max_memory_allocated() / 1e9),
                "\n>> Current VRAM usage:"
                "%4.2fG" % (torch.cuda.memory_allocated() / 1e9),
            )
        return model, width, height, model_hash

    def _load_ckpt_model(self, model_name, mconfig):
        config = mconfig.config
        weights = mconfig.weights
        vae = mconfig.get("vae")
        width = mconfig.width
        height = mconfig.height

        if not os.path.isabs(config):
            config = os.path.join(Globals.root, config)
        if not os.path.isabs(weights):
            weights = os.path.normpath(os.path.join(Globals.root, weights))

        # check whether this is a v2 file and force conversion
        convert = Globals.ckpt_convert or self.is_v2_config(config)

        if matching_config := self._scan_for_matching_file(Path(weights),suffixes=['.yaml']):
            print(f'   | Using external config file {matching_config}')
            config = matching_config

        # get the path to the custom vae, if any
        vae_path = None
        # first we use whatever is in the config file
        if vae:
            path = Path(vae if os.path.isabs(vae) else os.path.normpath(os.path.join(Globals.root, vae)))
            if path.exists():
                vae_path = path
        # then we look for a file with the same basename
        vae_path = vae_path or self._scan_for_matching_file(Path(weights))
            
        # if converting automatically to diffusers, then we do the conversion and return
        # a diffusers pipeline
        if convert:
            print(
                f">> Converting legacy checkpoint {model_name} into a diffusers model..."
            )
            from ldm.invoke.ckpt_to_diffuser import load_pipeline_from_original_stable_diffusion_ckpt

            try:
                if self.list_models()[self.current_model]['status'] == 'active':
                    self.offload_model(self.current_model)
            except Exception:
                pass
            
            if self._has_cuda():
                torch.cuda.empty_cache()
            pipeline = load_pipeline_from_original_stable_diffusion_ckpt(
                checkpoint_path=weights,
                original_config_file=config,
                vae_path=vae_path,
                return_generator_pipeline=True,
                precision=torch.float16
                if self.precision == "float16"
                else torch.float32,
            )
            if self.sequential_offload:
                pipeline.enable_offload_submodels(self.device)
            else:
                pipeline.to(self.device)

            return (
                pipeline,
                width,
                height,
                "NOHASH",
            )

        # for usage statistics
        if self._has_cuda():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # this does the work
        if not os.path.isabs(config):
            config = os.path.join(Globals.root, config)
        omega_config = OmegaConf.load(config)
        with open(weights, "rb") as f:
            weight_bytes = f.read()
        model_hash = self._cached_sha256(weights, weight_bytes)
        sd = None

        if weights.endswith(".ckpt"):
            self.scan_model(model_name, weights)
            sd = torch.load(io.BytesIO(weight_bytes), map_location="cpu")
        else:
            sd = safetensors.torch.load(weight_bytes)

        del weight_bytes
        # merged models from auto11 merge board are flat for some reason
        if "state_dict" in sd:
            sd = sd["state_dict"]

        print("   | Forcing garbage collection prior to loading new model")
        gc.collect()
        model = instantiate_from_config(omega_config.model)
        model.load_state_dict(sd, strict=False)

        if self.precision == "float16":
            print("   | Using faster float16 precision")
            model = model.to(torch.float16)
        else:
            print("   | Using more accurate float32 precision")

        # look and load a matching vae file. Code borrowed from AUTOMATIC1111 modules/sd_models.py
        if vae_path:
            print(f"   | Loading VAE weights from: {vae_path}")
            if vae_path.suffix in [".ckpt", ".pt"]:
                self.scan_model(vae_path.name, vae_path)
                vae_ckpt = torch.load(vae_path, map_location="cpu")
            else:
                vae_ckpt = safetensors.torch.load_file(vae_path)
            vae_dict = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss"}
            model.first_stage_model.load_state_dict(vae_dict, strict=False)
        else:
            print("   | Using VAE built into model.")

        model.to(self.device)
        # model.to doesn't change the cond_stage_model.device used to move the tokenizer output, so set it here
        model.cond_stage_model.device = self.device

        model.eval()

        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                module._orig_padding_mode = module.padding_mode
        return model, width, height, model_hash

    def _load_diffusers_model(self, mconfig):
        name_or_path = self.model_name_or_path(mconfig)
        using_fp16 = self.precision == "float16"

        print(f">> Loading diffusers model from {name_or_path}")
        if using_fp16:
            print("   | Using faster float16 precision")
        else:
            print("   | Using more accurate float32 precision")

        # TODO: scan weights maybe?
        pipeline_args: dict[str, Any] = dict(
            safety_checker=None, local_files_only=not Globals.internet_available
        )
        if "vae" in mconfig and mconfig["vae"] is not None:
            if vae := self._load_vae(mconfig["vae"]):
                pipeline_args.update(vae=vae)
        if not isinstance(name_or_path, Path):
            pipeline_args.update(cache_dir=global_cache_dir("hub"))
        if using_fp16:
            pipeline_args.update(torch_dtype=torch.float16)
            fp_args_list = [{"revision": "fp16"}, {}]
        else:
            fp_args_list = [{}]

        verbosity = dlogging.get_verbosity()
        dlogging.set_verbosity_error()

        pipeline = None
        for fp_args in fp_args_list:
            try:
                pipeline = StableDiffusionGeneratorPipeline.from_pretrained(
                    name_or_path,
                    **pipeline_args,
                    **fp_args,
                )
            except OSError as e:
                if str(e).startswith("fp16 is not a valid"):
                    pass
                else:
                    print(
                        f"** An unexpected error occurred while downloading the model: {e})"
                    )
            if pipeline:
                break

        dlogging.set_verbosity(verbosity)
        assert pipeline is not None, OSError(f'"{name_or_path}" could not be loaded')

        if self.sequential_offload:
            pipeline.enable_offload_submodels(self.device)
        else:
            pipeline.to(self.device)

        model_hash = self._diffuser_sha256(name_or_path)

        # square images???
        width = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
        height = width

        print(f"   | Default image dimensions = {width} x {height}")

        return pipeline, width, height, model_hash

    def is_v2_config(self, config: Path) -> bool:
        if not os.path.isabs(config):
            config = os.path.join(Globals.root, config)
        try:
            mconfig = OmegaConf.load(config)
            return (
                mconfig["model"]["params"]["unet_config"]["params"]["context_dim"] > 768
            )
        except:
            return False

    def model_name_or_path(self, model_name: Union[str, DictConfig]) -> str | Path:
        if isinstance(model_name, DictConfig) or isinstance(model_name, dict):
            mconfig = model_name
        elif model_name in self.config:
            mconfig = self.config[model_name]
        else:
            raise ValueError(
                f'"{model_name}" is not a known model name. Please check your models.yaml file'
            )

        if "path" in mconfig and mconfig["path"] is not None:
            path = Path(mconfig["path"])
            if not path.is_absolute():
                path = Path(Globals.root, path).resolve()
            return path
        elif "repo_id" in mconfig:
            return mconfig["repo_id"]
        else:
            raise ValueError("Model config must specify either repo_id or path.")

    def offload_model(self, model_name: str) -> None:
        """
        Offload the indicated model to CPU. Will call
        _make_cache_room() to free space if needed.
        """
        if model_name not in self.models:
            return

        print(f">> Offloading {model_name} to CPU")
        model = self.models[model_name]["model"]
        self.models[model_name]["model"] = self._model_to_cpu(model)

        gc.collect()
        if self._has_cuda():
            torch.cuda.empty_cache()

    @classmethod
    def scan_model(self, model_name, checkpoint):
        """
        Apply picklescanner to the indicated checkpoint and issue a warning
        and option to exit if an infected file is identified.
        """
        # scan model
        print(f"   | Scanning Model: {model_name}")
        scan_result = scan_file_path(checkpoint)
        if scan_result.infected_files != 0:
            if scan_result.infected_files == 1:
                print(f"\n### Issues Found In Model: {scan_result.issues_count}")
                print(
                    "### WARNING: The model you are trying to load seems to be infected."
                )
                print("### For your safety, InvokeAI will not load this model.")
                print("### Please use checkpoints from trusted sources.")
                print("### Exiting InvokeAI")
                sys.exit()
            else:
                print(
                    "\n### WARNING: InvokeAI was unable to scan the model you are using."
                )
                model_safe_check_fail = ask_user(
                    "Do you want to to continue loading the model?", ["y", "n"]
                )
                if model_safe_check_fail.lower() != "y":
                    print("### Exiting InvokeAI")
                    sys.exit()
        else:
            print("   | Model scanned ok")

    def import_diffuser_model(
        self,
        repo_or_path: Union[str, Path],
        model_name: str = None,
        description: str = None,
        vae: dict = None,
        commit_to_conf: Path = None,
    ) -> bool:
        """
        Attempts to install the indicated diffuser model and returns True if successful.

        "repo_or_path" can be either a repo-id or a path-like object corresponding to the
        top of a downloaded diffusers directory.

        You can optionally provide a model name and/or description. If not provided,
        then these will be derived from the repo name. If you provide a commit_to_conf
        path to the configuration file, then the new entry will be committed to the
        models.yaml file.
        """
        model_name = model_name or Path(repo_or_path).stem
        model_description = description or f"Imported diffusers model {model_name}"
        new_config = dict(
            description=model_description,
            vae=vae,
            format="diffusers",
        )
        if isinstance(repo_or_path, Path) and repo_or_path.exists():
            new_config.update(path=str(repo_or_path))
        else:
            new_config.update(repo_id=repo_or_path)

        self.add_model(model_name, new_config, True)
        if commit_to_conf:
            self.commit(commit_to_conf)
        return model_name

    def import_ckpt_model(
        self,
        weights: Union[str, Path],
        config: Union[str, Path] = "configs/stable-diffusion/v1-inference.yaml",
        vae: Union[str, Path] = None,
        model_name: str = None,
        model_description: str = None,
        commit_to_conf: Path = None,
    ) -> str:
        """
        Attempts to install the indicated ckpt file and returns True if successful.

        "weights" can be either a path-like object corresponding to a local .ckpt file
        or a http/https URL pointing to a remote model.

        "vae" is a Path or str object pointing to a ckpt or safetensors file to be used
        as the VAE for this model.

        "config" is the model config file to use with this ckpt file. It defaults to
        v1-inference.yaml. If a URL is provided, the config will be downloaded.

        You can optionally provide a model name and/or description. If not provided,
        then these will be derived from the weight file name. If you provide a commit_to_conf
        path to the configuration file, then the new entry will be committed to the
        models.yaml file.

        Return value is the name of the imported file, or None if an error occurred.
        """
        if str(weights).startswith(("http:", "https:")):
            model_name = model_name or url_attachment_name(weights)

        weights_path = self._resolve_path(weights, "models/ldm/stable-diffusion-v1")
        config_path = self._resolve_path(config, "configs/stable-diffusion")

        if weights_path is None or not weights_path.exists():
            return
        if config_path is None or not config_path.exists():
            return

        model_name = (
            model_name or Path(weights).stem
        )  # note this gives ugly pathnames if used on a URL without a Content-Disposition header
        model_description = (
            model_description or f"Imported stable diffusion weights file {model_name}"
        )
        new_config = dict(
            weights=str(weights_path),
            config=str(config_path),
            description=model_description,
            format="ckpt",
            width=512,
            height=512,
        )
        if vae:
            new_config["vae"] = vae
        self.add_model(model_name, new_config, True)
        if commit_to_conf:
            self.commit(commit_to_conf)
        return model_name

    @classmethod
    def probe_model_type(self, checkpoint: dict) -> SDLegacyType:
        """
        Given a pickle or safetensors model object, probes contents
        of the object and returns an SDLegacyType indicating its
        format. Valid return values include:
        SDLegacyType.V1
        SDLegacyType.V1_INPAINT
        SDLegacyType.V2     (V2 prediction type unknown)
        SDLegacyType.V2_e   (V2 using 'epsilon' prediction type)
        SDLegacyType.V2_v   (V2 using 'v_prediction' prediction type)
        SDLegacyType.UNKNOWN
        """
        global_step = checkpoint.get("global_step")
        state_dict = checkpoint.get("state_dict") or checkpoint

        try:
            state_dict = checkpoint.get("state_dict") or checkpoint
            key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
            if key_name in state_dict and state_dict[key_name].shape[-1] == 1024:
                if global_step == 220000:
                    return SDLegacyType.V2_e
                elif global_step == 110000:
                    return SDLegacyType.V2_v
                else:
                    return SDLegacyType.V2
            # otherwise we assume a V1 file
            in_channels = state_dict[
                "model.diffusion_model.input_blocks.0.0.weight"
            ].shape[1]
            if in_channels == 9:
                return SDLegacyType.V1_INPAINT
            elif in_channels == 4:
                return SDLegacyType.V1
            else:
                return SDLegacyType.UNKNOWN
        except KeyError:
            return SDLegacyType.UNKNOWN

    def heuristic_import(
        self,
        path_url_or_repo: str,
        convert: bool = False,
        model_name: str = None,
        description: str = None,
        model_config_file: Path = None,
        commit_to_conf: Path = None,
        config_file_callback: Callable[[Path], Path] = None,
    ) -> str:
        """
        Accept a string which could be:
           - a HF diffusers repo_id
           - a URL pointing to a legacy .ckpt or .safetensors file
           - a local path pointing to a legacy .ckpt or .safetensors file
           - a local directory containing .ckpt and .safetensors files
           - a local directory containing a diffusers model

        After determining the nature of the model and downloading it
        (if necessary), the file is probed to determine the correct
        configuration file (if needed) and it is imported.

        The model_name and/or description can be provided. If not, they will
        be generated automatically.

        If convert is true, legacy models will be converted to diffusers
        before importing.

        If commit_to_conf is provided, the newly loaded model will be written
        to the `models.yaml` file at the indicated path. Otherwise, the changes
        will only remain in memory.

        The (potentially derived) name of the model is returned on success, or None
        on failure. When multiple models are added from a directory, only the last
        imported one is returned.
        """
        model_path: Path = None
        thing = path_url_or_repo  # to save typing
        is_temporary = False

        print(f">> Probing {thing} for import")

        if thing.startswith(("http:", "https:", "ftp:")):
            print(f"   | {thing} appears to be a URL")
            model_path = self._resolve_path(
                thing, "models/ldm/stable-diffusion-v1"
            )  # _resolve_path does a download if needed
            is_temporary = True

        elif Path(thing).is_file() and thing.endswith((".ckpt", ".safetensors")):
            if Path(thing).stem in ["model", "diffusion_pytorch_model"]:
                print(
                    f"   | {Path(thing).name} appears to be part of a diffusers model. Skipping import"
                )
                return
            else:
                print(f"   | {thing} appears to be a checkpoint file on disk")
                model_path = self._resolve_path(thing, "models/ldm/stable-diffusion-v1")

        elif Path(thing).is_dir() and Path(thing, "model_index.json").exists():
            print(f"  | {thing} appears to be a diffusers file on disk")
            model_name = self.import_diffuser_model(
                thing,
                model_name=model_name,
                description=description,
                commit_to_conf=commit_to_conf,
            )

        elif Path(thing).is_dir():
            if (Path(thing) / "model_index.json").exists():
                print(f"  | {thing} appears to be a diffusers model.")
                model_name = self.import_diffuser_model(
                    thing, commit_to_conf=commit_to_conf
                )
            else:
                print(
                    f"  |{thing} appears to be a directory. Will scan for models to import"
                )
                for m in list(Path(thing).rglob("*.ckpt")) + list(
                    Path(thing).rglob("*.safetensors")
                ):
                    if model_name := self.heuristic_import(
                        str(m),
                        convert,
                        commit_to_conf=commit_to_conf,
                        config_file_callback=config_file_callback,
                    ):
                        print(f" >> {model_name} successfully imported")
                return model_name

        elif re.match(r"^[\w.+-]+/[\w.+-]+$", thing):
            print(f"  | {thing} appears to be a HuggingFace diffusers repo_id")
            model_name = self.import_diffuser_model(
                thing, commit_to_conf=commit_to_conf
            )
            pipeline, _, _, _ = self._load_diffusers_model(self.config[model_name])
            return model_name
        else:
            print(
                f"** {thing}: Unknown thing. Please provide a URL, file path, directory or HuggingFace repo_id"
            )

        # Model_path is set in the event of a legacy checkpoint file.
        # If not set, we're all done
        if not model_path:
            return

        if model_path.stem in self.config:  # already imported
            print("  | Already imported. Skipping")
            return model_path.stem

        # another round of heuristics to guess the correct config file.
        checkpoint = None
        if model_path.suffix.endswith((".ckpt", ".pt")):
            self.scan_model(model_path, model_path)
            checkpoint = torch.load(model_path)
        else:
            checkpoint = safetensors.torch.load_file(model_path)
        # additional probing needed if no config file provided
        if model_config_file is None:
            # Is there a like-named .yaml file in the same directory as the
            # weights file? If so, we treat this as our model
            if model_path.with_suffix(".yaml").exists():
                model_config_file = model_path.with_suffix(".yaml")
                print(f"   | Using config file {model_config_file.name}")
            else:
                model_type = self.probe_model_type(checkpoint)
                if model_type == SDLegacyType.V1:
                    print("   | SD-v1 model detected")
                    model_config_file = Path(
                        Globals.root, "configs/stable-diffusion/v1-inference.yaml"
                    )
                elif model_type == SDLegacyType.V1_INPAINT:
                    print("   | SD-v1 inpainting model detected")
                    model_config_file = Path(
                        Globals.root,
                        "configs/stable-diffusion/v1-inpainting-inference.yaml",
                    )
                elif model_type == SDLegacyType.V2_v:
                    print("   | SD-v2-v model detected")
                    model_config_file = Path(
                        Globals.root, "configs/stable-diffusion/v2-inference-v.yaml"
                    )
                elif model_type == SDLegacyType.V2_e:
                    print("   | SD-v2-e model detected")
                    model_config_file = Path(
                        Globals.root, "configs/stable-diffusion/v2-inference.yaml"
                    )
                elif model_type == SDLegacyType.V2:
                    print(
                        f"** {thing} is a V2 checkpoint file, but its parameterization cannot be determined. Please provide the configuration file type or path."
                    )
                else:
                    print(
                        f"** {thing} is a legacy checkpoint file but not a known Stable Diffusion model. Please provide the configuration file type or path."
                    )

            if not model_config_file and config_file_callback:
                model_config_file = config_file_callback(model_path)
            if not model_config_file:
                return

        if self.is_v2_config(model_config_file):
            convert = True
            print("   | This SD-v2 model will be converted to diffusers format for use")

        if (vae_path := self._scan_for_matching_file(model_path)):
            print(f"   | Using VAE file {vae_path.name}")
        
        if convert:
            diffuser_path = Path(
                Globals.root, "models", Globals.converted_ckpts_dir, model_path.stem
            )
            vae = None if vae_path else dict(repo_id="stabilityai/sd-vae-ft-mse")
            model_name = self.convert_and_import(
                model_path,
                diffusers_path=diffuser_path,
                vae=vae,
                vae_path=vae_path,
                model_name=model_name,
                model_description=description,
                original_config_file=model_config_file,
                commit_to_conf=commit_to_conf,
                scan_needed=False,
            )
            # in the event that this file was downloaded automatically prior to conversion
            # we do not keep the original .ckpt/.safetensors around
            if is_temporary:
                model_path.unlink(missing_ok=True)
        else:
            model_name = self.import_ckpt_model(
                model_path,
                config=model_config_file,
                model_name=model_name,
                model_description=description,
                vae=str(
                    vae_path
                    or Path(
                        Globals.root,
                        "models/ldm/stable-diffusion-v1/vae-ft-mse-840000-ema-pruned.ckpt",
                    )
                ),
                commit_to_conf=commit_to_conf,
            )
        if commit_to_conf:
            self.commit(commit_to_conf)
        return model_name

    def convert_and_import(
        self,
        ckpt_path: Path,
        diffusers_path: Path,
        model_name=None,
        model_description=None,
        vae: dict = None,
        vae_path: Path = None,
        original_config_file: Path = None,
        commit_to_conf: Path = None,
        scan_needed: bool = True,
    ) -> str:
        """
        Convert a legacy ckpt weights file to diffuser model and import
        into models.yaml.
        """
        ckpt_path = self._resolve_path(ckpt_path, "models/ldm/stable-diffusion-v1")
        if original_config_file:
            original_config_file = self._resolve_path(
                original_config_file, "configs/stable-diffusion"
            )

        new_config = None

        from ldm.invoke.ckpt_to_diffuser import convert_ckpt_to_diffusers

        if diffusers_path.exists():
            print(
                f"ERROR: The path {str(diffusers_path)} already exists. Please move or remove it and try again."
            )
            return

        model_name = model_name or diffusers_path.name
        model_description = model_description or f"Optimized version of {model_name}"
        print(f">> Optimizing {model_name} (30-60s)")
        try:
            # By passing the specified VAE to the conversion function, the autoencoder
            # will be built into the model rather than tacked on afterward via the config file
            vae_model=None
            if vae:
                vae_model=self._load_vae(vae)
                vae_path=None
            convert_ckpt_to_diffusers(
                ckpt_path,
                diffusers_path,
                extract_ema=True,
                original_config_file=original_config_file,
                vae=vae_model,
                vae_path=vae_path,
                scan_needed=scan_needed,
            )
            print(
                f"   | Success. Optimized model is now located at {str(diffusers_path)}"
            )
            print(f"   | Writing new config file entry for {model_name}")
            new_config = dict(
                path=str(diffusers_path),
                description=model_description,
                format="diffusers",
            )
            if model_name in self.config:
                self.del_model(model_name)
            self.add_model(model_name, new_config, True)
            if commit_to_conf:
                self.commit(commit_to_conf)
            print(">> Conversion succeeded")
        except Exception as e:
            print(f"** Conversion failed: {str(e)}")
            print(
                "** If you are trying to convert an inpainting or 2.X model, please indicate the correct config file (e.g. v1-inpainting-inference.yaml)"
            )

        return model_name

    def search_models(self, search_folder):
        print(f">> Finding Models In: {search_folder}")
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

    def _make_cache_room(self) -> None:
        num_loaded_models = len(self.models)
        if num_loaded_models >= self.max_loaded_models:
            least_recent_model = self._pop_oldest_model()
            print(
                f">> Cache limit (max={self.max_loaded_models}) reached. Purging {least_recent_model}"
            )
            if least_recent_model is not None:
                del self.models[least_recent_model]
                gc.collect()

    def print_vram_usage(self) -> None:
        if self._has_cuda:
            print(
                ">> Current VRAM usage: ",
                "%4.2fG" % (torch.cuda.memory_allocated() / 1e9),
            )

    def commit(self, config_file_path: str) -> None:
        """
        Write current configuration out to the indicated file.
        """
        yaml_str = OmegaConf.to_yaml(self.config)
        if not os.path.isabs(config_file_path):
            config_file_path = os.path.normpath(
                os.path.join(Globals.root, config_file_path)
            )
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

    @classmethod
    def migrate_models(cls):
        """
        Migrate the ~/invokeai/models directory from the legacy format used through 2.2.5
        to the 2.3.0 "diffusers" version. This should be a one-time operation, called at
        script startup time.
        """
        # Three transformer models to check: bert, clip and safety checker, and
        # the diffusers as well
        models_dir = Path(Globals.root, "models")
        legacy_locations = [
            Path(
                models_dir,
                "CompVis/stable-diffusion-safety-checker/models--CompVis--stable-diffusion-safety-checker",
            ),
            Path("bert-base-uncased/models--bert-base-uncased"),
            Path(
                "openai/clip-vit-large-patch14/models--openai--clip-vit-large-patch14"
            ),
        ]
        legacy_locations.extend(list(global_cache_dir("diffusers").glob("*")))
        legacy_layout = False
        for model in legacy_locations:
            legacy_layout = legacy_layout or model.exists()
        if not legacy_layout:
            return

        print(
            """
>> ALERT:
>> The location of your previously-installed diffusers models needs to move from
>> invokeai/models/diffusers to invokeai/models/hub due to a change introduced by
>> diffusers version 0.14. InvokeAI will now move all models from the "diffusers" directory
>> into "hub" and then remove the diffusers directory. This is a quick, safe, one-time
>> operation. However if you have customized either of these directories and need to
>> make adjustments, please press ctrl-C now to abort and relaunch InvokeAI when you are ready.
>> Otherwise press <enter> to continue."""
        )
        print("** This is a quick one-time operation.")
        input("continue> ")

        # transformer files get moved into the hub directory
        if cls._is_huggingface_hub_directory_present():
            hub = global_cache_dir("hub")
        else:
            hub = models_dir / "hub"

        os.makedirs(hub, exist_ok=True)
        for model in legacy_locations:
            source = models_dir / model
            dest = hub / model.stem
            if dest.exists() and not source.exists():
                continue
            print(f"** {source} => {dest}")
            if source.exists():
                if dest.is_symlink():
                    print(f"** Found symlink at {dest.name}. Not migrating.")
                elif dest.exists():
                    if source.is_dir():
                        rmtree(source)
                    else:
                        source.unlink()
                else:
                    move(source, dest)

        # now clean up by removing any empty directories
        empty = [
            root
            for root, dirs, files, in os.walk(models_dir)
            if not len(dirs) and not len(files)
        ]
        for d in empty:
            os.rmdir(d)
        print("** Migration is done. Continuing...")

    def _resolve_path(
        self, source: Union[str, Path], dest_directory: str
    ) -> Optional[Path]:
        resolved_path = None
        if str(source).startswith(("http:", "https:", "ftp:")):
            dest_directory = Path(dest_directory)
            if not dest_directory.is_absolute():
                dest_directory = Globals.root / dest_directory
            dest_directory.mkdir(parents=True, exist_ok=True)
            resolved_path = download_with_resume(str(source), dest_directory)
        else:
            if not os.path.isabs(source):
                source = os.path.join(Globals.root, source)
            resolved_path = Path(source)
        return resolved_path

    def _invalidate_cached_model(self, model_name: str) -> None:
        self.offload_model(model_name)
        if model_name in self.stack:
            self.stack.remove(model_name)
        self.models.pop(model_name, None)

    def _model_to_cpu(self, model):
        if self.device == CPU_DEVICE:
            return model

        if isinstance(model, StableDiffusionGeneratorPipeline):
            model.offload_all()
            return model

        model.cond_stage_model.device = CPU_DEVICE
        model.to(CPU_DEVICE)

        for submodel in ("first_stage_model", "cond_stage_model", "model"):
            try:
                getattr(model, submodel).to(CPU_DEVICE)
            except AttributeError:
                pass
        return model

    def _model_from_cpu(self, model):
        if self.device == CPU_DEVICE:
            return model

        if isinstance(model, StableDiffusionGeneratorPipeline):
            model.ready()
            return model

        model.to(self.device)
        model.cond_stage_model.device = self.device

        for submodel in ("first_stage_model", "cond_stage_model", "model"):
            try:
                getattr(model, submodel).to(self.device)
            except AttributeError:
                pass

        return model

    def _pop_oldest_model(self):
        """
        Remove the first element of the FIFO, which ought
        to be the least recently accessed model. Do not
        pop the last one, because it is in active use!
        """
        return self.stack.pop(0)

    def _push_newest_model(self, model_name: str) -> None:
        """
        Maintain a simple FIFO. First element is always the
        least recent, and last element is always the most recent.
        """
        with contextlib.suppress(ValueError):
            self.stack.remove(model_name)
        self.stack.append(model_name)

    def _has_cuda(self) -> bool:
        return self.device.type == "cuda"

    def _diffuser_sha256(
        self, name_or_path: Union[str, Path], chunksize=4096
    ) -> Union[str, bytes]:
        path = None
        if isinstance(name_or_path, Path):
            path = name_or_path
        else:
            owner, repo = name_or_path.split("/")
            path = Path(global_cache_dir("hub") / f"models--{owner}--{repo}")
        if not path.exists():
            return None
        hashpath = path / "checksum.sha256"
        if hashpath.exists() and path.stat().st_mtime <= hashpath.stat().st_mtime:
            with open(hashpath) as f:
                hash = f.read()
            return hash
        print("   | Calculating sha256 hash of model files")
        tic = time.time()
        sha = hashlib.sha256()
        count = 0
        for root, dirs, files in os.walk(path, followlinks=False):
            for name in files:
                count += 1
                with open(os.path.join(root, name), "rb") as f:
                    while chunk := f.read(chunksize):
                        sha.update(chunk)
        hash = sha.hexdigest()
        toc = time.time()
        print(f"   | sha256 = {hash} ({count} files hashed in", "%4.2fs)" % (toc - tic))
        with open(hashpath, "w") as f:
            f.write(hash)
        return hash

    def _cached_sha256(self, path, data) -> Union[str, bytes]:
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        base, _ = os.path.splitext(basename)
        hashpath = os.path.join(dirname, base + ".sha256")

        if os.path.exists(hashpath) and os.path.getmtime(path) <= os.path.getmtime(
            hashpath
        ):
            with open(hashpath) as f:
                hash = f.read()
            return hash

        print("   | Calculating sha256 hash of weights file")
        tic = time.time()
        sha = hashlib.sha256()
        sha.update(data)
        hash = sha.hexdigest()
        toc = time.time()
        print(f">> sha256 = {hash}", "(%4.2fs)" % (toc - tic))

        with open(hashpath, "w") as f:
            f.write(hash)
        return hash

    @classmethod
    def _scan_for_matching_file(
            self,model_path: Path,
            suffixes: List[str]=['.vae.pt','.vae.ckpt','.vae.safetensors']
    )->Path:
        """
        Find a file with same basename as the indicated model, but with one
        of the suffixes passed.
        """
        # look for a custom vae
        vae_path = None
        for suffix in suffixes:
            if model_path.with_suffix(suffix).exists():
                vae_path = model_path.with_suffix(suffix)
        return vae_path
                               
    def _load_vae(self, vae_config) -> AutoencoderKL:
        vae_args = {}
        try:
            name_or_path = self.model_name_or_path(vae_config)
        except Exception:
            return None
        if name_or_path is None:
            return None
        using_fp16 = self.precision == "float16"

        vae_args.update(
            cache_dir=global_cache_dir("hub"),
            local_files_only=not Globals.internet_available,
        )

        print(f"   | Loading diffusers VAE from {name_or_path}")
        if using_fp16:
            vae_args.update(torch_dtype=torch.float16)
            fp_args_list = [{"revision": "fp16"}, {}]
        else:
            print("   | Using more accurate float32 precision")
            fp_args_list = [{}]

        vae = None
        deferred_error = None

        # A VAE may be in a subfolder of a model's repository.
        if "subfolder" in vae_config:
            vae_args["subfolder"] = vae_config["subfolder"]

        for fp_args in fp_args_list:
            # At some point we might need to be able to use different classes here? But for now I think
            # all Stable Diffusion VAE are AutoencoderKL.
            try:
                vae = AutoencoderKL.from_pretrained(name_or_path, **vae_args, **fp_args)
            except OSError as e:
                if str(e).startswith("fp16 is not a valid"):
                    pass
                else:
                    deferred_error = e
            if vae:
                break

        if not vae and deferred_error:
            print(f"** Could not load VAE {name_or_path}: {str(deferred_error)}")

        return vae

    @staticmethod
    def _delete_model_from_cache(repo_id):
        cache_info = scan_cache_dir(global_cache_dir("diffusers"))

        # I'm sure there is a way to do this with comprehensions
        # but the code quickly became incomprehensible!
        hashes_to_delete = set()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                for revision in repo.revisions:
                    hashes_to_delete.add(revision.commit_hash)
        strategy = cache_info.delete_revisions(*hashes_to_delete)
        print(
            f"** Deletion of this model is expected to free {strategy.expected_freed_size_str}"
        )
        strategy.execute()

    @staticmethod
    def _abs_path(path: str | Path) -> Path:
        if path is None or Path(path).is_absolute():
            return path
        return Path(Globals.root, path).resolve()

    @staticmethod
    def _is_huggingface_hub_directory_present() -> bool:
        return (
            os.getenv("HF_HOME") is not None or os.getenv("XDG_CACHE_HOME") is not None
        )
