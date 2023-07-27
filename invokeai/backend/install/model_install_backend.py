"""
Utility (backend) functions used by model_install.py
"""
import os
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Callable, Union, Set

import requests
from diffusers import DiffusionPipeline
from diffusers import logging as dlogging
from huggingface_hub import hf_hub_url, HfFolder, HfApi
from omegaconf import OmegaConf
from tqdm import tqdm

import invokeai.configs as configs

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_management import ModelManager, ModelType, BaseModelType, ModelVariantType, AddModelResult
from invokeai.backend.model_management.model_probe import ModelProbe, SchedulerPredictionType, ModelProbeInfo
from invokeai.backend.util import download_with_resume
from ..util.logging import InvokeAILogger

warnings.filterwarnings("ignore")

# --------------------------globals-----------------------
config = InvokeAIAppConfig.get_config()
logger = InvokeAILogger.getLogger(name="InvokeAI")

# the initial "configs" dir is now bundled in the `invokeai.configs` package
Dataset_path = Path(configs.__path__[0]) / "INITIAL_MODELS.yaml"

Config_preamble = """
# This file describes the alternative machine learning models
# available to InvokeAI script.
#
# To add a new model, follow the examples below. Each
# model requires a model config file, a weights file,
# and the width and height of the images it
# was trained on.
"""

LEGACY_CONFIGS = {
    BaseModelType.StableDiffusion1: {
        ModelVariantType.Normal: "v1-inference.yaml",
        ModelVariantType.Inpaint: "v1-inpainting-inference.yaml",
    },
    BaseModelType.StableDiffusion2: {
        ModelVariantType.Normal: {
            SchedulerPredictionType.Epsilon: "v2-inference.yaml",
            SchedulerPredictionType.VPrediction: "v2-inference-v.yaml",
        },
        ModelVariantType.Inpaint: {
            SchedulerPredictionType.Epsilon: "v2-inpainting-inference.yaml",
            SchedulerPredictionType.VPrediction: "v2-inpainting-inference-v.yaml",
        },
    },
    BaseModelType.StableDiffusionXL: {
        ModelVariantType.Normal: "sd_xl_base.yaml",
    },
    BaseModelType.StableDiffusionXLRefiner: {
        ModelVariantType.Normal: "sd_xl_refiner.yaml",
    },
}


@dataclass
class ModelInstallList:
    """Class for listing models to be installed/removed"""

    install_models: List[str] = field(default_factory=list)
    remove_models: List[str] = field(default_factory=list)


@dataclass
class InstallSelections:
    install_models: List[str] = field(default_factory=list)
    remove_models: List[str] = field(default_factory=list)


@dataclass
class ModelLoadInfo:
    name: str
    model_type: ModelType
    base_type: BaseModelType
    path: Path = None
    repo_id: str = None
    description: str = ""
    installed: bool = False
    recommended: bool = False
    default: bool = False


class ModelInstall(object):
    def __init__(
        self,
        config: InvokeAIAppConfig,
        prediction_type_helper: Callable[[Path], SchedulerPredictionType] = None,
        model_manager: ModelManager = None,
        access_token: str = None,
    ):
        self.config = config
        self.mgr = model_manager or ModelManager(config.model_conf_path)
        self.datasets = OmegaConf.load(Dataset_path)
        self.prediction_helper = prediction_type_helper
        self.access_token = access_token or HfFolder.get_token()
        self.reverse_paths = self._reverse_paths(self.datasets)

    def all_models(self) -> Dict[str, ModelLoadInfo]:
        """
        Return dict of model_key=>ModelLoadInfo objects.
        This method consolidates and simplifies the entries in both
        models.yaml and INITIAL_MODELS.yaml so that they can
        be treated uniformly. It also sorts the models alphabetically
        by their name, to improve the display somewhat.
        """
        model_dict = dict()

        # first populate with the entries in INITIAL_MODELS.yaml
        for key, value in self.datasets.items():
            name, base, model_type = ModelManager.parse_key(key)
            value["name"] = name
            value["base_type"] = base
            value["model_type"] = model_type
            model_dict[key] = ModelLoadInfo(**value)

        # supplement with entries in models.yaml
        installed_models = self.mgr.list_models()

        for md in installed_models:
            base = md["base_model"]
            model_type = md["model_type"]
            name = md["model_name"]
            key = ModelManager.create_key(name, base, model_type)
            if key in model_dict:
                model_dict[key].installed = True
            else:
                model_dict[key] = ModelLoadInfo(
                    name=name,
                    base_type=base,
                    model_type=model_type,
                    path=value.get("path"),
                    installed=True,
                )
        return {x: model_dict[x] for x in sorted(model_dict.keys(), key=lambda y: model_dict[y].name.lower())}

    def list_models(self, model_type):
        installed = self.mgr.list_models(model_type=model_type)
        print(f"Installed models of type `{model_type}`:")
        for i in installed:
            print(f"{i['model_name']}\t{i['base_model']}\t{i['path']}")

    # logic here a little reversed to maintain backward compatibility
    def starter_models(self, all_models: bool = False) -> Set[str]:
        models = set()
        for key, value in self.datasets.items():
            name, base, model_type = ModelManager.parse_key(key)
            if all_models or model_type in [ModelType.Main, ModelType.Vae]:
                models.add(key)
        return models

    def recommended_models(self) -> Set[str]:
        starters = self.starter_models(all_models=True)
        return set([x for x in starters if self.datasets[x].get("recommended", False)])

    def default_model(self) -> str:
        starters = self.starter_models()
        defaults = [x for x in starters if self.datasets[x].get("default", False)]
        return defaults[0]

    def install(self, selections: InstallSelections):
        verbosity = dlogging.get_verbosity()  # quench NSFW nags
        dlogging.set_verbosity_error()

        job = 1
        jobs = len(selections.remove_models) + len(selections.install_models)

        # remove requested models
        for key in selections.remove_models:
            name, base, mtype = self.mgr.parse_key(key)
            logger.info(f"Deleting {mtype} model {name} [{job}/{jobs}]")
            try:
                self.mgr.del_model(name, base, mtype)
            except FileNotFoundError as e:
                logger.warning(e)
            job += 1

        # add requested models
        for path in selections.install_models:
            logger.info(f"Installing {path} [{job}/{jobs}]")
            try:
                self.heuristic_import(path)
            except (ValueError, KeyError) as e:
                logger.error(str(e))
            job += 1

        dlogging.set_verbosity(verbosity)
        self.mgr.commit()

    def heuristic_import(
        self,
        model_path_id_or_url: Union[str, Path],
        models_installed: Set[Path] = None,
    ) -> Dict[str, AddModelResult]:
        """
        :param model_path_id_or_url: A Path to a local model to import, or a string representing its repo_id or URL
        :param models_installed: Set of installed models, used for recursive invocation
        Returns a set of dict objects corresponding to newly-created stanzas in models.yaml.
        """

        if not models_installed:
            models_installed = dict()

        # A little hack to allow nested routines to retrieve info on the requested ID
        self.current_id = model_path_id_or_url
        path = Path(model_path_id_or_url)
        # checkpoint file, or similar
        if path.is_file():
            models_installed.update({str(path): self._install_path(path)})

        # folders style or similar
        elif path.is_dir() and any(
            [
                (path / x).exists()
                for x in {"config.json", "model_index.json", "learned_embeds.bin", "pytorch_lora_weights.bin"}
            ]
        ):
            models_installed.update({str(model_path_id_or_url): self._install_path(path)})

        # recursive scan
        elif path.is_dir():
            for child in path.iterdir():
                self.heuristic_import(child, models_installed=models_installed)

        # huggingface repo
        elif len(str(model_path_id_or_url).split("/")) == 2:
            models_installed.update({str(model_path_id_or_url): self._install_repo(str(model_path_id_or_url))})

        # a URL
        elif str(model_path_id_or_url).startswith(("http:", "https:", "ftp:")):
            models_installed.update({str(model_path_id_or_url): self._install_url(model_path_id_or_url)})

        else:
            raise KeyError(f"{str(model_path_id_or_url)} is not recognized as a local path, repo ID or URL. Skipping")

        return models_installed

    # install a model from a local path. The optional info parameter is there to prevent
    # the model from being probed twice in the event that it has already been probed.
    def _install_path(self, path: Path, info: ModelProbeInfo = None) -> AddModelResult:
        info = info or ModelProbe().heuristic_probe(path, self.prediction_helper)
        if not info:
            logger.warning(f"Unable to parse format of {path}")
            return None
        model_name = path.stem if path.is_file() else path.name
        if self.mgr.model_exists(model_name, info.base_type, info.model_type):
            raise ValueError(f'A model named "{model_name}" is already installed.')
        attributes = self._make_attributes(path, info)
        return self.mgr.add_model(
            model_name=model_name,
            base_model=info.base_type,
            model_type=info.model_type,
            model_attributes=attributes,
        )

    def _install_url(self, url: str) -> AddModelResult:
        with TemporaryDirectory(dir=self.config.models_path) as staging:
            location = download_with_resume(url, Path(staging))
            if not location:
                logger.error(f"Unable to download {url}. Skipping.")
            info = ModelProbe().heuristic_probe(location)
            dest = self.config.models_path / info.base_type.value / info.model_type.value / location.name
            models_path = shutil.move(location, dest)

        # staged version will be garbage-collected at this time
        return self._install_path(Path(models_path), info)

    def _install_repo(self, repo_id: str) -> AddModelResult:
        hinfo = HfApi().model_info(repo_id)

        # we try to figure out how to download this most economically
        # list all the files in the repo
        files = [x.rfilename for x in hinfo.siblings]
        location = None

        with TemporaryDirectory(dir=self.config.models_path) as staging:
            staging = Path(staging)
            if "model_index.json" in files:
                location = self._download_hf_pipeline(repo_id, staging)  # pipeline
            else:
                for suffix in ["safetensors", "bin"]:
                    if f"pytorch_lora_weights.{suffix}" in files:
                        location = self._download_hf_model(repo_id, ["pytorch_lora_weights.bin"], staging)  # LoRA
                        break
                    elif (
                        self.config.precision == "float16" and f"diffusion_pytorch_model.fp16.{suffix}" in files
                    ):  # vae, controlnet or some other standalone
                        files = ["config.json", f"diffusion_pytorch_model.fp16.{suffix}"]
                        location = self._download_hf_model(repo_id, files, staging)
                        break
                    elif f"diffusion_pytorch_model.{suffix}" in files:
                        files = ["config.json", f"diffusion_pytorch_model.{suffix}"]
                        location = self._download_hf_model(repo_id, files, staging)
                        break
                    elif f"learned_embeds.{suffix}" in files:
                        location = self._download_hf_model(repo_id, [f"learned_embeds.{suffix}"], staging)
                        break
            if not location:
                logger.warning(f"Could not determine type of repo {repo_id}. Skipping install.")
                return {}

            info = ModelProbe().heuristic_probe(location, self.prediction_helper)
            if not info:
                logger.warning(f"Could not probe {location}. Skipping install.")
                return {}
            dest = (
                self.config.models_path
                / info.base_type.value
                / info.model_type.value
                / self._get_model_name(repo_id, location)
            )
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(location, dest)
            return self._install_path(dest, info)

    def _get_model_name(self, path_name: str, location: Path) -> str:
        """
        Calculate a name for the model - primitive implementation.
        """
        if key := self.reverse_paths.get(path_name):
            (name, base, mtype) = ModelManager.parse_key(key)
            return name
        elif location.is_dir():
            return location.name
        else:
            return location.stem

    def _make_attributes(self, path: Path, info: ModelProbeInfo) -> dict:
        model_name = path.name if path.is_dir() else path.stem
        description = f"{info.base_type.value} {info.model_type.value} model {model_name}"
        if key := self.reverse_paths.get(self.current_id):
            if key in self.datasets:
                description = self.datasets[key].get("description") or description

        rel_path = self.relative_to_root(path)

        attributes = dict(
            path=str(rel_path),
            description=str(description),
            model_format=info.format,
        )
        legacy_conf = None
        if info.model_type == ModelType.Main:
            attributes.update(
                dict(
                    variant=info.variant_type,
                )
            )
            if info.format == "checkpoint":
                try:
                    possible_conf = path.with_suffix(".yaml")
                    if possible_conf.exists():
                        legacy_conf = str(self.relative_to_root(possible_conf))
                    elif info.base_type == BaseModelType.StableDiffusion2:
                        legacy_conf = Path(
                            self.config.legacy_conf_dir,
                            LEGACY_CONFIGS[info.base_type][info.variant_type][info.prediction_type],
                        )
                    else:
                        legacy_conf = Path(
                            self.config.legacy_conf_dir, LEGACY_CONFIGS[info.base_type][info.variant_type]
                        )
                except KeyError:
                    legacy_conf = Path(self.config.legacy_conf_dir, "v1-inference.yaml")  # best guess

        if info.model_type == ModelType.ControlNet and info.format == "checkpoint":
            possible_conf = path.with_suffix(".yaml")
            if possible_conf.exists():
                legacy_conf = str(self.relative_to_root(possible_conf))

        if legacy_conf:
            attributes.update(dict(config=str(legacy_conf)))
        return attributes

    def relative_to_root(self, path: Path) -> Path:
        root = self.config.root_path
        if path.is_relative_to(root):
            return path.relative_to(root)
        else:
            return path

    def _download_hf_pipeline(self, repo_id: str, staging: Path) -> Path:
        """
        This retrieves a StableDiffusion model from cache or remote and then
        does a save_pretrained() to the indicated staging area.
        """
        _, name = repo_id.split("/")
        revisions = ["fp16", "main"] if self.config.precision == "float16" else ["main"]
        model = None
        for revision in revisions:
            try:
                model = DiffusionPipeline.from_pretrained(repo_id, revision=revision, safety_checker=None)
            except:  # most errors are due to fp16 not being present. Fix this to catch other errors
                pass
            if model:
                break
        if not model:
            logger.error(f"Diffusers model {repo_id} could not be downloaded. Skipping.")
            return None
        model.save_pretrained(staging / name, safe_serialization=True)
        return staging / name

    def _download_hf_model(self, repo_id: str, files: List[str], staging: Path) -> Path:
        _, name = repo_id.split("/")
        location = staging / name
        paths = list()
        for filename in files:
            p = hf_download_with_resume(
                repo_id, model_dir=location, model_name=filename, access_token=self.access_token
            )
            if p:
                paths.append(p)
            else:
                logger.warning(f"Could not download {filename} from {repo_id}.")

        return location if len(paths) > 0 else None

    @classmethod
    def _reverse_paths(cls, datasets) -> dict:
        """
        Reverse mapping from repo_id/path to destination name.
        """
        return {v.get("path") or v.get("repo_id"): k for k, v in datasets.items()}


# -------------------------------------
def yes_or_no(prompt: str, default_yes=True):
    default = "y" if default_yes else "n"
    response = input(f"{prompt} [{default}] ") or default
    if default_yes:
        return response[0] not in ("n", "N")
    else:
        return response[0] in ("y", "Y")


# ---------------------------------------------
def hf_download_from_pretrained(model_class: object, model_name: str, destination: Path, **kwargs):
    logger = InvokeAILogger.getLogger("InvokeAI")
    logger.addFilter(lambda x: "fp16 is not a valid" not in x.getMessage())

    model = model_class.from_pretrained(
        model_name,
        resume_download=True,
        **kwargs,
    )
    model.save_pretrained(destination, safe_serialization=True)
    return destination


# ---------------------------------------------
def hf_download_with_resume(
    repo_id: str,
    model_dir: str,
    model_name: str,
    model_dest: Path = None,
    access_token: str = None,
) -> Path:
    model_dest = model_dest or Path(os.path.join(model_dir, model_name))
    os.makedirs(model_dir, exist_ok=True)

    url = hf_hub_url(repo_id, model_name)

    header = {"Authorization": f"Bearer {access_token}"} if access_token else {}
    open_mode = "wb"
    exist_size = 0

    if os.path.exists(model_dest):
        exist_size = os.path.getsize(model_dest)
        header["Range"] = f"bytes={exist_size}-"
        open_mode = "ab"

    resp = requests.get(url, headers=header, stream=True)
    total = int(resp.headers.get("content-length", 0))

    if resp.status_code == 416:  # "range not satisfiable", which means nothing to return
        logger.info(f"{model_name}: complete file found. Skipping.")
        return model_dest
    elif resp.status_code == 404:
        logger.warning("File not found")
        return None
    elif resp.status_code != 200:
        logger.warning(f"{model_name}: {resp.reason}")
    elif exist_size > 0:
        logger.info(f"{model_name}: partial file found. Resuming...")
    else:
        logger.info(f"{model_name}: Downloading...")

    try:
        with open(model_dest, open_mode) as file, tqdm(
            desc=model_name,
            initial=exist_size,
            total=total + exist_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1000,
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    except Exception as e:
        logger.error(f"An error occurred while downloading {model_name}: {str(e)}")
        return None
    return model_dest
