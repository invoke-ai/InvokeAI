"""
Utility (backend) functions used by model_install.py
"""
import os
import re
import shutil
import sys
import warnings
from dataclasses import dataclass,field
from pathlib import Path
from tempfile import TemporaryFile
from typing import List, Dict, Callable

import requests
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_url, HfFolder
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

import invokeai.configs as configs


from invokeai.app.services.config import InvokeAIAppConfig
from ..stable_diffusion import StableDiffusionGeneratorPipeline
from ..util.logging import InvokeAILogger

warnings.filterwarnings("ignore")

# --------------------------globals-----------------------
config = InvokeAIAppConfig.get_config()

Model_dir = "models"
Weights_dir = "ldm/stable-diffusion-v1/"

# the initial "configs" dir is now bundled in the `invokeai.configs` package
Dataset_path = Path(configs.__path__[0]) / "INITIAL_MODELS.yaml"

# initial models omegaconf
Datasets = None

# logger
logger = InvokeAILogger.getLogger(name='InvokeAI')

Config_preamble = """
# This file describes the alternative machine learning models
# available to InvokeAI script.
#
# To add a new model, follow the examples below. Each
# model requires a model config file, a weights file,
# and the width and height of the images it
# was trained on.
"""

@dataclass
class ModelInstallList:
    '''Class for listing models to be installed/removed'''
    install_models: List[str] = field(default_factory=list)
    remove_models: List[str] = field(default_factory=list)

@dataclass
class UserSelections():
    install_models: List[str]= field(default_factory=list)
    remove_models: List[str]=field(default_factory=list)
    purge_deleted_models: bool=field(default_factory=list)
    install_cn_models: List[str] = field(default_factory=list)
    remove_cn_models: List[str] = field(default_factory=list)
    install_lora_models: List[str] = field(default_factory=list)
    remove_lora_models: List[str] = field(default_factory=list)
    install_ti_models: List[str] = field(default_factory=list)
    remove_ti_models: List[str] = field(default_factory=list)
    scan_directory: Path = None
    autoscan_on_startup: bool=False
    import_model_paths: str=None
        
def default_config_file():
    return config.model_conf_path

def sd_configs():
    return config.legacy_conf_path

def initial_models():
    global Datasets
    if Datasets:
        return Datasets
    return (Datasets := OmegaConf.load(Dataset_path)['diffusers'])

def install_requested_models(
        diffusers: ModelInstallList = None,
        controlnet: ModelInstallList = None,
        lora: ModelInstallList = None,
        ti: ModelInstallList = None,
        cn_model_map: Dict[str,str] = None, # temporary - move to model manager
        scan_directory: Path = None,
        external_models: List[str] = None,
        scan_at_startup: bool = False,
        precision: str = "float16",
        purge_deleted: bool = False,
        config_file_path: Path = None,
        model_config_file_callback:  Callable[[Path],Path] = None
):
    """
    Entry point for installing/deleting starter models, or installing external models.
    """
    access_token = HfFolder.get_token()
    config_file_path = config_file_path or default_config_file()
    if not config_file_path.exists():
        open(config_file_path, "w")

    # prevent circular import here
    from ..model_management import ModelManager
    model_manager = ModelManager(OmegaConf.load(config_file_path), precision=precision)
    if controlnet:
        model_manager.install_controlnet_models(controlnet.install_models, access_token=access_token)
        model_manager.delete_controlnet_models(controlnet.remove_models)

    if lora:
        model_manager.install_lora_models(lora.install_models, access_token=access_token)
        model_manager.delete_lora_models(lora.remove_models)

    if ti:
        model_manager.install_ti_models(ti.install_models, access_token=access_token)
        model_manager.delete_ti_models(ti.remove_models)

    if diffusers:
        # TODO: Replace next three paragraphs with calls into new model manager
        if diffusers.remove_models and len(diffusers.remove_models) > 0:
            logger.info("Processing requested deletions")
            for model in diffusers.remove_models:
                logger.info(f"{model}...")
                model_manager.del_model(model, delete_files=purge_deleted)
            model_manager.commit(config_file_path)

        if diffusers.install_models and len(diffusers.install_models) > 0:
            logger.info("Installing requested models")
            downloaded_paths = download_weight_datasets(
                models=diffusers.install_models,
                access_token=None,
                precision=precision,
            )
            successful = {x:v for x,v in downloaded_paths.items() if v is not None}
            if len(successful) > 0:
                update_config_file(successful, config_file_path)
            if len(successful) < len(diffusers.install_models):
                unsuccessful = [x for x in downloaded_paths if downloaded_paths[x] is None]
                logger.warning(f"Some of the model downloads were not successful: {unsuccessful}")

    # due to above, we have to reload the model manager because conf file
    # was changed behind its back
    model_manager = ModelManager(OmegaConf.load(config_file_path), precision=precision)

    external_models = external_models or list()
    if scan_directory:
        external_models.append(str(scan_directory))

    if len(external_models) > 0:
        logger.info("INSTALLING EXTERNAL MODELS")
        for path_url_or_repo in external_models:
            try:
                logger.debug(f'In install_requested_models; callback = {model_config_file_callback}')
                model_manager.heuristic_import(
                    path_url_or_repo,
                    commit_to_conf=config_file_path,
                    config_file_callback = model_config_file_callback,
                )
            except KeyboardInterrupt:
                sys.exit(-1)
            except Exception:
                pass

    if scan_at_startup and scan_directory.is_dir():
        update_autoconvert_dir(scan_directory)
    else:
        update_autoconvert_dir(None)

def update_autoconvert_dir(autodir: Path):
    '''
    Update the "autoconvert_dir" option in invokeai.yaml
    '''
    invokeai_config_path = config.init_file_path
    conf = OmegaConf.load(invokeai_config_path)
    conf.InvokeAI.Paths.autoconvert_dir = str(autodir) if autodir else None
    yaml = OmegaConf.to_yaml(conf)
    tmpfile = invokeai_config_path.parent / "new_config.tmp"
    with open(tmpfile, "w", encoding="utf-8") as outfile:
        outfile.write(yaml)
    tmpfile.replace(invokeai_config_path)


# -------------------------------------
def yes_or_no(prompt: str, default_yes=True):
    default = "y" if default_yes else "n"
    response = input(f"{prompt} [{default}] ") or default
    if default_yes:
        return response[0] not in ("n", "N")
    else:
        return response[0] in ("y", "Y")

# ---------------------------------------------
def recommended_datasets() -> List['str']:
    datasets = set()
    for ds in initial_models().keys():
        if initial_models()[ds].get("recommended", False):
            datasets.add(ds)
    return list(datasets)

# ---------------------------------------------
def default_dataset() -> dict:
    datasets = set()
    for ds in initial_models().keys():
        if initial_models()[ds].get("default", False):
            datasets.add(ds)
    return list(datasets)


# ---------------------------------------------
def all_datasets() -> dict:
    datasets = dict()
    for ds in initial_models().keys():
        datasets[ds] = True
    return datasets


# ---------------------------------------------
# look for legacy model.ckpt in models directory and offer to
# normalize its name
def migrate_models_ckpt():
    model_path = os.path.join(config.root_dir, Model_dir, Weights_dir)
    if not os.path.exists(os.path.join(model_path, "model.ckpt")):
        return
    new_name = initial_models()["stable-diffusion-1.4"]["file"]
    logger.warning(
        'The Stable Diffusion v4.1 "model.ckpt" is already installed. The name will be changed to {new_name} to avoid confusion.'
    )
    logger.warning(f"model.ckpt => {new_name}")
    os.replace(
        os.path.join(model_path, "model.ckpt"), os.path.join(model_path, new_name)
    )


# ---------------------------------------------
def download_weight_datasets(
    models: List[str], access_token: str, precision: str = "float32"
):
    migrate_models_ckpt()
    successful = dict()
    for mod in models:
        logger.info(f"Downloading {mod}:")
        successful[mod] = _download_repo_or_file(
            initial_models()[mod], access_token, precision=precision
        )
    return successful


def _download_repo_or_file(
    mconfig: DictConfig, access_token: str, precision: str = "float32"
) -> Path:
    path = None
    if mconfig["format"] == "ckpt":
        path = _download_ckpt_weights(mconfig, access_token)
    else:
        path = _download_diffusion_weights(mconfig, access_token, precision=precision)
        if "vae" in mconfig and "repo_id" in mconfig["vae"]:
            _download_diffusion_weights(
                mconfig["vae"], access_token, precision=precision
            )
    return path

def _download_ckpt_weights(mconfig: DictConfig, access_token: str) -> Path:
    repo_id = mconfig["repo_id"]
    filename = mconfig["file"]
    cache_dir = os.path.join(config.root_dir, Model_dir, Weights_dir)
    return hf_download_with_resume(
        repo_id=repo_id,
        model_dir=cache_dir,
        model_name=filename,
        access_token=access_token,
    )


# ---------------------------------------------
def download_from_hf(
    model_class: object, model_name: str, **kwargs
):
    logger = InvokeAILogger.getLogger('InvokeAI')
    logger.addFilter(lambda x: 'fp16 is not a valid' not in x.getMessage())
    
    path = config.cache_dir
    model = model_class.from_pretrained(
        model_name,
        cache_dir=path,
        resume_download=True,
        **kwargs,
    )
    model_name = "--".join(("models", *model_name.split("/")))
    return path / model_name if model else None


def _download_diffusion_weights(
    mconfig: DictConfig, access_token: str, precision: str = "float32"
):
    repo_id = mconfig["repo_id"]
    model_class = (
        StableDiffusionGeneratorPipeline
        if mconfig.get("format", None) == "diffusers"
        else AutoencoderKL
    )
    extra_arg_list = [{"revision": "fp16"}, {}] if precision == "float16" else [{}]
    path = None
    for extra_args in extra_arg_list:
        try:
            path = download_from_hf(
                model_class,
                repo_id,
                safety_checker=None,
                **extra_args,
            )
        except OSError as e:
            if 'Revision Not Found' in str(e):
                pass
            else:
                logger.error(str(e))
        if path:
            break
    return path


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

    if (
        resp.status_code == 416
    ):  # "range not satisfiable", which means nothing to return
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


# ---------------------------------------------
def update_config_file(successfully_downloaded: dict, config_file: Path):
    config_file = (
        Path(config_file) if config_file is not None else default_config_file()
    )

    # In some cases (incomplete setup, etc), the default configs directory might be missing.
    # Create it if it doesn't exist.
    # this check is ignored if opt.config_file is specified - user is assumed to know what they
    # are doing if they are passing a custom config file from elsewhere.
    if config_file is default_config_file() and not config_file.parent.exists():
        configs_src = Dataset_path.parent
        configs_dest = default_config_file().parent
        shutil.copytree(configs_src, configs_dest, dirs_exist_ok=True)

    yaml = new_config_file_contents(successfully_downloaded, config_file)

    try:
        backup = None
        if os.path.exists(config_file):
            logger.warning(
                f"{config_file.name} exists. Renaming to {config_file.stem}.yaml.orig"
            )
            backup = config_file.with_suffix(".yaml.orig")
            ## Ugh. Windows is unable to overwrite an existing backup file, raises a WinError 183
            if sys.platform == "win32" and backup.is_file():
                backup.unlink()
            config_file.rename(backup)

        with TemporaryFile() as tmp:
            tmp.write(Config_preamble.encode())
            tmp.write(yaml.encode())

            with open(str(config_file.expanduser().resolve()), "wb") as new_config:
                tmp.seek(0)
                new_config.write(tmp.read())

    except Exception as e:
        logger.error(f"Error creating config file {config_file}: {str(e)}")
        if backup is not None:
            logger.info("restoring previous config file")
            ## workaround, for WinError 183, see above
            if sys.platform == "win32" and config_file.is_file():
                config_file.unlink()
            backup.rename(config_file)
        return
    
    logger.info(f"Successfully created new configuration file {config_file}")


# ---------------------------------------------
def new_config_file_contents(
    successfully_downloaded: dict,
    config_file: Path,
) -> str:
    if config_file.exists():
        conf = OmegaConf.load(str(config_file.expanduser().resolve()))
    else:
        conf = OmegaConf.create()

    default_selected = None
    for model in successfully_downloaded:
        # a bit hacky - what we are doing here is seeing whether a checkpoint
        # version of the model was previously defined, and whether the current
        # model is a diffusers (indicated with a path)
        if conf.get(model) and Path(successfully_downloaded[model]).is_dir():
            delete_weights(model, conf[model])

        stanza = {}
        mod = initial_models()[model]
        stanza["description"] = mod["description"]
        stanza["repo_id"] = mod["repo_id"]
        stanza["format"] = mod["format"]
        # diffusers don't need width and height (probably .ckpt doesn't either)
        # so we no longer require these in INITIAL_MODELS.yaml
        if "width" in mod:
            stanza["width"] = mod["width"]
        if "height" in mod:
            stanza["height"] = mod["height"]
        if "file" in mod:
            stanza["weights"] = os.path.relpath(
                successfully_downloaded[model], start=config.root_dir
            )
            stanza["config"] = os.path.normpath(
                os.path.join(sd_configs(), mod["config"])
            )
        if "vae" in mod:
            if "file" in mod["vae"]:
                stanza["vae"] = os.path.normpath(
                    os.path.join(Model_dir, Weights_dir, mod["vae"]["file"])
                )
            else:
                stanza["vae"] = mod["vae"]
        if mod.get("default", False):
            stanza["default"] = True
            default_selected = True

        conf[model] = stanza

    # if no default model was chosen, then we select the first
    # one in the list
    if not default_selected:
        conf[list(successfully_downloaded.keys())[0]]["default"] = True

    return OmegaConf.to_yaml(conf)


# ---------------------------------------------
def delete_weights(model_name: str, conf_stanza: dict):
    if not (weights := conf_stanza.get("weights")):
        return
    if re.match("/VAE/", conf_stanza.get("config")):
        return

    logger.warning(
        f"\nThe checkpoint version of {model_name} is superseded by the diffusers version. Deleting the original file {weights}?"
    )

    weights = Path(weights)
    if not weights.is_absolute():
        weights = config.root_dir / weights
        try:
            weights.unlink()
        except OSError as e:
            logger.error(str(e))
