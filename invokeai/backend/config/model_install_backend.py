"""
Utility (backend) functions used by model_install.py
"""
import os
import re
import shutil
import sys
import warnings
from pathlib import Path
from tempfile import TemporaryFile
from typing import List

import requests
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_url
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

import invokeai.configs as configs

from invokeai.app.services.config import get_invokeai_config
from ..model_management import ModelManager
from ..stable_diffusion import StableDiffusionGeneratorPipeline


warnings.filterwarnings("ignore")

# --------------------------globals-----------------------
config = get_invokeai_config(argv=[])
Model_dir = "models"
Weights_dir = "ldm/stable-diffusion-v1/"

# the initial "configs" dir is now bundled in the `invokeai.configs` package
Dataset_path = Path(configs.__path__[0]) / "INITIAL_MODELS.yaml"

# initial models omegaconf
Datasets = None

Config_preamble = """
# This file describes the alternative machine learning models
# available to InvokeAI script.
#
# To add a new model, follow the examples below. Each
# model requires a model config file, a weights file,
# and the width and height of the images it
# was trained on.
"""


def default_config_file():
    return config.model_conf_path


def sd_configs():
    return config.legacy_conf_path

def initial_models():
    global Datasets
    if Datasets:
        return Datasets
    return (Datasets := OmegaConf.load(Dataset_path))


def install_requested_models(
    install_initial_models: List[str] = None,
    remove_models: List[str] = None,
    scan_directory: Path = None,
    external_models: List[str] = None,
    scan_at_startup: bool = False,
    precision: str = "float16",
    purge_deleted: bool = False,
    config_file_path: Path = None,
):
    """
    Entry point for installing/deleting starter models, or installing external models.
    """
    config_file_path = config_file_path or default_config_file()
    if not config_file_path.exists():
        open(config_file_path, "w")

    model_manager = ModelManager(OmegaConf.load(config_file_path), precision=precision)

    if remove_models and len(remove_models) > 0:
        print("== DELETING UNCHECKED STARTER MODELS ==")
        for model in remove_models:
            print(f"{model}...")
            model_manager.del_model(model, delete_files=purge_deleted)
        model_manager.commit(config_file_path)

    if install_initial_models and len(install_initial_models) > 0:
        print("== INSTALLING SELECTED STARTER MODELS ==")
        successfully_downloaded = download_weight_datasets(
            models=install_initial_models,
            access_token=None,
            precision=precision,
        )  # FIX: for historical reasons, we don't use model manager here
        update_config_file(successfully_downloaded, config_file_path)
        if len(successfully_downloaded) < len(install_initial_models):
            print("** Some of the model downloads were not successful")

    # due to above, we have to reload the model manager because conf file
    # was changed behind its back
    model_manager = ModelManager(OmegaConf.load(config_file_path), precision=precision)

    external_models = external_models or list()
    if scan_directory:
        external_models.append(str(scan_directory))

    if len(external_models) > 0:
        print("== INSTALLING EXTERNAL MODELS ==")
        for path_url_or_repo in external_models:
            try:
                model_manager.heuristic_import(
                    path_url_or_repo,
                    commit_to_conf=config_file_path,
                )
            except KeyboardInterrupt:
                sys.exit(-1)
            except Exception:
                pass

    if scan_at_startup and scan_directory.is_dir():
        argument = "--autoconvert"
        print('** The global initfile is no longer supported; rewrite to support new yaml format **')
        initfile = Path(config.root, 'invokeai.init')
        replacement = Path(config.root, f"invokeai.init.new")
        directory = str(scan_directory).replace("\\", "/")
        with open(initfile, "r") as input:
            with open(replacement, "w") as output:
                while line := input.readline():
                    if not line.startswith(argument):
                        output.writelines([line])
                output.writelines([f"{argument} {directory}"])
        os.replace(replacement, initfile)


# -------------------------------------
def yes_or_no(prompt: str, default_yes=True):
    default = "y" if default_yes else "n"
    response = input(f"{prompt} [{default}] ") or default
    if default_yes:
        return response[0] not in ("n", "N")
    else:
        return response[0] in ("y", "Y")


# -------------------------------------
def get_root(root: str = None) -> str:
    if root:
        return root
    elif os.environ.get("INVOKEAI_ROOT"):
        return os.environ.get("INVOKEAI_ROOT")
    else:
        return config.root


# ---------------------------------------------
def recommended_datasets() -> dict:
    datasets = dict()
    for ds in initial_models().keys():
        if initial_models()[ds].get("recommended", False):
            datasets[ds] = True
    return datasets


# ---------------------------------------------
def default_dataset() -> dict:
    datasets = dict()
    for ds in initial_models().keys():
        if initial_models()[ds].get("default", False):
            datasets[ds] = True
    return datasets


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
    model_path = os.path.join(config.root, Model_dir, Weights_dir)
    if not os.path.exists(os.path.join(model_path, "model.ckpt")):
        return
    new_name = initial_models()["stable-diffusion-1.4"]["file"]
    print(
        'The Stable Diffusion v4.1 "model.ckpt" is already installed. The name will be changed to {new_name} to avoid confusion.'
    )
    print(f"model.ckpt => {new_name}")
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
        print(f"Downloading {mod}:")
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
    cache_dir = os.path.join(config.root, Model_dir, Weights_dir)
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
            if str(e).startswith("fp16 is not a valid"):
                pass
            else:
                print(f"An unexpected error occurred while downloading the model: {e})")
        if path:
            break
    return path


# ---------------------------------------------
def hf_download_with_resume(
    repo_id: str, model_dir: str, model_name: str, access_token: str = None
) -> Path:
    model_dest = Path(os.path.join(model_dir, model_name))
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
        print(f"* {model_name}: complete file found. Skipping.")
        return model_dest
    elif resp.status_code != 200:
        print(f"** An error occurred during downloading {model_name}: {resp.reason}")
    elif exist_size > 0:
        print(f"* {model_name}: partial file found. Resuming...")
    else:
        print(f"* {model_name}: Downloading...")

    try:
        if total < 2000:
            print(f"*** ERROR DOWNLOADING {model_name}: {resp.text}")
            return None

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
        print(f"An error occurred while downloading {model_name}: {str(e)}")
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
            print(
                f"** {config_file.name} exists. Renaming to {config_file.stem}.yaml.orig"
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
        print(f"**Error creating config file {config_file}: {str(e)} **")
        if backup is not None:
            print("restoring previous config file")
            ## workaround, for WinError 183, see above
            if sys.platform == "win32" and config_file.is_file():
                config_file.unlink()
            backup.rename(config_file)
        return

    print(f"Successfully created new configuration file {config_file}")


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
                successfully_downloaded[model], start=config.root
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

    print(
        f"\n** The checkpoint version of {model_name} is superseded by the diffusers version. Deleting the original file {weights}?"
    )

    weights = Path(weights)
    if not weights.is_absolute():
        weights = Path(config.root) / weights
        try:
            weights.unlink()
        except OSError as e:
            print(str(e))
