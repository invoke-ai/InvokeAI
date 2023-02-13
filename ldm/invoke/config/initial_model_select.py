#!/usr/bin/env python
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)
# Before running stable-diffusion on an internet-isolated machine,
# run this script from one with internet connectivity. The
# two machines must share a common .cache directory.
#
# Coauthor: Kevin Turner http://github.com/keturn
#
import argparse
import os
import re
import shutil
import sys
import traceback
import warnings
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryFile

import npyscreen
import requests
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_url
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

import invokeai.configs as configs
from ldm.invoke.devices import choose_precision, choose_torch_device
from ldm.invoke.generator.diffusers_pipeline import StableDiffusionGeneratorPipeline
from ldm.invoke.globals import Globals, global_cache_dir, global_config_dir
from ldm.invoke.readline import generic_completer

warnings.filterwarnings("ignore")
import torch

# --------------------------globals-----------------------
Model_dir = "models"
Weights_dir = "ldm/stable-diffusion-v1/"

# the initial "configs" dir is now bundled in the `invokeai.configs` package
Dataset_path = Path(configs.__path__[0]) / "INITIAL_MODELS.yaml"

Default_config_file = Path(global_config_dir()) / "models.yaml"
SD_Configs = Path(global_config_dir()) / "stable-diffusion"

Datasets = OmegaConf.load(Dataset_path)
completer = generic_completer(["yes", "no"])

Config_preamble = """# This file describes the alternative machine learning models
# available to InvokeAI script.
#
# To add a new model, follow the examples below. Each
# model requires a model config file, a weights file,
# and the width and height of the images it
# was trained on.
"""


# -------------------------------------
def get_root(root: str = None) -> str:
    if root:
        return root
    elif os.environ.get("INVOKEAI_ROOT"):
        return os.environ.get("INVOKEAI_ROOT")
    else:
        return Globals.root


class addRemoveModelsForm(npyscreen.FormMultiPageAction):
    def __init__(self, parentApp, name):
        self.initial_models = OmegaConf.load(Dataset_path)
        try:
            self.existing_models = OmegaConf.load(Default_config_file)
        except:
            self.existing_models = dict()
        self.starter_model_list = [
            x for x in list(self.initial_models.keys()) if x not in self.existing_models
        ]
        super().__init__(parentApp, name)

    def create(self):
        starter_model_labels = [
            "%-30s %-50s" % (x, self.initial_models[x].description)
            for x in self.starter_model_list
        ]
        recommended_models = [
            x
            for x in self.starter_model_list
            if self.initial_models[x].get("recommended", False)
        ]
        previously_installed_models = [
            x for x in list(self.initial_models.keys()) if x in self.existing_models
        ]
        self.add_widget_intelligent(
            npyscreen.TitleText,
            name="This is a starter set of Stable Diffusion models from HuggingFace",
            editable=False,
            color="CONTROL",
        )
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="Select models to install:",
            editable=False,
            color="LABELBOLD",
        )
        self.add_widget_intelligent(npyscreen.FixedText, value="", editable=False),
        self.models_selected = self.add_widget_intelligent(
            npyscreen.MultiSelect,
            name="Install/Remove Models",
            values=starter_model_labels,
            value=[
                self.starter_model_list.index(x)
                for x in self.initial_models
                if x in recommended_models
            ],
            max_height=len(starter_model_labels) + 1,
            scroll_exit=True,
        )
        if len(previously_installed_models) > 0:
            self.add_widget_intelligent(
                npyscreen.TitleText,
                name="These starter models are already installed. Use the command-line or Web UIs to manage them:",
                editable=False,
                color="CONTROL",
            )
            for m in previously_installed_models:
                self.add_widget_intelligent(
                    npyscreen.FixedText,
                    value=m,
                    editable=False,
                    relx=10,
                )
        self.models_selected.editing = True

    def on_ok(self):
        self.parentApp.setNextForm(None)
        self.editing = False
        self.parentApp.selected_models = [
            self.starter_model_list[x] for x in self.models_selected.value
        ]
        npyscreen.notify(f"Installing selected {self.parentApp.selected_models}")

    def on_cancel(self):
        self.parentApp.setNextForm(None)
        self.parentApp.selected_models = None
        self.editing = False


class AddRemoveModelApplication(npyscreen.NPSAppManaged):
    def __init__(self, saved_args=None):
        super().__init__()
        self.models_to_install = None

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.DefaultTheme)
        self.main = self.addForm(
            "MAIN",
            addRemoveModelsForm,
            name="Add/Remove Models",
        )


# ---------------------------------------------
def yes_or_no(prompt: str, default_yes=True):
    completer.set_options(["yes", "no"])
    completer.complete_extensions(None)  # turn off path-completion mode
    default = "y" if default_yes else "n"
    response = input(f"{prompt} [{default}] ") or default
    if default_yes:
        return response[0] not in ("n", "N")
    else:
        return response[0] in ("y", "Y")


# ---------------------------------------------
def recommended_datasets() -> dict:
    datasets = dict()
    for ds in Datasets.keys():
        if Datasets[ds].get("recommended", False):
            datasets[ds] = True
    return datasets


# ---------------------------------------------
def default_dataset() -> dict:
    datasets = dict()
    for ds in Datasets.keys():
        if Datasets[ds].get("default", False):
            datasets[ds] = True
    return datasets


# ---------------------------------------------
def all_datasets() -> dict:
    datasets = dict()
    for ds in Datasets.keys():
        datasets[ds] = True
    return datasets


# ---------------------------------------------
# look for legacy model.ckpt in models directory and offer to
# normalize its name
def migrate_models_ckpt():
    model_path = os.path.join(Globals.root, Model_dir, Weights_dir)
    if not os.path.exists(os.path.join(model_path, "model.ckpt")):
        return
    new_name = Datasets["stable-diffusion-1.4"]["file"]
    print('You seem to have the Stable Diffusion v4.1 "model.ckpt" already installed.')
    rename = yes_or_no(f'Ok to rename it to "{new_name}" for future reference?')
    if rename:
        print(f"model.ckpt => {new_name}")
        os.replace(
            os.path.join(model_path, "model.ckpt"), os.path.join(model_path, new_name)
        )


# ---------------------------------------------
def download_weight_datasets(
    models: dict, access_token: str, precision: str = "float32"
):
    migrate_models_ckpt()
    successful = dict()
    for mod in models.keys():
        print(f"Downloading {mod}:")
        successful[mod] = _download_repo_or_file(
            Datasets[mod], access_token, precision=precision
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
    cache_dir = os.path.join(Globals.root, Model_dir, Weights_dir)
    return hf_download_with_resume(
        repo_id=repo_id,
        model_dir=cache_dir,
        model_name=filename,
        access_token=access_token,
    )


# ---------------------------------------------
def download_from_hf(
    model_class: object, model_name: str, cache_subdir: Path = Path("hub"), **kwargs
):
    print("", file=sys.stderr)  # to prevent tqdm from overwriting
    path = global_cache_dir(cache_subdir)
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
                cache_subdir="diffusers",
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
def update_config_file(successfully_downloaded: dict, opt: dict):
    config_file = (
        Path(opt.config_file) if opt.config_file is not None else Default_config_file
    )

    # In some cases (incomplete setup, etc), the default configs directory might be missing.
    # Create it if it doesn't exist.
    # this check is ignored if opt.config_file is specified - user is assumed to know what they
    # are doing if they are passing a custom config file from elsewhere.
    if config_file is Default_config_file and not config_file.parent.exists():
        configs_src = Dataset_path.parent
        configs_dest = Default_config_file.parent
        shutil.copytree(configs_src, configs_dest, dirs_exist_ok=True)

    yaml = new_config_file_contents(successfully_downloaded, config_file, opt)

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
    successfully_downloaded: dict, config_file: Path, opt: dict
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
            offer_to_delete_weights(model, conf[model], opt.yes_to_all)

        stanza = {}
        mod = Datasets[model]
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
                successfully_downloaded[model], start=Globals.root
            )
            stanza["config"] = os.path.normpath(os.path.join(SD_Configs, mod["config"]))
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
def offer_to_delete_weights(model_name: str, conf_stanza: dict, yes_to_all: bool):
    if not (weights := conf_stanza.get("weights")):
        return
    if re.match("/VAE/", conf_stanza.get("config")):
        return
    if yes_to_all or yes_or_no(
        f"\n** The checkpoint version of {model_name} is superseded by the diffusers version. Delete the original file {weights}?",
        default_yes=False,
    ):
        weights = Path(weights)
        if not weights.is_absolute():
            weights = Path(Globals.root) / weights
        try:
            weights.unlink()
        except OSError as e:
            print(str(e))


# --------------------------------------------------------
def select_and_download_models(opt: Namespace):
    if opt.default_only:
        models_to_download = default_dataset()
    else:
        myapplication = AddRemoveModelApplication()
        myapplication.run()
        models_to_download = dict(map(lambda x: (x, True), myapplication.selected_models))

    if not models_to_download:
        print(
            '** No models were selected. To run this program again, select "Install initial models" from the invoke script.'
        )
        return

    print("** Downloading and installing the selected models.")
    precision = (
        "float32"
        if opt.full_precision
        else choose_precision(torch.device(choose_torch_device()))
    )
    successfully_downloaded = download_weight_datasets(
        models=models_to_download,
        access_token=None,
        precision=precision,
    )

    update_config_file(successfully_downloaded, opt)
    if len(successfully_downloaded) < len(models_to_download):
        print("** Some of the model downloads were not successful")

    print(
        "\nYour starting models were installed. To find and add more models, see https://invoke-ai.github.io/InvokeAI/installation/050_INSTALLING_MODELS"
    )


# -------------------------------------
def main():
    parser = argparse.ArgumentParser(description="InvokeAI model downloader")
    parser.add_argument(
        "--full-precision",
        dest="full_precision",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="use 32-bit weights instead of faster 16-bit weights",
    )
    parser.add_argument(
        "--yes",
        "-y",
        dest="yes_to_all",
        action="store_true",
        help='answer "yes" to all prompts',
    )
    parser.add_argument(
        "--default_only",
        action="store_true",
        help="only install the default model",
    )
    parser.add_argument(
        "--config_file",
        "-c",
        dest="config_file",
        type=str,
        default=None,
        help="path to configuration file to create",
    )
    parser.add_argument(
        "--root_dir",
        dest="root",
        type=str,
        default=None,
        help="path to root of install directory",
    )
    opt = parser.parse_args()

    # setting a global here
    Globals.root = os.path.expanduser(get_root(opt.root) or "")

    try:
        select_and_download_models(opt)
    except KeyboardInterrupt:
        print("\nGoodbye! Come back soon.")
    except Exception as e:
        print(f'\nA problem occurred during initialization.\nThe error was: "{str(e)}"')
        print(traceback.format_exc())


# -------------------------------------
if __name__ == "__main__":
    main()
