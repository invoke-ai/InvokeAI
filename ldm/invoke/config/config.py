#!/usr/bin/env python
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)
# Before running stable-diffusion on an internet-isolated machine,
# run this script from one with internet connectivity. The
# two machines must share a common .cache directory.
#
# Coauthors:
#   Kevin Turner https://github.com/keturn
#   Eugene Brodsky https://github.com/ebr
#

import argparse
import io
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Union
from urllib import request

import requests
import transformers
from diffusers import AutoencoderKL
from getpass_asterisk import getpass_asterisk
from huggingface_hub import HfFolder, hf_hub_url
from huggingface_hub import login as hf_hub_login
from huggingface_hub import whoami as hf_whoami
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from ldm.invoke.config.paths import InvokePaths
from ldm.invoke.config.runtime_dir import RuntimeDir
from ldm.invoke.devices import choose_precision, choose_torch_device
from ldm.invoke.generator.diffusers_pipeline import StableDiffusionGeneratorPipeline

warnings.filterwarnings("ignore")
import torch

transformers.logging.set_verbosity_error()

Paths = InvokePaths()
console = Console()

## TODO move this to the configfile manager
Config_preamble = """# This file describes the alternative machine learning models
# available to InvokeAI script.
#
# To add a new model, follow the examples below. Each
# model requires a model config file, a weights file,
# and the width and height of the images it
# was trained on.
"""

# ---------------------------------------------
def introduction():
    console.print(
        Panel(
            title="Welcome to InvokeAI",
            renderable="[bold]This script will help download the Stable Diffusion weight files and other large models that are needed for text to image generation. At any point you may interrupt this program and resume later.",
            style="grey23 on grey78",
        )
    )
    console.line()


# --------------------------------------------
def postscript(errors: None):
    if not any(errors):
        message_title = "Model Installation Successful"
        message = """
You're all set!

If you installed using one of the automated installation scripts,
execute 'invoke.sh' (Linux/macOS) or 'invoke.bat' (Windows) to
start InvokeAI.

If you installed manually, activate the 'invokeai' environment
(e.g. 'conda activate invokeai'), then run one of the following
commands to start InvokeAI.

Web UI:
    python scripts/invoke.py --web # (connect to http://localhost:9090)
Command-line interface:
   python scripts/invoke.py

Have fun!
"""
    else:
        message_title = "There were errors during installation"
        message = "It is possible some of the models were not fully downloaded."
        for err in errors:
            message += f"\t - {err}\n"
        message += "Please check the logs above and correct any issues."

    console.print(Panel(title=message_title, renderable=message))


# ---------------------------------------------
def user_wants_to_download_weights() -> str:
    """
    Returns one of "skip", "recommended" or "customized"
    """
    console.print(
        Panel(
            """You can download and configure the weights files manually or let this
script do it for you. Manual installation is described at:

https://invoke-ai.github.io/InvokeAI/installation/INSTALLING_MODELS/

You may download the recommended models (about 10GB total), select a customized set, or
completely skip this step.
"""
        )
    )

    choice = Prompt.ask(
        "Download <r>ecommended models, <a>ll models, <c>ustomized list, or <s>kip this step? [r]:",
        default="r",
        choices=["r", "a", "c", "s"],
    )

    if choice.startswith(("r", "R")) or len(choice) == 0:
        selection = "recommended"
    elif choice.startswith(("c", "C")):
        selection = "customized"
    elif choice.startswith(("a", "A")):
        selection = "all"
    elif choice.startswith(("s", "S")):
        selection = "skip"
    else:
        selection = "recommended"
    return selection


# ---------------------------------------------
def select_datasets(action: str):
    done = False
    while not done:
        datasets = dict()
        default = None  # the first model selected will be the default; TODO let user change
        counter = 1

        if action == "customized":
            print(
                """
Choose the weight file(s) you wish to download. Before downloading you
will be given the option to view and change your selections.
"""
            )
            for ds in Datasets.keys():
                recommended = Datasets[ds].get("recommended", False)
                r_str = "(recommended)" if recommended else ""
                print(f'[{counter}] {ds}:\n    {Datasets[ds]["description"]} {r_str}')
                if Confirm.ask("Download?", default=recommended):
                    datasets[ds] = counter
                    counter += 1
        else:
            for ds in Datasets.keys():
                if Datasets[ds].get("recommended", False):
                    datasets[ds] = counter
                    counter += 1

        print("The following weight files will be downloaded:")
        for ds in datasets:
            default = "*" if default is None else ""
            print(f"   [{datasets[ds]}] {ds}{default}")
        print("*default")
        ok_to_download = Confirm.ask("Ok to download?")
        if not ok_to_download:
            if Confirm.ask("Change your selection?"):
                action = "customized"
                pass
            else:
                done = True
        else:
            done = True
    return datasets if ok_to_download else None


# ---------------------------------------------
def recommended_datasets() -> dict:
    datasets = dict()
    for ds in Datasets.keys():
        if Datasets[ds].get("recommended", False):
            datasets[ds] = True
    return datasets


# ---------------------------------------------
def all_datasets() -> dict:
    datasets = dict()
    for ds in Datasets.keys():
        datasets[ds] = True
    return datasets


# ---------------------------------------------
def HfLogin(access_token) -> str:
    """
    Helper for logging in to Huggingface
    The stdout capture is needed to hide the irrelevant "git credential helper" warning
    """

    capture = io.StringIO()
    sys.stdout = capture
    try:
        hf_hub_login(token=access_token, add_to_git_credential=False)
        sys.stdout = sys.__stdout__
    except Exception as exc:
        sys.stdout = sys.__stdout__
        print(exc)
        raise exc


# -------------------------------Authenticate against Hugging Face
def authenticate(yes_to_all=False):
    console.print(
        Panel(
            title="License Agreement for Weight Files",
            renderable="""
By downloading the Stable Diffusion weight files from the official Hugging Face
repository, you agree to have read and accepted the CreativeML Responsible AI License.
The license terms are located here:

   https://huggingface.co/spaces/CompVis/stable-diffusion-license

""",
        )
    )

    if not yes_to_all:
        accepted = False
        while not accepted:
            accepted = Confirm.ask("Accept the above License terms?")
            if not accepted:
                console.print("Please accept the License or Ctrl+C to exit.")
            else:
                console.print("Thank you!")
    else:
        console.print(
            "The program was started with a '--yes' flag, which indicates user's acceptance of the above License terms."
        )

    # Authenticate to Huggingface using environment variables.
    # If successful, authentication will persist for either interactive or non-interactive use.
    # Default env var expected by HuggingFace is HUGGING_FACE_HUB_TOKEN.
    console.rule(title="Authenticating to Huggingface")
    hf_envvars = ["HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"]
    token_found = False
    if not (access_token := HfFolder.get_token()):
        console.print(f"Huggingface token not found in cache.")

        for ev in hf_envvars:
            if access_token := os.getenv(ev):
                console.print(f"Token was found in the {ev} environment variable.... Logging in.")
                try:
                    HfLogin(access_token)
                    continue
                except ValueError:
                    console.print(f"Login failed due to invalid token found in {ev}")
            else:
                console.print(f"Token was not found in the environment variable {ev}.")
    else:
        console.print(f"Huggingface token found in cache.")
        try:
            HfLogin(access_token)
            token_found = True
        except ValueError:
            console.print(f"Login failed due to invalid token found in cache")
            HfFolder.delete_token()

    if not yes_to_all:
        console.print(
            Panel(
                """
You may optionally enter your Huggingface token now. InvokeAI *will* work without it, but some functionality may be limited.
See https://invoke-ai.github.io/InvokeAI/features/CONCEPTS/#using-a-hugging-face-concept for more information.

Visit https://huggingface.co/settings/tokens to generate a token. (Sign up for an account if needed).

Paste the token below using Ctrl-Shift-V (macOS/Linux) or right-click (Windows), and/or 'Enter' to continue.
You may re-run the configuration script again in the future if you do not wish to set the token right now.
        """
            )
        )
        again = True
        while again:
            try:
                access_token = getpass_asterisk.getpass_asterisk(prompt="HF Token ⮞ ")
                HfLogin(access_token)
                access_token = HfFolder.get_token()
                again = False
            except ValueError:
                again = Confirm.ask("Failed to log in to Huggingface. Would you like to try again?")
                if not again:
                    console.line()
                    console.print("Re-run the configuration script whenever you wish to set the token.")
                    console.print("...Continuing...")
            except EOFError:
                # this happens if the user pressed Enter on the prompt without any input; assume this means they don't want to input a token
                # safety net needed against accidental "Enter"?
                console.print("None provided - continuing")
                again = False

    elif access_token is None:
        console.print(
            Panel(
                f"HuggingFace login did not succeed. Some functionality may be limited; see https://invoke-ai.github.io/InvokeAI/features/CONCEPTS/#using-a-hugging-face-concept for more information. Re-run the configuration script without '--yes' to set the HuggingFace token interactively, or use one of the environment variables: {', '.join(hf_envvars)}"
            )
        )

    console.rule()

    return access_token


# ---------------------------------------------
# look for legacy model.ckpt in models directory and offer to
# normalize its name
def migrate_models_ckpt():

    model = Paths.default_weights.location / "model.ckpt"
    if not model.is_file():
        return
    new_name = Datasets["stable-diffusion-1.4"]["file"]
    print('You seem to have the Stable Diffusion v4.1 "model.ckpt" already installed.')
    rename = Confirm.ask(f'Ok to rename it to "{new_name}" for future reference?')
    if rename:
        print(f"model.ckpt => {new_name}")
        os.replace(model, Paths.default_weights.location / new_name)


# ---------------------------------------------
def download_weight_datasets(models: dict, access_token: str, precision: str = "float32"):
    migrate_models_ckpt()
    successful = dict()
    for mod in models.keys():
        print(f"{mod}...", file=sys.stderr, end="")
        successful[mod] = _download_repo_or_file(Datasets[mod], access_token, precision=precision)
    return successful


def _download_repo_or_file(mconfig: DictConfig, access_token: str, precision: str = "float32") -> Path:
    path = None
    if mconfig["format"] == "ckpt":
        path = _download_ckpt_weights(mconfig, access_token)
    else:
        path = _download_diffusion_weights(mconfig, access_token, precision=precision)
        if "vae" in mconfig and "repo_id" in mconfig["vae"]:
            _download_diffusion_weights(mconfig["vae"], access_token, precision=precision)
    return path


def _download_ckpt_weights(mconfig: DictConfig, access_token: str) -> Path:
    repo_id = mconfig["repo_id"]
    filename = mconfig["file"]
    cache_dir = Paths.default_weights.location
    return hf_download_with_resume(repo_id=repo_id, model_dir=cache_dir, model_name=filename, access_token=access_token)


def _download_diffusion_weights(mconfig: DictConfig, access_token: str, precision: str = "float32"):
    repo_id = mconfig["repo_id"]
    model_class = StableDiffusionGeneratorPipeline if mconfig.get("format", None) == "diffusers" else AutoencoderKL
    extra_arg_list = [{"revision": "fp16"}, {}] if precision == "float16" else [{}]
    model = None
    for extra_args in extra_arg_list:
        try:
            model = download_from_hf(
                model_class,
                repo_id,
                safety_checker=None,
                **extra_args,
            )
        except OSError as e:
            if str(e).startswith("fp16 is not a valid"):
                print(f"Could not fetch half-precision version of model {repo_id}; fetching full-precision instead")
            else:
                print(f"An unexpected error occurred while downloading the model: {e})")
        if model:
            break
    return model


# ---------------------------------------------
def hf_download_with_resume(repo_id: str, model_dir: str, model_name: str, access_token: str = None) -> Path:
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

    if resp.status_code == 416:  # "range not satisfiable", which means nothing to return
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


# -----------------------------------------------------------------------------------
# ---------------------------------------------
def is_huggingface_authenticated():
    # huggingface_hub 0.10 API isn't great for this, it could be OSError, ValueError,
    # maybe other things, not all end-user-friendly.
    # noinspection PyBroadException
    try:
        response = hf_whoami()
        if response.get("id") is not None:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------
def download_with_progress_bar(model_url: str, model_dest: str, label: str = "the"):
    try:
        print(f"Installing {label} model file {model_url}...", end="", file=sys.stderr)
        if not os.path.exists(model_dest):
            os.makedirs(os.path.dirname(model_dest), exist_ok=True)
            print("", file=sys.stderr)
            request.urlretrieve(model_url, model_dest, ProgressBar(os.path.basename(model_dest)))
            print("...downloaded successfully", file=sys.stderr)
        else:
            print("...exists", file=sys.stderr)
    except Exception:
        print("...download failed")
        print(f"Error downloading {label} model")
        print(traceback.format_exc())


# ---------------------------------------------
def update_config_file(successfully_downloaded: dict):
    config_file = Paths.config.location

    yaml = new_config_file_contents(successfully_downloaded)

    try:
        if config_file.exists():
            print(f"** {config_file} exists. Renaming to {config_file}.orig")
            os.replace(config_file, f"{config_file}.orig")
        tmpfile = config_file.parent / "new_config.tmp"
        with open(tmpfile, "w") as outfile:
            outfile.write(Config_preamble)
            outfile.write(yaml)
        os.replace(tmpfile, config_file)

    except Exception as e:
        print(f"**Error creating config file {config_file}: {str(e)} **")
        return

    print(f"Successfully created new configuration file {config_file}")


# ---------------------------------------------
def new_config_file_contents(successfully_downloaded: dict) -> str:
    config_file = Paths.config.location
    if config_file.exists():
        conf = OmegaConf.load(config_file)
    else:
        conf = OmegaConf.create()

    # find the VAE file, if there is one
    vaes = {}
    default_selected = False

    for model in successfully_downloaded:
        stanza = conf[model] if model in conf else {}
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
            stanza["weights"] = os.path.relpath(successfully_downloaded[model], start=Paths.root)
            stanza["config"] = Paths.sd_configs_dir.location / mod["config"]
        if "vae" in mod:
            if "file" in mod["vae"]:
                stanza["vae"] = Paths.default_weights / mod["vae"]["file"]
            else:
                stanza["vae"] = mod["vae"]
        stanza.pop("default", None)  # this will be set later

        # BUG - the first stanza is always the default. User should select.
        if not default_selected:
            stanza["default"] = True
            default_selected = True
        conf[model] = stanza
    return OmegaConf.to_yaml(conf)


# ---------------------------------------------
# this will preload the Bert tokenizer fles
def download_bert():
    print("Installing bert tokenizer (ignore deprecation errors)...", end="", file=sys.stderr)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from transformers import BertTokenizerFast

        download_from_hf(BertTokenizerFast, "bert-base-uncased")
        print("...success", file=sys.stderr)


# ---------------------------------------------
def download_from_hf(model_class: object, model_name: str, **kwargs):
    print("", file=sys.stderr)  # to prevent tqdm from overwriting
    return model_class.from_pretrained(
        model_name,
        cache_dir=Paths.models_dir.location / model_name,
        resume_download=True,
        **kwargs,
    )


# ---------------------------------------------
def download_clip():
    print("Installing CLIP model (ignore deprecation errors)...", file=sys.stderr)
    version = "openai/clip-vit-large-patch14"
    print("Tokenizer...", file=sys.stderr, end="")
    download_from_hf(CLIPTokenizer, version)
    print("Text model...", file=sys.stderr, end="")
    download_from_hf(CLIPTextModel, version)
    print("...success", file=sys.stderr)


# ---------------------------------------------
def download_realesrgan():
    print("Installing models from RealESRGAN...", file=sys.stderr)
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    model_dest = Paths.models_dir.location / "realesrgan/realesr-general-x4v3.pth"
    download_with_progress_bar(model_url, model_dest, "RealESRGAN")


def download_gfpgan():
    print("Installing GFPGAN models...", file=sys.stderr)
    for model in (
        ["https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth", "gfpgan/GFPGANv1.4.pth"],
        [
            "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            "gfpgan/weights/detection_Resnet50_Final.pth",
        ],
        [
            "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
            "gfpgan/weights/parsing_parsenet.pth",
        ],
    ):
        model_url, model_dest = model[0], Paths.models_dir.location / model[1]
        download_with_progress_bar(model_url, model_dest, "GFPGAN weights")


# ---------------------------------------------
def download_codeformer():
    print("Installing CodeFormer model file...", file=sys.stderr)
    model_url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    model_dest = Paths.models_dir.location / "codeformer/codeformer.pth"
    download_with_progress_bar(model_url, model_dest, "CodeFormer")


# ---------------------------------------------
def download_clipseg():
    print("Installing clipseg model for text-based masking...", end="", file=sys.stderr)
    import zipfile

    try:
        model_url = "https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download"
        model_dest = Paths.models_dir.location / "clipseg/clipseg_weights"
        weights_zip = "clipseg/weights.zip"

        if not model_dest.is_dir():
            model_dest.mkdir(exist_ok=True, parents=True)
        if not (model_dest / "rd64-uni-refined.pth").is_file():
            dest = Paths.models_dir.location / weights_zip
            request.urlretrieve(model_url, dest)
            with zipfile.ZipFile(dest, "r") as zip:
                zip.extractall(Paths.models_dir.location / "clipseg")
            os.remove(str(dest.expanduser().absolute()))

            from clipseg.clipseg import CLIPDensePredT

            model = CLIPDensePredT(
                version="ViT-B/16",
                reduce_dim=64,
            )
            model.eval()
            model.load_state_dict(
                torch.load(
                    Paths.models_dir.location / "clipseg/clipseg_weights/rd64-uni-refined.pth",
                    map_location=torch.device("cpu"),
                ),
                strict=False,
            )
    except Exception:
        print("Error installing clipseg model:")
        print(traceback.format_exc())
    print("...success", file=sys.stderr)


# -------------------------------------
def download_safety_checker():
    print("Installing model for NSFW content detection...", file=sys.stderr)
    try:
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        from transformers import AutoFeatureExtractor
    except ModuleNotFoundError:
        print("Error installing NSFW checker model:")
        print(traceback.format_exc())
        return
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    print("AutoFeatureExtractor...", end="", file=sys.stderr)
    download_from_hf(AutoFeatureExtractor, safety_model_id)
    print("StableDiffusionSafetyChecker...", end="", file=sys.stderr)
    download_from_hf(StableDiffusionSafetyChecker, safety_model_id)
    print("...success", file=sys.stderr)


# -------------------------------------
def download_weights(opt: dict) -> Union[str, None]:

    precision = "float32" if opt.full_precision else choose_precision(torch.device(choose_torch_device()))

    if opt.yes_to_all:
        models = recommended_datasets()
        access_token = authenticate(opt.yes_to_all)
        if len(models) > 0:
            successfully_downloaded = download_weight_datasets(models, access_token, precision=precision)
            update_config_file(successfully_downloaded, opt)
            return

    else:
        choice = user_wants_to_download_weights()

    if choice == "recommended":
        models = recommended_datasets()
    elif choice == "all":
        models = all_datasets()
    elif choice == "customized":
        models = select_datasets(choice)
        if models is None and Confirm.ask("Quit?", default_yes=False):
            sys.exit(0)
    else:  # 'skip'
        return

    # We are either already authenticated, or will be asked to provide the token interactively
    access_token = authenticate()
    console.rule("DOWNLOADING WEIGHTS")
    successfully_downloaded = download_weight_datasets(models, access_token, precision=precision)

    update_config_file(successfully_downloaded)
    if len(successfully_downloaded) < len(models):
        return "some of the model weights downloads were not successful"


# -------------------------------------
class ProgressBar:
    def __init__(self, model_name="file"):
        self.pbar = None
        self.name = model_name

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(desc=self.name, initial=0, unit="iB", unit_scale=True, unit_divisor=1000, total=total_size)
        self.pbar.update(block_size)


# -------------------------------------
def main():
    parser = argparse.ArgumentParser(description="InvokeAI model downloader")
    parser.add_argument(
        "--interactive",
        dest="interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run in interactive mode (default) - DEPRECATED",
    )
    parser.add_argument(
        "--skip-sd-weights",
        dest="skip_sd_weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip downloading the large Stable Diffusion weight files",
    )
    parser.add_argument(
        "--full-precision",
        dest="full_precision",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="use 32-bit weights instead of faster 16-bit weights",
    )
    parser.add_argument("--yes", "-y", dest="yes_to_all", action="store_true", help='answer "yes" to all prompts')
    parser.add_argument(
        "--config_file", "-c", dest="config_file", type=str, default=None, help="path to configuration file to create"
    )
    parser.add_argument("--root_dir", dest="root", type=str, default=None, help="path to root of install directory")
    parser.add_argument("--outdir", dest="outdir", type=str, default=None, help="path to image output directory")

    opt = parser.parse_args()

    try:
        introduction()

        Paths = InvokePaths()

        Paths.root = opt.root
        Paths.config = opt.config_file
        Paths.outdir = opt.outdir

        # WHY do we need to pass Paths here if it's a singleton???
        runtime_dir = RuntimeDir(Paths)

        if not runtime_dir.validate():
            runtime_dir.initialize(yes_to_all=opt.yes_to_all)

        global Datasets
        Datasets = OmegaConf.load(Paths.initial_models_config.location)

        # Optimistically try to download all required assets. If any errors occur, add them and proceed anyway.
        errors = set()

        if not opt.interactive:
            console.print(
                "WARNING: The --(no)-interactive argument is deprecated and will be removed. Use --skip-sd-weights."
            )
            opt.skip_sd_weights = True
        if opt.skip_sd_weights:
            console.rule("SKIPPING DIFFUSION WEIGHTS DOWNLOAD PER USER REQUEST")
        else:
            console.rule("DOWNLOADING DIFFUSION WEIGHTS")
            errors.add(download_weights(opt))
        console.rule("DOWNLOADING SUPPORT MODELS")
        download_bert()
        download_clip()
        download_realesrgan()
        download_gfpgan()
        download_codeformer()
        download_clipseg()
        download_safety_checker()
        postscript(errors=errors)
    except KeyboardInterrupt:
        console.print("Goodbye! Come back soon.")
    except Exception as e:
        console.print(f'A problem occurred during initialization.\nThe error was: "{str(e)}"')
        print(traceback.format_exc())


# -------------------------------------
if __name__ == "__main__":
    main()
