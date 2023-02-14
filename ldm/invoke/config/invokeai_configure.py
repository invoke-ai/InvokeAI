#!/usr/bin/env python
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)
# Before running stable-diffusion on an internet-isolated machine,
# run this script from one with internet connectivity. The
# two machines must share a common .cache directory.
#
# Coauthor: Kevin Turner http://github.com/keturn
#
print("Loading Python libraries...\n")
import argparse
import io
import os
import shutil
import sys
import traceback
import warnings
from pathlib import Path
from urllib import request

import transformers
from getpass_asterisk import getpass_asterisk
from huggingface_hub import HfFolder
from huggingface_hub import login as hf_hub_login
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    CLIPSegForImageSegmentation,
    CLIPTextModel,
    CLIPTokenizer,
)

import invokeai.configs as configs
from ldm.invoke.config.model_install import (
    download_from_hf,
    select_and_download_models,
)
from ldm.invoke.globals import Globals, global_config_dir
from ldm.invoke.readline import generic_completer

warnings.filterwarnings("ignore")

transformers.logging.set_verbosity_error()

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


# --------------------------------------------
def postscript(errors: None):
    if not any(errors):
        message = f"""
** Model Installation Successful **

You're all set!

---
If you installed manually from source or with 'pip install': activate the virtual environment
then run one of the following commands to start InvokeAI.

Web UI:
   invokeai --web # (connect to http://localhost:9090)
   invokeai --web --host 0.0.0.0 # (connect to http://your-lan-ip:9090 from another computer on the local network)

Command-line interface:
   invokeai
---

If you installed using an installation script, run:

{Globals.root}/invoke.{"bat" if sys.platform == "win32" else "sh"}

Add the '--help' argument to see all of the command-line switches available for use.

Have fun!
"""

    else:
        message = "\n** There were errors during installation. It is possible some of the models were not fully downloaded.\n"
        for err in errors:
            message += f"\t - {err}\n"
        message += "Please check the logs above and correct any issues."

    print(message)

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
def save_hf_token(yes_to_all=False):
    print("** LICENSE AGREEMENT FOR WEIGHT FILES **")
    print("=" * shutil.get_terminal_size()[0])
    print(
        """
By downloading the Stable Diffusion weight files from the official Hugging Face
repository, you agree to have read and accepted the CreativeML Responsible AI License.
The license terms are located here:

   https://huggingface.co/spaces/CompVis/stable-diffusion-license

"""
    )
    print("=" * shutil.get_terminal_size()[0])

    if not yes_to_all:
        accepted = False
        while not accepted:
            accepted = yes_or_no("Accept the above License terms?")
            if not accepted:
                print("Please accept the License or Ctrl+C to exit.")
            else:
                print("Thank you!")
    else:
        print(
            "The program was started with a '--yes' flag, which indicates user's acceptance of the above License terms."
        )

    # Authenticate to Huggingface using environment variables.
    # If successful, authentication will persist for either interactive or non-interactive use.
    # Default env var expected by HuggingFace is HUGGING_FACE_HUB_TOKEN.
    print("=" * shutil.get_terminal_size()[0])
    print("Authenticating to Huggingface")
    hf_envvars = ["HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"]
    token_found = False
    if not (access_token := HfFolder.get_token()):
        print("Huggingface token not found in cache.")

        for ev in hf_envvars:
            if access_token := os.getenv(ev):
                print(
                    f"Token was found in the {ev} environment variable.... Logging in."
                )
                try:
                    HfLogin(access_token)
                    continue
                except ValueError:
                    print(f"Login failed due to invalid token found in {ev}")
            else:
                print(f"Token was not found in the environment variable {ev}.")
    else:
        print("Huggingface token found in cache.")
        try:
            HfLogin(access_token)
            token_found = True
        except ValueError:
            print("Login failed due to invalid token found in cache")

    if not (yes_to_all or token_found):
        print(
            f""" You may optionally enter your Huggingface token now. InvokeAI
*will* work without it but you will not be able to automatically
download some of the Hugging Face style concepts.  See
https://invoke-ai.github.io/InvokeAI/features/CONCEPTS/#using-a-hugging-face-concept
for more information.

Visit https://huggingface.co/settings/tokens to generate a token. (Sign up for an account if needed).

Paste the token below using {"Ctrl+Shift+V" if sys.platform == "linux" else "Command+V" if sys.platform == "darwin" else "Ctrl+V, right-click, or Edit>Paste"}.

Alternatively, press 'Enter' to skip this step and continue.

You may re-run the configuration script again in the future if you do not wish to set the token right now.
        """
        )
        again = True
        while again:
            try:
                access_token = getpass_asterisk.getpass_asterisk(prompt="HF Token ❯ ")
                if access_token is None or len(access_token) == 0:
                    raise EOFError
                HfLogin(access_token)
                access_token = HfFolder.get_token()
                again = False
            except ValueError:
                again = yes_or_no(
                    "Failed to log in to Huggingface. Would you like to try again?"
                )
                if not again:
                    print(
                        "\nRe-run the configuration script whenever you wish to set the token."
                    )
                    print("...Continuing...")
            except EOFError:
                # this happens if the user pressed Enter on the prompt without any input; assume this means they don't want to input a token
                # safety net needed against accidental "Enter"?
                print("None provided - continuing")
                again = False

    elif access_token is None:
        print()
        print(
            "HuggingFace login did not succeed. Some functionality may be limited; see https://invoke-ai.github.io/InvokeAI/features/CONCEPTS/#using-a-hugging-face-concept for more information"
        )
        print()
        print(
            f"Re-run the configuration script without '--yes' to set the HuggingFace token interactively, or use one of the environment variables: {', '.join(hf_envvars)}"
        )

    print("=" * shutil.get_terminal_size()[0])

    return access_token


# ---------------------------------------------
def download_with_progress_bar(model_url: str, model_dest: str, label: str = "the"):
    try:
        print(f"Installing {label} model file {model_url}...", end="", file=sys.stderr)
        if not os.path.exists(model_dest):
            os.makedirs(os.path.dirname(model_dest), exist_ok=True)
            print("", file=sys.stderr)
            request.urlretrieve(
                model_url, model_dest, ProgressBar(os.path.basename(model_dest))
            )
            print("...downloaded successfully", file=sys.stderr)
        else:
            print("...exists", file=sys.stderr)
    except Exception:
        print("...download failed")
        print(f"Error downloading {label} model")
        print(traceback.format_exc())

# ---------------------------------------------
# this will preload the Bert tokenizer fles
def download_bert():
    print(
        "Installing bert tokenizer (ignore deprecation errors)...",
        end="",
        file=sys.stderr,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from transformers import BertTokenizerFast

        download_from_hf(BertTokenizerFast, "bert-base-uncased")
        print("...success", file=sys.stderr)


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
    wdn_model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"

    model_dest = os.path.join(
        Globals.root, "models/realesrgan/realesr-general-x4v3.pth"
    )

    wdn_model_dest = os.path.join(
        Globals.root, "models/realesrgan/realesr-general-wdn-x4v3.pth"
    )

    download_with_progress_bar(model_url, model_dest, "RealESRGAN")
    download_with_progress_bar(wdn_model_url, wdn_model_dest, "RealESRGANwdn")


def download_gfpgan():
    print("Installing GFPGAN models...", file=sys.stderr)
    for model in (
        [
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            "./models/gfpgan/GFPGANv1.4.pth",
        ],
        [
            "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            "./models/gfpgan/weights/detection_Resnet50_Final.pth",
        ],
        [
            "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
            "./models/gfpgan/weights/parsing_parsenet.pth",
        ],
    ):
        model_url, model_dest = model[0], os.path.join(Globals.root, model[1])
        download_with_progress_bar(model_url, model_dest, "GFPGAN weights")


# ---------------------------------------------
def download_codeformer():
    print("Installing CodeFormer model file...", file=sys.stderr)
    model_url = (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    )
    model_dest = os.path.join(Globals.root, "models/codeformer/codeformer.pth")
    download_with_progress_bar(model_url, model_dest, "CodeFormer")


# ---------------------------------------------
def download_clipseg():
    print("Installing clipseg model for text-based masking...", end="", file=sys.stderr)
    CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"
    try:
        download_from_hf(AutoProcessor, CLIPSEG_MODEL)
        download_from_hf(CLIPSegForImageSegmentation, CLIPSEG_MODEL)
    except Exception:
        print("Error installing clipseg model:")
        print(traceback.format_exc())
    print("...success", file=sys.stderr)


# -------------------------------------
def download_safety_checker():
    print("Installing model for NSFW content detection...", file=sys.stderr)
    try:
        from diffusers.pipelines.stable_diffusion.safety_checker import (
            StableDiffusionSafetyChecker,
        )
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
def get_root(root: str = None) -> str:
    if root:
        return root
    elif os.environ.get("INVOKEAI_ROOT"):
        return os.environ.get("INVOKEAI_ROOT")
    else:
        return Globals.root


# -------------------------------------
def select_root(root: str, yes_to_all: bool = False):
    default = root or os.path.expanduser("~/invokeai")
    if yes_to_all:
        return default
    completer.set_default_dir(default)
    completer.complete_extensions(())
    completer.set_line(default)
    directory = input(
        f"Select a directory in which to install InvokeAI's models and configuration files [{default}]: "
    ).strip(" \\")
    return directory or default


# -------------------------------------
def select_outputs(root: str, yes_to_all: bool = False):
    default = os.path.normpath(os.path.join(root, "outputs"))
    if yes_to_all:
        return default
    completer.set_default_dir(os.path.expanduser("~"))
    completer.complete_extensions(())
    completer.set_line(default)
    directory = input(
        f"Select the default directory for image outputs [{default}]: "
    ).strip(" \\")
    return directory or default


# -------------------------------------
def initialize_rootdir(root: str, yes_to_all: bool = False):
    print("** INITIALIZING INVOKEAI RUNTIME DIRECTORY **")
    root_selected = False
    while not root_selected:
        outputs = select_outputs(root, yes_to_all)
        outputs = (
            outputs
            if os.path.isabs(outputs)
            else os.path.abspath(os.path.join(Globals.root, outputs))
        )

        print(f'\nInvokeAI image outputs will be placed into "{outputs}".')
        if not yes_to_all:
            root_selected = yes_or_no("Accept this location?")
        else:
            root_selected = True

    print(
        f'\nYou may change the chosen output directory at any time by editing the --outdir options in "{Globals.initfile}",'
    )
    print(
        "You may also change the runtime directory by setting the environment variable INVOKEAI_ROOT.\n"
    )

    enable_safety_checker = True
    if not yes_to_all:
        print(
            "The NSFW (not safe for work) checker blurs out images that potentially contain sexual imagery."
        )
        print(
            "It can be selectively enabled at run time with --nsfw_checker, and disabled with --no-nsfw_checker."
        )
        print(
            "The following option will set whether the checker is enabled by default. Like other options, you can"
        )
        print(f"change this setting later by editing the file {Globals.initfile}.")
        print(
            "This is NOT recommended for systems with less than 6G VRAM because of the checker's memory requirements."
        )
        enable_safety_checker = yes_or_no(
            "Enable the NSFW checker by default?", enable_safety_checker
        )

    safety_checker = "--nsfw_checker" if enable_safety_checker else "--no-nsfw_checker"

    for name in (
        "models",
        "configs",
        "embeddings",
        "text-inversion-data",
        "text-inversion-training-data",
    ):
        os.makedirs(os.path.join(root, name), exist_ok=True)

    configs_src = Path(configs.__path__[0])
    configs_dest = Path(root) / "configs"
    if not os.path.samefile(configs_src, configs_dest):
        shutil.copytree(configs_src, configs_dest, dirs_exist_ok=True)

    init_file = os.path.join(Globals.root, Globals.initfile)

    print(f'Creating the initialization file at "{init_file}".\n')
    with open(init_file, "w") as f:
        f.write(
            f"""# InvokeAI initialization file
# This is the InvokeAI initialization file, which contains command-line default values.
# Feel free to edit. If anything goes wrong, you can re-initialize this file by deleting
# or renaming it and then running invokeai-configure again.

# the --outdir option controls the default location of image files.
--outdir="{outputs}"

# generation arguments
{safety_checker}

# You may place other  frequently-used startup commands here, one or more per line.
# Examples:
# --web --host=0.0.0.0
# --steps=20
# -Ak_euler_a -C10.0
#
"""
        )


# -------------------------------------
class ProgressBar:
    def __init__(self, model_name="file"):
        self.pbar = None
        self.name = model_name

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(
                desc=self.name,
                initial=0,
                unit="iB",
                unit_scale=True,
                unit_divisor=1000,
                total=total_size,
            )
        self.pbar.update(block_size)


# -------------------------------------
def main():
    parser = argparse.ArgumentParser(description="InvokeAI model downloader")
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
        help="when --yes specified, only install the default model",
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

    errors = set()

    try:
        # We check for to see if the runtime directory is correctly initialized.
        if Globals.root == "" or not os.path.exists(
            os.path.join(Globals.root, "invokeai.init")
        ):
            initialize_rootdir(Globals.root, opt.yes_to_all)
            save_hf_token(opt.yes_to_all)

        print("\n** DOWNLOADING SUPPORT MODELS **")
        download_bert()
        download_clip()
        download_realesrgan()
        download_gfpgan()
        download_codeformer()
        download_clipseg()
        download_safety_checker()

        if opt.skip_sd_weights:
            print("** SKIPPING DIFFUSION WEIGHTS DOWNLOAD PER USER REQUEST **")
        else:
            print("** DOWNLOADING DIFFUSION WEIGHTS **")
            errors.add(select_and_download_models(opt))

        postscript(errors=errors)
    except KeyboardInterrupt:
        print("\nGoodbye! Come back soon.")
    except Exception as e:
        print(f'\nA problem occurred during initialization.\nThe error was: "{str(e)}"')
        print(traceback.format_exc())


# -------------------------------------
if __name__ == "__main__":
    main()
