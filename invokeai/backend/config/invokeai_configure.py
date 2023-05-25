#!/usr/bin/env python
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)
# Before running stable-diffusion on an internet-isolated machine,
# run this script from one with internet connectivity. The
# two machines must share a common .cache directory.
#
# Coauthor: Kevin Turner http://github.com/keturn
#
import sys
print("Loading Python libraries...\n",file=sys.stderr)

import argparse
import io
import os
import re
import shutil
import traceback
import warnings
from argparse import Namespace
from pathlib import Path
from shutil import get_terminal_size
from typing import get_type_hints
from urllib import request

import npyscreen
import transformers
from diffusers import AutoencoderKL
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

from invokeai.app.services.config import (
    get_invokeai_config,
    InvokeAIAppConfig,
)
from invokeai.frontend.install.model_install import addModelsForm, process_and_execute
from invokeai.frontend.install.widgets import (
    CenteredButtonPress,
    IntTitleSlider,
    set_min_terminal_size,
)

from invokeai.backend.config.legacy_arg_parsing import legacy_parser
from invokeai.backend.config.model_install_backend import (
    default_dataset,
    download_from_hf,
    hf_download_with_resume,
    recommended_datasets,
)

warnings.filterwarnings("ignore")

transformers.logging.set_verbosity_error()


# --------------------------globals-----------------------

config = get_invokeai_config(argv=[])

Model_dir = "models"
Weights_dir = "ldm/stable-diffusion-v1/"

# the initial "configs" dir is now bundled in the `invokeai.configs` package
Dataset_path = Path(configs.__path__[0]) / "INITIAL_MODELS.yaml"

Default_config_file = config.model_conf_path
SD_Configs = config.legacy_conf_path

Datasets = OmegaConf.load(Dataset_path)

# minimum size for the UI
MIN_COLS = 135
MIN_LINES = 45

PRECISION_CHOICES = ['auto','float16','float32','autocast']

INIT_FILE_PREAMBLE = """# InvokeAI initialization file
# This is the InvokeAI initialization file, which contains command-line default values.
# Feel free to edit. If anything goes wrong, you can re-initialize this file by deleting
# or renaming it and then running invokeai-configure again.
"""


# --------------------------------------------
def postscript(errors: None):
    if not any(errors):
        message = f"""
** INVOKEAI INSTALLATION SUCCESSFUL **
If you installed manually from source or with 'pip install': activate the virtual environment
then run one of the following commands to start InvokeAI.

Web UI:
   invokeai-web

Command-line client:
   invokeai

If you installed using an installation script, run:
  {config.root}/invoke.{"bat" if sys.platform == "win32" else "sh"}

Add the '--help' argument to see all of the command-line switches available for use.
"""

    else:
        message = "\n** There were errors during installation. It is possible some of the models were not fully downloaded.\n"
        for err in errors:
            message += f"\t - {err}\n"
        message += "Please check the logs above and correct any issues."

    print(message)


# ---------------------------------------------
def yes_or_no(prompt: str, default_yes=True):
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


# ---------------------------------------------
def download_with_progress_bar(model_url: str, model_dest: str, label: str = "the"):
    try:
        print(f"Installing {label} model file {model_url}...", end="", file=sys.stderr)
        if not os.path.exists(model_dest):
            os.makedirs(os.path.dirname(model_dest), exist_ok=True)
            request.urlretrieve(
                model_url, model_dest, ProgressBar(os.path.basename(model_dest))
            )
            print("...downloaded successfully", file=sys.stderr)
        else:
            print("...exists", file=sys.stderr)
    except Exception:
        print("...download failed", file=sys.stderr)
        print(f"Error downloading {label} model", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


# ---------------------------------------------
# this will preload the Bert tokenizer fles
def download_bert():
    print("Installing bert tokenizer...", file=sys.stderr)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from transformers import BertTokenizerFast

        download_from_hf(BertTokenizerFast, "bert-base-uncased")


# ---------------------------------------------
def download_sd1_clip():
    print("Installing SD1 clip model...", file=sys.stderr)
    version = "openai/clip-vit-large-patch14"
    download_from_hf(CLIPTokenizer, version)
    download_from_hf(CLIPTextModel, version)


# ---------------------------------------------
def download_sd2_clip():
    version = "stabilityai/stable-diffusion-2"
    print("Installing SD2 clip model...", file=sys.stderr)
    download_from_hf(CLIPTokenizer, version, subfolder="tokenizer")
    download_from_hf(CLIPTextModel, version, subfolder="text_encoder")


# ---------------------------------------------
def download_realesrgan():
    print("Installing models from RealESRGAN...", file=sys.stderr)
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    wdn_model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"

    model_dest = os.path.join(
        config.root, "models/realesrgan/realesr-general-x4v3.pth"
    )

    wdn_model_dest = os.path.join(
        config.root, "models/realesrgan/realesr-general-wdn-x4v3.pth"
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
        model_url, model_dest = model[0], os.path.join(config.root, model[1])
        download_with_progress_bar(model_url, model_dest, "GFPGAN weights")


# ---------------------------------------------
def download_codeformer():
    print("Installing CodeFormer model file...", file=sys.stderr)
    model_url = (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    )
    model_dest = os.path.join(config.root, "models/codeformer/codeformer.pth")
    download_with_progress_bar(model_url, model_dest, "CodeFormer")


# ---------------------------------------------
def download_clipseg():
    print("Installing clipseg model for text-based masking...", file=sys.stderr)
    CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"
    try:
        download_from_hf(AutoProcessor, CLIPSEG_MODEL)
        download_from_hf(CLIPSegForImageSegmentation, CLIPSEG_MODEL)
    except Exception:
        print("Error installing clipseg model:")
        print(traceback.format_exc())


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
    print("AutoFeatureExtractor...", file=sys.stderr)
    download_from_hf(AutoFeatureExtractor, safety_model_id)
    print("StableDiffusionSafetyChecker...", file=sys.stderr)
    download_from_hf(StableDiffusionSafetyChecker, safety_model_id)


# -------------------------------------
def download_vaes():
    print("Installing stabilityai VAE...", file=sys.stderr)
    try:
        # first the diffusers version
        repo_id = "stabilityai/sd-vae-ft-mse"
        args = dict(
            cache_dir=config.cache_dir,
        )
        if not AutoencoderKL.from_pretrained(repo_id, **args):
            raise Exception(f"download of {repo_id} failed")

        repo_id = "stabilityai/sd-vae-ft-mse-original"
        model_name = "vae-ft-mse-840000-ema-pruned.ckpt"
        # next the legacy checkpoint version
        if not hf_download_with_resume(
            repo_id=repo_id,
            model_name=model_name,
            model_dir=str(config.root / Model_dir / Weights_dir),
        ):
            raise Exception(f"download of {model_name} failed")
    except Exception as e:
        print(f"Error downloading StabilityAI standard VAE: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


# -------------------------------------
def get_root(root: str = None) -> str:
    if root:
        return root
    elif os.environ.get("INVOKEAI_ROOT"):
        return os.environ.get("INVOKEAI_ROOT")
    else:
        return config.root

# -------------------------------------
class editOptsForm(npyscreen.FormMultiPage):
    # for responsive resizing - disabled
    # FIX_MINIMUM_SIZE_WHEN_CREATED = False

    def create(self):
        program_opts = self.parentApp.program_opts
        old_opts = self.parentApp.invokeai_opts
        first_time = not (config.root / 'invokeai.yaml').exists()
        access_token = HfFolder.get_token()
        window_width, window_height = get_terminal_size()
        for i in [
            "Configure startup settings. You can come back and change these later.",
            "Use ctrl-N and ctrl-P to move to the <N>ext and <P>revious fields.",
            "Use cursor arrows to make a checkbox selection, and space to toggle.",
        ]:
            self.add_widget_intelligent(
                npyscreen.FixedText,
                value=i,
                editable=False,
                color="CONTROL",
            )

        self.nextrely += 1
        self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name="== BASIC OPTIONS ==",
            begin_entry_at=0,
            editable=False,
            color="CONTROL",
            scroll_exit=True,
        )
        self.nextrely -= 1
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="Select an output directory for images:",
            editable=False,
            color="CONTROL",
        )
        self.outdir = self.add_widget_intelligent(
            npyscreen.TitleFilename,
            name="(<tab> autocompletes, ctrl-N advances):",
            value=str(old_opts.outdir) or str(default_output_dir()),
            select_dir=True,
            must_exist=False,
            use_two_lines=False,
            labelColor="GOOD",
            begin_entry_at=40,
            scroll_exit=True,
        )
        self.nextrely += 1
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="Activate the NSFW checker to blur images showing potential sexual imagery:",
            editable=False,
            color="CONTROL",
        )
        self.nsfw_checker = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="NSFW checker",
            value=old_opts.nsfw_checker,
            relx=5,
            scroll_exit=True,
        )
        self.nextrely += 1
        for i in [
            "If you have an account at HuggingFace you may optionally paste your access token here",
            'to allow InvokeAI to download restricted styles & subjects from the "Concept Library".',
            "See https://huggingface.co/settings/tokens",
        ]:
            self.add_widget_intelligent(
                npyscreen.FixedText,
                value=i,
                editable=False,
                color="CONTROL",
            )

        self.hf_token = self.add_widget_intelligent(
            npyscreen.TitlePassword,
            name="Access Token (ctrl-shift-V pastes):",
            value=access_token,
            begin_entry_at=42,
            use_two_lines=False,
            scroll_exit=True,
        )
        self.nextrely += 1
        self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name="== ADVANCED OPTIONS ==",
            begin_entry_at=0,
            editable=False,
            color="CONTROL",
            scroll_exit=True,
        )
        self.nextrely -= 1
        self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name="GPU Management",
            begin_entry_at=0,
            editable=False,
            color="CONTROL",
            scroll_exit=True,
        )
        self.nextrely -= 1
        self.free_gpu_mem = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Free GPU memory after each generation",
            value=old_opts.free_gpu_mem,
            relx=5,
            scroll_exit=True,
        )
        self.xformers_enabled = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Enable xformers support if available",
            value=old_opts.xformers_enabled,
            relx=5,
            scroll_exit=True,
        )
        self.always_use_cpu = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Force CPU to be used on GPU systems",
            value=old_opts.always_use_cpu,
            relx=5,
            scroll_exit=True,
        )
        precision = old_opts.precision or (
            "float32" if program_opts.full_precision else "auto"
        )
        self.precision = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name="Precision",
            values=PRECISION_CHOICES,
            value=PRECISION_CHOICES.index(precision),
            begin_entry_at=3,
            max_height=len(PRECISION_CHOICES) + 1,
            scroll_exit=True,
        )
        self.max_loaded_models = self.add_widget_intelligent(
            IntTitleSlider,
            name="Number of models to cache in CPU memory (each will use 2-4 GB!)",
            value=old_opts.max_loaded_models,
            out_of=10,
            lowest=1,
            begin_entry_at=4,
            scroll_exit=True,
        )
        self.nextrely += 1
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="Directories containing textual inversion and LoRA models (<tab> autocompletes, ctrl-N advances):",
            editable=False,
            color="CONTROL",
        )
        self.embedding_dir = self.add_widget_intelligent(
            npyscreen.TitleFilename,
            name=" Textual Inversion Embeddings:",
            value=str(default_embedding_dir()),
            select_dir=True,
            must_exist=False,
            use_two_lines=False,
            labelColor="GOOD",
            begin_entry_at=32,
            scroll_exit=True,
        )
        self.lora_dir = self.add_widget_intelligent(
            npyscreen.TitleFilename,
            name="             LoRA and LyCORIS:",
            value=str(default_lora_dir()),
            select_dir=True,
            must_exist=False,
            use_two_lines=False,
            labelColor="GOOD",
            begin_entry_at=32,
            scroll_exit=True,
        )
        self.nextrely += 1
        self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name="== LICENSE ==",
            begin_entry_at=0,
            editable=False,
            color="CONTROL",
            scroll_exit=True,
        )
        self.nextrely -= 1
        for i in [
            "BY DOWNLOADING THE STABLE DIFFUSION WEIGHT FILES, YOU AGREE TO HAVE READ",
            "AND ACCEPTED THE CREATIVEML RESPONSIBLE AI LICENSE LOCATED AT",
            "https://huggingface.co/spaces/CompVis/stable-diffusion-license",
        ]:
            self.add_widget_intelligent(
                npyscreen.FixedText,
                value=i,
                editable=False,
                color="CONTROL",
            )
        self.license_acceptance = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="I accept the CreativeML Responsible AI License",
            value=not first_time,
            relx=2,
            scroll_exit=True,
        )
        self.nextrely += 1
        label = (
            "DONE"
            if program_opts.skip_sd_weights or program_opts.default_only
            else "NEXT"
        )
        self.ok_button = self.add_widget_intelligent(
            CenteredButtonPress,
            name=label,
            relx=(window_width - len(label)) // 2,
            rely=-3,
            when_pressed_function=self.on_ok,
        )

    def on_ok(self):
        options = self.marshall_arguments()
        if self.validate_field_values(options):
            self.parentApp.new_opts = options
            if hasattr(self.parentApp, "model_select"):
                self.parentApp.setNextForm("MODELS")
            else:
                self.parentApp.setNextForm(None)
            self.editing = False
        else:
            self.editing = True

    def validate_field_values(self, opt: Namespace) -> bool:
        bad_fields = []
        if not opt.license_acceptance:
            bad_fields.append(
                "Please accept the license terms before proceeding to model downloads"
            )
        if not Path(opt.outdir).parent.exists():
            bad_fields.append(
                f"The output directory does not seem to be valid. Please check that {str(Path(opt.outdir).parent)} is an existing directory."
            )
        if not Path(opt.embedding_dir).parent.exists():
            bad_fields.append(
                f"The embedding directory does not seem to be valid. Please check that {str(Path(opt.embedding_dir).parent)} is an existing directory."
            )
        if len(bad_fields) > 0:
            message = "The following problems were detected and must be corrected:\n"
            for problem in bad_fields:
                message += f"* {problem}\n"
            npyscreen.notify_confirm(message)
            return False
        else:
            return True

    def marshall_arguments(self):
        new_opts = Namespace()

        for attr in [
                "outdir",
                "nsfw_checker",
                "free_gpu_mem",
                "max_loaded_models",
                "xformers_enabled",
                "always_use_cpu",
                "embedding_dir",
                "lora_dir",
        ]:
            setattr(new_opts, attr, getattr(self, attr).value)

        new_opts.hf_token = self.hf_token.value
        new_opts.license_acceptance = self.license_acceptance.value
        new_opts.precision = PRECISION_CHOICES[self.precision.value[0]]
        
        # widget library workaround to make max_loaded_models an int rather than a float
        new_opts.max_loaded_models = int(new_opts.max_loaded_models)

        return new_opts


class EditOptApplication(npyscreen.NPSAppManaged):
    def __init__(self, program_opts: Namespace, invokeai_opts: Namespace):
        super().__init__()
        self.program_opts = program_opts
        self.invokeai_opts = invokeai_opts
        self.user_cancelled = False
        self.user_selections = default_user_selections(program_opts)

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.DefaultTheme)
        self.options = self.addForm(
            "MAIN",
            editOptsForm,
            name="InvokeAI Startup Options",
        )
        if not (self.program_opts.skip_sd_weights or self.program_opts.default_only):
            self.model_select = self.addForm(
                "MODELS",
                addModelsForm,
                name="Install Stable Diffusion Models",
                multipage=True,
            )

    def new_opts(self):
        return self.options.marshall_arguments()


def edit_opts(program_opts: Namespace, invokeai_opts: Namespace) -> argparse.Namespace:
    editApp = EditOptApplication(program_opts, invokeai_opts)
    editApp.run()
    return editApp.new_opts()


def default_startup_options(init_file: Path) -> Namespace:
    opts = InvokeAIAppConfig(argv=[])
    outdir = Path(opts.outdir)
    if not outdir.is_absolute():
        opts.outdir = str(config.root / opts.outdir)
    if not init_file.exists():
        opts.nsfw_checker = True
    return opts

def default_user_selections(program_opts: Namespace) -> Namespace:
    return Namespace(
        starter_models=default_dataset()
        if program_opts.default_only
        else recommended_datasets()
        if program_opts.yes_to_all
        else dict(),
        purge_deleted_models=False,
        scan_directory=None,
        autoscan_on_startup=None,
        import_model_paths=None,
        convert_to_diffusers=None,
    )


# -------------------------------------
def initialize_rootdir(root: str, yes_to_all: bool = False):
    print("** INITIALIZING INVOKEAI RUNTIME DIRECTORY **")

    for name in (
        "models",
        "configs",
        "embeddings",
        "text-inversion-output",
        "text-inversion-training-data",
    ):
        os.makedirs(os.path.join(root, name), exist_ok=True)

    configs_src = Path(configs.__path__[0])
    configs_dest = Path(root) / "configs"
    if not os.path.samefile(configs_src, configs_dest):
        shutil.copytree(configs_src, configs_dest, dirs_exist_ok=True)


# -------------------------------------
def run_console_ui(
    program_opts: Namespace, initfile: Path = None
) -> (Namespace, Namespace):
    # parse_args() will read from init file if present
    invokeai_opts = default_startup_options(initfile)

    set_min_terminal_size(MIN_COLS, MIN_LINES)
    editApp = EditOptApplication(program_opts, invokeai_opts)
    editApp.run()
    if editApp.user_cancelled:
        return (None, None)
    else:
        return (editApp.new_opts, editApp.user_selections)


# -------------------------------------
def write_opts(opts: Namespace, init_file: Path):
    """
    Update the invokeai.yaml file with values from current settings.
    """

    # this will load current settings
    config = InvokeAIAppConfig(argv=[])
    for key,value in opts.__dict__.items():
        if hasattr(config,key):
            setattr(config,key,value)

    with open(init_file,'w', encoding='utf-8') as file:
        file.write(config.to_yaml())

# -------------------------------------
def default_output_dir() -> Path:
    return config.root / "outputs"

# -------------------------------------
def default_embedding_dir() -> Path:
    return config.root / "embeddings"

# -------------------------------------
def default_lora_dir() -> Path:
    return config.root / "loras"

# -------------------------------------
def write_default_options(program_opts: Namespace, initfile: Path):
    opt = default_startup_options(initfile)
    write_opts(opt, initfile)

# -------------------------------------
# Here we bring in
# the legacy Args object in order to parse
# the old init file and write out the new
# yaml format.
def migrate_init_file(legacy_format:Path):
    old = legacy_parser.parse_args([f'@{str(legacy_format)}'])
    new = InvokeAIAppConfig(conf={})

    fields = list(get_type_hints(InvokeAIAppConfig).keys())
    for attr in fields:
        if hasattr(old,attr):
            setattr(new,attr,getattr(old,attr))

    # a few places where the field names have changed and we have to
    # manually add in the new names/values
    new.nsfw_checker = old.safety_checker
    new.xformers_enabled = old.xformers
    new.conf_path = old.conf
    new.embedding_dir = old.embedding_path

    invokeai_yaml = legacy_format.parent / 'invokeai.yaml'
    with open(invokeai_yaml,"w", encoding="utf-8") as outfile:
        outfile.write(new.to_yaml())

    legacy_format.replace(legacy_format.parent / 'invokeai.init.old')

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
        "--skip-support-models",
        dest="skip_support_models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip downloading the support models",
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
    global config
    config.root = Path(os.path.expanduser(get_root(opt.root) or ""))

    errors = set()

    try:
        models_to_download = default_user_selections(opt)

        # We check for to see if the runtime directory is correctly initialized.
        old_init_file = Path(config.root, 'invokeai.init')
        new_init_file = Path(config.root, 'invokeai.yaml')
        if old_init_file.exists() and not new_init_file.exists():
            print('** Migrating invokeai.init to invokeai.yaml')
            migrate_init_file(old_init_file)
            config = get_invokeai_config(argv=[])  # reread defaults


        if not config.model_conf_path.exists():
            initialize_rootdir(config.root, opt.yes_to_all)

        if opt.yes_to_all:
            write_default_options(opt, new_init_file)
            init_options = Namespace(
                precision="float32" if opt.full_precision else "float16"
            )
        else:
            init_options, models_to_download = run_console_ui(opt, new_init_file)
            if init_options:
                write_opts(init_options, new_init_file)
            else:
                print(
                    '\n** CANCELLED AT USER\'S REQUEST. USE THE "invoke.sh" LAUNCHER TO RUN LATER **\n'
                )
                sys.exit(0)

        if opt.skip_support_models:
            print("\n** SKIPPING SUPPORT MODEL DOWNLOADS PER USER REQUEST **")
        else:
            print("\n** DOWNLOADING SUPPORT MODELS **")
            download_bert()
            download_sd1_clip()
            download_sd2_clip()
            download_realesrgan()
            download_gfpgan()
            download_codeformer()
            download_clipseg()
            download_safety_checker()
            download_vaes()

        if opt.skip_sd_weights:
            print("\n** SKIPPING DIFFUSION WEIGHTS DOWNLOAD PER USER REQUEST **")
        elif models_to_download:
            print("\n** DOWNLOADING DIFFUSION WEIGHTS **")
            process_and_execute(opt, models_to_download)

        postscript(errors=errors)
    except KeyboardInterrupt:
        print("\nGoodbye! Come back soon.")


# -------------------------------------
if __name__ == "__main__":
    main()
