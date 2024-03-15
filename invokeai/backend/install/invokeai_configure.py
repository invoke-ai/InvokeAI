#!/usr/bin/env python
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)
# Before running stable-diffusion on an internet-isolated machine,
# run this script from one with internet connectivity. The
# two machines must share a common .cache directory.
#
# Coauthor: Kevin Turner http://github.com/keturn
#
import argparse
import io
import os
import shutil
import sys
import textwrap
import traceback
import warnings
from argparse import Namespace
from enum import Enum
from pathlib import Path
from shutil import copy, get_terminal_size, move
from typing import Any, Optional, Tuple, Type, get_args, get_type_hints
from urllib import request

import npyscreen
import psutil
import torch
import transformers
from diffusers import ModelMixin
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from huggingface_hub import HfFolder
from huggingface_hub import login as hf_hub_login
from tqdm import tqdm
from transformers import AutoFeatureExtractor

import invokeai.configs as model_configs
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.config.config_default import get_config
from invokeai.backend.install.install_helper import InstallHelper, InstallSelections
from invokeai.backend.model_manager import ModelType
from invokeai.backend.util import choose_precision, choose_torch_device
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.frontend.install.model_install import addModelsForm

# TO DO - Move all the frontend code into invokeai.frontend.install
from invokeai.frontend.install.widgets import (
    MIN_COLS,
    MIN_LINES,
    CenteredButtonPress,
    CyclingForm,
    FileBox,
    MultiSelectColumns,
    SingleSelectColumnsSimple,
    WindowTooSmallException,
    set_min_terminal_size,
)

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


def get_literal_fields(field: str) -> Tuple[Any]:
    return get_args(get_type_hints(InvokeAIAppConfig).get(field))


# --------------------------globals-----------------------

config = get_config()

PRECISION_CHOICES = get_literal_fields("precision")
DEVICE_CHOICES = get_literal_fields("device")
ATTENTION_CHOICES = get_literal_fields("attention_type")
ATTENTION_SLICE_CHOICES = get_literal_fields("attention_slice_size")
GENERATION_OPT_CHOICES = ["sequential_guidance", "force_tiled_decode", "lazy_offload"]
GB = 1073741824  # GB in bytes
HAS_CUDA = torch.cuda.is_available()
_, MAX_VRAM = torch.cuda.mem_get_info() if HAS_CUDA else (0.0, 0.0)

MAX_VRAM /= GB
MAX_RAM = psutil.virtual_memory().total / GB

FORCE_FULL_PRECISION = False

INIT_FILE_PREAMBLE = """# InvokeAI initialization file
# This is the InvokeAI initialization file, which contains command-line default values.
# Feel free to edit. If anything goes wrong, you can re-initialize this file by deleting
# or renaming it and then running invokeai-configure again.
"""

logger = InvokeAILogger.get_logger()


class DummyWidgetValue(Enum):
    """Dummy widget values."""

    zero = 0
    true = True
    false = False


# --------------------------------------------
def postscript(errors: set[str]) -> None:
    if not any(errors):
        message = f"""
** INVOKEAI INSTALLATION SUCCESSFUL **
If you installed manually from source or with 'pip install': activate the virtual environment
then run one of the following commands to start InvokeAI.

Web UI:
   invokeai-web

If you installed using an installation script, run:
  {config.root_path}/invoke.{"bat" if sys.platform == "win32" else "sh"}

Add the '--help' argument to see all of the command-line switches available for use.
"""

    else:
        message = (
            "\n** There were errors during installation. It is possible some of the models were not fully downloaded.\n"
        )
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
def HfLogin(access_token) -> None:
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
    def __init__(self, model_name: str = "file"):
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
def hf_download_from_pretrained(model_class: Type[ModelMixin], model_name: str, destination: Path, **kwargs: Any):
    filter = lambda x: "fp16 is not a valid" not in x.getMessage()  # noqa E731
    logger.addFilter(filter)
    try:
        model = model_class.from_pretrained(
            model_name,
            resume_download=True,
            **kwargs,
        )
        model.save_pretrained(destination, safe_serialization=True)
    finally:
        logger.removeFilter(filter)
    return destination


# ---------------------------------------------
def download_with_progress_bar(model_url: str, model_dest: str | Path, label: str = "the"):
    try:
        logger.info(f"Installing {label} model file {model_url}...")
        if not os.path.exists(model_dest):
            os.makedirs(os.path.dirname(model_dest), exist_ok=True)
            request.urlretrieve(model_url, model_dest, ProgressBar(os.path.basename(model_dest)))
            logger.info("...downloaded successfully")
        else:
            logger.info("...exists")
    except Exception:
        logger.info("...download failed")
        logger.info(f"Error downloading {label} model")
        print(traceback.format_exc(), file=sys.stderr)


def download_safety_checker():
    target_dir = config.models_path / "core/convert"
    kwargs = {}  # for future use
    try:
        # safety checking
        logger.info("Downloading safety checker")
        repo_id = "CompVis/stable-diffusion-safety-checker"
        pipeline = AutoFeatureExtractor.from_pretrained(repo_id, **kwargs)
        pipeline.save_pretrained(target_dir / "stable-diffusion-safety-checker", safe_serialization=True)
        pipeline = StableDiffusionSafetyChecker.from_pretrained(repo_id, **kwargs)
        pipeline.save_pretrained(target_dir / "stable-diffusion-safety-checker", safe_serialization=True)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(str(e))


# ---------------------------------------------
# TO DO: use the download queue here.
def download_realesrgan():
    logger.info("Installing ESRGAN Upscaling models...")
    URLs = [
        {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "dest": "core/upscaling/realesrgan/RealESRGAN_x4plus.pth",
            "description": "RealESRGAN_x4plus.pth",
        },
        {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "dest": "core/upscaling/realesrgan/RealESRGAN_x4plus_anime_6B.pth",
            "description": "RealESRGAN_x4plus_anime_6B.pth",
        },
        {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
            "dest": "core/upscaling/realesrgan/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
            "description": "ESRGAN_SRx4_DF2KOST_official.pth",
        },
        {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "dest": "core/upscaling/realesrgan/RealESRGAN_x2plus.pth",
            "description": "RealESRGAN_x2plus.pth",
        },
    ]
    for model in URLs:
        download_with_progress_bar(model["url"], config.models_path / model["dest"], model["description"])


# ---------------------------------------------
def download_lama():
    logger.info("Installing lama infill model")
    download_with_progress_bar(
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        config.models_path / "core/misc/lama/lama.pt",
        "lama infill model",
    )


# ---------------------------------------------
def download_support_models() -> None:
    download_realesrgan()
    download_lama()
    download_safety_checker()


# -------------------------------------
def get_root(root: Optional[str] = None) -> str:
    if root:
        return root
    elif root := os.environ.get("INVOKEAI_ROOT"):
        assert root is not None
        return root
    else:
        return str(config.root_path)


# -------------------------------------
class editOptsForm(CyclingForm, npyscreen.FormMultiPage):
    # for responsive resizing - disabled
    # FIX_MINIMUM_SIZE_WHEN_CREATED = False

    def create(self):
        program_opts = self.parentApp.program_opts
        old_opts: InvokeAIAppConfig = self.parentApp.invokeai_opts
        first_time = not (config.root_path / "invokeai.yaml").exists()
        access_token = HfFolder.get_token()
        window_width, window_height = get_terminal_size()
        label = """Configure startup settings. You can come back and change these later.
Use ctrl-N and ctrl-P to move to the <N>ext and <P>revious fields.
Use cursor arrows to make a checkbox selection, and space to toggle.
"""
        self.nextrely -= 1
        for i in textwrap.wrap(label, width=window_width - 6):
            self.add_widget_intelligent(
                npyscreen.FixedText,
                value=i,
                editable=False,
                color="CONTROL",
            )

        self.nextrely += 1
        label = """HuggingFace access token (OPTIONAL) for automatic model downloads. See https://huggingface.co/settings/tokens."""
        for line in textwrap.wrap(label, width=window_width - 6):
            self.add_widget_intelligent(
                npyscreen.FixedText,
                value=line,
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

        # old settings for defaults
        precision = old_opts.precision or ("float32" if program_opts.full_precision else "auto")
        device = old_opts.device
        attention_type = old_opts.attention_type
        attention_slice_size = old_opts.attention_slice_size
        self.nextrely += 1
        self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name="Image Generation Options:",
            editable=False,
            color="CONTROL",
            scroll_exit=True,
        )
        self.nextrely -= 2
        self.generation_options = self.add_widget_intelligent(
            MultiSelectColumns,
            columns=3,
            values=GENERATION_OPT_CHOICES,
            value=[GENERATION_OPT_CHOICES.index(x) for x in GENERATION_OPT_CHOICES if getattr(old_opts, x)],
            relx=30,
            max_height=2,
            max_width=80,
            scroll_exit=True,
        )

        self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name="Floating Point Precision:",
            begin_entry_at=0,
            editable=False,
            color="CONTROL",
            scroll_exit=True,
        )
        self.nextrely -= 2
        self.precision = self.add_widget_intelligent(
            SingleSelectColumnsSimple,
            columns=len(PRECISION_CHOICES),
            name="Precision",
            values=PRECISION_CHOICES,
            value=PRECISION_CHOICES.index(precision),
            begin_entry_at=3,
            max_height=2,
            relx=30,
            max_width=80,
            scroll_exit=True,
        )
        self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name="Generation Device:",
            begin_entry_at=0,
            editable=False,
            color="CONTROL",
            scroll_exit=True,
        )
        self.nextrely -= 2
        self.device = self.add_widget_intelligent(
            SingleSelectColumnsSimple,
            columns=len(DEVICE_CHOICES),
            values=DEVICE_CHOICES,
            value=[DEVICE_CHOICES.index(device)],
            begin_entry_at=3,
            relx=30,
            max_height=2,
            max_width=60,
            scroll_exit=True,
        )
        self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name="Attention Type:",
            begin_entry_at=0,
            editable=False,
            color="CONTROL",
            scroll_exit=True,
        )
        self.nextrely -= 2
        self.attention_type = self.add_widget_intelligent(
            SingleSelectColumnsSimple,
            columns=len(ATTENTION_CHOICES),
            values=ATTENTION_CHOICES,
            value=[ATTENTION_CHOICES.index(attention_type)],
            begin_entry_at=3,
            max_height=2,
            relx=30,
            max_width=80,
            scroll_exit=True,
        )
        self.attention_type.on_changed = self.show_hide_slice_sizes
        self.attention_slice_label = self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name="Attention Slice Size:",
            relx=5,
            editable=False,
            hidden=attention_type != "sliced",
            color="CONTROL",
            scroll_exit=True,
        )
        self.nextrely -= 2
        self.attention_slice_size = self.add_widget_intelligent(
            SingleSelectColumnsSimple,
            columns=len(ATTENTION_SLICE_CHOICES),
            values=ATTENTION_SLICE_CHOICES,
            value=[ATTENTION_SLICE_CHOICES.index(attention_slice_size)],
            relx=30,
            hidden=attention_type != "sliced",
            max_height=2,
            max_width=110,
            scroll_exit=True,
        )
        self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name="Model disk conversion cache size (GB). This is used to cache safetensors files that need to be converted to diffusers..",
            begin_entry_at=0,
            editable=False,
            color="CONTROL",
            scroll_exit=True,
        )
        self.nextrely -= 1
        self.disk = self.add_widget_intelligent(
            npyscreen.Slider,
            value=clip(old_opts.convert_cache, range=(0, 100), step=0.5),
            out_of=100,
            lowest=0.0,
            step=0.5,
            relx=8,
            scroll_exit=True,
        )
        self.nextrely += 1
        self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name="Model RAM cache size (GB). Make this at least large enough to hold a single full model (2GB for SD-1, 6GB for SDXL).",
            begin_entry_at=0,
            editable=False,
            color="CONTROL",
            scroll_exit=True,
        )
        self.nextrely -= 1
        self.ram = self.add_widget_intelligent(
            npyscreen.Slider,
            value=clip(old_opts.ram, range=(3.0, MAX_RAM), step=0.5),
            out_of=round(MAX_RAM),
            lowest=0.0,
            step=0.5,
            relx=8,
            scroll_exit=True,
        )
        if HAS_CUDA:
            self.nextrely += 1
            self.add_widget_intelligent(
                npyscreen.TitleFixedText,
                name="Model VRAM cache size (GB). Reserving a small amount of VRAM will modestly speed up the start of image generation.",
                begin_entry_at=0,
                editable=False,
                color="CONTROL",
                scroll_exit=True,
            )
            self.nextrely -= 1
            self.vram = self.add_widget_intelligent(
                npyscreen.Slider,
                value=clip(old_opts.vram, range=(0, MAX_VRAM), step=0.25),
                out_of=round(MAX_VRAM * 2) / 2,
                lowest=0.0,
                relx=8,
                step=0.25,
                scroll_exit=True,
            )
        else:
            self.vram = DummyWidgetValue.zero

        self.nextrely += 1
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="Location of the database used to store model path and configuration information:",
            editable=False,
            color="CONTROL",
        )
        self.nextrely += 1
        self.outdir = self.add_widget_intelligent(
            FileBox,
            name="Output directory for images (<tab> autocompletes, ctrl-N advances):",
            value=str(default_output_dir()),
            select_dir=True,
            must_exist=False,
            use_two_lines=False,
            labelColor="GOOD",
            begin_entry_at=40,
            max_height=3,
            max_width=127,
            scroll_exit=True,
        )
        self.autoimport_dirs = {}
        self.autoimport_dirs["autoimport_dir"] = self.add_widget_intelligent(
            FileBox,
            name="Optional folder to scan for new checkpoints, ControlNets, LoRAs and TI models",
            value=str(config.autoimport_path),
            select_dir=True,
            must_exist=False,
            use_two_lines=False,
            labelColor="GOOD",
            begin_entry_at=32,
            max_height=3,
            max_width=127,
            scroll_exit=True,
        )
        self.nextrely += 1
        label = """BY DOWNLOADING THE STABLE DIFFUSION WEIGHT FILES, YOU AGREE TO HAVE READ
AND ACCEPTED THE CREATIVEML RESPONSIBLE AI LICENSES LOCATED AT
https://huggingface.co/spaces/CompVis/stable-diffusion-license and
https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md
"""
        for i in textwrap.wrap(label, width=window_width - 6):
            self.add_widget_intelligent(
                npyscreen.FixedText,
                value=i,
                editable=False,
                color="CONTROL",
            )
        self.license_acceptance = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="I accept the CreativeML Responsible AI Licenses",
            value=not first_time,
            relx=2,
            scroll_exit=True,
        )
        self.nextrely += 1
        label = "DONE" if program_opts.skip_sd_weights or program_opts.default_only else "NEXT"
        self.ok_button = self.add_widget_intelligent(
            CenteredButtonPress,
            name=label,
            relx=(window_width - len(label)) // 2,
            when_pressed_function=self.on_ok,
        )

    def show_hide_slice_sizes(self, value):
        show = ATTENTION_CHOICES[value[0]] == "sliced"
        self.attention_slice_label.hidden = not show
        self.attention_slice_size.hidden = not show

    def show_hide_model_conf_override(self, value):
        self.model_conf_override.hidden = value
        self.model_conf_override.display()

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
            bad_fields.append("Please accept the license terms before proceeding to model downloads")
        if not Path(opt.outdir).parent.exists():
            bad_fields.append(
                f"The output directory does not seem to be valid. Please check that {str(Path(opt.outdir).parent)} is an existing directory."
            )
        if len(bad_fields) > 0:
            message = "The following problems were detected and must be corrected:\n"
            for problem in bad_fields:
                message += f"* {problem}\n"
            npyscreen.notify_confirm(message)
            return False
        else:
            return True

    def marshall_arguments(self) -> Namespace:
        new_opts = Namespace()

        for attr in [
            "ram",
            "vram",
            "convert_cache",
            "outdir",
        ]:
            if hasattr(self, attr):
                setattr(new_opts, attr, getattr(self, attr).value)

        for attr in self.autoimport_dirs:
            if not self.autoimport_dirs[attr].value:
                continue
            directory = Path(self.autoimport_dirs[attr].value)
            if directory.is_relative_to(config.root_path):
                directory = directory.relative_to(config.root_path)
            setattr(new_opts, attr, directory)

        new_opts.hf_token = self.hf_token.value
        new_opts.license_acceptance = self.license_acceptance.value
        new_opts.precision = PRECISION_CHOICES[self.precision.value[0]]
        new_opts.device = DEVICE_CHOICES[self.device.value[0]]
        new_opts.attention_type = ATTENTION_CHOICES[self.attention_type.value[0]]
        new_opts.attention_slice_size = ATTENTION_SLICE_CHOICES[self.attention_slice_size.value[0]]
        generation_options = [GENERATION_OPT_CHOICES[x] for x in self.generation_options.value]
        for v in GENERATION_OPT_CHOICES:
            setattr(new_opts, v, v in generation_options)
        return new_opts


class EditOptApplication(npyscreen.NPSAppManaged):
    def __init__(self, program_opts: Namespace, invokeai_opts: InvokeAIAppConfig, install_helper: InstallHelper):
        super().__init__()
        self.program_opts = program_opts
        self.invokeai_opts = invokeai_opts
        self.user_cancelled = False
        self.autoload_pending = True
        self.install_helper = install_helper
        self.install_selections = default_user_selections(program_opts, install_helper)

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.DefaultTheme)
        self.options = self.addForm(
            "MAIN",
            editOptsForm,
            name="InvokeAI Startup Options",
            cycle_widgets=False,
        )
        if not (self.program_opts.skip_sd_weights or self.program_opts.default_only):
            self.model_select = self.addForm(
                "MODELS",
                addModelsForm,
                name="Install Stable Diffusion Models",
                multipage=True,
                cycle_widgets=False,
            )


def get_default_ram_cache_size() -> float:
    """Run a heuristic for the default RAM cache based on installed RAM."""

    # Note that on my 64 GB machine, psutil.virtual_memory().total gives 62 GB,
    # So we adjust everthing down a bit.
    return (
        15.0 if MAX_RAM >= 60 else 7.5 if MAX_RAM >= 30 else 4 if MAX_RAM >= 14 else 2.1
    )  # 2.1 is just large enough for sd 1.5 ;-)


def get_default_config() -> InvokeAIAppConfig:
    """Builds a new config object, setting the ram and precision using the appropriate heuristic."""
    config = InvokeAIAppConfig()
    config.ram = get_default_ram_cache_size()
    config.precision = "float32" if FORCE_FULL_PRECISION else choose_precision(torch.device(choose_torch_device()))
    return config


def default_user_selections(program_opts: Namespace, install_helper: InstallHelper) -> InstallSelections:
    default_model = install_helper.default_model()
    assert default_model is not None
    default_models = [default_model] if program_opts.default_only else install_helper.recommended_models()
    return InstallSelections(
        install_models=default_models if program_opts.yes_to_all else [],
    )


# -------------------------------------
def clip(value: float, range: tuple[float, float], step: float) -> float:
    minimum, maximum = range
    if value < minimum:
        value = minimum
    if value > maximum:
        value = maximum
    return round(value / step) * step


# -------------------------------------
def initialize_rootdir(root: Path, yes_to_all: bool = False):
    logger.info("Initializing InvokeAI runtime directory")
    for name in ("models", "databases", "text-inversion-output", "text-inversion-training-data", "configs"):
        os.makedirs(os.path.join(root, name), exist_ok=True)
    for model_type in ModelType:
        Path(root, "autoimport", model_type.value).mkdir(parents=True, exist_ok=True)

    configs_src = Path(model_configs.__path__[0])
    configs_dest = root / "configs"
    if not os.path.samefile(configs_src, configs_dest):
        shutil.copytree(configs_src, configs_dest, dirs_exist_ok=True)

    dest = root / "models"
    dest.mkdir(parents=True, exist_ok=True)


# -------------------------------------
def run_console_ui(
    program_opts: Namespace, install_helper: InstallHelper
) -> Tuple[Optional[Namespace], Optional[InstallSelections]]:
    first_time = not config.init_file_path.exists()
    config_opts = get_default_config() if first_time else config
    if program_opts.root:
        config_opts.set_root(Path(program_opts.root))

    if not set_min_terminal_size(MIN_COLS, MIN_LINES):
        raise WindowTooSmallException(
            "Could not increase terminal size. Try running again with a larger window or smaller font size."
        )

    editApp = EditOptApplication(program_opts, config_opts, install_helper)
    editApp.run()
    if editApp.user_cancelled:
        return (None, None)
    else:
        return (editApp.new_opts, editApp.install_selections)


# -------------------------------------
def default_output_dir() -> Path:
    return config.root_path / "outputs"


def is_v2_install(root: Path) -> bool:
    # We check for to see if the runtime directory is correctly initialized.
    old_init_file = root / "invokeai.init"
    new_init_file = root / "invokeai.yaml"
    old_hub = root / "models/hub"
    is_v2 = (old_init_file.exists() and not new_init_file.exists()) and old_hub.exists()
    return is_v2


# -------------------------------------
def main() -> None:
    global FORCE_FULL_PRECISION  # FIXME
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
        "--root_dir",
        dest="root",
        type=str,
        default=None,
        help="path to root of install directory",
    )
    opt = parser.parse_args()
    updates: dict[str, Any] = {}
    if opt.root:
        config.set_root(Path(opt.root))
    if opt.full_precision:
        updates["precision"] = "float32"

    try:
        # Attempt to read the config file into the config object
        config.merge_from_file()
    except FileNotFoundError:
        # No config file, first time running the app
        pass

    config.update_config(updates)
    logger = InvokeAILogger().get_logger(config=config)

    errors: set[str] = set()
    FORCE_FULL_PRECISION = opt.full_precision  # FIXME global

    # Before we write anything else, make a backup of the existing init file
    new_init_file = config.init_file_path
    backup_init_file = new_init_file.with_suffix(".bak")
    if new_init_file.exists():
        copy(new_init_file, backup_init_file)

    try:
        # v2.3 -> v4.0.0 upgrade is no longer supported
        if is_v2_install(config.root_path):
            logger.error("Migration from v2.3 to v4.0.0 is no longer supported. Please install a fresh copy.")
            sys.exit(0)

        # run this unconditionally in case new directories need to be added
        initialize_rootdir(config.root_path, opt.yes_to_all)

        # this will initialize and populate the models tables if not present
        install_helper = InstallHelper(config, logger)

        models_to_download = default_user_selections(opt, install_helper)

        if opt.yes_to_all:
            # We will not show the UI - just write the default config to the file and move on to installing models.
            get_default_config().write_file(new_init_file)
        else:
            # Run the UI to get the user's options & model choices
            user_opts, models_to_download = run_console_ui(opt, install_helper)
            if user_opts:
                # Create a dict of the user's opts, omitting any fields that are not config settings (like `hf_token`)
                user_opts_dict = {k: v for k, v in vars(user_opts).items() if k in config.model_fields}
                # Merge the user's opts back into the config object & write it
                config.update_config(user_opts_dict)
                config.write_file(config.init_file_path)

                if hasattr(user_opts, "hf_token") and user_opts.hf_token:
                    HfLogin(user_opts.hf_token)
            else:
                logger.info('\n** CANCELLED AT USER\'S REQUEST. USE THE "invoke.sh" LAUNCHER TO RUN LATER **\n')
                sys.exit(0)

        if opt.skip_support_models:
            logger.info("Skipping support models at user's request")
        else:
            logger.info("Installing support models")
            download_support_models()

        if opt.skip_sd_weights:
            logger.warning("Skipping diffusion weights download per user request")
        elif models_to_download:
            install_helper.add_or_delete(models_to_download)

        postscript(errors=errors)

        if not opt.yes_to_all:
            input("Press any key to continue...")
    except WindowTooSmallException as e:
        logger.error(str(e))
        if backup_init_file.exists():
            move(backup_init_file, new_init_file)
    except KeyboardInterrupt:
        print("\nGoodbye! Come back soon.")
        if backup_init_file.exists():
            move(backup_init_file, new_init_file)
    except Exception:
        print("An error occurred during installation.")
        if backup_init_file.exists():
            move(backup_init_file, new_init_file)
        print(traceback.format_exc(), file=sys.stderr)


# -------------------------------------
if __name__ == "__main__":
    main()
