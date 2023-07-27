#!/usr/bin/env python

"""
This is the frontend to "textual_inversion_training.py".

Copyright (c) 2023 Lincoln Stein and the InvokeAI Development Team
"""


import os
import re
import shutil
import sys
import traceback
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple

import npyscreen
from npyscreen import widget
from omegaconf import OmegaConf

import invokeai.backend.util.logging as logger

from invokeai.app.services.config import InvokeAIAppConfig
from ...backend.training import do_textual_inversion_training, parse_args

TRAINING_DATA = "text-inversion-training-data"
TRAINING_DIR = "text-inversion-output"
CONF_FILE = "preferences.conf"
config = None


class textualInversionForm(npyscreen.FormMultiPageAction):
    resolutions = [512, 768, 1024]
    lr_schedulers = [
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ]
    precisions = ["no", "fp16", "bf16"]
    learnable_properties = ["object", "style"]

    def __init__(self, parentApp, name, saved_args=None):
        self.saved_args = saved_args or {}
        super().__init__(parentApp, name)

    def afterEditing(self):
        self.parentApp.setNextForm(None)

    def create(self):
        self.model_names, default = self.get_model_names()
        default_initializer_token = "★"
        default_placeholder_token = ""
        saved_args = self.saved_args

        try:
            default = self.model_names.index(saved_args["model"])
        except:
            pass

        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="Use ctrl-N and ctrl-P to move to the <N>ext and <P>revious fields, cursor arrows to make a selection, and space to toggle checkboxes.",
            editable=False,
        )

        self.model = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name="Model Name:",
            values=self.model_names,
            value=default,
            max_height=len(self.model_names) + 1,
            scroll_exit=True,
        )
        self.placeholder_token = self.add_widget_intelligent(
            npyscreen.TitleText,
            name="Trigger Term:",
            value="",  # saved_args.get('placeholder_token',''), # to restore previous term
            scroll_exit=True,
        )
        self.placeholder_token.when_value_edited = self.initializer_changed
        self.nextrely -= 1
        self.nextrelx += 30
        self.prompt_token = self.add_widget_intelligent(
            npyscreen.FixedText,
            name="Trigger term for use in prompt",
            value="",
            editable=False,
            scroll_exit=True,
        )
        self.nextrelx -= 30
        self.initializer_token = self.add_widget_intelligent(
            npyscreen.TitleText,
            name="Initializer:",
            value=saved_args.get("initializer_token", default_initializer_token),
            scroll_exit=True,
        )
        self.resume_from_checkpoint = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Resume from last saved checkpoint",
            value=False,
            scroll_exit=True,
        )
        self.learnable_property = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name="Learnable property:",
            values=self.learnable_properties,
            value=self.learnable_properties.index(saved_args.get("learnable_property", "object")),
            max_height=4,
            scroll_exit=True,
        )
        self.train_data_dir = self.add_widget_intelligent(
            npyscreen.TitleFilename,
            name="Data Training Directory:",
            select_dir=True,
            must_exist=False,
            value=str(
                saved_args.get(
                    "train_data_dir",
                    config.root_dir / TRAINING_DATA / default_placeholder_token,
                )
            ),
            scroll_exit=True,
        )
        self.output_dir = self.add_widget_intelligent(
            npyscreen.TitleFilename,
            name="Output Destination Directory:",
            select_dir=True,
            must_exist=False,
            value=str(
                saved_args.get(
                    "output_dir",
                    config.root_dir / TRAINING_DIR / default_placeholder_token,
                )
            ),
            scroll_exit=True,
        )
        self.resolution = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name="Image resolution (pixels):",
            values=self.resolutions,
            value=self.resolutions.index(saved_args.get("resolution", 512)),
            max_height=4,
            scroll_exit=True,
        )
        self.center_crop = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Center crop images before resizing to resolution",
            value=saved_args.get("center_crop", False),
            scroll_exit=True,
        )
        self.mixed_precision = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name="Mixed Precision:",
            values=self.precisions,
            value=self.precisions.index(saved_args.get("mixed_precision", "fp16")),
            max_height=4,
            scroll_exit=True,
        )
        self.num_train_epochs = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name="Number of training epochs:",
            out_of=1000,
            step=50,
            lowest=1,
            value=saved_args.get("num_train_epochs", 100),
            scroll_exit=True,
        )
        self.max_train_steps = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name="Max Training Steps:",
            out_of=10000,
            step=500,
            lowest=1,
            value=saved_args.get("max_train_steps", 3000),
            scroll_exit=True,
        )
        self.train_batch_size = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name="Batch Size (reduce if you run out of memory):",
            out_of=50,
            step=1,
            lowest=1,
            value=saved_args.get("train_batch_size", 8),
            scroll_exit=True,
        )
        self.gradient_accumulation_steps = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name="Gradient Accumulation Steps (may need to decrease this to resume from a checkpoint):",
            out_of=10,
            step=1,
            lowest=1,
            value=saved_args.get("gradient_accumulation_steps", 4),
            scroll_exit=True,
        )
        self.lr_warmup_steps = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name="Warmup Steps:",
            out_of=100,
            step=1,
            lowest=0,
            value=saved_args.get("lr_warmup_steps", 0),
            scroll_exit=True,
        )
        self.learning_rate = self.add_widget_intelligent(
            npyscreen.TitleText,
            name="Learning Rate:",
            value=str(
                saved_args.get("learning_rate", "5.0e-04"),
            ),
            scroll_exit=True,
        )
        self.scale_lr = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Scale learning rate by number GPUs, steps and batch size",
            value=saved_args.get("scale_lr", True),
            scroll_exit=True,
        )
        self.enable_xformers_memory_efficient_attention = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Use xformers acceleration",
            value=saved_args.get("enable_xformers_memory_efficient_attention", False),
            scroll_exit=True,
        )
        self.lr_scheduler = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name="Learning rate scheduler:",
            values=self.lr_schedulers,
            max_height=7,
            value=self.lr_schedulers.index(saved_args.get("lr_scheduler", "constant")),
            scroll_exit=True,
        )
        self.model.editing = True

    def initializer_changed(self):
        placeholder = self.placeholder_token.value
        self.prompt_token.value = f"(Trigger by using <{placeholder}> in your prompts)"
        self.train_data_dir.value = str(config.root_dir / TRAINING_DATA / placeholder)
        self.output_dir.value = str(config.root_dir / TRAINING_DIR / placeholder)
        self.resume_from_checkpoint.value = Path(self.output_dir.value).exists()

    def on_ok(self):
        if self.validate_field_values():
            self.parentApp.setNextForm(None)
            self.editing = False
            self.parentApp.ti_arguments = self.marshall_arguments()
            npyscreen.notify("Launching textual inversion training. This will take a while...")
        else:
            self.editing = True

    def ok_cancel(self):
        sys.exit(0)

    def validate_field_values(self) -> bool:
        bad_fields = []
        if self.model.value is None:
            bad_fields.append("Model Name must correspond to a known model in models.yaml")
        if not re.match("^[a-zA-Z0-9.-]+$", self.placeholder_token.value):
            bad_fields.append("Trigger term must only contain alphanumeric characters, the dot and hyphen")
        if self.train_data_dir.value is None:
            bad_fields.append("Data Training Directory cannot be empty")
        if self.output_dir.value is None:
            bad_fields.append("The Output Destination Directory cannot be empty")
        if len(bad_fields) > 0:
            message = "The following problems were detected and must be corrected:"
            for problem in bad_fields:
                message += f"\n* {problem}"
            npyscreen.notify_confirm(message)
            return False
        else:
            return True

    def get_model_names(self) -> Tuple[List[str], int]:
        conf = OmegaConf.load(config.root_dir / "configs/models.yaml")
        model_names = [idx for idx in sorted(list(conf.keys())) if conf[idx].get("format", None) == "diffusers"]
        defaults = [idx for idx in range(len(model_names)) if "default" in conf[model_names[idx]]]
        default = defaults[0] if len(defaults) > 0 else 0
        return (model_names, default)

    def marshall_arguments(self) -> dict:
        args = dict()

        # the choices
        args.update(
            model=self.model_names[self.model.value[0]],
            resolution=self.resolutions[self.resolution.value[0]],
            lr_scheduler=self.lr_schedulers[self.lr_scheduler.value[0]],
            mixed_precision=self.precisions[self.mixed_precision.value[0]],
            learnable_property=self.learnable_properties[self.learnable_property.value[0]],
        )

        # all the strings and booleans
        for attr in (
            "initializer_token",
            "placeholder_token",
            "train_data_dir",
            "output_dir",
            "scale_lr",
            "center_crop",
            "enable_xformers_memory_efficient_attention",
        ):
            args[attr] = getattr(self, attr).value

        # all the integers
        for attr in (
            "train_batch_size",
            "gradient_accumulation_steps",
            "num_train_epochs",
            "max_train_steps",
            "lr_warmup_steps",
        ):
            args[attr] = int(getattr(self, attr).value)

        # the floats (just one)
        args.update(learning_rate=float(self.learning_rate.value))

        # a special case
        if self.resume_from_checkpoint.value and Path(self.output_dir.value).exists():
            args["resume_from_checkpoint"] = "latest"

        return args


class MyApplication(npyscreen.NPSAppManaged):
    def __init__(self, saved_args=None):
        super().__init__()
        self.ti_arguments = None
        self.saved_args = saved_args

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.DefaultTheme)
        self.main = self.addForm(
            "MAIN",
            textualInversionForm,
            name="Textual Inversion Settings",
            saved_args=self.saved_args,
        )


def copy_to_embeddings_folder(args: dict):
    """
    Copy learned_embeds.bin into the embeddings folder, and offer to
    delete the full model and checkpoints.
    """
    source = Path(args["output_dir"], "learned_embeds.bin")
    dest_dir_name = args["placeholder_token"].strip("<>")
    destination = config.root_dir / "embeddings" / dest_dir_name
    os.makedirs(destination, exist_ok=True)
    logger.info(f"Training completed. Copying learned_embeds.bin into {str(destination)}")
    shutil.copy(source, destination)
    if (input("Delete training logs and intermediate checkpoints? [y] ") or "y").startswith(("y", "Y")):
        shutil.rmtree(Path(args["output_dir"]))
    else:
        logger.info(f'Keeping {args["output_dir"]}')


def save_args(args: dict):
    """
    Save the current argument values to an omegaconf file
    """
    dest_dir = config.root_dir / TRAINING_DIR
    os.makedirs(dest_dir, exist_ok=True)
    conf_file = dest_dir / CONF_FILE
    conf = OmegaConf.create(args)
    OmegaConf.save(config=conf, f=conf_file)


def previous_args() -> dict:
    """
    Get the previous arguments used.
    """
    conf_file = config.root_dir / TRAINING_DIR / CONF_FILE
    try:
        conf = OmegaConf.load(conf_file)
        conf["placeholder_token"] = conf["placeholder_token"].strip("<>")
    except:
        conf = None

    return conf


def do_front_end(args: Namespace):
    saved_args = previous_args()
    myapplication = MyApplication(saved_args=saved_args)
    myapplication.run()

    if args := myapplication.ti_arguments:
        os.makedirs(args["output_dir"], exist_ok=True)

        # Automatically add angle brackets around the trigger
        if not re.match("^<.+>$", args["placeholder_token"]):
            args["placeholder_token"] = f"<{args['placeholder_token']}>"

        args["only_save_embeds"] = True
        save_args(args)

        try:
            do_textual_inversion_training(InvokeAIAppConfig.get_config(), **args)
            copy_to_embeddings_folder(args)
        except Exception as e:
            logger.error("An exception occurred during training. The exception was:")
            logger.error(str(e))
            logger.error("DETAILS:")
            logger.error(traceback.format_exc())


def main():
    global config

    args = parse_args()
    config = InvokeAIAppConfig.get_config()

    # change root if needed
    if args.root_dir:
        config.root = args.root_dir

    try:
        if args.front_end:
            do_front_end(args)
        else:
            do_textual_inversion_training(config, **vars(args))
    except AssertionError as e:
        logger.error(e)
        sys.exit(-1)
    except KeyboardInterrupt:
        pass
    except (widget.NotEnoughSpaceForWidget, Exception) as e:
        if str(e).startswith("Height of 1 allocated"):
            logger.error("You need to have at least one diffusers models defined in models.yaml in order to train")
        elif str(e).startswith("addwstr"):
            logger.error("Not enough window space for the interface. Please make your window larger and try again.")
        else:
            logger.error(e)
        sys.exit(-1)


if __name__ == "__main__":
    main()
