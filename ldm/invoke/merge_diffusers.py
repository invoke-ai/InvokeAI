"""
ldm.invoke.merge_diffusers exports a single function call merge_diffusion_models()
used to merge 2-3 models together and create a new InvokeAI-registered diffusion model.

Copyright (c) 2023 Lincoln Stein and the InvokeAI Development Team
"""
import argparse
import curses
import os
import sys
import traceback
import warnings
from argparse import Namespace
from pathlib import Path
from typing import List, Union

import npyscreen
from diffusers import DiffusionPipeline
from diffusers import logging as dlogging
from npyscreen import widget
from omegaconf import OmegaConf

from ldm.invoke.globals import (Globals, global_cache_dir, global_config_file,
                                global_models_dir, global_set_root)
from ldm.invoke.model_manager import ModelManager

DEST_MERGED_MODEL_DIR = "merged_models"


def merge_diffusion_models(
    model_ids_or_paths: List[Union[str, Path]],
    alpha: float = 0.5,
    interp: str = None,
    force: bool = False,
    **kwargs,
) -> DiffusionPipeline:
    """
    model_ids_or_paths - up to three models, designated by their local paths or HuggingFace repo_ids
    alpha  - The interpolation parameter. Ranges from 0 to 1.  It affects the ratio in which the checkpoints are merged. A 0.8 alpha
               would mean that the first model checkpoints would affect the final result far less than an alpha of 0.2
    interp - The interpolation method to use for the merging. Supports "sigmoid", "inv_sigmoid", "add_difference" and None.
               Passing None uses the default interpolation which is weighted sum interpolation. For merging three checkpoints, only "add_difference" is supported.
    force  - Whether to ignore mismatch in model_config.json for the current models. Defaults to False.

    **kwargs - the default DiffusionPipeline.get_config_dict kwargs:
         cache_dir, resume_download, force_download, proxies, local_files_only, use_auth_token, revision, torch_dtype, device_map
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        verbosity = dlogging.get_verbosity()
        dlogging.set_verbosity_error()

        pipe = DiffusionPipeline.from_pretrained(
            model_ids_or_paths[0],
            cache_dir=kwargs.get("cache_dir", global_cache_dir()),
            custom_pipeline="checkpoint_merger",
        )
        merged_pipe = pipe.merge(
            pretrained_model_name_or_path_list=model_ids_or_paths,
            alpha=alpha,
            interp=interp,
            force=force,
            **kwargs,
        )
        dlogging.set_verbosity(verbosity)
    return merged_pipe


def merge_diffusion_models_and_commit(
    models: List["str"],
    merged_model_name: str,
    alpha: float = 0.5,
    interp: str = None,
    force: bool = False,
    **kwargs,
):
    """
    models - up to three models, designated by their InvokeAI models.yaml model name
    merged_model_name = name for new model
    alpha  - The interpolation parameter. Ranges from 0 to 1.  It affects the ratio in which the checkpoints are merged. A 0.8 alpha
               would mean that the first model checkpoints would affect the final result far less than an alpha of 0.2
    interp - The interpolation method to use for the merging. Supports "sigmoid", "inv_sigmoid", "add_difference" and None.
               Passing None uses the default interpolation which is weighted sum interpolation. For merging three checkpoints, only "add_difference" is supported.
    force  - Whether to ignore mismatch in model_config.json for the current models. Defaults to False.

    **kwargs - the default DiffusionPipeline.get_config_dict kwargs:
         cache_dir, resume_download, force_download, proxies, local_files_only, use_auth_token, revision, torch_dtype, device_map
    """
    config_file = global_config_file()
    model_manager = ModelManager(OmegaConf.load(config_file))
    for mod in models:
        assert mod in model_manager.model_names(), f'** Unknown model "{mod}"'
        assert (
            model_manager.model_info(mod).get("format", None) == "diffusers"
        ), f"** {mod} is not a diffusers model. It must be optimized before merging."
    model_ids_or_paths = [model_manager.model_name_or_path(x) for x in models]

    merged_pipe = merge_diffusion_models(
        model_ids_or_paths, alpha, interp, force, **kwargs
    )
    dump_path = global_models_dir() / DEST_MERGED_MODEL_DIR

    os.makedirs(dump_path, exist_ok=True)
    dump_path = dump_path / merged_model_name
    merged_pipe.save_pretrained(dump_path, safe_serialization=1)
    import_args = dict(
        model_name=merged_model_name, description=f'Merge of models {", ".join(models)}'
    )
    if vae := model_manager.config[models[0]].get("vae", None):
        print(f">> Using configured VAE assigned to {models[0]}")
        import_args.update(vae=vae)
    model_manager.import_diffuser_model(dump_path, **import_args)
    model_manager.commit(config_file)


def _parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="InvokeAI model merging")
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=Globals.root,
        help="Path to the invokeai runtime directory",
    )
    parser.add_argument(
        "--front_end",
        "--gui",
        dest="front_end",
        action="store_true",
        default=False,
        help="Activate the text-based graphical front end for collecting parameters. Aside from --root_dir, other parameters will be ignored.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Two to three model names to be merged",
    )
    parser.add_argument(
        "--merged_model_name",
        "--destination",
        dest="merged_model_name",
        type=str,
        help="Name of the output model. If not specified, will be the concatenation of the input model names.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="The interpolation parameter, ranging from 0 to 1. It affects the ratio in which the checkpoints are merged. Higher values give more weight to the 2d and 3d models",
    )
    parser.add_argument(
        "--interpolation",
        dest="interp",
        type=str,
        choices=["weighted_sum", "sigmoid", "inv_sigmoid", "add_difference"],
        default="weighted_sum",
        help='Interpolation method to use. If three models are present, only "add_difference" will work.',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Try to merge models even if they are incompatible with each other",
    )
    parser.add_argument(
        "--clobber",
        "--overwrite",
        dest="clobber",
        action="store_true",
        help="Overwrite the merged model if --merged_model_name already exists",
    )
    return parser.parse_args()


# ------------------------- GUI HERE -------------------------
class FloatSlider(npyscreen.Slider):
    # this is supposed to adjust display precision, but doesn't
    def translate_value(self):
        stri = "%3.2f / %3.2f" % (self.value, self.out_of)
        l = (len(str(self.out_of))) * 2 + 4
        stri = stri.rjust(l)
        return stri


class FloatTitleSlider(npyscreen.TitleText):
    _entry_type = FloatSlider


class mergeModelsForm(npyscreen.FormMultiPageAction):
    interpolations = ["weighted_sum", "sigmoid", "inv_sigmoid", "add_difference"]

    def __init__(self, parentApp, name):
        self.parentApp = parentApp
        self.ALLOW_RESIZE = True
        self.FIX_MINIMUM_SIZE_WHEN_CREATED = False
        super().__init__(parentApp, name)

    @property
    def model_manager(self):
        return self.parentApp.model_manager

    def afterEditing(self):
        self.parentApp.setNextForm(None)

    def create(self):
        window_height, window_width = curses.initscr().getmaxyx()

        self.model_names = self.get_model_names()
        max_width = max([len(x) for x in self.model_names])
        max_width += 6
        horizontal_layout = max_width * 3 < window_width

        self.add_widget_intelligent(
            npyscreen.FixedText,
            color="CONTROL",
            value=f"Select two models to merge and optionally a third.",
            editable=False,
        )
        self.add_widget_intelligent(
            npyscreen.FixedText,
            color="CONTROL",
            value=f"Use up and down arrows to move, <space> to select an item, <tab> and <shift-tab> to move from one field to the next.",
            editable=False,
        )
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="MODEL 1",
            color="GOOD",
            editable=False,
            rely=4 if horizontal_layout else None,
        )
        self.model1 = self.add_widget_intelligent(
            npyscreen.SelectOne,
            values=self.model_names,
            value=0,
            max_height=len(self.model_names),
            max_width=max_width,
            scroll_exit=True,
            rely=5,
        )
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="MODEL 2",
            color="GOOD",
            editable=False,
            relx=max_width + 3 if horizontal_layout else None,
            rely=4 if horizontal_layout else None,
        )
        self.model2 = self.add_widget_intelligent(
            npyscreen.SelectOne,
            name="(2)",
            values=self.model_names,
            value=1,
            max_height=len(self.model_names),
            max_width=max_width,
            relx=max_width + 3 if horizontal_layout else None,
            rely=5 if horizontal_layout else None,
            scroll_exit=True,
        )
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="MODEL 3",
            color="GOOD",
            editable=False,
            relx=max_width * 2 + 3 if horizontal_layout else None,
            rely=4 if horizontal_layout else None,
        )
        models_plus_none = self.model_names.copy()
        models_plus_none.insert(0, "None")
        self.model3 = self.add_widget_intelligent(
            npyscreen.SelectOne,
            name="(3)",
            values=models_plus_none,
            value=0,
            max_height=len(self.model_names) + 1,
            max_width=max_width,
            scroll_exit=True,
            relx=max_width * 2 + 3 if horizontal_layout else None,
            rely=5 if horizontal_layout else None,
        )
        for m in [self.model1, self.model2, self.model3]:
            m.when_value_edited = self.models_changed
        self.merged_model_name = self.add_widget_intelligent(
            npyscreen.TitleText,
            name="Name for merged model:",
            labelColor="CONTROL",
            value="",
            scroll_exit=True,
        )
        self.force = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Force merge of incompatible models",
            labelColor="CONTROL",
            value=False,
            scroll_exit=True,
        )
        self.merge_method = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name="Merge Method:",
            values=self.interpolations,
            value=0,
            labelColor="CONTROL",
            max_height=len(self.interpolations) + 1,
            scroll_exit=True,
        )
        self.alpha = self.add_widget_intelligent(
            FloatTitleSlider,
            name="Weight (alpha) to assign to second and third models:",
            out_of=1,
            step=0.05,
            lowest=0,
            value=0.5,
            labelColor="CONTROL",
            scroll_exit=True,
        )
        self.model1.editing = True

    def models_changed(self):
        models = self.model1.values
        selected_model1 = self.model1.value[0]
        selected_model2 = self.model2.value[0]
        selected_model3 = self.model3.value[0]
        merged_model_name = f"{models[selected_model1]}+{models[selected_model2]}"
        self.merged_model_name.value = merged_model_name

        if selected_model3 > 0:
            self.merge_method.values = (["add_difference"],)
            self.merged_model_name.value += f"+{models[selected_model3]}"
        else:
            self.merge_method.values = self.interpolations
        self.merge_method.value = 0

    def on_ok(self):
        if self.validate_field_values() and self.check_for_overwrite():
            self.parentApp.setNextForm(None)
            self.editing = False
            self.parentApp.merge_arguments = self.marshall_arguments()
            npyscreen.notify("Starting the merge...")
        else:
            self.editing = True

    def on_cancel(self):
        sys.exit(0)

    def marshall_arguments(self) -> dict:
        model_names = self.model_names
        models = [
            model_names[self.model1.value[0]],
            model_names[self.model2.value[0]],
        ]
        if self.model3.value[0] > 0:
            models.append(model_names[self.model3.value[0] - 1])

        args = dict(
            models=models,
            alpha=self.alpha.value,
            interp=self.interpolations[self.merge_method.value[0]],
            force=self.force.value,
            merged_model_name=self.merged_model_name.value,
        )
        return args

    def check_for_overwrite(self) -> bool:
        model_out = self.merged_model_name.value
        if model_out not in self.model_names:
            return True
        else:
            return npyscreen.notify_yes_no(
                f"The chosen merged model destination, {model_out}, is already in use. Overwrite?"
            )

    def validate_field_values(self) -> bool:
        bad_fields = []
        model_names = self.model_names
        selected_models = set(
            (model_names[self.model1.value[0]], model_names[self.model2.value[0]])
        )
        if self.model3.value[0] > 0:
            selected_models.add(model_names[self.model3.value[0] - 1])
        if len(selected_models) < 2:
            bad_fields.append(
                f"Please select two or three DIFFERENT models to compare. You selected {selected_models}"
            )
        if len(bad_fields) > 0:
            message = "The following problems were detected and must be corrected:"
            for problem in bad_fields:
                message += f"\n* {problem}"
            npyscreen.notify_confirm(message)
            return False
        else:
            return True

    def get_model_names(self) -> List[str]:
        model_names = [
            name
            for name in self.model_manager.model_names()
            if self.model_manager.model_info(name).get("format") == "diffusers"
        ]
        print(model_names)
        return sorted(model_names)


class Mergeapp(npyscreen.NPSAppManaged):
    def __init__(self):
        super().__init__()
        conf = OmegaConf.load(global_config_file())
        self.model_manager = ModelManager(
            conf, "cpu", "float16"
        )  # precision doesn't really matter here

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.ElegantTheme)
        self.main = self.addForm("MAIN", mergeModelsForm, name="Merge Models Settings")


def run_gui(args: Namespace):
    mergeapp = Mergeapp()
    mergeapp.run()

    args = mergeapp.merge_arguments
    merge_diffusion_models_and_commit(**args)
    print(f'>> Models merged into new model: "{args["merged_model_name"]}".')


def run_cli(args: Namespace):
    assert args.alpha >= 0 and args.alpha <= 1.0, "alpha must be between 0 and 1"
    assert (
        args.models and len(args.models) >= 1 and len(args.models) <= 3
    ), "Please provide the --models argument to list 2 to 3 models to merge. Use --help for full usage."

    if not args.merged_model_name:
        args.merged_model_name = "+".join(args.models)
        print(
            f'>> No --merged_model_name provided. Defaulting to "{args.merged_model_name}"'
        )

        model_manager = ModelManager(OmegaConf.load(global_config_file()))
        assert (
            args.clobber or args.merged_model_name not in model_manager.model_names()
        ), f'A model named "{args.merged_model_name}" already exists. Use --clobber to overwrite.'

        merge_diffusion_models_and_commit(**vars(args))
        print(f'>> Models merged into new model: "{args.merged_model_name}".')


def main():
    args = _parse_args()
    global_set_root(args.root_dir)

    cache_dir = str(global_cache_dir("diffusers"))
    os.environ[
        "HF_HOME"
    ] = cache_dir  # because not clear the merge pipeline is honoring cache_dir
    args.cache_dir = cache_dir

    try:
        if args.front_end:
            run_gui(args)
        else:
            run_cli(args)
    except widget.NotEnoughSpaceForWidget as e:
        if str(e).startswith("Height of 1 allocated"):
            print(
                "** You need to have at least two diffusers models defined in models.yaml in order to merge"
            )
        else:
            print(f"** A layout error has occurred: {str(e)}")
        sys.exit(-1)
    except Exception as e:
        print(">> An error occurred:")
        traceback.print_exc()
        sys.exit(-1)
    except KeyboardInterrupt:
        sys.exit(-1)


if __name__ == "__main__":
    main()
