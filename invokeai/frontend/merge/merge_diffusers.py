"""
invokeai.frontend.merge exports a single function call merge_diffusion_models()
used to merge 2-3 models together and create a new InvokeAI-registered diffusion model.

Copyright (c) 2023 Lincoln Stein and the InvokeAI Development Team
"""
import argparse
import curses
import sys
from argparse import Namespace
from pathlib import Path
from typing import List, Optional

import npyscreen
from npyscreen import widget

import invokeai.backend.util.logging as logger
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_management import (
    ModelMerger,
    ModelManager,
    ModelType,
    BaseModelType,
)
from invokeai.frontend.install.widgets import FloatTitleSlider, TextBox, SingleSelectColumns

config = InvokeAIAppConfig.get_config()


def _parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="InvokeAI model merging")
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=config.root,
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
        dest="model_names",
        type=str,
        nargs="+",
        help="Two to three model names to be merged",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        choices=[x.value for x in BaseModelType],
        help="The base model shared by the models to be merged",
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
class mergeModelsForm(npyscreen.FormMultiPageAction):
    interpolations = ["weighted_sum", "sigmoid", "inv_sigmoid"]

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
        self.current_base = 0
        max_width = max([len(x) for x in self.model_names])
        max_width += 6
        horizontal_layout = max_width * 3 < window_width

        self.add_widget_intelligent(
            npyscreen.FixedText,
            color="CONTROL",
            value="Select two models to merge and optionally a third.",
            editable=False,
        )
        self.add_widget_intelligent(
            npyscreen.FixedText,
            color="CONTROL",
            value="Use up and down arrows to move, <space> to select an item, <tab> and <shift-tab> to move from one field to the next.",
            editable=False,
        )
        self.nextrely += 1
        self.base_select = self.add_widget_intelligent(
            SingleSelectColumns,
            values=[
                "Models Built on SD-1.x",
                "Models Built on SD-2.x",
            ],
            value=[self.current_base],
            columns=4,
            max_height=2,
            relx=8,
            scroll_exit=True,
        )
        self.base_select.on_changed = self._populate_models
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="MODEL 1",
            color="GOOD",
            editable=False,
            rely=6 if horizontal_layout else None,
        )
        self.model1 = self.add_widget_intelligent(
            npyscreen.SelectOne,
            values=self.model_names,
            value=0,
            max_height=len(self.model_names),
            max_width=max_width,
            scroll_exit=True,
            rely=7,
        )
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="MODEL 2",
            color="GOOD",
            editable=False,
            relx=max_width + 3 if horizontal_layout else None,
            rely=6 if horizontal_layout else None,
        )
        self.model2 = self.add_widget_intelligent(
            npyscreen.SelectOne,
            name="(2)",
            values=self.model_names,
            value=1,
            max_height=len(self.model_names),
            max_width=max_width,
            relx=max_width + 3 if horizontal_layout else None,
            rely=7 if horizontal_layout else None,
            scroll_exit=True,
        )
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="MODEL 3",
            color="GOOD",
            editable=False,
            relx=max_width * 2 + 3 if horizontal_layout else None,
            rely=6 if horizontal_layout else None,
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
            rely=7 if horizontal_layout else None,
        )
        for m in [self.model1, self.model2, self.model3]:
            m.when_value_edited = self.models_changed
        self.merged_model_name = self.add_widget_intelligent(
            TextBox,
            name="Name for merged model:",
            labelColor="CONTROL",
            max_height=3,
            value="",
            scroll_exit=True,
        )
        self.force = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Force merge of models created by different diffusers library versions",
            labelColor="CONTROL",
            value=True,
            scroll_exit=True,
        )
        self.nextrely += 1
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
            out_of=1.0,
            step=0.01,
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
            self.merge_method.values = ["add_difference ( A+(B-C) )"]
            self.merged_model_name.value += f"+{models[selected_model3 -1]}"  # In model3 there is one more element in the list (None). So we have to subtract one.
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
            interp = "add_difference"
        else:
            interp = self.interpolations[self.merge_method.value[0]]

        args = dict(
            model_names=models,
            base_model=tuple(BaseModelType)[self.base_select.value[0]],
            alpha=self.alpha.value,
            interp=interp,
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
        selected_models = set((model_names[self.model1.value[0]], model_names[self.model2.value[0]]))
        if self.model3.value[0] > 0:
            selected_models.add(model_names[self.model3.value[0] - 1])
        if len(selected_models) < 2:
            bad_fields.append(f"Please select two or three DIFFERENT models to compare. You selected {selected_models}")
        if len(bad_fields) > 0:
            message = "The following problems were detected and must be corrected:"
            for problem in bad_fields:
                message += f"\n* {problem}"
            npyscreen.notify_confirm(message)
            return False
        else:
            return True

    def get_model_names(self, base_model: Optional[BaseModelType] = None) -> List[str]:
        model_names = [
            info["model_name"]
            for info in self.model_manager.list_models(model_type=ModelType.Main, base_model=base_model)
            if info["model_format"] == "diffusers"
        ]
        return sorted(model_names)

    def _populate_models(self, value=None):
        base_model = tuple(BaseModelType)[value[0]]
        self.model_names = self.get_model_names(base_model)

        models_plus_none = self.model_names.copy()
        models_plus_none.insert(0, "None")
        self.model1.values = self.model_names
        self.model2.values = self.model_names
        self.model3.values = models_plus_none

        self.display()


class Mergeapp(npyscreen.NPSAppManaged):
    def __init__(self, model_manager: ModelManager):
        super().__init__()
        self.model_manager = model_manager

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.ElegantTheme)
        self.main = self.addForm("MAIN", mergeModelsForm, name="Merge Models Settings")


def run_gui(args: Namespace):
    model_manager = ModelManager(config.model_conf_path)
    mergeapp = Mergeapp(model_manager)
    mergeapp.run()

    args = mergeapp.merge_arguments
    merger = ModelMerger(model_manager)
    merger.merge_diffusion_models_and_save(**args)
    logger.info(f'Models merged into new model: "{args["merged_model_name"]}".')


def run_cli(args: Namespace):
    assert args.alpha >= 0 and args.alpha <= 1.0, "alpha must be between 0 and 1"
    assert (
        args.model_names and len(args.model_names) >= 1 and len(args.model_names) <= 3
    ), "Please provide the --models argument to list 2 to 3 models to merge. Use --help for full usage."

    if not args.merged_model_name:
        args.merged_model_name = "+".join(args.model_names)
        logger.info(f'No --merged_model_name provided. Defaulting to "{args.merged_model_name}"')

    model_manager = ModelManager(config.model_conf_path)
    assert (
        not model_manager.model_exists(args.merged_model_name, args.base_model, ModelType.Main) or args.clobber
    ), f'A model named "{args.merged_model_name}" already exists. Use --clobber to overwrite.'

    merger = ModelMerger(model_manager)
    merger.merge_diffusion_models_and_save(**vars(args))
    logger.info(f'Models merged into new model: "{args.merged_model_name}".')


def main():
    args = _parse_args()
    if args.root_dir:
        config.parse_args(["--root", str(args.root_dir)])

    try:
        if args.front_end:
            run_gui(args)
        else:
            run_cli(args)
    except widget.NotEnoughSpaceForWidget as e:
        if str(e).startswith("Height of 1 allocated"):
            logger.error("You need to have at least two diffusers models defined in models.yaml in order to merge")
        else:
            logger.error("Not enough room for the user interface. Try making this window larger.")
        sys.exit(-1)
    except Exception as e:
        logger.error(e)
        sys.exit(-1)
    except KeyboardInterrupt:
        sys.exit(-1)


if __name__ == "__main__":
    main()
