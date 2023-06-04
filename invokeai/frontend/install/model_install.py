#!/usr/bin/env python
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)
# Before running stable-diffusion on an internet-isolated machine,
# run this script from one with internet connectivity. The
# two machines must share a common .cache directory.

"""
This is the npyscreen frontend to the model installation application.
The work is actually done in backend code in model_install_backend.py.
"""

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path
from shutil import get_terminal_size
from typing import List

import npyscreen
import torch
from npyscreen import widget
from omegaconf import OmegaConf

import invokeai.backend.util.logging as logger

from ...backend.config.model_install_backend import (
    Dataset_path,
    default_config_file,
    default_dataset,
    get_root,
    install_requested_models,
    recommended_datasets,
)
from ...backend.util import choose_precision, choose_torch_device
from .widgets import (
    CenteredTitleText,
    MultiSelectColumns,
    OffsetButtonPress,
    TextBox,
    set_min_terminal_size,
)
from invokeai.app.services.config import get_invokeai_config

# minimum size for the UI
MIN_COLS = 120
MIN_LINES = 45

config = get_invokeai_config(argv=[])

class addModelsForm(npyscreen.FormMultiPage):
    # for responsive resizing - disabled
    # FIX_MINIMUM_SIZE_WHEN_CREATED = False

    def __init__(self, parentApp, name, multipage=False, *args, **keywords):
        self.multipage = multipage
        self.initial_models = OmegaConf.load(Dataset_path)
        try:
            self.existing_models = OmegaConf.load(default_config_file())
        except:
            self.existing_models = dict()
        self.starter_model_list = [
            x for x in list(self.initial_models.keys()) if x not in self.existing_models
        ]
        self.installed_models = dict()
        super().__init__(parentApp=parentApp, name=name, *args, **keywords)

    def create(self):
        window_width, window_height = get_terminal_size()
        starter_model_labels = self._get_starter_model_labels()
        recommended_models = [
            x
            for x in self.starter_model_list
            if self.initial_models[x].get("recommended", False)
        ]
        self.installed_models = sorted(
            [x for x in list(self.initial_models.keys()) if x in self.existing_models]
        )
        self.nextrely -= 1
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="Use ctrl-N and ctrl-P to move to the <N>ext and <P>revious fields,",
            editable=False,
            color="CAUTION",
        )
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="Use cursor arrows to make a selection, and space to toggle checkboxes.",
            editable=False,
            color="CAUTION",
        )
        self.nextrely += 1
        if len(self.installed_models) > 0:
            self.add_widget_intelligent(
                CenteredTitleText,
                name="== INSTALLED STARTER MODELS ==",
                editable=False,
                color="CONTROL",
            )
            self.nextrely -= 1
            self.add_widget_intelligent(
                CenteredTitleText,
                name="Currently installed starter models. Uncheck to delete:",
                editable=False,
                labelColor="CAUTION",
            )
            self.nextrely -= 1
            columns = self._get_columns()
            self.previously_installed_models = self.add_widget_intelligent(
                MultiSelectColumns,
                columns=columns,
                values=self.installed_models,
                value=[x for x in range(0, len(self.installed_models))],
                max_height=1 + len(self.installed_models) // columns,
                relx=4,
                slow_scroll=True,
                scroll_exit=True,
            )
            self.purge_deleted = self.add_widget_intelligent(
                npyscreen.Checkbox,
                name="Purge deleted models from disk",
                value=False,
                scroll_exit=True,
                relx=4,
            )
        self.nextrely += 1
        if len(self.starter_model_list) > 0:
            self.add_widget_intelligent(
                CenteredTitleText,
                name="== STARTER MODELS (recommended ones selected) ==",
                editable=False,
                color="CONTROL",
            )
            self.nextrely -= 1
            self.add_widget_intelligent(
                CenteredTitleText,
                name="Select from a starter set of Stable Diffusion models from HuggingFace.",
                editable=False,
                labelColor="CAUTION",
            )
            self.nextrely -= 1
            # if user has already installed some initial models, then don't patronize them
            # by showing more recommendations
            show_recommended = not self.existing_models
            self.models_selected = self.add_widget_intelligent(
                npyscreen.MultiSelect,
                name="Install Starter Models",
                values=starter_model_labels,
                value=[
                    self.starter_model_list.index(x)
                    for x in self.starter_model_list
                    if show_recommended and x in recommended_models
                ],
                max_height=len(starter_model_labels) + 1,
                relx=4,
                scroll_exit=True,
            )
        self.add_widget_intelligent(
            CenteredTitleText,
            name="== IMPORT LOCAL AND REMOTE MODELS ==",
            editable=False,
            color="CONTROL",
        )
        self.nextrely -= 1

        for line in [
            "In the box below, enter URLs, file paths, or HuggingFace repository IDs.",
            "Separate model names by lines or whitespace (Use shift-control-V to paste):",
        ]:
            self.add_widget_intelligent(
                CenteredTitleText,
                name=line,
                editable=False,
                labelColor="CONTROL",
                relx=4,
            )
            self.nextrely -= 1
        self.import_model_paths = self.add_widget_intelligent(
            TextBox, max_height=7, scroll_exit=True, editable=True, relx=4
        )
        self.nextrely += 1
        self.show_directory_fields = self.add_widget_intelligent(
            npyscreen.FormControlCheckbox,
            name="Select a directory for models to import",
            value=False,
        )
        self.autoload_directory = self.add_widget_intelligent(
            npyscreen.TitleFilename,
            name="Directory (<tab> autocompletes):",
            select_dir=True,
            must_exist=True,
            use_two_lines=False,
            labelColor="DANGER",
            begin_entry_at=34,
            scroll_exit=True,
        )
        self.autoscan_on_startup = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Scan this directory each time InvokeAI starts for new models to import",
            value=False,
            relx=4,
            scroll_exit=True,
        )
        self.cancel = self.add_widget_intelligent(
            npyscreen.ButtonPress,
            name="CANCEL",
            rely=-3,
            when_pressed_function=self.on_cancel,
        )
        done_label = "DONE"
        back_label = "BACK"
        button_length = len(done_label)
        button_offset = 0
        if self.multipage:
            button_length += len(back_label) + 1
            button_offset += len(back_label) + 1
            self.back_button = self.add_widget_intelligent(
                OffsetButtonPress,
                name=back_label,
                relx=(window_width - button_length) // 2,
                offset=-3,
                rely=-3,
                when_pressed_function=self.on_back,
            )
        self.ok_button = self.add_widget_intelligent(
            OffsetButtonPress,
            name=done_label,
            offset=+3,
            relx=button_offset + 1 + (window_width - button_length) // 2,
            rely=-3,
            when_pressed_function=self.on_ok,
        )

        for i in [self.autoload_directory, self.autoscan_on_startup]:
            self.show_directory_fields.addVisibleWhenSelected(i)

        self.show_directory_fields.when_value_edited = self._clear_scan_directory

    def resize(self):
        super().resize()
        if hasattr(self, "models_selected"):
            self.models_selected.values = self._get_starter_model_labels()

    def _clear_scan_directory(self):
        if not self.show_directory_fields.value:
            self.autoload_directory.value = ""

    def _get_starter_model_labels(self) -> List[str]:
        window_width, window_height = get_terminal_size()
        label_width = 25
        checkbox_width = 4
        spacing_width = 2
        description_width = window_width - label_width - checkbox_width - spacing_width
        im = self.initial_models
        names = self.starter_model_list
        descriptions = [
            im[x].description[0 : description_width - 3] + "..."
            if len(im[x].description) > description_width
            else im[x].description
            for x in names
        ]
        return [
            f"%-{label_width}s %s" % (names[x], descriptions[x])
            for x in range(0, len(names))
        ]

    def _get_columns(self) -> int:
        window_width, window_height = get_terminal_size()
        cols = (
            4
            if window_width > 240
            else 3
            if window_width > 160
            else 2
            if window_width > 80
            else 1
        )
        return min(cols, len(self.installed_models))

    def on_ok(self):
        self.parentApp.setNextForm(None)
        self.editing = False
        self.parentApp.user_cancelled = False
        self.marshall_arguments()

    def on_back(self):
        self.parentApp.switchFormPrevious()
        self.editing = False

    def on_cancel(self):
        if npyscreen.notify_yes_no(
            "Are you sure you want to cancel?\nYou may re-run this script later using the invoke.sh or invoke.bat command.\n"
        ):
            self.parentApp.setNextForm(None)
            self.parentApp.user_cancelled = True
            self.editing = False

    def marshall_arguments(self):
        """
        Assemble arguments and store as attributes of the application:
        .starter_models: dict of model names to install from INITIAL_CONFIGURE.yaml
                         True  => Install
                         False => Remove
        .scan_directory: Path to a directory of models to scan and import
        .autoscan_on_startup:  True if invokeai should scan and import at startup time
        .import_model_paths:   list of URLs, repo_ids and file paths to import
        """
        # we're using a global here rather than storing the result in the parentapp
        # due to some bug in npyscreen that is causing attributes to be lost
        selections = self.parentApp.user_selections

        # starter models to install/remove
        if hasattr(self, "models_selected"):
            starter_models = dict(
                map(
                    lambda x: (self.starter_model_list[x], True),
                    self.models_selected.value,
                )
            )
        else:
            starter_models = dict()
        selections.purge_deleted_models = False
        if hasattr(self, "previously_installed_models"):
            unchecked = [
                self.previously_installed_models.values[x]
                for x in range(0, len(self.previously_installed_models.values))
                if x not in self.previously_installed_models.value
            ]
            starter_models.update(map(lambda x: (x, False), unchecked))
            selections.purge_deleted_models = self.purge_deleted.value
        selections.starter_models = starter_models

        # load directory and whether to scan on startup
        if self.show_directory_fields.value:
            selections.scan_directory = self.autoload_directory.value
            selections.autoscan_on_startup = self.autoscan_on_startup.value
        else:
            selections.scan_directory = None
            selections.autoscan_on_startup = False

        # URLs and the like
        selections.import_model_paths = self.import_model_paths.value.split()


class AddModelApplication(npyscreen.NPSAppManaged):
    def __init__(self):
        super().__init__()
        self.user_cancelled = False
        self.user_selections = Namespace(
            starter_models=None,
            purge_deleted_models=False,
            scan_directory=None,
            autoscan_on_startup=None,
            import_model_paths=None,
        )

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.DefaultTheme)
        self.main_form = self.addForm(
            "MAIN", addModelsForm, name="Install Stable Diffusion Models"
        )


# --------------------------------------------------------
def process_and_execute(opt: Namespace, selections: Namespace):
    models_to_remove = [
        x for x in selections.starter_models if not selections.starter_models[x]
    ]
    models_to_install = [
        x for x in selections.starter_models if selections.starter_models[x]
    ]
    directory_to_scan = selections.scan_directory
    scan_at_startup = selections.autoscan_on_startup
    potential_models_to_install = selections.import_model_paths

    install_requested_models(
        install_initial_models=models_to_install,
        remove_models=models_to_remove,
        scan_directory=Path(directory_to_scan) if directory_to_scan else None,
        external_models=potential_models_to_install,
        scan_at_startup=scan_at_startup,
        precision="float32"
        if opt.full_precision
        else choose_precision(torch.device(choose_torch_device())),
        purge_deleted=selections.purge_deleted_models,
        config_file_path=Path(opt.config_file) if opt.config_file else None,
    )


# --------------------------------------------------------
def select_and_download_models(opt: Namespace):
    precision = (
        "float32"
        if opt.full_precision
        else choose_precision(torch.device(choose_torch_device()))
    )
    if opt.default_only:
        install_requested_models(
            install_initial_models=default_dataset(),
            precision=precision,
        )
    elif opt.yes_to_all:
        install_requested_models(
            install_initial_models=recommended_datasets(),
            precision=precision,
        )
    else:
        set_min_terminal_size(MIN_COLS, MIN_LINES)
        installApp = AddModelApplication()
        installApp.run()

        if not installApp.user_cancelled:
            process_and_execute(opt, installApp.user_selections)


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
    config.root = os.path.expanduser(get_root(opt.root) or "")

    if not (config.conf_path / '..' ).exists():
        logger.info(
            "Your InvokeAI root directory is not set up. Calling invokeai-configure."
        )
        from invokeai.frontend.install import invokeai_configure

        invokeai_configure()
        sys.exit(0)

    try:
        select_and_download_models(opt)
    except AssertionError as e:
        logger.error(e)
        sys.exit(-1)
    except KeyboardInterrupt:
        logger.info("Goodbye! Come back soon.")
    except widget.NotEnoughSpaceForWidget as e:
        if str(e).startswith("Height of 1 allocated"):
            logger.error(
                "Insufficient vertical space for the interface. Please make your window taller and try again"
            )
        elif str(e).startswith("addwstr"):
            logger.error(
                "Insufficient horizontal space for the interface. Please make your window wider and try again."
            )


# -------------------------------------
if __name__ == "__main__":
    main()
