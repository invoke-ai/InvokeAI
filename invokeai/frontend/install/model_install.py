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

from ...backend.install.model_install_backend import (
    Dataset_path,
    default_config_file,
    default_dataset,
    install_requested_models,
    recommended_datasets,
    ModelInstallList,
    dataclass,
)
from ...backend import ModelManager
from ...backend.util import choose_precision, choose_torch_device
from .widgets import (
    CenteredTitleText,
    MultiSelectColumns,
    SingleSelectColumns,
    OffsetButtonPress,
    TextBox,
    set_min_terminal_size,
)
from invokeai.app.services.config import get_invokeai_config

# minimum size for the UI
MIN_COLS = 120
MIN_LINES = 50

config = get_invokeai_config()

class addModelsForm(npyscreen.FormMultiPage):
    # for responsive resizing - disabled
    # FIX_MINIMUM_SIZE_WHEN_CREATED = False

    def __init__(self, parentApp, name, multipage=False, *args, **keywords):
        self.multipage = multipage

        model_manager = ModelManager(config.model_conf_path)
        
        self.initial_models = OmegaConf.load(Dataset_path)['diffusers']
        self.installed_cn_models = model_manager.list_controlnet_models()

        try:
            self.existing_models = OmegaConf.load(default_config_file())
        except:
            self.existing_models = dict()
        self.starter_model_list = list(self.initial_models.keys())
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

        cn_model_list = sorted(self.installed_cn_models.keys())
        
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
        if len(self.starter_model_list) > 0:
            self.add_widget_intelligent(
                CenteredTitleText,
                name="== DIFFUSERS MODEL STARTER PACK ==",
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
                    if (show_recommended and x in recommended_models)\
                    or (x in self.existing_models)
                ],
                max_height=len(starter_model_labels) + 1,
                relx=4,
                scroll_exit=True,
            )
            self.purge_deleted = self.add_widget_intelligent(
                npyscreen.Checkbox,
                name="Purge unchecked diffusers models from disk",
                value=False,
                scroll_exit=True,
                relx=4,
            )
        self.nextrely += 1            
        self.add_widget_intelligent(
            CenteredTitleText,
            name="== IMPORT MORE DIFFUSERS MODELS FROM YOUR LOCAL DISK OR THE INTERNET ==",
            editable=False,
            color="CONTROL",
        )
        self.nextrely -= 1

        for line in [
            "Enter URLs, file paths, or HuggingFace repository IDs, separated by spaces. Use shift-control-V to paste:",
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
            TextBox, max_height=4, scroll_exit=True, editable=True, relx=4
        )
        self.nextrely += 1
        self.show_directory_fields = self.add_widget_intelligent(
            npyscreen.FormControlCheckbox,
            name="Select a directory for models to import automatically at startup",
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

        self.add_widget_intelligent(
            CenteredTitleText,
            name='_' * (window_width-5),
            editable=False,
            labelColor='CAUTION'
        )
            
        self.nextrely += 1
        self.tabs = self.add_widget_intelligent(
            SingleSelectColumns,
            values=['ADD CONTROLNET MODELS','ADD LORA/LYCORIS MODELS', 'ADD TEXTUAL INVERSION MODELS'],
            value=0,
            columns = 4,
            max_height = 2,
            relx=8,
            scroll_exit = True,
        )
        # self.add_widget_intelligent(
        #     CenteredTitleText,
        #     name="== CONTROLNET MODELS ==",
        #     editable=False,
        #     color="CONTROL",
        # )
        top_of_table = self.nextrely
        self.cn_label_1 = self.add_widget_intelligent(
            CenteredTitleText,
            name="Select the desired models to install. Unchecked models will be purged from disk.",
            editable=False,
            labelColor="CAUTION",
        )
        columns=6
        self.cn_models_selected = self.add_widget_intelligent(
            MultiSelectColumns,
            columns=columns,
            name="Install ControlNet Models",
            values=cn_model_list,
            value=[
                cn_model_list.index(x)
                for x in cn_model_list
                if self.installed_cn_models[x]
            ],
            max_height=len(cn_model_list)//columns + 1,
            relx=4,
            scroll_exit=True,
        )
        self.nextrely += 1        
        self.cn_label_2 = self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name='Additional ControlNet HuggingFace repo_ids to install (Space separated. Use shift-control-V to paste):',
            relx=4,
            color='CONTROL',
            editable=False,
            scroll_exit=True
        )
        self.nextrely -= 1        
        self.additional_controlnet_ids = self.add_widget_intelligent(
            TextBox, max_height=2, scroll_exit=True, editable=True, relx=4
        )

        bottom_of_table = self.nextrely
        self.nextrely = top_of_table
        self.lora_label_1 = self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name='LoRA/LYCORIS models to download and install (Space separated. Use shift-control-V to paste):',
            relx=4,
            color='CONTROL',
            editable=False,
            hidden=True,
            scroll_exit=True
        )
        self.nextrely -= 1
        self.loras = self.add_widget_intelligent(
            TextBox,
            max_height=2,
            scroll_exit=True,
            editable=True,
            relx=4,
            hidden=True,
        )
        self.nextrely = top_of_table
        self.ti_label_1 = self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name='Textual Inversion models to download and install (Space separated. Use shift-control-V to paste):',
            relx=4,
            color='CONTROL',
            editable=False,
            hidden=True,
            scroll_exit=True
        )
        self.nextrely -= 1
        self.tis = self.add_widget_intelligent(
            TextBox,
            max_height=2,
            scroll_exit=True,
            editable=True,
            relx=4,
            hidden=True,
        )
        self.nextrely = bottom_of_table
        self.nextrely += 1
        self.add_widget_intelligent(
            CenteredTitleText,
            name='_' * (window_width-5),
            editable=False,
            labelColor='CAUTION'
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

        # self.tabs.when_value_edited = self._toggle_tables
        self.tabs.on_changed = self._toggle_tables

        self.show_directory_fields.when_value_edited = self._clear_scan_directory

    def resize(self):
        super().resize()
        if hasattr(self, "models_selected"):
            self.models_selected.values = self._get_starter_model_labels()

    def _toggle_tables(self, value=None):
        selected_tab = value[0] if value else self.tabs.value[0]
        widgets = [
            [self.cn_label_1, self.cn_models_selected, self.cn_label_2, self.additional_controlnet_ids],
            [self.lora_label_1,self.loras],
            [self.ti_label_1,self.tis],
        ]

        for group in widgets:
            for w in group:
                w.hidden = True
        for w in widgets[selected_tab]:
            w.hidden = False
        self.display()

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

    def _get_installed_cn_models(self)->list[str]:
        cn_dir = config.controlnet_path
        installed_cn_models = set()
        for root, dirs, files in os.walk(cn_dir):
            for name in dirs:
                if Path(root, name, '.download_complete').exists():
                    installed_cn_models.add(name.replace('--','/'))
        return installed_cn_models

    def _add_additional_cn_models(self, known_models: dict, installed_models: set):
        for i in installed_models:
            if i in known_models:
                continue
            # translate from name to repo_id
            repo_id = i.replace('--','/')
            known_models.update({i: repo_id})
            
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
        selections.purge_deleted_models = self.purge_deleted.value
        
        selections.install_models = [x for x in starter_models if x not in self.existing_models]
        selections.remove_models = [x for x in self.starter_model_list if x in self.existing_models and x not in starter_models]

        selections.install_cn_models = [self.cn_models_selected.values[x]
                                        for x in self.cn_models_selected.value
                                        if not self.installed_cn_models[self.cn_models_selected.values[x]]
                                        ]
        selections.remove_cn_models = [x
                                       for x in self.cn_models_selected.values
                                       if self.installed_cn_models[x]
                                       and self.cn_models_selected.values.index(x) not in self.cn_models_selected.value
                                       ]
        if (additional_cns := self.additional_controlnet_ids.value.split()):
            valid_cns = [x for x in additional_cns if '/' in x]
            selections.install_cn_models.extend(valid_cns)

        # load directory and whether to scan on startup
        if self.show_directory_fields.value:
            selections.scan_directory = self.autoload_directory.value
            selections.autoscan_on_startup = self.autoscan_on_startup.value
        else:
            selections.scan_directory = None
            selections.autoscan_on_startup = False

        # URLs and the like
        selections.import_model_paths = self.import_model_paths.value.split()

@dataclass
class UserSelections():
    install_models: List[str]=None
    remove_models: List[str]=None
    purge_deleted_models: bool=False,
    install_cn_models: List[str] = None,
    remove_cn_models: List[str] = None,
    scan_directory: Path=None,
    autoscan_on_startup: bool=False,
    import_model_paths: str=None,
        
class AddModelApplication(npyscreen.NPSAppManaged):
    def __init__(self):
        super().__init__()
        self.user_cancelled = False
        self.user_selections = UserSelections()

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.DefaultTheme)
        self.main_form = self.addForm(
            "MAIN", addModelsForm, name="Install Stable Diffusion Models"
        )


# --------------------------------------------------------
def process_and_execute(opt: Namespace, selections: Namespace):
    models_to_install = selections.install_models
    models_to_remove = selections.remove_models
    directory_to_scan = selections.scan_directory
    scan_at_startup = selections.autoscan_on_startup
    potential_models_to_install = selections.import_model_paths
    install_requested_models(
        diffusers = ModelInstallList(models_to_install, models_to_remove),
        controlnet = ModelInstallList(selections.install_cn_models, selections.remove_cn_models),
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
    if opt.root and Path(opt.root).exists():
        config.root = Path(opt.root)

    if not (config.root_dir / config.conf_path.parent).exists():
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
