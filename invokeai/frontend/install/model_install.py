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
import curses
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
from dataclasses import dataclass,field

from ...backend.install.model_install_backend import (
    Dataset_path,
    default_config_file,
    default_dataset,
    install_requested_models,
    recommended_datasets,
    ModelInstallList,
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
    
    # for persistence
    current_tab = 0

    def __init__(self, parentApp, name, multipage=False, *args, **keywords):
        self.multipage = multipage

        model_manager = ModelManager(config.model_conf_path)
        
        self.initial_models = OmegaConf.load(Dataset_path)['diffusers']
        self.installed_cn_models = model_manager.list_controlnet_models()
        self.installed_lora_models = model_manager.list_lora_models()
        self.installed_ti_models = model_manager.list_ti_models()

        try:
            self.existing_models = OmegaConf.load(default_config_file())
        except:
            self.existing_models = dict()
        self.starter_model_list = list(self.initial_models.keys())
        self.installed_models = dict()
        super().__init__(parentApp=parentApp, name=name, *args, **keywords)

    def create(self):
        window_width, window_height = get_terminal_size()

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
        self.tabs = self.add_widget_intelligent(
            SingleSelectColumns,
            values=[
                'DIFFUSERS MODELS',
                'CONTROLNET MODELS',
                'LORA/LYCORIS MODELS',
                'TEXTUAL INVERSION MODELS'
            ],
            value=[self.current_tab],
            columns = 4,
            max_height = 2,
            relx=8,
            scroll_exit = True,
        )
        self.tabs.on_changed = self._toggle_tables
        
        top_of_table = self.nextrely
        self.diffusers_models = self.add_diffusers()
        bottom_of_table = self.nextrely

        self.nextrely = top_of_table
        self.controlnet_models = self.add_controlnets()

        self.nextrely = top_of_table
        self.lora_models = self.add_loras()

        self.nextrely = top_of_table
        self.ti_models = self.add_tis()
                
        self.nextrely = bottom_of_table
        
        self.nextrely += 1
        done_label = "INSTALL/REMOVE"
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

        self.cancel = self.add_widget_intelligent(
            npyscreen.ButtonPress,
            name="QUIT",
            rely=-3,
            relx=window_width-20,
            when_pressed_function=self.on_cancel,
        )

        # This restores the selected page on return from an installation
        for i in range(1,self.current_tab+1):
            self.tabs.h_cursor_line_down(1)
        self._toggle_tables([self.current_tab])

    def add_diffusers(self)->dict[str, npyscreen.widget]:
        '''Add widgets responsible for selecting diffusers models'''
        widgets = dict()

        starter_model_labels = self._get_starter_model_labels()
        recommended_models = [
            x
            for x in self.starter_model_list
            if self.initial_models[x].get("recommended", False)
        ]
        self.installed_models = sorted(
            [x for x in list(self.initial_models.keys()) if x in self.existing_models]
        )

        widgets.update(
            label1 = self.add_widget_intelligent(
                CenteredTitleText,
                name="Select from a starter set of Stable Diffusion models from HuggingFace.",
                editable=False,
                labelColor="CAUTION",
            )
        )
        
        self.nextrely -= 1
        # if user has already installed some initial models, then don't patronize them
        # by showing more recommendations
        show_recommended = not self.existing_models
        widgets.update(
            models_selected = self.add_widget_intelligent(
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
        )

        widgets.update(
            purge_deleted = self.add_widget_intelligent(
                npyscreen.Checkbox,
                name="Purge unchecked diffusers models from disk",
                value=False,
                scroll_exit=True,
                relx=4,
            )
        )
        
        self.nextrely += 1
        widgets.update(
            label3 = self.add_widget_intelligent(
                CenteredTitleText,
                name="== IMPORT MORE DIFFUSERS MODELS FROM YOUR LOCAL DISK OR THE INTERNET ==",
                editable=False,
                color="CONTROL",
            )
        )

        self.nextrely -= 1
        widgets.update(
            label4 = self.add_widget_intelligent(
                CenteredTitleText,
                name="Enter URLs, file paths, or HuggingFace repository IDs, separated by spaces. Use shift-control-V to paste:",
                editable=False,
                labelColor="CONTROL",
                relx=4,
            )
        )

        self.nextrely -= 1
        widgets.update(
            download_ids = self.add_widget_intelligent(
                TextBox, max_height=4, scroll_exit=True, editable=True, relx=4
            )
        )
        
        self.nextrely += 1
        
        widgets.update(
            autoload_directory = self.add_widget_intelligent(
                npyscreen.TitleFilename,
                name="Directory to scan for models to import (<tab> autocompletes):",
                select_dir=True,
                must_exist=True,
                use_two_lines=False,
                labelColor="DANGER",
                begin_entry_at=65,
                scroll_exit=True,
            )
        )

        widgets.update(
            autoscan_on_startup = self.add_widget_intelligent(
                npyscreen.Checkbox,
                name="Scan and import from this directory each time InvokeAI starts",
                value=False,
                relx=4,
                scroll_exit=True,
            )
        )
            
        return widgets


    def add_controlnets(self)->dict[str, npyscreen.widget]:
        widgets = dict()
        cn_model_list = sorted(self.installed_cn_models.keys())

        widgets.update(
            label1 = self.add_widget_intelligent(
                CenteredTitleText,
                name="Select the desired ControlNet models to install. Unchecked models will be purged from disk.",
                editable=False,
                labelColor="CAUTION",
            )
        )
        
        columns=6
        widgets.update(
            models_selected = self.add_widget_intelligent(
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
        )
        
        self.nextrely += 1
        widgets.update(
            label2 = self.add_widget_intelligent(
                npyscreen.TitleFixedText,
                name='Additional ControlNet HuggingFace repo_ids to install (Space separated. Use shift-control-V to paste):',
                relx=4,
                color='CONTROL',
                editable=False,
                scroll_exit=True
            )
        )

        self.nextrely -= 1
        widgets.update(
            download_ids = self.add_widget_intelligent(
                TextBox,
                max_height=4,
                scroll_exit=True,
                editable=True,
                relx=4
            )
        )
        return widgets

    # TO DO - create generic function for loras and textual inversions
    def add_loras(self)->dict[str,npyscreen.widget]:
        widgets = dict()
        
        model_list = sorted(self.installed_lora_models.keys())
        widgets.update(
            label1 = self.add_widget_intelligent(
                CenteredTitleText,
                name="Select the desired LoRA/LyCORIS models to install. Unchecked models will be purged from disk.",
                editable=False,
                labelColor="CAUTION",
            )
        )

        columns=min(len(model_list),3) or 1
        widgets.update(
            models_selected = self.add_widget_intelligent(
                MultiSelectColumns,
                columns=columns,
                name="Install ControlNet Models",
                values=model_list,
                value=[
                    model_list.index(x)
                    for x in model_list
                    if self.installed_lora_models[x]
                ],
                max_height=len(model_list)//columns + 1,
                relx=4,
                scroll_exit=True,
            )
        )

        self.nextrely += 1
        widgets.update(
            label2 = self.add_widget_intelligent(
                npyscreen.TitleFixedText,
                name='URLs for new LoRA/LYCORIS models to download and install (Space separated. Use shift-control-V to paste):',
                relx=4,
                color='CONTROL',
                editable=False,
                hidden=True,
                scroll_exit=True
            )
        )

        self.nextrely -= 1
        widgets.update(
            download_ids = self.add_widget_intelligent(
                TextBox,
                max_height=4,
                scroll_exit=True,
                editable=True,
                relx=4,
                hidden=True,
            )
        )
        return widgets

    def add_tis(self)->dict[str, npyscreen.widget]:
        widgets = dict()
        model_list = sorted(self.installed_ti_models.keys())

        widgets.update(
            label1 = self.add_widget_intelligent(
                CenteredTitleText,
                name="Select the desired models to install. Unchecked models will be purged from disk.",
                editable=False,
                labelColor="CAUTION",
            )
        )
        
        columns=min(len(model_list),6) or 1
        widgets.update(
            models_selected = self.add_widget_intelligent(
                MultiSelectColumns,
                columns=columns,
                name="Install Textual Inversion Embeddings",
                values=model_list,
                value=[
                    model_list.index(x)
                    for x in model_list
                    if self.installed_ti_models[x]
                ],
                max_height=len(model_list)//columns + 1,
                relx=4,
                scroll_exit=True,
            )
        )
        
        widgets.update(
            label2 = self.add_widget_intelligent(
                npyscreen.TitleFixedText,
                name='Textual Inversion models to download, use URLs or HugggingFace repo_ids  (Space separated. Use shift-control-V to paste):',
                relx=4,
                color='CONTROL',
                editable=False,
                hidden=True,
                scroll_exit=True
            )
        )
        
        self.nextrely -= 1
        widgets.update(
            download_ids = self.add_widget_intelligent(
                TextBox,
                max_height=4,
                scroll_exit=True,
                editable=True,
                relx=4,
                hidden=True,
            )
        )
        return widgets

    def resize(self):
        super().resize()
        if (s := self.diffusers_models.get("models_selected")):
            s.values = self._get_starter_model_labels()

    def _toggle_tables(self, value=None):
        selected_tab = value[0]
        widgets = [
            self.diffusers_models,
            self.controlnet_models,
            self.lora_models,
            self.ti_models,
        ]

        for group in widgets:
            for k,v in group.items():
                v.hidden = True
        for k,v in widgets[selected_tab].items():
            v.hidden = False
        self.__class__.current_tab = selected_tab  # for persistence
        self.display()

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
        starter_models = dict(
            map(
                lambda x: (self.starter_model_list[x], True),
                self.diffusers_models['models_selected'].value,
            )
        )
        selections.purge_deleted_models = self.diffusers_models['purge_deleted'].value
        
        selections.install_models = [x for x in starter_models if x not in self.existing_models]
        selections.remove_models = [x for x in self.starter_model_list if x in self.existing_models and x not in starter_models]

        # TODO: REFACTOR THIS REPETITIVE CODE
        cn_models_selected = self.controlnet_models['models_selected']
        selections.install_cn_models = [cn_models_selected.values[x]
                                        for x in cn_models_selected.value
                                        if not self.installed_cn_models[cn_models_selected.values[x]]
                                        ]
        selections.remove_cn_models = [x
                                       for x in cn_models_selected.values
                                       if self.installed_cn_models[x]
                                       and cn_models_selected.values.index(x) not in cn_models_selected.value
                                       ]
        if (additional_cns := self.controlnet_models['download_ids'].value.split()):
            valid_cns = [x for x in additional_cns if '/' in x]
            selections.install_cn_models.extend(valid_cns)

        # same thing, for LoRAs
        loras_selected = self.lora_models['models_selected']
        selections.install_lora_models = [loras_selected.values[x]
                                          for x in loras_selected.value
                                          if not self.installed_lora_models[loras_selected.values[x]]
                                          ]
        selections.remove_lora_models = [x
                                         for x in loras_selected.values
                                         if self.installed_lora_models[x]
                                         and loras_selected.values.index(x) not in loras_selected.value
                                         ]
                
        if (additional_loras := self.lora_models['download_ids'].value.split()):
            selections.install_lora_models.extend(additional_loras)

        # same thing, for TIs
        # TODO: refactor
        tis_selected = self.ti_models['models_selected']
        selections.install_ti_models = [tis_selected.values[x]
                                        for x in tis_selected.value
                                        if not self.installed_ti_models[tis_selected.values[x]]
                                        ]
        selections.remove_ti_models = [x
                                       for x in tis_selected.values
                                       if self.installed_ti_models[x]
                                       and tis_selected.values.index(x) not in tis_selected.value
                                       ]
                
        if (additional_tis := self.ti_models['download_ids'].value.split()):
            selections.install_ti_models.extend(additional_tis)
            
        # load directory and whether to scan on startup
        selections.scan_directory = self.diffusers_models['autoload_directory'].value
        selections.autoscan_on_startup = self.diffusers_models['autoscan_on_startup'].value

        # URLs and the like
        selections.import_model_paths = self.diffusers_models['download_ids'].value.split()


@dataclass
class UserSelections():
    install_models: List[str]= field(default_factory=list)
    remove_models: List[str]=field(default_factory=list)
    purge_deleted_models: bool=field(default_factory=list)
    install_cn_models: List[str] = field(default_factory=list)
    remove_cn_models: List[str] = field(default_factory=list)
    install_lora_models: List[str] = field(default_factory=list)
    remove_lora_models: List[str] = field(default_factory=list)
    install_ti_models: List[str] = field(default_factory=list)
    remove_ti_models: List[str] = field(default_factory=list)
    scan_directory: Path = None
    autoscan_on_startup: bool=False
    import_model_paths: str=None
        
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
        lora = ModelInstallList(selections.install_lora_models, selections.remove_lora_models),
        ti = ModelInstallList(selections.install_ti_models, selections.remove_ti_models),
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
        curses.nocbreak()
        curses.echo()
        curses.endwin()
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
