#!/usr/bin/env python
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)
# Before running stable-diffusion on an internet-isolated machine,
# run this script from one with internet connectivity. The
# two machines must share a common .cache directory.

import argparse
import curses
import os
import sys
import traceback
from argparse import Namespace
from typing import List

import npyscreen
import torch
from npyscreen import widget
from omegaconf import OmegaConf

from ..devices import choose_precision, choose_torch_device
from ..globals import Globals
from .widgets import MultiSelectColumns, TextBox
from .model_install_util import (Dataset_path, Default_config_file,
                                 default_dataset, download_weight_datasets,
                                 update_config_file, get_root
                                 )

class addModelsForm(npyscreen.FormMultiPageAction):
    def __init__(self, parentApp, name):
        self.initial_models = OmegaConf.load(Dataset_path)
        try:
            self.existing_models = OmegaConf.load(Default_config_file)
        except:
            self.existing_models = dict()
        self.starter_model_list = [
            x for x in list(self.initial_models.keys()) if x not in self.existing_models
        ]
        self.installed_models=dict()
        super().__init__(parentApp, name)

    def create(self):
        window_height, window_width = curses.initscr().getmaxyx()
        starter_model_labels = self._get_starter_model_labels()
        recommended_models = [
            x
            for x in self.starter_model_list
            if self.initial_models[x].get("recommended", False)
        ]
        self.installed_models = sorted(
            [
                x for x in list(self.initial_models.keys()) if x in self.existing_models
            ]
        )
        
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value='Use ctrl-N and ctrl-P to move to the <N>ext and <P>revious fields,',
            editable=False,
        )
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value='cursor arrows to make a selection, and space to toggle checkboxes.',
            editable=False,
        )
         
        if len(self.installed_models) > 0:
            self.add_widget_intelligent(
                npyscreen.TitleFixedText,
                name="== INSTALLED STARTER MODELS ==",
                value="Currently installed starter models. Uncheck to delete:",
                begin_entry_at=2,
                editable=False,
                color="CONTROL",
            )
            columns = self._get_columns()
            self.previously_installed_models = self.add_widget_intelligent(
                MultiSelectColumns,
                columns=columns,
                values=self.installed_models,
                value=[x for x in range(0,len(self.installed_models))],
                max_height=2+len(self.installed_models) // columns,
                relx = 4,
                slow_scroll=True,
                scroll_exit = True,
            )
            
        self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name="== UNINSTALLED STARTER MODELS ==",
            value="Select from a starter set of Stable Diffusion models from HuggingFace:",
            begin_entry_at=2,
            editable=False,
            color="CONTROL",
        )
        self.nextrely -= 1
        self.models_selected = self.add_widget_intelligent(
            npyscreen.MultiSelect,
            name="Install Starter Models",
            values=starter_model_labels,
            value=[
                self.starter_model_list.index(x)
                for x in self.initial_models
                if x in recommended_models
            ],
            max_height=len(starter_model_labels) + 1,
            relx = 4,
            scroll_exit=True,
        )
        self.add_widget_intelligent(
            npyscreen.TitleFixedText,
            name='== MODEL IMPORT DIRECTORY ==',
            value='Import all models found in this directory (<tab> autocompletes):',
            begin_entry_at=2,
            editable=False,
            color="CONTROL",
        )
        self.autoload_directory = self.add_widget_intelligent(
            npyscreen.TitleFilename,
            name='Directory:',
            select_dir=True,
            must_exist=True,
            use_two_lines=False,
            relx = 4,
            labelColor='DANGER',
            scroll_exit=True,
        )
        self.autoscan_on_startup = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name='Scan this directory each time InvokeAI starts for new models to import.',
            value=False,
            relx = 4,
            scroll_exit=True,
        )
        self.nextrely += 1
        for line in [
                '== INDIVIDUAL MODELS TO IMPORT ==',
                'Enter list of URLs, paths models or HuggingFace diffusers repository IDs.',
                'Use control-V or shift-control-V to paste:'
        ]:
            self.add_widget_intelligent(
                npyscreen.TitleText,
                name=line,
                editable=False,
                color="CONTROL",
            )
            self.nextrely -= 1
        self.import_model_paths = self.add_widget_intelligent(
            TextBox,
            max_height=8,
            scroll_exit=True,
            editable=True,
            relx=4
        )
        self.nextrely += 2
        self.convert_models = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='== CONVERT IMPORTED MODELS INTO DIFFUSERS==',
            values=['Keep original format','Convert to diffusers'],
            value=0,
            begin_entry_at=4,
            scroll_exit=True,
        )

    def resize(self):
        super().resize()
        self.models_selected.values = self._get_starter_model_labels()
        # thought this would dynamically resize the widget, but no luck
        # self.previously_installed_models.columns = self._get_columns()
        # self.previously_installed_models.max_height = 2+len(self.installed_models) // self._get_columns()
        # self.previously_installed_models.make_contained_widgets()
        # self.previously_installed_models.display()
        
    def _get_starter_model_labels(self)->List[str]:
        window_height, window_width = curses.initscr().getmaxyx()
        label_width = 25
        checkbox_width = 4
        spacing_width = 2
        description_width = window_width - label_width - checkbox_width - spacing_width
        im = self.initial_models
        names = list(im.keys())
        descriptions = [im[x].description [0:description_width-3]+'...'
                        if len(im[x].description) > description_width
                        else im[x].description
                        for x in im]
        return [
            f"%-{label_width}s %s" % (names[x], descriptions[x]) for x in range(0,len(im))
        ]

    def _get_columns(self)->int:
        window_height, window_width = curses.initscr().getmaxyx()
        return 4 if window_width > 240 else 3 if window_width>160 else 2 if window_width>80 else 1

    def on_ok(self):
        self.parentApp.setNextForm('MONITOR_OUTPUT')
        self.editing = False
        self.parentApp.user_cancelled = False
        self.marshall_arguments()

    def on_cancel(self):
        self.parentApp.setNextForm(None)
        self.ParentApp.user_cancelled = True
        self.editing = False

    def marshall_arguments(self):
        '''
        Assemble arguments and store as attributes of the application:
        .starter_models: dict of model names to install from INITIAL_CONFIGURE.yaml
                         True  => Install
                         False => Remove
        .scan_directory: Path to a directory of models to scan and import
        .autoscan_on_startup:  True if invokeai should scan and import at startup time
        .import_model_paths:   list of URLs, repo_ids and file paths to import
        .convert_to_diffusers: if True, convert legacy checkpoints into diffusers
        '''
        # starter models to install/remove
        model_names = list(self.initial_models.keys())
        starter_models = dict(map(lambda x: (model_names[x], True), self.models_selected.value))
        if hasattr(self,'previously_installed_models'):
            unchecked = [
                self.previously_installed_models.values[x]
                for x in range(0,len(self.previously_installed_models.values))
                if x not in self.previously_installed_models.value
            ]
            starter_models.update(
                map(lambda x: (x, False), unchecked)
            )
        self.parentApp.starter_models=starter_models

        # load directory and whether to scan on startup
        self.parentApp.scan_directory = self.autoload_directory.value
        self.parentApp.autoscan_on_startup = self.autoscan_on_startup.value

        # URLs and the like
        self.parentApp.import_model_paths = self.import_model_paths.value.split()
        self.parentApp.convert_to_diffusers = self.convert_models.value != 0

class Log(object):
    def __init__(self, writable):
        self.writable = writable
        
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self.writable
        return self
    def __exit__(self, *args):
        sys.stdout = self._stdout

class outputForm(npyscreen.ActionForm):
    def create(self):
        self.buffer = self.add_widget(
            npyscreen.BufferPager,
            editable=False,
        )

    def write(self,string):
        if string != '\n':
            self.buffer.buffer([string])

    def beforeEditing(self):
        myapplication = self.parentApp
        with Log(self):
            print(f'DEBUG: these models will be removed: {[x for x in myapplication.starter_models if not myapplication.starter_models[x]]}')
            print(f'DEBUG: these models will be installed: {[x for x in myapplication.starter_models if myapplication.starter_models[x]]}')
            print(f'DEBUG: this directory will be scanned: {myapplication.scan_directory}')
            print(f'DEBUG: scan at startup time? {myapplication.autoscan_on_startup}')
            print(f'DEBUG: these things will be downloaded: {myapplication.import_model_paths}')
            print(f'DEBUG: convert to diffusers? {myapplication.convert_to_diffusers}')

    def on_ok(self):
        self.buffer.buffer(['goodbye!'])
        self.parentApp.setNextForm(None)
        self.editing = False

class AddModelApplication(npyscreen.NPSAppManaged):
    def __init__(self, saved_args=None):
        super().__init__()
        self.models_to_install = None

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.DefaultTheme)
        self.main = self.addForm(
            "MAIN",
            addModelsForm,
            name="Add/Remove Models",
        )
        self.output = self.addForm(
            'MONITOR_OUTPUT',
            outputForm,
            name='Model Install Output'
        )

# --------------------------------------------------------
def select_and_download_models(opt: Namespace):
    if opt.default_only:
        models_to_download = default_dataset()
    else:
        myapplication = AddModelApplication()
        myapplication.run()
        if not myapplication.user_cancelled:
            print(f'DEBUG: these models will be removed: {[x for x in myapplication.starter_models if not myapplication.starter_models[x]]}')
            print(f'DEBUG: these models will be installed: {[x for x in myapplication.starter_models if myapplication.starter_models[x]]}')
            print(f'DEBUG: this directory will be scanned: {myapplication.scan_directory}')
            print(f'DEBUG: scan at startup time? {myapplication.autoscan_on_startup}')
            print(f'DEBUG: these things will be downloaded: {myapplication.import_model_paths}')
            print(f'DEBUG: convert to diffusers? {myapplication.convert_to_diffusers}')
        sys.exit(0)

    if not models_to_download:
        print(
            '** No models were selected. To run this program again, select "Install initial models" from the invoke script.'
        )
        return

    print("** Downloading and installing the selected models.")
    precision = (
        "float32"
        if opt.full_precision
        else choose_precision(torch.device(choose_torch_device()))
    )
    successfully_downloaded = download_weight_datasets(
        models=models_to_download,
        access_token=None,
        precision=precision,
    )

    update_config_file(successfully_downloaded, opt)
    if len(successfully_downloaded) < len(models_to_download):
        print("** Some of the model downloads were not successful")

    print(
        "\nYour starting models were installed. To find and add more models, see https://invoke-ai.github.io/InvokeAI/installation/050_INSTALLING_MODELS"
    )


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
    Globals.root = os.path.expanduser(get_root(opt.root) or "")

    try:
        select_and_download_models(opt)
    except AssertionError as e:
        print(str(e))
        sys.exit(-1)
    except KeyboardInterrupt:
        print("\nGoodbye! Come back soon.")
    except (widget.NotEnoughSpaceForWidget, Exception) as e:
        if str(e).startswith("Height of 1 allocated"):
            print(
                "** Insufficient vertical space for the interface. Please make your window taller and try again"
            )
        elif str(e).startswith('addwstr'):
            print(
                '** Insufficient horizontal space for the interface. Please make your window wider and try again.'
            )
        else:
            print(f"** A layout error has occurred: {str(e)}")
            traceback.print_exc()
        sys.exit(-1)

# -------------------------------------
if __name__ == "__main__":
    main()
