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
import textwrap
import traceback
from argparse import Namespace
from multiprocessing import Process
from multiprocessing.connection import Connection, Pipe
from pathlib import Path
from shutil import get_terminal_size
from typing import List

import logging
import npyscreen
import torch
from npyscreen import widget
from omegaconf import OmegaConf

from invokeai.backend.util.logging import InvokeAILogger

from invokeai.backend.install.model_install_backend import (
    Dataset_path,  # most of these should go!!
    default_config_file,
    default_dataset,
    install_requested_models,
    recommended_datasets,
    ModelInstallList,
    UserSelections,
    ModelInstall
)
from invokeai.backend.model_management import ModelManager, BaseModelType, ModelType
from invokeai.backend.util import choose_precision, choose_torch_device
from invokeai.frontend.install.widgets import (
    CenteredTitleText,
    MultiSelectColumns,
    SingleSelectColumns,
    TextBox,
    BufferBox,
    FileBox,
    set_min_terminal_size,
    select_stable_diffusion_config_file,
    CyclingForm,
    MIN_COLS,
    MIN_LINES,
)
from invokeai.app.services.config import InvokeAIAppConfig

config = InvokeAIAppConfig.get_config()
logger = InvokeAILogger.getLogger()

# build a table mapping all non-printable characters to None
# for stripping control characters
# from https://stackoverflow.com/questions/92438/stripping-non-printable-characters-from-a-string-in-python
NOPRINT_TRANS_TABLE = {
    i: None for i in range(0, sys.maxunicode + 1) if not chr(i).isprintable()
}

def make_printable(s:str)->str:
    '''Replace non-printable characters in a string'''
    return s.translate(NOPRINT_TRANS_TABLE)

class addModelsForm(CyclingForm, npyscreen.FormMultiPage):
    # for responsive resizing - disabled
    # FIX_MINIMUM_SIZE_WHEN_CREATED = False
    
    # for persistence
    current_tab = 0

    def __init__(self, parentApp, name, multipage=False, *args, **keywords):
        self.multipage = multipage
        self.subprocess = None
        super().__init__(parentApp=parentApp, name=name, *args, **keywords)

    def create(self):
        self.keypress_timeout = 10
        self.counter = 0
        self.subprocess_connection = None

        if not config.model_conf_path.exists():
            with open(config.model_conf_path,'w') as file:
                print('# InvokeAI model configuration file',file=file)
        self.installer = ModelInstall(config)
        self.all_models = self.installer.all_models()
        self.starter_models = self.installer.starter_models()
        self.model_labels = self._get_model_labels()        
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
                'STARTER MODELS',
                'MORE MODELS',
                'CONTROLNETS',
                'LORA/LYCORIS',
                'TEXTUAL INVERSION',
            ],
            value=[self.current_tab],
            columns = 5,
            max_height = 2,
            relx=8,
            scroll_exit = True,
        )
        self.tabs.on_changed = self._toggle_tables

        top_of_table = self.nextrely
        self.starter_pipelines = self.add_starter_pipelines()
        bottom_of_table = self.nextrely

        self.nextrely = top_of_table
        self.pipeline_models = self.add_model_widgets(
            model_type=ModelType.Pipeline,
            window_width=window_width,
            exclude = self.starter_models
        )
        bottom_of_table = max(bottom_of_table,self.nextrely)

        self.nextrely = top_of_table
        self.controlnet_models = self.add_model_widgets(
            model_type=ModelType.ControlNet,
            window_width=window_width,
        )
        bottom_of_table = max(bottom_of_table,self.nextrely)

        self.nextrely = top_of_table
        self.lora_models = self.add_model_widgets(
            model_type=ModelType.Lora,
            window_width=window_width,
        )
        bottom_of_table = max(bottom_of_table,self.nextrely)

        self.nextrely = top_of_table
        self.ti_models = self.add_model_widgets(
            model_type=ModelType.TextualInversion,
            window_width=window_width,
        )
        bottom_of_table = max(bottom_of_table,self.nextrely)
                
        self.nextrely = bottom_of_table+1

        self.monitor = self.add_widget_intelligent(
            BufferBox,
            name='Log Messages',
            editable=False,
            max_height = 16,
        )
        
        self.nextrely += 1
        done_label = "APPLY CHANGES"
        back_label = "BACK"
        if self.multipage:
            self.back_button = self.add_widget_intelligent(
                npyscreen.ButtonPress,
                name=back_label,
                rely=-3,
                when_pressed_function=self.on_back,
            )
        self.ok_button = self.add_widget_intelligent(
            npyscreen.ButtonPress,
            name=done_label,
            relx=(window_width - len(done_label)) // 2,
            rely=-3,
            when_pressed_function=self.on_execute
        )

        label = "APPLY CHANGES & EXIT"
        self.done = self.add_widget_intelligent(
            npyscreen.ButtonPress,
            name=label,
            rely=-3,
            relx=window_width-len(label)-15,
            when_pressed_function=self.on_done,
        )

        # This restores the selected page on return from an installation
        for i in range(1,self.current_tab+1):
            self.tabs.h_cursor_line_down(1)
        self._toggle_tables([self.current_tab])

    ############# diffusers tab ##########        
    def add_starter_pipelines(self)->dict[str, npyscreen.widget]:
        '''Add widgets responsible for selecting diffusers models'''
        widgets = dict()
        models = self.all_models
        starters = self.starter_models
        starter_model_labels = self.model_labels
        
        recommended_models = set([
            x
            for x in starters
            if models[x].recommended
        ])
        self.installed_models = sorted(
            [x for x in starters if models[x].installed]
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
        show_recommended = len(self.installed_models)==0
        keys = [x for x in models.keys() if x in starters]
        widgets.update(
            models_selected = self.add_widget_intelligent(
                MultiSelectColumns,
                columns=1,
                name="Install Starter Models",
                values=[starter_model_labels[x] for x in keys],
                value=[
                    keys.index(x)
                    for x in keys
                    if (show_recommended and models[x].recommended) \
                    or (x in self.installed_models)
                ],
                max_height=len(starters) + 1,
                relx=4,
                scroll_exit=True,
            ),
            models = keys,
        )

        self.nextrely += 1
        return widgets

    ############# Add a set of model install widgets ########
    def add_model_widgets(self,
                          model_type: ModelType,
                          window_width: int=120,
                          install_prompt: str=None,
                          exclude: set=set(),
                          )->dict[str,npyscreen.widget]:
        '''Generic code to create model selection widgets'''
        widgets = dict()
        model_list = [x for x in self.all_models if self.all_models[x].model_type==model_type and not x in exclude]
        model_labels = [self.model_labels[x] for x in model_list]
        if len(model_list) > 0:
            max_width = max([len(x) for x in model_labels])
            columns = window_width // (max_width+8)  # 8 characters for "[x] " and padding
            columns = min(len(model_list),columns) or 1
            prompt = install_prompt or f"Select the desired {model_type.value.title()} models to install. Unchecked models will be purged from disk."

            widgets.update(
                label1 = self.add_widget_intelligent(
                    CenteredTitleText,
                    name=prompt,
                    editable=False,
                    labelColor="CAUTION",
                )
            )

            widgets.update(
                models_selected = self.add_widget_intelligent(
                    MultiSelectColumns,
                    columns=columns,
                    name=f"Install {model_type} Models",
                    values=model_labels,
                    value=[
                        model_list.index(x)
                        for x in model_list
                        if self.all_models[x].installed
                    ],
                    max_height=len(model_list)//columns + 1,
                    relx=4,
                    scroll_exit=True,
                ),
                models = model_list,
            )

        self.nextrely += 1
        widgets.update(
            download_ids = self.add_widget_intelligent(
                TextBox,
                name = "Additional URLs, or HuggingFace repo_ids to install (Space separated. Use shift-control-V to paste):",
                max_height=4,
                scroll_exit=True,
                editable=True,
            )
        )
        return widgets

    ### Tab for arbitrary diffusers widgets ###
    def add_diffusers_widgets(self,
                              model_type: ModelType=ModelType.Pipeline,
                              window_width: int=120,
                              )->dict[str,npyscreen.widget]:
        '''Similar to add_model_widgets() but adds some additional widgets at the bottom
        to support the autoload directory'''
        widgets = self.add_model_widgets(
            model_type = model_type,
            window_width = window_width,
            install_prompt=f"Additional {model_type.value.title()} models already installed.",
        )

        label = "Directory to scan for models to automatically import (<tab> autocompletes):"
        self.nextrely += 1
        widgets.update(
            autoload_directory = self.add_widget_intelligent(
                FileBox,
                max_height=3,
                name=label,
                value=str(config.autoconvert_dir) if config.autoconvert_dir else None,
                select_dir=True,
                must_exist=True,
                use_two_lines=False,
                labelColor="DANGER",
                begin_entry_at=len(label)+1,
                scroll_exit=True,
            )
        )
        widgets.update(
            autoscan_on_startup = self.add_widget_intelligent(
                npyscreen.Checkbox,
                name="Scan and import from this directory each time InvokeAI starts",
                value=config.autoconvert_dir is not None,
                relx=4,
                scroll_exit=True,
            )
        )
        return widgets

    def resize(self):
        super().resize()
        if (s := self.starter_pipelines.get("models_selected")):
            keys = [x for x in self.all_models.keys() if x in self.starter_models]
            s.values = [self.model_labels[x] for x in keys]

    def _toggle_tables(self, value=None):
        selected_tab = value[0]
        widgets = [
            self.starter_pipelines,
            self.pipeline_models,
            self.controlnet_models,
            self.lora_models,
            self.ti_models,
        ]

        for group in widgets:
            for k,v in group.items():
                try:
                    v.hidden = True
                    v.editable = False
                except:
                    pass
        for k,v in widgets[selected_tab].items():
            try:
                v.hidden = False
                if not isinstance(v,(npyscreen.FixedText, npyscreen.TitleFixedText, CenteredTitleText)):
                    v.editable = True
            except:
                pass
        self.__class__.current_tab = selected_tab  # for persistence
        self.display()

    def _get_model_labels(self) -> dict[str,str]:
        window_width, window_height = get_terminal_size()
        checkbox_width = 4
        spacing_width = 2
        
        models = self.all_models
        label_width = max([len(models[x].name) for x in models])
        description_width = window_width - label_width - checkbox_width - spacing_width

        result = dict()
        for x in models.keys():
            description = models[x].description
            description = description[0 : description_width - 3] + "..." \
                if description and len(description) > description_width \
                   else description if description else ''
            result[x] =  f"%-{label_width}s %s" % (models[x].name, description)
        return result
            
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

    def on_execute(self):
        self.monitor.entry_widget.buffer(['Processing...'],scroll_end=True)
        self.marshall_arguments()
        app = self.parentApp
        self.ok_button.hidden = True
        self.display()
        
        # for communication with the subprocess
        parent_conn, child_conn = Pipe()
        p = Process(
            target = process_and_execute,
            kwargs=dict(
                opt = app.program_opts,
                selections = app.user_selections,
                conn_out = child_conn,
            )
        )
        p.start()
        child_conn.close()
        self.subprocess_connection = parent_conn
        self.subprocess = p
        app.user_selections = UserSelections()
        # process_and_execute(app.opt, app.user_selections)

    def on_back(self):
        self.parentApp.switchFormPrevious()
        self.editing = False

    def on_cancel(self):
        self.parentApp.setNextForm(None)
        self.parentApp.user_cancelled = True
        self.editing = False
        
    def on_done(self):
        self.marshall_arguments()
        self.parentApp.setNextForm(None)
        self.parentApp.user_cancelled = False
        self.editing = False

    ########## This routine monitors the child process that is performing model installation and removal #####
    def while_waiting(self):
        '''Called during idle periods. Main task is to update the Log Messages box with messages
        from the child process that does the actual installation/removal'''
        c = self.subprocess_connection
        if not c:
            return
        
        monitor_widget = self.monitor.entry_widget
        while c.poll():
            try:
                data = c.recv_bytes().decode('utf-8')
                data.strip('\n')

                # processing child is requesting user input to select the
                # right configuration file
                if data.startswith('*need v2 config'):
                    _,model_path,*_ = data.split(":",2)
                    self._return_v2_config(model_path)

                # processing child is done
                elif data=='*done*':
                    self._close_subprocess_and_regenerate_form()
                    break

                # update the log message box
                else:
                    data=make_printable(data)
                    data=data.replace('[A','')
                    monitor_widget.buffer(
                        textwrap.wrap(data,
                                      width=monitor_widget.width,
                                      subsequent_indent='   ',
                                      ),
                        scroll_end=True
                    )
                    self.display()
            except (EOFError,OSError):
                self.subprocess_connection = None

    def _return_v2_config(self,model_path: str):
        c = self.subprocess_connection
        model_name = Path(model_path).name
        message = select_stable_diffusion_config_file(model_name=model_name)
        c.send_bytes(message.encode('utf-8'))

    def _close_subprocess_and_regenerate_form(self):
        app = self.parentApp
        self.subprocess_connection.close()
        self.subprocess_connection = None
        self.monitor.entry_widget.buffer(['** Action Complete **'])
        self.display()
        
        # rebuild the form, saving and restoring some of the fields that need to be preserved.
        saved_messages = self.monitor.entry_widget.values
        autoload_dir = self.pipeline_models['autoload_directory'].value
        autoscan = self.pipeline_models['autoscan_on_startup'].value
        
        app.main_form = app.addForm(
            "MAIN", addModelsForm, name="Install Stable Diffusion Models", multipage=self.multipage,
        )
        app.switchForm("MAIN")
        
        app.main_form.monitor.entry_widget.values = saved_messages
        app.main_form.monitor.entry_widget.buffer([''],scroll_end=True)
        app.main_form.pipeline_models['autoload_directory'].value = autoload_dir
        app.main_form.pipeline_models['autoscan_on_startup'].value = autoscan
        
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

        # Starter models to install/remove
        # TO DO - turn these into a dict so we don't have to hard-code the attributes
        print(f'installed={[x for x in self.all_models if self.all_models[x].installed]}',file=f)
        for section in [self.starter_pipelines, self.pipeline_models,
                        self.controlnet_models, self.lora_models, self.ti_models]:
            selected = set([section['models'][x] for x in section['models_selected'].value])
            models_to_install = [x for x in selected if not self.all_models[x].installed]
            models_to_remove = [x for x in section['models'] if x not in selected and self.all_models[x].installed]
                            
        # "More" models
        selections.import_model_paths = self.pipeline_models['download_ids'].value.split()
        if diffusers_selected := self.pipeline_models.get('models_selected'):
            selections.remove_models.extend([x
                                             for x in diffusers_selected.values
                                             if self.installed_pipeline_models[x]
                                             and diffusers_selected.values.index(x) not in diffusers_selected.value
                                             ]
                                            )
                                        
        # TODO: REFACTOR THIS REPETITIVE CODE
        if cn_models_selected := self.controlnet_models.get('models_selected'):
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
        if loras_selected := self.lora_models.get('models_selected'):
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
        if tis_selected := self.ti_models.get('models_selected'):
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
        selections.scan_directory = self.pipeline_models['autoload_directory'].value
        selections.autoscan_on_startup = self.pipeline_models['autoscan_on_startup'].value

class AddModelApplication(npyscreen.NPSAppManaged):
    def __init__(self,opt):
        super().__init__()
        self.program_opts = opt
        self.user_cancelled = False
        self.user_selections = UserSelections()

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.DefaultTheme)
        self.main_form = self.addForm(
            "MAIN", addModelsForm, name="Install Stable Diffusion Models", cycle_widgets=True,
        )

class StderrToMessage():
    def __init__(self, connection: Connection):
        self.connection = connection

    def write(self, data:str):
        self.connection.send_bytes(data.encode('utf-8'))

    def flush(self):
        pass

# --------------------------------------------------------
def ask_user_for_config_file(model_path: Path,
                             tui_conn: Connection=None
                             )->Path:
    if tui_conn:
        logger.debug('Waiting for user response...')
        return _ask_user_for_cf_tui(model_path, tui_conn)        
    else:
        return _ask_user_for_cf_cmdline(model_path)

def _ask_user_for_cf_cmdline(model_path):
    choices = [
        config.legacy_conf_path / x
        for x in ['v2-inference.yaml','v2-inference-v.yaml']
    ]
    choices.extend([None])
    print(
f"""
Please select the type of the V2 checkpoint named {model_path.name}:
[1] A Stable Diffusion v2.x base model (512 pixels; there should be no 'parameterization:' line in its yaml file)
[2] A Stable Diffusion v2.x v-predictive model (768 pixels; look for a 'parameterization: "v"' line in its yaml file)
[3] Skip this model and come back later.
"""
        )
    choice = None
    ok = False
    while not ok:
        try:
            choice = input('select> ').strip()
            choice = choices[int(choice)-1]
            ok = True
        except (ValueError, IndexError):
            print(f'{choice} is not a valid choice')
        except EOFError:
            return
    return choice
        
def _ask_user_for_cf_tui(model_path: Path, tui_conn: Connection)->Path:
    try:
        tui_conn.send_bytes(f'*need v2 config for:{model_path}'.encode('utf-8'))
        # note that we don't do any status checking here
        response = tui_conn.recv_bytes().decode('utf-8')
        if response is None:
            return None
        elif response == 'epsilon':
            return config.legacy_conf_path / 'v2-inference.yaml'
        elif response == 'v':
            return config.legacy_conf_path  / 'v2-inference-v.yaml'
        elif response == 'abort':
            logger.info('Conversion aborted')
            return None
        else:
            return Path(response)
    except:
        return None
        
# --------------------------------------------------------
def process_and_execute(opt: Namespace,
                        selections: UserSelections,
                        conn_out: Connection=None,
                        ):
    # set up so that stderr is sent to conn_out
    if conn_out:
        translator = StderrToMessage(conn_out)
        sys.stderr = translator
        sys.stdout = translator
        logger = InvokeAILogger.getLogger()
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler(translator))
    
    models_to_install = selections.install_models
    models_to_remove = selections.remove_models
    directory_to_scan = selections.scan_directory
    scan_at_startup = selections.autoscan_on_startup
    potential_models_to_install = selections.import_model_paths
    name_map = selections.model_name_map

    install_requested_models(
        diffusers = ModelInstallList(models_to_install, [name_map[ModelType.Pipeline][x] for x in models_to_remove]),
        controlnet = ModelInstallList(selections.install_cn_models, [name_map[ModelType.ControlNet][x] for x in selections.remove_cn_models]),
        lora = ModelInstallList(selections.install_lora_models, [name_map[ModelType.Lora][x] for x in selections.remove_lora_models]),
        ti = ModelInstallList(selections.install_ti_models, [name_map[ModelType.TextualInversion][x] for x in selections.remove_ti_models]),
        scan_directory=Path(directory_to_scan) if directory_to_scan else None,
        external_models=potential_models_to_install,
        scan_at_startup=scan_at_startup,
        precision="float32"
        if opt.full_precision
        else choose_precision(torch.device(choose_torch_device())),
        config_file_path=Path(opt.config_file) if opt.config_file else config.model_conf_path,
        model_config_file_callback = lambda x: ask_user_for_config_file(x,conn_out)
    )

    if conn_out:
        conn_out.send_bytes('*done*'.encode('utf-8'))
        conn_out.close()


def do_listings(opt)->bool:
    """List installed models of various sorts, and return
    True if any were requested."""
    model_manager = ModelManager(config.model_conf_path)
    if opt.list_models == 'diffusers':
        print("Diffuser models:")
        model_manager.print_models()
    elif opt.list_models == 'controlnets':
        print("Installed Controlnet Models:")
        cnm = model_manager.list_controlnet_models()
        print(textwrap.indent("\n".join([x for x in cnm if cnm[x]]),prefix='   '))
    elif opt.list_models == 'loras':
        print("Installed LoRA/LyCORIS Models:")
        cnm = model_manager.list_lora_models()
        print(textwrap.indent("\n".join([x for x in cnm if cnm[x]]),prefix='   '))
    elif opt.list_models == 'tis':
        print("Installed Textual Inversion Embeddings:")
        cnm = model_manager.list_ti_models()
        print(textwrap.indent("\n".join([x for x in cnm if cnm[x]]),prefix='   '))
    else:
        return False
    return True

# --------------------------------------------------------
def select_and_download_models(opt: Namespace):
    precision = (
        "float32"
        if opt.full_precision
        else choose_precision(torch.device(choose_torch_device()))
    )

    if do_listings(opt):
        pass
    # this processes command line additions/removals
    elif opt.diffusers or opt.controlnets or opt.textual_inversions or opt.loras:
        action = 'remove_models' if opt.delete else 'install_models'
        diffusers_args = {'diffusers':ModelInstallList(remove_models=opt.diffusers or [])} \
                          if opt.delete \
                          else {'external_models':opt.diffusers or []} 
        install_requested_models(
            **diffusers_args,
            controlnet=ModelInstallList(**{action:opt.controlnets or []}),
            ti=ModelInstallList(**{action:opt.textual_inversions or []}),
            lora=ModelInstallList(**{action:opt.loras or []}),
            precision=precision,
            model_config_file_callback=lambda x: ask_user_for_config_file(x),
        )
    elif opt.default_only:
        install_requested_models(
            diffusers=ModelInstallList(install_models=default_dataset()),
            precision=precision,
        )
    elif opt.yes_to_all:
        install_requested_models(
            diffusers=ModelInstallList(install_models=recommended_datasets()),
            precision=precision,
        )

    # this is where the TUI is called
    else:
        # needed because the torch library is loaded, even though we don't use it
        torch.multiprocessing.set_start_method("spawn")

        # the third argument is needed in the Windows 11 environment in
        # order to launch and resize a console window running this program
        set_min_terminal_size(MIN_COLS, MIN_LINES,'invokeai-model-install')
        installApp = AddModelApplication(opt)
        try:
            installApp.run()
        except KeyboardInterrupt as e:
            if hasattr(installApp,'main_form'):
                if installApp.main_form.subprocess \
                   and installApp.main_form.subprocess.is_alive():
                    logger.info('Terminating subprocesses')
                    installApp.main_form.subprocess.terminate()
                    installApp.main_form.subprocess = None
            raise e
        process_and_execute(opt, installApp.user_selections)

# -------------------------------------
def main():
    parser = argparse.ArgumentParser(description="InvokeAI model downloader")
    parser.add_argument(
        "--diffusers",
        nargs="*",
        help="List of URLs or repo_ids of diffusers to install/delete",
    )
    parser.add_argument(
        "--loras",
        nargs="*",
        help="List of URLs or repo_ids of LoRA/LyCORIS models to install/delete",
    )
    parser.add_argument(
        "--controlnets",
        nargs="*",
        help="List of URLs or repo_ids of controlnet models to install/delete",
    )
    parser.add_argument(
        "--textual-inversions",
        nargs="*",
        help="List of URLs or repo_ids of textual inversion embeddings to install/delete",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete models listed on command line rather than installing them",
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
        help="only install the default model",
    )
    parser.add_argument(
        "--list-models",
        choices=["diffusers","loras","controlnets","tis"],
        help="list installed models",
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
    
    invoke_args = []
    if opt.root:
        invoke_args.extend(['--root',opt.root])
    if opt.full_precision:
        invoke_args.extend(['--precision','float32'])
    config.parse_args(invoke_args)
    logger = InvokeAILogger().getLogger(config=config)

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
    except Exception as e:
        print(f'An exception has occurred: {str(e)} Details:')
        print(traceback.format_exc(), file=sys.stderr)
        input('Press any key to continue...')
    

# -------------------------------------
if __name__ == "__main__":
    main()
