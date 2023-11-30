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
import logging
import sys
import textwrap
import traceback
from argparse import Namespace
from multiprocessing import Process
from multiprocessing.connection import Connection, Pipe
from pathlib import Path
from shutil import get_terminal_size
from typing import Optional

import npyscreen
import torch
from npyscreen import widget

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.install.model_install_backend import InstallSelections, ModelInstall, SchedulerPredictionType
from invokeai.backend.model_management import ModelManager, ModelType
from invokeai.backend.util import choose_precision, choose_torch_device
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.frontend.install.widgets import (
    MIN_COLS,
    MIN_LINES,
    BufferBox,
    CenteredTitleText,
    CyclingForm,
    MultiSelectColumns,
    SingleSelectColumns,
    TextBox,
    WindowTooSmallException,
    select_stable_diffusion_config_file,
    set_min_terminal_size,
)

config = InvokeAIAppConfig.get_config()
logger = InvokeAILogger.get_logger()

# build a table mapping all non-printable characters to None
# for stripping control characters
# from https://stackoverflow.com/questions/92438/stripping-non-printable-characters-from-a-string-in-python
NOPRINT_TRANS_TABLE = {i: None for i in range(0, sys.maxunicode + 1) if not chr(i).isprintable()}

# maximum number of installed models we can display before overflowing vertically
MAX_OTHER_MODELS = 72


def make_printable(s: str) -> str:
    """Replace non-printable characters in a string"""
    return s.translate(NOPRINT_TRANS_TABLE)


class addModelsForm(CyclingForm, npyscreen.FormMultiPage):
    # for responsive resizing set to False, but this seems to cause a crash!
    FIX_MINIMUM_SIZE_WHEN_CREATED = True

    # for persistence
    current_tab = 0

    def __init__(self, parentApp, name, multipage=False, *args, **keywords):
        self.multipage = multipage
        self.subprocess = None
        super().__init__(parentApp=parentApp, name=name, *args, **keywords)  # noqa: B026 # TODO: maybe this is bad?

    def create(self):
        self.keypress_timeout = 10
        self.counter = 0
        self.subprocess_connection = None

        if not config.model_conf_path.exists():
            with open(config.model_conf_path, "w") as file:
                print("# InvokeAI model configuration file", file=file)
        self.installer = ModelInstall(config)
        self.all_models = self.installer.all_models()
        self.starter_models = self.installer.starter_models()
        self.model_labels = self._get_model_labels()
        window_width, window_height = get_terminal_size()

        self.nextrely -= 1
        self.add_widget_intelligent(
            npyscreen.FixedText,
            value="Use ctrl-N and ctrl-P to move to the <N>ext and <P>revious fields. Cursor keys navigate, and <space> selects.",
            editable=False,
            color="CAUTION",
        )
        self.nextrely += 1
        self.tabs = self.add_widget_intelligent(
            SingleSelectColumns,
            values=[
                "STARTERS",
                "MAINS",
                "CONTROLNETS",
                "T2I-ADAPTERS",
                "IP-ADAPTERS",
                "LORAS",
                "TI EMBEDDINGS",
            ],
            value=[self.current_tab],
            columns=7,
            max_height=2,
            relx=8,
            scroll_exit=True,
        )
        self.tabs.on_changed = self._toggle_tables

        top_of_table = self.nextrely
        self.starter_pipelines = self.add_starter_pipelines()
        bottom_of_table = self.nextrely

        self.nextrely = top_of_table
        self.pipeline_models = self.add_pipeline_widgets(
            model_type=ModelType.Main, window_width=window_width, exclude=self.starter_models
        )
        # self.pipeline_models['autoload_pending'] = True
        bottom_of_table = max(bottom_of_table, self.nextrely)

        self.nextrely = top_of_table
        self.controlnet_models = self.add_model_widgets(
            model_type=ModelType.ControlNet,
            window_width=window_width,
        )
        bottom_of_table = max(bottom_of_table, self.nextrely)

        self.nextrely = top_of_table
        self.t2i_models = self.add_model_widgets(
            model_type=ModelType.T2IAdapter,
            window_width=window_width,
        )
        bottom_of_table = max(bottom_of_table, self.nextrely)
        self.nextrely = top_of_table
        self.ipadapter_models = self.add_model_widgets(
            model_type=ModelType.IPAdapter,
            window_width=window_width,
        )
        bottom_of_table = max(bottom_of_table, self.nextrely)

        self.nextrely = top_of_table
        self.lora_models = self.add_model_widgets(
            model_type=ModelType.Lora,
            window_width=window_width,
        )
        bottom_of_table = max(bottom_of_table, self.nextrely)

        self.nextrely = top_of_table
        self.ti_models = self.add_model_widgets(
            model_type=ModelType.TextualInversion,
            window_width=window_width,
        )
        bottom_of_table = max(bottom_of_table, self.nextrely)

        self.nextrely = bottom_of_table + 1

        self.monitor = self.add_widget_intelligent(
            BufferBox,
            name="Log Messages",
            editable=False,
            max_height=6,
        )

        self.nextrely += 1
        done_label = "APPLY CHANGES"
        back_label = "BACK"
        cancel_label = "CANCEL"
        current_position = self.nextrely
        if self.multipage:
            self.back_button = self.add_widget_intelligent(
                npyscreen.ButtonPress,
                name=back_label,
                when_pressed_function=self.on_back,
            )
        else:
            self.nextrely = current_position
            self.cancel_button = self.add_widget_intelligent(
                npyscreen.ButtonPress, name=cancel_label, when_pressed_function=self.on_cancel
            )
            self.nextrely = current_position
            self.ok_button = self.add_widget_intelligent(
                npyscreen.ButtonPress,
                name=done_label,
                relx=(window_width - len(done_label)) // 2,
                when_pressed_function=self.on_execute,
            )

        label = "APPLY CHANGES & EXIT"
        self.nextrely = current_position
        self.done = self.add_widget_intelligent(
            npyscreen.ButtonPress,
            name=label,
            relx=window_width - len(label) - 15,
            when_pressed_function=self.on_done,
        )

        # This restores the selected page on return from an installation
        for _i in range(1, self.current_tab + 1):
            self.tabs.h_cursor_line_down(1)
        self._toggle_tables([self.current_tab])

    ############# diffusers tab ##########
    def add_starter_pipelines(self) -> dict[str, npyscreen.widget]:
        """Add widgets responsible for selecting diffusers models"""
        widgets = {}
        models = self.all_models
        starters = self.starter_models
        starter_model_labels = self.model_labels

        self.installed_models = sorted([x for x in starters if models[x].installed])

        widgets.update(
            label1=self.add_widget_intelligent(
                CenteredTitleText,
                name="Select from a starter set of Stable Diffusion models from HuggingFace.",
                editable=False,
                labelColor="CAUTION",
            )
        )

        self.nextrely -= 1
        # if user has already installed some initial models, then don't patronize them
        # by showing more recommendations
        show_recommended = len(self.installed_models) == 0
        keys = [x for x in models.keys() if x in starters]
        widgets.update(
            models_selected=self.add_widget_intelligent(
                MultiSelectColumns,
                columns=1,
                name="Install Starter Models",
                values=[starter_model_labels[x] for x in keys],
                value=[
                    keys.index(x)
                    for x in keys
                    if (show_recommended and models[x].recommended) or (x in self.installed_models)
                ],
                max_height=len(starters) + 1,
                relx=4,
                scroll_exit=True,
            ),
            models=keys,
        )

        self.nextrely += 1
        return widgets

    ############# Add a set of model install widgets ########
    def add_model_widgets(
        self,
        model_type: ModelType,
        window_width: int = 120,
        install_prompt: str = None,
        exclude: set = None,
    ) -> dict[str, npyscreen.widget]:
        """Generic code to create model selection widgets"""
        if exclude is None:
            exclude = set()
        widgets = {}
        model_list = [x for x in self.all_models if self.all_models[x].model_type == model_type and x not in exclude]
        model_labels = [self.model_labels[x] for x in model_list]

        show_recommended = len(self.installed_models) == 0
        truncated = False
        if len(model_list) > 0:
            max_width = max([len(x) for x in model_labels])
            columns = window_width // (max_width + 8)  # 8 characters for "[x] " and padding
            columns = min(len(model_list), columns) or 1
            prompt = (
                install_prompt
                or f"Select the desired {model_type.value.title()} models to install. Unchecked models will be purged from disk."
            )

            widgets.update(
                label1=self.add_widget_intelligent(
                    CenteredTitleText,
                    name=prompt,
                    editable=False,
                    labelColor="CAUTION",
                )
            )

            if len(model_labels) > MAX_OTHER_MODELS:
                model_labels = model_labels[0:MAX_OTHER_MODELS]
                truncated = True

            widgets.update(
                models_selected=self.add_widget_intelligent(
                    MultiSelectColumns,
                    columns=columns,
                    name=f"Install {model_type} Models",
                    values=model_labels,
                    value=[
                        model_list.index(x)
                        for x in model_list
                        if (show_recommended and self.all_models[x].recommended) or self.all_models[x].installed
                    ],
                    max_height=len(model_list) // columns + 1,
                    relx=4,
                    scroll_exit=True,
                ),
                models=model_list,
            )

        if truncated:
            widgets.update(
                warning_message=self.add_widget_intelligent(
                    npyscreen.FixedText,
                    value=f"Too many models to display (max={MAX_OTHER_MODELS}). Some are not displayed.",
                    editable=False,
                    color="CAUTION",
                )
            )

        self.nextrely += 1
        widgets.update(
            download_ids=self.add_widget_intelligent(
                TextBox,
                name="Additional URLs, or HuggingFace repo_ids to install (Space separated. Use shift-control-V to paste):",
                max_height=4,
                scroll_exit=True,
                editable=True,
            )
        )
        return widgets

    ### Tab for arbitrary diffusers widgets ###
    def add_pipeline_widgets(
        self,
        model_type: ModelType = ModelType.Main,
        window_width: int = 120,
        **kwargs,
    ) -> dict[str, npyscreen.widget]:
        """Similar to add_model_widgets() but adds some additional widgets at the bottom
        to support the autoload directory"""
        widgets = self.add_model_widgets(
            model_type=model_type,
            window_width=window_width,
            install_prompt=f"Installed {model_type.value.title()} models. Unchecked models in the InvokeAI root directory will be deleted. Enter URLs, paths or repo_ids to import.",
            **kwargs,
        )

        return widgets

    def resize(self):
        super().resize()
        if s := self.starter_pipelines.get("models_selected"):
            keys = [x for x in self.all_models.keys() if x in self.starter_models]
            s.values = [self.model_labels[x] for x in keys]

    def _toggle_tables(self, value=None):
        selected_tab = value[0]
        widgets = [
            self.starter_pipelines,
            self.pipeline_models,
            self.controlnet_models,
            self.t2i_models,
            self.ipadapter_models,
            self.lora_models,
            self.ti_models,
        ]

        for group in widgets:
            for _k, v in group.items():
                try:
                    v.hidden = True
                    v.editable = False
                except Exception:
                    pass
        for _k, v in widgets[selected_tab].items():
            try:
                v.hidden = False
                if not isinstance(v, (npyscreen.FixedText, npyscreen.TitleFixedText, CenteredTitleText)):
                    v.editable = True
            except Exception:
                pass
        self.__class__.current_tab = selected_tab  # for persistence
        self.display()

    def _get_model_labels(self) -> dict[str, str]:
        window_width, window_height = get_terminal_size()
        checkbox_width = 4
        spacing_width = 2

        models = self.all_models
        label_width = max([len(models[x].name) for x in models])
        description_width = window_width - label_width - checkbox_width - spacing_width

        result = {}
        for x in models.keys():
            description = models[x].description
            description = (
                description[0 : description_width - 3] + "..."
                if description and len(description) > description_width
                else description
                if description
                else ""
            )
            result[x] = f"%-{label_width}s %s" % (models[x].name, description)
        return result

    def _get_columns(self) -> int:
        window_width, window_height = get_terminal_size()
        cols = 4 if window_width > 240 else 3 if window_width > 160 else 2 if window_width > 80 else 1
        return min(cols, len(self.installed_models))

    def confirm_deletions(self, selections: InstallSelections) -> bool:
        remove_models = selections.remove_models
        if len(remove_models) > 0:
            mods = "\n".join([ModelManager.parse_key(x)[0] for x in remove_models])
            return npyscreen.notify_ok_cancel(
                f"These unchecked models will be deleted from disk. Continue?\n---------\n{mods}"
            )
        else:
            return True

    def on_execute(self):
        self.marshall_arguments()
        app = self.parentApp
        if not self.confirm_deletions(app.install_selections):
            return

        self.monitor.entry_widget.buffer(["Processing..."], scroll_end=True)
        self.ok_button.hidden = True
        self.display()

        # TO DO: Spawn a worker thread, not a subprocess
        parent_conn, child_conn = Pipe()
        p = Process(
            target=process_and_execute,
            kwargs={
                "opt": app.program_opts,
                "selections": app.install_selections,
                "conn_out": child_conn,
            },
        )
        p.start()
        child_conn.close()
        self.subprocess_connection = parent_conn
        self.subprocess = p
        app.install_selections = InstallSelections()

    def on_back(self):
        self.parentApp.switchFormPrevious()
        self.editing = False

    def on_cancel(self):
        self.parentApp.setNextForm(None)
        self.parentApp.user_cancelled = True
        self.editing = False

    def on_done(self):
        self.marshall_arguments()
        if not self.confirm_deletions(self.parentApp.install_selections):
            return
        self.parentApp.setNextForm(None)
        self.parentApp.user_cancelled = False
        self.editing = False

    ########## This routine monitors the child process that is performing model installation and removal #####
    def while_waiting(self):
        """Called during idle periods. Main task is to update the Log Messages box with messages
        from the child process that does the actual installation/removal"""
        c = self.subprocess_connection
        if not c:
            return

        monitor_widget = self.monitor.entry_widget
        while c.poll():
            try:
                data = c.recv_bytes().decode("utf-8")
                data.strip("\n")

                # processing child is requesting user input to select the
                # right configuration file
                if data.startswith("*need v2 config"):
                    _, model_path, *_ = data.split(":", 2)
                    self._return_v2_config(model_path)

                # processing child is done
                elif data == "*done*":
                    self._close_subprocess_and_regenerate_form()
                    break

                # update the log message box
                else:
                    data = make_printable(data)
                    data = data.replace("[A", "")
                    monitor_widget.buffer(
                        textwrap.wrap(
                            data,
                            width=monitor_widget.width,
                            subsequent_indent="   ",
                        ),
                        scroll_end=True,
                    )
                    self.display()
            except (EOFError, OSError):
                self.subprocess_connection = None

    def _return_v2_config(self, model_path: str):
        c = self.subprocess_connection
        model_name = Path(model_path).name
        message = select_stable_diffusion_config_file(model_name=model_name)
        c.send_bytes(message.encode("utf-8"))

    def _close_subprocess_and_regenerate_form(self):
        app = self.parentApp
        self.subprocess_connection.close()
        self.subprocess_connection = None
        self.monitor.entry_widget.buffer(["** Action Complete **"])
        self.display()

        # rebuild the form, saving and restoring some of the fields that need to be preserved.
        saved_messages = self.monitor.entry_widget.values

        app.main_form = app.addForm(
            "MAIN",
            addModelsForm,
            name="Install Stable Diffusion Models",
            multipage=self.multipage,
        )
        app.switchForm("MAIN")

        app.main_form.monitor.entry_widget.values = saved_messages
        app.main_form.monitor.entry_widget.buffer([""], scroll_end=True)
        # app.main_form.pipeline_models['autoload_directory'].value = autoload_dir
        # app.main_form.pipeline_models['autoscan_on_startup'].value = autoscan

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
        selections = self.parentApp.install_selections
        all_models = self.all_models

        # Defined models (in INITIAL_CONFIG.yaml or models.yaml) to add/remove
        ui_sections = [
            self.starter_pipelines,
            self.pipeline_models,
            self.controlnet_models,
            self.t2i_models,
            self.ipadapter_models,
            self.lora_models,
            self.ti_models,
        ]
        for section in ui_sections:
            if "models_selected" not in section:
                continue
            selected = {section["models"][x] for x in section["models_selected"].value}
            models_to_install = [x for x in selected if not self.all_models[x].installed]
            models_to_remove = [x for x in section["models"] if x not in selected and self.all_models[x].installed]
            selections.remove_models.extend(models_to_remove)
            selections.install_models.extend(
                all_models[x].path or all_models[x].repo_id
                for x in models_to_install
                if all_models[x].path or all_models[x].repo_id
            )

        # models located in the 'download_ids" section
        for section in ui_sections:
            if downloads := section.get("download_ids"):
                selections.install_models.extend(downloads.value.split())

        # NOT NEEDED - DONE IN BACKEND NOW
        # # special case for the ipadapter_models. If any of the adapters are
        # # chosen, then we add the corresponding encoder(s) to the install list.
        # section = self.ipadapter_models
        # if section.get("models_selected"):
        #     selected_adapters = [
        #         self.all_models[section["models"][x]].name for x in section.get("models_selected").value
        #     ]
        #     encoders = []
        #     if any(["sdxl" in x for x in selected_adapters]):
        #         encoders.append("ip_adapter_sdxl_image_encoder")
        #     if any(["sd15" in x for x in selected_adapters]):
        #         encoders.append("ip_adapter_sd_image_encoder")
        #     for encoder in encoders:
        #         key = f"any/clip_vision/{encoder}"
        #         repo_id = f"InvokeAI/{encoder}"
        #         if key not in self.all_models:
        #             selections.install_models.append(repo_id)


class AddModelApplication(npyscreen.NPSAppManaged):
    def __init__(self, opt):
        super().__init__()
        self.program_opts = opt
        self.user_cancelled = False
        # self.autoload_pending = True
        self.install_selections = InstallSelections()

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.DefaultTheme)
        self.main_form = self.addForm(
            "MAIN",
            addModelsForm,
            name="Install Stable Diffusion Models",
            cycle_widgets=False,
        )


class StderrToMessage:
    def __init__(self, connection: Connection):
        self.connection = connection

    def write(self, data: str):
        self.connection.send_bytes(data.encode("utf-8"))

    def flush(self):
        pass


# --------------------------------------------------------
def ask_user_for_prediction_type(model_path: Path, tui_conn: Connection = None) -> SchedulerPredictionType:
    if tui_conn:
        logger.debug("Waiting for user response...")
        return _ask_user_for_pt_tui(model_path, tui_conn)
    else:
        return _ask_user_for_pt_cmdline(model_path)


def _ask_user_for_pt_cmdline(model_path: Path) -> Optional[SchedulerPredictionType]:
    choices = [SchedulerPredictionType.Epsilon, SchedulerPredictionType.VPrediction, None]
    print(
        f"""
Please select the scheduler prediction type of the checkpoint named {model_path.name}:
[1] "epsilon" - most v1.5 models and v2 models trained on 512 pixel images
[2] "vprediction" - v2 models trained on 768 pixel images and a few v1.5 models
[3] Accept the best guess;  you can fix it in the Web UI later
"""
    )
    choice = None
    ok = False
    while not ok:
        try:
            choice = input("select [3]> ").strip()
            if not choice:
                return None
            choice = choices[int(choice) - 1]
            ok = True
        except (ValueError, IndexError):
            print(f"{choice} is not a valid choice")
        except EOFError:
            return
    return choice


def _ask_user_for_pt_tui(model_path: Path, tui_conn: Connection) -> SchedulerPredictionType:
    tui_conn.send_bytes(f"*need v2 config for:{model_path}".encode("utf-8"))
    # note that we don't do any status checking here
    response = tui_conn.recv_bytes().decode("utf-8")
    if response is None:
        return None
    elif response == "epsilon":
        return SchedulerPredictionType.epsilon
    elif response == "v":
        return SchedulerPredictionType.VPrediction
    elif response == "guess":
        return None
    else:
        return None


# --------------------------------------------------------
def process_and_execute(
    opt: Namespace,
    selections: InstallSelections,
    conn_out: Connection = None,
):
    # need to reinitialize config in subprocess
    config = InvokeAIAppConfig.get_config()
    args = ["--root", opt.root] if opt.root else []
    config.parse_args(args)

    # set up so that stderr is sent to conn_out
    if conn_out:
        translator = StderrToMessage(conn_out)
        sys.stderr = translator
        sys.stdout = translator
        logger = InvokeAILogger.get_logger()
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler(translator))

    installer = ModelInstall(config, prediction_type_helper=lambda x: ask_user_for_prediction_type(x, conn_out))
    installer.install(selections)

    if conn_out:
        conn_out.send_bytes("*done*".encode("utf-8"))
        conn_out.close()


# --------------------------------------------------------
def select_and_download_models(opt: Namespace):
    precision = "float32" if opt.full_precision else choose_precision(torch.device(choose_torch_device()))
    config.precision = precision
    installer = ModelInstall(config, prediction_type_helper=ask_user_for_prediction_type)
    if opt.list_models:
        installer.list_models(opt.list_models)
    elif opt.add or opt.delete:
        selections = InstallSelections(install_models=opt.add or [], remove_models=opt.delete or [])
        installer.install(selections)
    elif opt.default_only:
        selections = InstallSelections(install_models=installer.default_model())
        installer.install(selections)
    elif opt.yes_to_all:
        selections = InstallSelections(install_models=installer.recommended_models())
        installer.install(selections)

    # this is where the TUI is called
    else:
        # needed to support the probe() method running under a subprocess
        torch.multiprocessing.set_start_method("spawn")

        if not set_min_terminal_size(MIN_COLS, MIN_LINES):
            raise WindowTooSmallException(
                "Could not increase terminal size. Try running again with a larger window or smaller font size."
            )

        installApp = AddModelApplication(opt)
        try:
            installApp.run()
        except KeyboardInterrupt as e:
            if hasattr(installApp, "main_form"):
                if installApp.main_form.subprocess and installApp.main_form.subprocess.is_alive():
                    logger.info("Terminating subprocesses")
                    installApp.main_form.subprocess.terminate()
                    installApp.main_form.subprocess = None
            raise e
        process_and_execute(opt, installApp.install_selections)


# -------------------------------------
def main():
    parser = argparse.ArgumentParser(description="InvokeAI model downloader")
    parser.add_argument(
        "--add",
        nargs="*",
        help="List of URLs, local paths or repo_ids of models to install",
    )
    parser.add_argument(
        "--delete",
        nargs="*",
        help="List of names of models to idelete",
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
        help="Only install the default model",
    )
    parser.add_argument(
        "--list-models",
        choices=[x.value for x in ModelType],
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
        invoke_args.extend(["--root", opt.root])
    if opt.full_precision:
        invoke_args.extend(["--precision", "float32"])
    config.parse_args(invoke_args)
    logger = InvokeAILogger().get_logger(config=config)

    if not config.model_conf_path.exists():
        logger.info("Your InvokeAI root directory is not set up. Calling invokeai-configure.")
        from invokeai.frontend.install.invokeai_configure import invokeai_configure

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
    except WindowTooSmallException as e:
        logger.error(str(e))
    except widget.NotEnoughSpaceForWidget as e:
        if str(e).startswith("Height of 1 allocated"):
            logger.error("Insufficient vertical space for the interface. Please make your window taller and try again")
        input("Press any key to continue...")
    except Exception as e:
        if str(e).startswith("addwstr"):
            logger.error(
                "Insufficient horizontal space for the interface. Please make your window wider and try again."
            )
        else:
            print(f"An exception has occurred: {str(e)} Details:")
            print(traceback.format_exc(), file=sys.stderr)
        input("Press any key to continue...")


# -------------------------------------
if __name__ == "__main__":
    main()
