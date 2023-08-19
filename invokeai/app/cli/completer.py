"""
Readline helper functions for cli_app.py
You may import the global singleton `completer` to get access to the
completer object.
"""
import atexit
import readline
import shlex

from pathlib import Path
from typing import List, Dict, Literal, get_args, get_type_hints, get_origin

import invokeai.backend.util.logging as logger
from ...backend import ModelManager
from ..invocations.baseinvocation import BaseInvocation
from .commands import BaseCommand
from ..services.invocation_services import InvocationServices

# singleton object, class variable
completer = None


class Completer(object):
    def __init__(self, model_manager: ModelManager):
        self.commands = self.get_commands()
        self.matches = None
        self.linebuffer = None
        self.manager = model_manager
        return

    def complete(self, text, state):
        """
        Complete commands and switches fromm the node CLI command line.
        Switches are determined in a context-specific manner.
        """

        buffer = readline.get_line_buffer()
        if state == 0:
            options = None
            try:
                current_command, current_switch = self.get_current_command(buffer)
                options = self.get_command_options(current_command, current_switch)
            except IndexError:
                pass
            options = options or list(self.parse_commands().keys())

            if not text:  # first time
                self.matches = options
            else:
                self.matches = [s for s in options if s and s.startswith(text)]

        try:
            match = self.matches[state]
        except IndexError:
            match = None
        return match

    @classmethod
    def get_commands(self) -> List[object]:
        """
        Return a list of all the client commands and invocations.
        """
        return BaseCommand.get_commands() + BaseInvocation.get_invocations()

    def get_current_command(self, buffer: str) -> tuple[str, str]:
        """
        Parse the readline buffer to find the most recent command and its switch.
        """
        if len(buffer) == 0:
            return None, None
        tokens = shlex.split(buffer)
        command = None
        switch = None
        for t in tokens:
            if t[0].isalpha():
                if switch is None:
                    command = t
            else:
                switch = t
        # don't try to autocomplete switches that are already complete
        if switch and buffer.endswith(" "):
            switch = None
        return command or "", switch or ""

    def parse_commands(self) -> Dict[str, List[str]]:
        """
        Return a dict in which the keys are the command name
        and the values are the parameters the command takes.
        """
        result = dict()
        for command in self.commands:
            hints = get_type_hints(command)
            name = get_args(hints["type"])[0]
            result.update({name: hints})
        return result

    def get_command_options(self, command: str, switch: str) -> List[str]:
        """
        Return all the parameters that can be passed to the command as
        command-line switches. Returns None if the command is unrecognized.
        """
        parsed_commands = self.parse_commands()
        if command not in parsed_commands:
            return None

        # handle switches in the format "-foo=bar"
        argument = None
        if switch and "=" in switch:
            switch, argument = switch.split("=")

        parameter = switch.strip("-")
        if parameter in parsed_commands[command]:
            if argument is None:
                return self.get_parameter_options(parameter, parsed_commands[command][parameter])
            else:
                return [
                    f"--{parameter}={x}"
                    for x in self.get_parameter_options(parameter, parsed_commands[command][parameter])
                ]
        else:
            return [f"--{x}" for x in parsed_commands[command].keys()]

    def get_parameter_options(self, parameter: str, typehint) -> List[str]:
        """
        Given a parameter type (such as Literal), offers autocompletions.
        """
        if get_origin(typehint) == Literal:
            return get_args(typehint)
        if parameter == "model":
            return self.manager.model_names()

    def _pre_input_hook(self):
        if self.linebuffer:
            readline.insert_text(self.linebuffer)
            readline.redisplay()
            self.linebuffer = None


def set_autocompleter(services: InvocationServices) -> Completer:
    global completer

    if completer:
        return completer

    completer = Completer(services.model_manager)

    readline.set_completer(completer.complete)
    # pyreadline3 does not have a set_auto_history() method
    try:
        readline.set_auto_history(True)
    except:
        pass
    readline.set_pre_input_hook(completer._pre_input_hook)
    readline.set_completer_delims(" ")
    readline.parse_and_bind("tab: complete")
    readline.parse_and_bind("set print-completions-horizontally off")
    readline.parse_and_bind("set page-completions on")
    readline.parse_and_bind("set skip-completed-text on")
    readline.parse_and_bind("set show-all-if-ambiguous on")

    histfile = Path(services.configuration.root_dir / ".invoke_history")
    try:
        readline.read_history_file(histfile)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass
    except OSError:  # file likely corrupted
        newname = f"{histfile}.old"
        logger.error(f"Your history file {histfile} couldn't be loaded and may be corrupted. Renaming it to {newname}")
        histfile.replace(Path(newname))
    atexit.register(readline.write_history_file, histfile)
