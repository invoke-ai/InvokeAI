# Copyright (c) 2023 Lincoln Stein (https://github.com/lstein) and the InvokeAI Development Team

"""
Base class for the InvokeAI configuration system.
It defines a type of pydantic BaseSettings object that
is able to read and write from an omegaconf-based config file,
with overriding of settings from environment variables and/or
the command line.
"""

from __future__ import annotations

import argparse
import pydoc


class PagingArgumentParser(argparse.ArgumentParser):
    """
    A custom ArgumentParser that uses pydoc to page its output.
    It also supports reading defaults from an init file.
    """

    def print_help(self, file=None) -> None:
        text = self.format_help()
        pydoc.pager(text)
