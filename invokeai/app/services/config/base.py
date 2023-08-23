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
import os
import pydoc
import sys
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig, ListConfig
from pathlib import Path
from pydantic import BaseSettings
from typing import ClassVar, Dict, List, Literal, Union, get_origin, get_type_hints, get_args


class PagingArgumentParser(argparse.ArgumentParser):
    """
    A custom ArgumentParser that uses pydoc to page its output.
    It also supports reading defaults from an init file.
    """

    def print_help(self, file=None):
        text = self.format_help()
        pydoc.pager(text)


class InvokeAISettings(BaseSettings):
    """
    Runtime configuration settings in which default values are
    read from an omegaconf .yaml file.
    """

    initconf: ClassVar[DictConfig] = None
    argparse_groups: ClassVar[Dict] = {}

    def parse_args(self, argv: list = sys.argv[1:]):
        parser = self.get_parser()
        opt = parser.parse_args(argv)
        for name in self.__fields__:
            if name not in self._excluded():
                value = getattr(opt, name)
                if isinstance(value, ListConfig):
                    value = list(value)
                elif isinstance(value, DictConfig):
                    value = dict(value)
                setattr(self, name, value)

    def to_yaml(self) -> str:
        """
        Return a YAML string representing our settings. This can be used
        as the contents of `invokeai.yaml` to restore settings later.
        """
        cls = self.__class__
        type = get_args(get_type_hints(cls)["type"])[0]
        field_dict = dict({type: dict()})
        for name, field in self.__fields__.items():
            if name in cls._excluded_from_yaml():
                continue
            category = field.field_info.extra.get("category") or "Uncategorized"
            value = getattr(self, name)
            if category not in field_dict[type]:
                field_dict[type][category] = dict()
            # keep paths as strings to make it easier to read
            field_dict[type][category][name] = str(value) if isinstance(value, Path) else value
        conf = OmegaConf.create(field_dict)
        return OmegaConf.to_yaml(conf)

    @classmethod
    def add_parser_arguments(cls, parser):
        if "type" in get_type_hints(cls):
            settings_stanza = get_args(get_type_hints(cls)["type"])[0]
        else:
            settings_stanza = "Uncategorized"

        env_prefix = cls.Config.env_prefix if hasattr(cls.Config, "env_prefix") else settings_stanza.upper()

        initconf = (
            cls.initconf.get(settings_stanza)
            if cls.initconf and settings_stanza in cls.initconf
            else OmegaConf.create()
        )

        # create an upcase version of the environment in
        # order to achieve case-insensitive environment
        # variables (the way Windows does)
        upcase_environ = dict()
        for key, value in os.environ.items():
            upcase_environ[key.upper()] = value

        fields = cls.__fields__
        cls.argparse_groups = {}

        for name, field in fields.items():
            if name not in cls._excluded():
                current_default = field.default

                category = field.field_info.extra.get("category", "Uncategorized")
                env_name = env_prefix + "_" + name
                if category in initconf and name in initconf.get(category):
                    field.default = initconf.get(category).get(name)
                if env_name.upper() in upcase_environ:
                    field.default = upcase_environ[env_name.upper()]
                cls.add_field_argument(parser, name, field)

                field.default = current_default

    @classmethod
    def cmd_name(self, command_field: str = "type") -> str:
        hints = get_type_hints(self)
        if command_field in hints:
            return get_args(hints[command_field])[0]
        else:
            return "Uncategorized"

    @classmethod
    def get_parser(cls) -> ArgumentParser:
        parser = PagingArgumentParser(
            prog=cls.cmd_name(),
            description=cls.__doc__,
        )
        cls.add_parser_arguments(parser)
        return parser

    @classmethod
    def add_subparser(cls, parser: argparse.ArgumentParser):
        parser.add_parser(cls.cmd_name(), help=cls.__doc__)

    @classmethod
    def _excluded(self) -> List[str]:
        # internal fields that shouldn't be exposed as command line options
        return ["type", "initconf"]

    @classmethod
    def _excluded_from_yaml(self) -> List[str]:
        # combination of deprecated parameters and internal ones that shouldn't be exposed as invokeai.yaml options
        return [
            "type",
            "initconf",
            "version",
            "from_file",
            "model",
            "root",
            "max_cache_size",
            "max_vram_cache_size",
            "always_use_cpu",
            "free_gpu_mem",
            "xformers_enabled",
            "tiled_decode",
        ]

    class Config:
        env_file_encoding = "utf-8"
        arbitrary_types_allowed = True
        case_sensitive = True

    @classmethod
    def add_field_argument(cls, command_parser, name: str, field, default_override=None):
        field_type = get_type_hints(cls).get(name)
        default = (
            default_override
            if default_override is not None
            else field.default
            if field.default_factory is None
            else field.default_factory()
        )
        if category := field.field_info.extra.get("category"):
            if category not in cls.argparse_groups:
                cls.argparse_groups[category] = command_parser.add_argument_group(category)
            argparse_group = cls.argparse_groups[category]
        else:
            argparse_group = command_parser

        if get_origin(field_type) == Literal:
            allowed_values = get_args(field.type_)
            allowed_types = set()
            for val in allowed_values:
                allowed_types.add(type(val))
            allowed_types_list = list(allowed_types)
            field_type = allowed_types_list[0] if len(allowed_types) == 1 else int_or_float_or_str

            argparse_group.add_argument(
                f"--{name}",
                dest=name,
                type=field_type,
                default=default,
                choices=allowed_values,
                help=field.field_info.description,
            )

        elif get_origin(field_type) == Union:
            argparse_group.add_argument(
                f"--{name}",
                dest=name,
                type=int_or_float_or_str,
                default=default,
                help=field.field_info.description,
            )

        elif get_origin(field_type) == list:
            argparse_group.add_argument(
                f"--{name}",
                dest=name,
                nargs="*",
                type=field.type_,
                default=default,
                action=argparse.BooleanOptionalAction if field.type_ == bool else "store",
                help=field.field_info.description,
            )
        else:
            argparse_group.add_argument(
                f"--{name}",
                dest=name,
                type=field.type_,
                default=default,
                action=argparse.BooleanOptionalAction if field.type_ == bool else "store",
                help=field.field_info.description,
            )


def int_or_float_or_str(value: str) -> Union[int, float, str]:
    """
    Workaround for argparse type checking.
    """
    try:
        return int(value)
    except Exception as e:  # noqa F841
        pass
    try:
        return float(value)
    except Exception as e:  # noqa F841
        pass
    return str(value)
