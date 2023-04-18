# Copyright (c) 2023 Lincoln Stein (https://github.com/lstein)

'''Beginnings of a flexible invokeai configuration system.

Arguments and fields are taken from the pydantic definition of the
model.  Defaults can be set by creating a yaml configuration file that
has top-level keys corresponding to an invocation name, a command, or
"app_settings" for global values such as `xformers_enabled`.

[file: invokeai.yaml]

  app_settings:
    outdir: ~/invokeai/outputs
    precision: float16
    xformers_enabled: false

  txt2img:
    scheduler: k_euler_a
    steps: 30

  img2img:
    strength: 0.6

  inpaint:
    strength: 0.75

Configuration file defaults take precedence over pydantic defaults,
and command-line options take precedence over config file options.

Typical usage:

from invokeai.app.services.config_management import get_app_config

# returns a singleton object that uses `./invokeai.yaml` for default
# values
conf = get_app_config('./invokeai.yaml')

# parse arguments on the command line
conf.parse_args()

# print a value
print(conf.precision)

'''
import argparse
import os
import uuid
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from pydantic import BaseSettings, Field
from typing import Any, ClassVar, List, Literal, Union, get_origin, get_type_hints, get_args

INIT_FILE = Path('invokeai.yaml')

def _get_root_directory()->Path:
    root = None
    if os.environ.get("INVOKEAI_ROOT"):
        root = Path(os.environ.get("INVOKEAI_ROOT")).resolve()
    elif (
            os.environ.get("VIRTUAL_ENV")
            and Path(os.environ.get("VIRTUAL_ENV"), "..", INIT_FILE).exists()
    ):
        root = Path(os.environ.get("VIRTUAL_ENV"), "..").resolve()
    else:
        root = Path("~/invokeai").expanduser().resolve()
    return root

class InvokeAISettings(BaseSettings):
    '''
    Runtime configuration settings in which default values are
    read from an omegaconf .yaml file.
    '''
    initconf             : ClassVar[DictConfig] = None

    def parse_args(self, argv: list=None):
        parser = self.get_parser()
        opt = parser.parse_args(argv)
        for name in self.__fields__:
            if name not in self._excluded():
                setattr(self, name, getattr(opt,name))

    def get_parser(self)->ArgumentParser:
        parser = ArgumentParser(
            prog=__name__,
            description='InvokeAI application',
        )
        env_prefix = self.Config.env_prefix
        default_settings_stanza = get_args(get_type_hints(self)['type'])[0]
        initconf = self.initconf.get(default_settings_stanza) if self.initconf and default_settings_stanza in self.initconf else None

        fields = self.__fields__
        for name, field in fields.items():
            if name not in self._excluded():
                env_name = env_prefix+name
                if initconf and name in initconf:
                    field.default = initconf.get(name)
                if env_name in os.environ:
                    field.default = os.environ[env_name]
                add_field_argument(parser, name, field)
        return parser

    @classmethod
    def _excluded(self)->List[str]:
        return ['type','initconf']
    
    class Config:
        env_file_encoding = 'utf-8'
        arbitrary_types_allowed = True
        env_prefix = 'INVOKEAI_'
        extra = 'allow'
        class_sensitive = True
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                InvokeAISettings._omegaconf_settings_source,
                env_settings,
                file_secret_settings,
            )

    @classmethod
    def _omegaconf_settings_source(cls, settings: BaseSettings) -> dict[str, Any]:
        if initconf := cls.initconf:
            hints = get_type_hints(settings)
            name = get_args(hints['type'])[0]
            return initconf.get(name,{})
        else:
            return {}
        
def add_field_argument(command_parser, name: str, field, default_override = None):
    default = default_override if default_override is not None else field.default if field.default_factory is None else field.default_factory()
    if get_origin(field.type_) == Literal:
        allowed_values = get_args(field.type_)
        allowed_types = set()
        for val in allowed_values:
            allowed_types.add(type(val))
        allowed_types_list = list(allowed_types)
        field_type = allowed_types_list[0] if len(allowed_types) == 1 else Union[allowed_types_list]  # type: ignore

        command_parser.add_argument(
            f"--{name}",
            dest=name,
            type=field_type,
            default=default,
            choices=allowed_values,
            help=field.field_info.description,
        )
    else:
        command_parser.add_argument(
            f"--{name}",
            dest=name,
            type=field.type_,
            default=default,
            action=argparse.BooleanOptionalAction if field.type_==bool else 'store',
            help=field.field_info.description,
        )

def get_configuration(
        object_type: InvokeAISettings,
        root: Path = _get_root_directory(),
)->InvokeAISettings:
    conf_file = root / INIT_FILE
    try:  # setting shared class variable
        InvokeAISettings.initconf = OmegaConf.load(conf_file)
    except OSError as e:
        print(f'** Initialization file could not be read. {str(e)}')
    return object_type(id=uuid.uuid4().hex) \
        if 'id' in get_type_hints(object_type) \
           else object_type()

