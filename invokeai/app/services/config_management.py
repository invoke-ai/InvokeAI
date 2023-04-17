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

from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from pydantic import BaseModel, BaseSettings, Field
from typing import Any, ClassVar, List, Literal, get_type_hints, get_args

from ..cli.commands import add_field_argument

_invokeai_config = None

class InvokeAIAppConfig(BaseSettings):
    '''
    Runtime configuration settings that are not node-specific.
    '''
    type: Literal["app_settings"] = "app_settings"
    precision           : Literal[tuple(['auto','float16','float32','autocast'])] = 'float16'
    outdir              : Path = Field(default='~/invokeai/outputs', description='Default folder for output images')
    xformers_enabled    : bool = Field(default=True, description="Whether to enable memory-efficient attention")
    sequential_guidance : bool = Field(default=False, description="Whether to calculate guidance in serial instead of in parallel, lowering memory requirements")
    max_loaded_models   : int = Field(default=2, gt=0, description="Maximum number of models to keep in memory for rapid switching")
    nsfw_checker        : bool = Field(default=True, description="Whether to enable the NSFW checker")
    conf                : ClassVar[DictConfig] = None

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
        default_settings_stanza = get_args(get_type_hints(self)['type'])[0]
        conf = self.conf.get(default_settings_stanza) if self.conf and default_settings_stanza in self.conf else None

        fields = self.__fields__
        for name, field in fields.items():
            if name not in self._excluded():
                if conf and name in conf:
                    field.default = conf.get(name) 
                add_field_argument(parser, name, field)
        return parser

    @classmethod
    def _excluded(self)->List[str]:
        return ['type','conf']
    
    class Config:
        env_file_encoding = 'utf-8'
        arbitrary_types_allowed = True
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                InvokeAIAppConfig._omegaconf_settings_source,
                env_settings,
                file_secret_settings,
            )

    @classmethod
    def _omegaconf_settings_source(cls, settings: BaseSettings) -> dict[str, Any]:
        if conf := cls.conf:
            hints = get_type_hints(settings)
            name = get_args(hints['type'])[0]
            return conf.get(name,{})
        else:
            return {}
        
def get_app_config(conf_file: Path = Path('./invokeai.yaml'))->InvokeAIAppConfig:
    global _invokeai_config
    if not _invokeai_config:
        InvokeAIAppConfig.conf = OmegaConf.load(conf_file)
        _invokeai_config = InvokeAIAppConfig()
    return _invokeai_config        


