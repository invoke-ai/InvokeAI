# Copyright (c) 2023 Lincoln Stein (https://github.com/lstein)

'''Invokeai configuration system.

Arguments and fields are taken from the pydantic definition of the
model.  Defaults can be set by creating a yaml configuration file that
has top-level keys corresponding to an invocation name, a command, or
"globals" for global values such as `xformers_enabled`. Currently
graphs cannot be configured this way, but their constituents can be.

[file: invokeai.yaml]

 globals:
   nsfw_checker: False
   max_loaded_models: 5
 
 txt2img:
   steps: 20
   scheduler: k_heun
   width: 768
 
 img2img:
   width: 1024
   height: 1024

Any value can be overwritten by setting an environment variable of
form: "INVOKEAI_<command>_<value>":

  export INVOKEAI_txt2img_steps=30

Order of precedence (from highest):
   1) command line options
   2) environment variable options
   3) config file options
   4) pydantic defaults

Typical usage:

 from invokeai.app.services.config import InvokeAIAppConfig
 from invokeai.invocations.generate import TextToImageInvocation

 # get global configuration and print its nsfw_checker value
 conf = InvokeAIAppConfig()
 print(conf.nsfw_checker)

 # get the text2image invocation and print its step value
 text2image = TextToImageInvocation()
 print(text2image.steps)

'''
import argparse
import os
import sys
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from pydantic import BaseSettings, Field
from typing import Any, ClassVar, List, Literal, Union, get_origin, get_type_hints, get_args

INIT_FILE = Path('invokeai.yaml')

class InvokeAISettings(BaseSettings):
    '''
    Runtime configuration settings in which default values are
    read from an omegaconf .yaml file.
    '''
    initconf             : ClassVar[DictConfig] = None

    def parse_args(self, argv: list=sys.argv[1:]):
        parser = self.get_parser()
        opt = parser.parse_args(argv)
        for name in self.__fields__:
            if name not in self._excluded():
                setattr(self, name, getattr(opt,name))

    @classmethod
    def add_parser_arguments(cls, parser):
        env_prefix = cls.Config.env_prefix
        default_settings_stanza = get_args(get_type_hints(cls)['type'])[0]
        initconf = cls.initconf.get(default_settings_stanza) if cls.initconf and default_settings_stanza in cls.initconf else None

        fields = cls.__fields__
        for name, field in fields.items():
            if name not in cls._excluded():
                env_name = env_prefix+name
                if initconf and name in initconf:
                    field.default = initconf.get(name)
                if env_name in os.environ:
                    field.default = os.environ[env_name]
                cls.add_field_argument(parser, name, field)


    @classmethod
    def cmd_name(self, command_field: str='type')->str:
        hints = get_type_hints(self)
        return get_args(hints[command_field])[0]

    @classmethod
    def get_parser(cls)->ArgumentParser:
        parser = ArgumentParser(
            prog=cls.cmd_name(),
            description=cls.__doc__,
        )
        cls.add_parser_arguments(parser)
        return parser

    @classmethod
    def add_subparser(cls, parser: argparse.ArgumentParser):
        parser.add_parser(cls.cmd_name(), help=cls.__doc__)

    @classmethod
    def _excluded(self)->List[str]:
        return ['type','initconf']
    
    class Config:
        env_file_encoding = 'utf-8'
        arbitrary_types_allowed = True
        env_prefix = 'INVOKEAI_'
        extra = 'allow'         # TODO fix me. this is sloppy
        #class_sensitive = True   ???
        case_sensitive = True
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                cls._omegaconf_settings_source,
                env_settings,
                file_secret_settings,
            )

        @classmethod
        def _omegaconf_settings_source(cls, settings: BaseSettings) -> dict[str, Any]:
            if initconf := InvokeAISettings.initconf:
                return initconf.get(settings.cmd_name(),{})
            else:
                return {}

    @staticmethod
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

class InvokeAIAppConfig(InvokeAISettings):
    '''
    Application-wide settings.
    '''
    #fmt: off
    type: Literal["globals"] = "globals"
    precision           : Literal[tuple(['auto','float16','float32','autocast'])] = 'float16'
    conf                : Path = Field(default='configs/models.yaml', description='Path to models definition file')
    outdir              : Path = Field(default='outputs', description='Default folder for output images')
    root                : Path = Field(default='~/invokeai', description='InvokeAI runtime root directory')
    embedding_dir       : Path = Field(default='embeddings', description='Path to InvokeAI embeddings directory')
    autoconvert_dir     : Path = Field(default=None, description='Path to a directory of ckpt files to be converted into diffusers and imported on startup.')
    gfpgan_model_dir    : Path = Field(default="./models/gfpgan/GFPGANv1.4.pth", description='Path to GFPGAN models directory.')
    embeddings          : bool = Field(default=True, description='Load contents of embeddings directory')
    xformers_enabled    : bool = Field(default=True, description="Enable/disable memory-efficient attention")
    sequential_guidance : bool = Field(default=False, description="Whether to calculate guidance in serial instead of in parallel, lowering memory requirements")
    max_loaded_models   : int = Field(default=2, gt=0, description="Maximum number of models to keep in memory for rapid switching")
    nsfw_checker        : bool = Field(default=True, description="Enable/disable the NSFW checker")
    restore             : bool = Field(default=True, description="Enable/disable face restoration code")
    esrgan              : bool = Field(default=True, description="Enable/disable upscaling code")
    #fmt: on

    def __init__(self, conf: DictConfig = None, argv: List[str]=None):
        super().__init__()
        
        # Set the runtime root directory. We parse command-line switches here
        # in order to pick up the --root_dir option.
        self.parse_args(argv)
        if not self.root:
            self.root = self._find_root()
        if not conf:
            try:
                conf = OmegaConf.load(self.root_dir / INIT_FILE)
            except:
                pass
        InvokeAISettings.initconf = conf
        # parse args again in order to pick up settings in configuration file
        self.parse_args(argv)

    @property
    def root_dir(self)->Path:
        return self.root.expanduser()

    def _resolve(self,partial_path:Path)->Path:
        return (self.root_dir / partial_path).resolve()

    @property
    def output_path(self)->Path:
        return self._resolve(self.outdir)

    @property
    def model_conf_path(self)->Path:
        return self._resolve(self.conf)

    @property
    def embedding_path(self)->Path:
        return self._resolve(self.embedding_dir) if self.embedding_dir else None

    @property
    def autoconvert_path(self)->Path:
        return self._resolve(self.autoconvert_dir) if self.autoconvert_dir else None

    @property
    def gfpgan_model_path(self)->Path:
        return self._resolve(self.gfpgan_model_dir) if self.gfpgan_model_dir else None

    @staticmethod
    def find_root()->Path:
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


class InvokeAIWebConfig(InvokeAIAppConfig):
    '''
    Web-specific settings
    '''
    #fmt: off
    type               : Literal["web"] = "web"
    allow_origins      : List = Field(default=[], description='Allowed CORS origins')
    allow_credentials  : bool = Field(default=True, description='Allow CORS credentials')
    allow_methods      : List = Field(default=["*"], description='Methods allowed for CORS')
    allow_headers      : List = Field(default=["*"], description='Headers allowed for CORS')
    #fmt: on
