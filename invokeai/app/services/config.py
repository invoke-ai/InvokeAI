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

The default name of the configuration file is `invokeai.yaml`, located
in INVOKEAI_ROOT. You can use any OmegaConf dictionary by passing it
to the config object at initialization time:

 omegaconf = OmegaConf.load('/tmp/init.yaml')
 conf = InvokeAIAppConfig(conf=omegaconf)

By default, InvokeAIAppConfig will parse the contents of argv at
initialization time. You may pass a list of strings in the optional
`argv` argument to use instead of the system argv:

 conf = InvokeAIAppConfig(arg=['--xformers_enabled'])

It is also possible to set a value at initialization time. This value
has highest priority.

 conf = InvokeAIAppConfig(xformers_enabled=True)

Any setting can be overwritten by setting an environment variable of
form: "INVOKEAI_<command>_<value>", as in:

  export INVOKEAI_txt2img_steps=30

Order of precedence (from highest):
   1) initialization options
   2) command line options
   3) environment variable options
   4) config file options
   5) pydantic defaults

Typical usage:

 from invokeai.app.services.config import InvokeAIAppConfig
 from invokeai.invocations.generate import TextToImageInvocation

 # get global configuration and print its nsfw_checker value
 conf = InvokeAIAppConfig()
 print(conf.nsfw_checker)

 # get the text2image invocation and print its step value
 text2image = TextToImageInvocation()
 print(text2image.steps)

Computed properties:

The InvokeAIAppConfig object has a series of properties that
resolve paths relative to the runtime root directory. They each return
a Path object:

 root_path          - path to InvokeAI root
 output_path        - path to default outputs directory
 model_conf_path    - path to models.yaml
 conf               - alias for the above
 embedding_path     - path to the embeddings directory
 lora_path          - path to the LoRA directory
 

'''
import argparse
import os
import sys
from argparse import ArgumentParser
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from pydantic import BaseSettings, Field, parse_obj_as
from typing import Any, ClassVar, List, Literal, Union, get_origin, get_type_hints, get_args

INIT_FILE = Path('invokeai.yaml')
LEGACY_INIT_FILE = Path('invokeai.init')

class InvokeAISettings(BaseSettings):
    '''
    Runtime configuration settings in which default values are
    read from an omegaconf .yaml file.
    '''
    initconf             : ClassVar[DictConfig] = None

    def parse_args(self, argv: list=sys.argv[1:]):
        parser = self.get_parser()
        opt, _ = parser.parse_known_args(argv)
        for name in self.__fields__:
            if name not in self._excluded():
                setattr(self, name, getattr(opt,name))

    @classmethod
    def add_parser_arguments(cls, parser):
        env_prefix = cls.Config.env_prefix if hasattr(cls.Config,'env_prefix') else 'INVOKEAI_'
        default_settings_stanza = get_args(get_type_hints(cls)['type'])[0]
        initconf = cls.initconf.get(default_settings_stanza) if cls.initconf and default_settings_stanza in cls.initconf else None

        fields = cls.__fields__
        for name, field in fields.items():
            if name not in cls._excluded():
                env_name = env_prefix+f'{cls.cmd_name()}_{name}'
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
def _find_root()->Path:
    if os.environ.get("INVOKEAI_ROOT"):
        root = Path(os.environ.get("INVOKEAI_ROOT")).resolve()
    elif (
            os.environ.get("VIRTUAL_ENV")
            and (Path(os.environ.get("VIRTUAL_ENV"), "..", INIT_FILE).exists()
                 or
                 Path(os.environ.get("VIRTUAL_ENV"), "..", LEGACY_INIT_FILE).exists()
                 )
    ):
        root = Path(os.environ.get("VIRTUAL_ENV"), "..").resolve()
    else:
        root = Path("~/invokeai").expanduser().resolve()
    return root

class InvokeAIAppConfig(InvokeAISettings):
    '''
    Application-wide settings.
    '''
    #fmt: off
    type: Literal["globals"] = "globals"
    root                : Path = Field(default=_find_root(), description='InvokeAI runtime root directory')
    infile              : Path = Field(default=None, description='Path to a file of prompt commands to bulk generate from')
    precision           : Literal[tuple(['auto','float16','float32','autocast'])] = 'float16'
    conf_path           : Path = Field(default='configs/models.yaml', description='Path to models definition file')
    model               : str = Field(default='stable-diffusion-1.5', description='Initial model name')
    outdir              : Path = Field(default='outputs', description='Default folder for output images')
    embedding_dir       : Path = Field(default='embeddings', description='Path to InvokeAI textual inversion aembeddings directory')
    lora_dir            : Path = Field(default='loras', description='Path to InvokeAI LoRA model directory')
    autoconvert_dir     : Path = Field(default=None, description='Path to a directory of ckpt files to be converted into diffusers and imported on startup.')
    gfpgan_model_dir    : Path = Field(default="./models/gfpgan/GFPGANv1.4.pth", description='Path to GFPGAN models directory.')
    embeddings          : bool = Field(default=True, description='Load contents of embeddings directory')
    xformers_enabled    : bool = Field(default=True, description="Enable/disable memory-efficient attention")
    sequential_guidance : bool = Field(default=False, description="Whether to calculate guidance in serial instead of in parallel, lowering memory requirements")
    max_loaded_models   : int = Field(default=2, gt=0, description="Maximum number of models to keep in memory for rapid switching")
    nsfw_checker        : bool = Field(default=True, description="Enable/disable the NSFW checker")
    restore             : bool = Field(default=True, description="Enable/disable face restoration code")
    esrgan              : bool = Field(default=True, description="Enable/disable upscaling code")
    patchmatch          : bool = Field(default=True, description="Enable/disable patchmatch inpaint code")
    internet_available  : bool = Field(default=True, description="If true, attempt to download models on the fly; otherwise only use local models")
    always_use_cpu      : bool = Field(default=False, description="If true, use the CPU for rendering even if a GPU is available.")
    free_gpu_mem        : bool = Field(default=False, description="If true, purge model from GPU after each generation.")
    log_tokenization    : bool = Field(default=False, description="Enable logging of parsed prompt tokens.")
    #fmt: on

    def __init__(self, conf: DictConfig = None, argv: List[str]=None, **kwargs):
        '''
        Initialize InvokeAIAppconfig.
        :param conf: alternate Omegaconf dictionary object
        :param argv: aternate sys.argv list
        :param **kwargs: attributes to initialize with
        '''
        super().__init__(**kwargs)
        
        # Set the runtime root directory. We parse command-line switches here
        # in order to pick up the --root_dir option.
        self.parse_args(argv)
        if not conf:
            try:
                conf = OmegaConf.load(self.root_dir / INIT_FILE)
            except:
                pass
        InvokeAISettings.initconf = conf

        # parse args again in order to pick up settings in configuration file
        self.parse_args(argv)

        # restore initialization values
        hints = get_type_hints(self)
        for k in kwargs:
            setattr(self,k,parse_obj_as(hints[k],kwargs[k]))

    @property
    def root_path(self)->Path:
        '''
        Path to the runtime root directory
        '''
        if self.root:
            return self.root.expanduser()
        else:
            return self.find_root()

    @property
    def root_dir(self)->Path:
        '''
        Alias for above.
        '''
        return self.root_path

    def _resolve(self,partial_path:Path)->Path:
        return (self.root_path / partial_path).resolve()

    @property
    def output_path(self)->Path:
        '''
        Path to defaults outputs directory.
        '''
        return self._resolve(self.outdir)

    @property
    def model_conf_path(self)->Path:
        '''
        Path to models configuration file.
        '''
        return self._resolve(self.conf_path)

    @property
    def conf(self)->Path:
        '''
        Path to models configuration file (alias for model_conf_path).
        '''
        return self.model_conf_path

    @property
    def embedding_path(self)->Path:
        '''
        Path to the textual inversion embeddings directory.
        '''
        return self._resolve(self.embedding_dir) if self.embedding_dir else None
    
    @property
    def lora_path(self)->Path:
        '''
        Path to the LoRA models directory.
        '''
        return self._resolve(self.lora_dir) if self.lora_dir else None

    @property
    def autoconvert_path(self)->Path:
        '''
        Path to the directory containing models to be imported automatically at startup.
        '''
        return self._resolve(self.autoconvert_dir) if self.autoconvert_dir else None

    @property
    def gfpgan_model_path(self)->Path:
        '''
        Path to the GFPGAN model.
        '''
        return self._resolve(self.gfpgan_model_dir) if self.gfpgan_model_dir else None

    @staticmethod
    def find_root()->Path:
        '''
        Choose the runtime root directory when not specified on command line or
        init file.
        '''
        return _find_root()

class InvokeAIWebConfig(InvokeAIAppConfig):
    '''
    Web-specific settings
    '''
    #fmt: off
    type               : Literal["web"] = "web"
    allow_origins      : List = Field(default=[], description="Allowed CORS origins")
    allow_credentials  : bool = Field(default=True, description="Allow CORS credentials")
    allow_methods      : List = Field(default=["*"], description="Methods allowed for CORS")
    allow_headers      : List = Field(default=["*"], description="Headers allowed for CORS")
    host               : str = Field(default="127.0.0.1", description="IP address to bind to")
    port               : int = Field(default=9090, description="Port to bind to")
    #fmt: on
