# Copyright (c) 2023 Lincoln Stein (https://github.com/lstein) and the InvokeAI Development Team

"""Invokeai configuration system.

Arguments and fields are taken from the pydantic definition of the
model.  Defaults can be set by creating a yaml configuration file that
has a top-level key of "InvokeAI" and subheadings for each of the
categories returned by `invokeai --help`. The file looks like this:

[file: invokeai.yaml]

InvokeAI:
  Web Server:
    host: 127.0.0.1
    port: 9090
    allow_origins: []
    allow_credentials: true
    allow_methods:
    - '*'
    allow_headers:
    - '*'
  Features:
    esrgan: true
    internet_available: true
    log_tokenization: false
    patchmatch: true
    ignore_missing_core_models: false
  Paths:
    autoimport_dir: autoimport
    lora_dir: null
    embedding_dir: null
    controlnet_dir: null
    conf_path: configs/models.yaml
    models_dir: models
    legacy_conf_dir: configs/stable-diffusion
    db_dir: databases
    outdir: /home/lstein/invokeai-main/outputs
    use_memory_db: false
  Logging:
    log_handlers:
    - console
    log_format: plain
    log_level: info
  Model Cache:
    ram: 13.5
    vram: 0.25
    lazy_offload: true
  Device:
    device: auto
    precision: auto
  Generation:
    sequential_guidance: false
    attention_type: xformers
    attention_slice_size: auto
    force_tiled_decode: false

The default name of the configuration file is `invokeai.yaml`, located
in INVOKEAI_ROOT. You can replace supersede this by providing any
OmegaConf dictionary object initialization time:

 omegaconf = OmegaConf.load('/tmp/init.yaml')
 conf = InvokeAIAppConfig()
 conf.parse_args(conf=omegaconf)

InvokeAIAppConfig.parse_args() will parse the contents of `sys.argv`
at initialization time. You may pass a list of strings in the optional
`argv` argument to use instead of the system argv:

 conf.parse_args(argv=['--log_tokenization'])

It is also possible to set a value at initialization time. However, if
you call parse_args() it may be overwritten.

 conf = InvokeAIAppConfig(log_tokenization=True)
 conf.parse_args(argv=['--no-log_tokenization'])
 conf.log_tokenization
 # False

To avoid this, use `get_config()` to retrieve the application-wide
configuration object. This will retain any properties set at object
creation time:

 conf = InvokeAIAppConfig.get_config(log_tokenization=True)
 conf.parse_args(argv=['--no-log_tokenization'])
 conf.log_tokenization
 # True

Any setting can be overwritten by setting an environment variable of
form: "INVOKEAI_<setting>", as in:

  export INVOKEAI_port=8080

Order of precedence (from highest):
   1) initialization options
   2) command line options
   3) environment variable options
   4) config file options
   5) pydantic defaults

Typical usage at the top level file:

 from invokeai.app.services.config import InvokeAIAppConfig

 # get global configuration and print its cache size
 conf = InvokeAIAppConfig.get_config()
 conf.parse_args()
 print(conf.ram_cache_size)

Typical usage in a backend module:

 from invokeai.app.services.config import InvokeAIAppConfig

 # get global configuration and print its cache size value
 conf = InvokeAIAppConfig.get_config()
 print(conf.ram_cache_size)

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

In most cases, you will want to create a single InvokeAIAppConfig
object for the entire application. The InvokeAIAppConfig.get_config() function
does this:

  config = InvokeAIAppConfig.get_config()
  config.parse_args()   # read values from the command line/config file
  print(config.root)

# Subclassing

If you wish to create a similar class, please subclass the
`InvokeAISettings` class and define a Literal field named "type",
which is set to the desired top-level name.  For example, to create a
"InvokeBatch" configuration, define like this:

  class InvokeBatch(InvokeAISettings):
     type: Literal["InvokeBatch"] = "InvokeBatch"
     node_count : int = Field(default=1, description="Number of nodes to run on", category='Resources')
     cpu_count  : int = Field(default=8, description="Number of GPUs to run on per node", category='Resources')

This will now read and write from the "InvokeBatch" section of the
config file, look for environment variables named INVOKEBATCH_*, and
accept the command-line arguments `--node_count` and `--cpu_count`. The
two configs are kept in separate sections of the config file:

  # invokeai.yaml

  InvokeBatch:
     Resources:
        node_count: 1
        cpu_count: 8

  InvokeAI:
     Paths:
        root: /home/lstein/invokeai-main
        conf_path: configs/models.yaml
        legacy_conf_dir: configs/stable-diffusion
        outdir: outputs
     ...

"""
from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Union, get_type_hints, Optional

from omegaconf import OmegaConf, DictConfig
from pydantic import Field, parse_obj_as

from .base import InvokeAISettings

INIT_FILE = Path("invokeai.yaml")
DB_FILE = Path("invokeai.db")
LEGACY_INIT_FILE = Path("invokeai.init")
DEFAULT_MAX_VRAM = 0.5


class InvokeAIAppConfig(InvokeAISettings):
    """
    Generate images using Stable Diffusion. Use "invokeai" to launch
    the command-line client (recommended for experts only), or
    "invokeai-web" to launch the web server. Global options
    can be changed by editing the file "INVOKEAI_ROOT/invokeai.yaml" or by
    setting environment variables INVOKEAI_<setting>.
    """

    singleton_config: ClassVar[InvokeAIAppConfig] = None
    singleton_init: ClassVar[Dict] = None

    # fmt: off
    type: Literal["InvokeAI"] = "InvokeAI"

    # WEB
    host                : str = Field(default="127.0.0.1", description="IP address to bind to", category='Web Server')
    port                : int = Field(default=9090, description="Port to bind to", category='Web Server')
    allow_origins       : List[str] = Field(default=[], description="Allowed CORS origins", category='Web Server')
    allow_credentials   : bool = Field(default=True, description="Allow CORS credentials", category='Web Server')
    allow_methods       : List[str] = Field(default=["*"], description="Methods allowed for CORS", category='Web Server')
    allow_headers       : List[str] = Field(default=["*"], description="Headers allowed for CORS", category='Web Server')

    # FEATURES
    esrgan              : bool = Field(default=True, description="Enable/disable upscaling code", category='Features')
    internet_available  : bool = Field(default=True, description="If true, attempt to download models on the fly; otherwise only use local models", category='Features')
    log_tokenization    : bool = Field(default=False, description="Enable logging of parsed prompt tokens.", category='Features')
    patchmatch          : bool = Field(default=True, description="Enable/disable patchmatch inpaint code", category='Features')
    ignore_missing_core_models : bool = Field(default=False, description='Ignore missing models in models/core/convert', category='Features')

    # PATHS
    root                : Path = Field(default=None, description='InvokeAI runtime root directory', category='Paths')
    autoimport_dir      : Path = Field(default='autoimport', description='Path to a directory of models files to be imported on startup.', category='Paths')
    lora_dir            : Path = Field(default=None, description='Path to a directory of LoRA/LyCORIS models to be imported on startup.', category='Paths')
    embedding_dir       : Path = Field(default=None, description='Path to a directory of Textual Inversion embeddings to be imported on startup.', category='Paths')
    controlnet_dir      : Path = Field(default=None, description='Path to a directory of ControlNet embeddings to be imported on startup.', category='Paths')
    conf_path           : Path = Field(default='configs/models.yaml', description='Path to models definition file', category='Paths')
    models_dir          : Path = Field(default='models', description='Path to the models directory', category='Paths')
    legacy_conf_dir     : Path = Field(default='configs/stable-diffusion', description='Path to directory of legacy checkpoint config files', category='Paths')
    db_dir              : Path = Field(default='databases', description='Path to InvokeAI databases directory', category='Paths')
    outdir              : Path = Field(default='outputs', description='Default folder for output images', category='Paths')
    use_memory_db       : bool = Field(default=False, description='Use in-memory database for storing image metadata', category='Paths')
    from_file           : Path = Field(default=None, description='Take command input from the indicated file (command-line client only)', category='Paths')

    # LOGGING
    log_handlers        : List[str] = Field(default=["console"], description='Log handler. Valid options are "console", "file=<path>", "syslog=path|address:host:port", "http=<url>"', category="Logging")
    # note - would be better to read the log_format values from logging.py, but this creates circular dependencies issues
    log_format          : Literal['plain', 'color', 'syslog', 'legacy'] = Field(default="color", description='Log format. Use "plain" for text-only, "color" for colorized output, "legacy" for 2.3-style logging and "syslog" for syslog-style', category="Logging")
    log_level           : Literal["debug", "info", "warning", "error", "critical"] = Field(default="info", description="Emit logging messages at this level or  higher", category="Logging")

    dev_reload          : bool = Field(default=False, description="Automatically reload when Python sources are changed.", category="Development")

    version             : bool = Field(default=False, description="Show InvokeAI version and exit", category="Other")

    # CACHE
    ram                 : Union[float, Literal["auto"]] = Field(default=6.0, gt=0, description="Maximum memory amount used by model cache for rapid switching (floating point number or 'auto')", category="Model Cache", )
    vram                : Union[float, Literal["auto"]] = Field(default=0.25, ge=0, description="Amount of VRAM reserved for model storage (floating point number or 'auto')", category="Model Cache", )
    lazy_offload        : bool = Field(default=True, description="Keep models in VRAM until their space is needed", category="Model Cache", )

    # DEVICE
    device              : Literal[tuple(["auto", "cpu", "cuda", "cuda:1", "mps"])] = Field(default="auto", description="Generation device", category="Device", )
    precision: Literal[tuple(["auto", "float16", "float32", "autocast"])] = Field(default="auto", description="Floating point precision", category="Device", )

    # GENERATION
    sequential_guidance : bool = Field(default=False, description="Whether to calculate guidance in serial instead of in parallel, lowering memory requirements", category="Generation", )
    attention_type      : Literal[tuple(["auto", "normal", "xformers", "sliced", "torch-sdp"])] = Field(default="auto", description="Attention type", category="Generation", )
    attention_slice_size: Literal[tuple(["auto", "balanced", "max", 1, 2, 3, 4, 5, 6, 7, 8])] = Field(default="auto", description='Slice size, valid when attention_type=="sliced"', category="Generation", )
    force_tiled_decode: bool = Field(default=False, description="Whether to enable tiled VAE decode (reduces memory consumption with some performance penalty)", category="Generation",)

    # DEPRECATED FIELDS - STILL HERE IN ORDER TO OBTAN VALUES FROM PRE-3.1 CONFIG FILES
    always_use_cpu      : bool = Field(default=False, description="If true, use the CPU for rendering even if a GPU is available.", category='Memory/Performance')
    free_gpu_mem        : Optional[bool] = Field(default=None, description="If true, purge model from GPU after each generation.", category='Memory/Performance')
    max_cache_size      : Optional[float] = Field(default=None, gt=0, description="Maximum memory amount used by model cache for rapid switching", category='Memory/Performance')
    max_vram_cache_size : Optional[float] = Field(default=None, ge=0, description="Amount of VRAM reserved for model storage", category='Memory/Performance')
    xformers_enabled    : bool = Field(default=True, description="Enable/disable memory-efficient attention", category='Memory/Performance')
    tiled_decode        : bool = Field(default=False, description="Whether to enable tiled VAE decode (reduces memory consumption with some performance penalty)", category='Memory/Performance')

    # See InvokeAIAppConfig subclass below for CACHE and DEVICE categories
    # fmt: on

    class Config:
        validate_assignment = True

    def parse_args(self, argv: List[str] = None, conf: DictConfig = None, clobber=False):
        """
        Update settings with contents of init file, environment, and
        command-line settings.
        :param conf: alternate Omegaconf dictionary object
        :param argv: aternate sys.argv list
        :param clobber: ovewrite any initialization parameters passed during initialization
        """
        # Set the runtime root directory. We parse command-line switches here
        # in order to pick up the --root_dir option.
        super().parse_args(argv)
        if conf is None:
            try:
                conf = OmegaConf.load(self.root_dir / INIT_FILE)
            except Exception:
                pass
        InvokeAISettings.initconf = conf

        # parse args again in order to pick up settings in configuration file
        super().parse_args(argv)

        if self.singleton_init and not clobber:
            hints = get_type_hints(self.__class__)
            for k in self.singleton_init:
                setattr(self, k, parse_obj_as(hints[k], self.singleton_init[k]))

    @classmethod
    def get_config(cls, **kwargs) -> InvokeAIAppConfig:
        """
        This returns a singleton InvokeAIAppConfig configuration object.
        """
        if (
            cls.singleton_config is None
            or type(cls.singleton_config) is not cls
            or (kwargs and cls.singleton_init != kwargs)
        ):
            cls.singleton_config = cls(**kwargs)
            cls.singleton_init = kwargs
        return cls.singleton_config

    @property
    def root_path(self) -> Path:
        """
        Path to the runtime root directory
        """
        if self.root:
            root = Path(self.root).expanduser().absolute()
        else:
            root = self.find_root().expanduser().absolute()
        self.root = root  # insulate ourselves from relative paths that may change
        return root

    @property
    def root_dir(self) -> Path:
        """
        Alias for above.
        """
        return self.root_path

    def _resolve(self, partial_path: Path) -> Path:
        return (self.root_path / partial_path).resolve()

    @property
    def init_file_path(self) -> Path:
        """
        Path to invokeai.yaml
        """
        return self._resolve(INIT_FILE)

    @property
    def output_path(self) -> Path:
        """
        Path to defaults outputs directory.
        """
        return self._resolve(self.outdir)

    @property
    def db_path(self) -> Path:
        """
        Path to the invokeai.db file.
        """
        return self._resolve(self.db_dir) / DB_FILE

    @property
    def model_conf_path(self) -> Path:
        """
        Path to models configuration file.
        """
        return self._resolve(self.conf_path)

    @property
    def legacy_conf_path(self) -> Path:
        """
        Path to directory of legacy configuration files (e.g. v1-inference.yaml)
        """
        return self._resolve(self.legacy_conf_dir)

    @property
    def models_path(self) -> Path:
        """
        Path to the models directory
        """
        return self._resolve(self.models_dir)

    @property
    def autoconvert_path(self) -> Path:
        """
        Path to the directory containing models to be imported automatically at startup.
        """
        return self._resolve(self.autoconvert_dir) if self.autoconvert_dir else None

    # the following methods support legacy calls leftover from the Globals era
    @property
    def full_precision(self) -> bool:
        """Return true if precision set to float32"""
        return self.precision == "float32"

    @property
    def try_patchmatch(self) -> bool:
        """Return true if patchmatch true"""
        return self.patchmatch

    @property
    def nsfw_checker(self) -> bool:
        """NSFW node is always active and disabled from Web UIe"""
        return True

    @property
    def invisible_watermark(self) -> bool:
        """invisible watermark node is always active and disabled from Web UIe"""
        return True

    @property
    def ram_cache_size(self) -> float:
        return self.max_cache_size or self.ram

    @property
    def vram_cache_size(self) -> float:
        return self.max_vram_cache_size or self.vram

    @property
    def use_cpu(self) -> bool:
        return self.always_use_cpu or self.device == "cpu"

    @property
    def disable_xformers(self) -> bool:
        """
        Return true if enable_xformers is false (reversed logic)
        and attention type is not set to xformers.
        """
        disabled_in_config = not self.xformers_enabled
        return disabled_in_config and self.attention_type != "xformers"

    @staticmethod
    def find_root() -> Path:
        """
        Choose the runtime root directory when not specified on command line or
        init file.
        """
        return _find_root()


def get_invokeai_config(**kwargs) -> InvokeAIAppConfig:
    """
    Legacy function which returns InvokeAIAppConfig.get_config()
    """
    return InvokeAIAppConfig.get_config(**kwargs)


def _find_root() -> Path:
    venv = Path(os.environ.get("VIRTUAL_ENV") or ".")
    if os.environ.get("INVOKEAI_ROOT"):
        root = Path(os.environ["INVOKEAI_ROOT"])
    elif any([(venv.parent / x).exists() for x in [INIT_FILE, LEGACY_INIT_FILE]]):
        root = (venv.parent).resolve()
    else:
        root = Path("~/invokeai").expanduser().resolve()
    return root
