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
    log_memory_usage: false
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
     node_count : int = Field(default=1, description="Number of nodes to run on", json_schema_extra=dict(category='Resources'))
     cpu_count  : int = Field(default=8, description="Number of GPUs to run on per node", json_schema_extra=dict(category='Resources'))

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
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from omegaconf import DictConfig, OmegaConf
from pydantic import Field
from pydantic.config import JsonDict
from pydantic_settings import SettingsConfigDict

from .config_base import InvokeAISettings

INIT_FILE = Path("invokeai.yaml")
DB_FILE = Path("invokeai.db")
LEGACY_INIT_FILE = Path("invokeai.init")
DEFAULT_MAX_VRAM = 0.5


class Categories(object):
    """Category headers for configuration variable groups."""

    WebServer: JsonDict = {"category": "Web Server"}
    Features: JsonDict = {"category": "Features"}
    Paths: JsonDict = {"category": "Paths"}
    Logging: JsonDict = {"category": "Logging"}
    Development: JsonDict = {"category": "Development"}
    Other: JsonDict = {"category": "Other"}
    ModelCache: JsonDict = {"category": "Model Cache"}
    Device: JsonDict = {"category": "Device"}
    Generation: JsonDict = {"category": "Generation"}
    Queue: JsonDict = {"category": "Queue"}
    Nodes: JsonDict = {"category": "Nodes"}
    MemoryPerformance: JsonDict = {"category": "Memory/Performance"}


class InvokeAIAppConfig(InvokeAISettings):
    """Configuration object for InvokeAI App."""

    singleton_config: ClassVar[Optional[InvokeAIAppConfig]] = None
    singleton_init: ClassVar[Optional[Dict[str, Any]]] = None

    # fmt: off
    type: Literal["InvokeAI"] = "InvokeAI"

    # WEB
    host                : str = Field(default="127.0.0.1", description="IP address to bind to", json_schema_extra=Categories.WebServer)
    port                : int = Field(default=9090, description="Port to bind to", json_schema_extra=Categories.WebServer)
    allow_origins       : List[str] = Field(default=[], description="Allowed CORS origins", json_schema_extra=Categories.WebServer)
    allow_credentials   : bool = Field(default=True, description="Allow CORS credentials", json_schema_extra=Categories.WebServer)
    allow_methods       : List[str] = Field(default=["*"], description="Methods allowed for CORS", json_schema_extra=Categories.WebServer)
    allow_headers       : List[str] = Field(default=["*"], description="Headers allowed for CORS", json_schema_extra=Categories.WebServer)
    # SSL options correspond to https://www.uvicorn.org/settings/#https
    ssl_certfile        : Optional[Path] = Field(default=None, description="SSL certificate file (for HTTPS)", json_schema_extra=Categories.WebServer)
    ssl_keyfile         : Optional[Path] = Field(default=None, description="SSL key file", json_schema_extra=Categories.WebServer)

    # FEATURES
    esrgan              : bool = Field(default=True, description="Enable/disable upscaling code", json_schema_extra=Categories.Features)
    internet_available  : bool = Field(default=True, description="If true, attempt to download models on the fly; otherwise only use local models", json_schema_extra=Categories.Features)
    log_tokenization    : bool = Field(default=False, description="Enable logging of parsed prompt tokens.", json_schema_extra=Categories.Features)
    patchmatch          : bool = Field(default=True, description="Enable/disable patchmatch inpaint code", json_schema_extra=Categories.Features)
    ignore_missing_core_models : bool = Field(default=False, description='Ignore missing models in models/core/convert', json_schema_extra=Categories.Features)

    # PATHS
    root                : Optional[Path] = Field(default=None, description='InvokeAI runtime root directory', json_schema_extra=Categories.Paths)
    autoimport_dir      : Path = Field(default=Path('autoimport'), description='Path to a directory of models files to be imported on startup.', json_schema_extra=Categories.Paths)
    conf_path           : Path = Field(default=Path('configs/models.yaml'), description='Path to models definition file', json_schema_extra=Categories.Paths)
    models_dir          : Path = Field(default=Path('models'), description='Path to the models directory', json_schema_extra=Categories.Paths)
    legacy_conf_dir     : Path = Field(default=Path('configs/stable-diffusion'), description='Path to directory of legacy checkpoint config files', json_schema_extra=Categories.Paths)
    db_dir              : Path = Field(default=Path('databases'), description='Path to InvokeAI databases directory', json_schema_extra=Categories.Paths)
    outdir              : Path = Field(default=Path('outputs'), description='Default folder for output images', json_schema_extra=Categories.Paths)
    use_memory_db       : bool = Field(default=False, description='Use in-memory database for storing image metadata', json_schema_extra=Categories.Paths)
    custom_nodes_dir    : Path = Field(default=Path('nodes'), description='Path to directory for custom nodes', json_schema_extra=Categories.Paths)
    from_file           : Optional[Path] = Field(default=None, description='Take command input from the indicated file (command-line client only)', json_schema_extra=Categories.Paths)

    # LOGGING
    log_handlers        : List[str] = Field(default=["console"], description='Log handler. Valid options are "console", "file=<path>", "syslog=path|address:host:port", "http=<url>"', json_schema_extra=Categories.Logging)
    # note - would be better to read the log_format values from logging.py, but this creates circular dependencies issues
    log_format          : Literal['plain', 'color', 'syslog', 'legacy'] = Field(default="color", description='Log format. Use "plain" for text-only, "color" for colorized output, "legacy" for 2.3-style logging and "syslog" for syslog-style', json_schema_extra=Categories.Logging)
    log_level           : Literal["debug", "info", "warning", "error", "critical"] = Field(default="info", description="Emit logging messages at this level or  higher", json_schema_extra=Categories.Logging)
    log_sql             : bool = Field(default=False, description="Log SQL queries", json_schema_extra=Categories.Logging)

    # Development
    dev_reload          : bool = Field(default=False, description="Automatically reload when Python sources are changed.", json_schema_extra=Categories.Development)
    profile_graphs      : bool = Field(default=False, description="Enable graph profiling", json_schema_extra=Categories.Development)
    profile_prefix      : Optional[str] = Field(default=None, description="An optional prefix for profile output files.", json_schema_extra=Categories.Development)
    profiles_dir        : Path = Field(default=Path('profiles'), description="Directory for graph profiles", json_schema_extra=Categories.Development)

    version             : bool = Field(default=False, description="Show InvokeAI version and exit", json_schema_extra=Categories.Other)

    # CACHE
    ram                 : float = Field(default=7.5, gt=0, description="Maximum memory amount used by model cache for rapid switching (floating point number, GB)", json_schema_extra=Categories.ModelCache, )
    vram                : float = Field(default=0.25, ge=0, description="Amount of VRAM reserved for model storage (floating point number, GB)", json_schema_extra=Categories.ModelCache, )
    lazy_offload        : bool = Field(default=True, description="Keep models in VRAM until their space is needed", json_schema_extra=Categories.ModelCache, )
    log_memory_usage    : bool = Field(default=False, description="If True, a memory snapshot will be captured before and after every model cache operation, and the result will be logged (at debug level). There is a time cost to capturing the memory snapshots, so it is recommended to only enable this feature if you are actively inspecting the model cache's behaviour.", json_schema_extra=Categories.ModelCache)

    # DEVICE
    device              : Literal["auto", "cpu", "cuda", "cuda:1", "mps"] = Field(default="auto", description="Generation device", json_schema_extra=Categories.Device)
    precision           : Literal["auto", "float16", "bfloat16", "float32", "autocast"] = Field(default="auto", description="Floating point precision", json_schema_extra=Categories.Device)

    # GENERATION
    sequential_guidance : bool = Field(default=False, description="Whether to calculate guidance in serial instead of in parallel, lowering memory requirements", json_schema_extra=Categories.Generation)
    attention_type      : Literal["auto", "normal", "xformers", "sliced", "torch-sdp"] = Field(default="auto", description="Attention type", json_schema_extra=Categories.Generation)
    attention_slice_size: Literal["auto", "balanced", "max", 1, 2, 3, 4, 5, 6, 7, 8] = Field(default="auto", description='Slice size, valid when attention_type=="sliced"', json_schema_extra=Categories.Generation)
    force_tiled_decode  : bool = Field(default=False, description="Whether to enable tiled VAE decode (reduces memory consumption with some performance penalty)", json_schema_extra=Categories.Generation)
    png_compress_level  : int = Field(default=1, description="The compress_level setting of PIL.Image.save(), used for PNG encoding. All settings are lossless. 0 = fastest, largest filesize, 9 = slowest, smallest filesize", json_schema_extra=Categories.Generation)

    # QUEUE
    max_queue_size      : int = Field(default=10000, gt=0, description="Maximum number of items in the session queue", json_schema_extra=Categories.Queue)

    # NODES
    allow_nodes         : Optional[List[str]] = Field(default=None, description="List of nodes to allow. Omit to allow all.", json_schema_extra=Categories.Nodes)
    deny_nodes          : Optional[List[str]] = Field(default=None, description="List of nodes to deny. Omit to deny none.", json_schema_extra=Categories.Nodes)
    node_cache_size     : int = Field(default=512, description="How many cached nodes to keep in memory", json_schema_extra=Categories.Nodes)

    # MODEL IMPORT
    civitai_api_key       : Optional[str] = Field(default=os.environ.get("CIVITAI_API_KEY"), description="API key for CivitAI", json_schema_extra=Categories.Other)

    # DEPRECATED FIELDS - STILL HERE IN ORDER TO OBTAN VALUES FROM PRE-3.1 CONFIG FILES
    always_use_cpu      : bool = Field(default=False, description="If true, use the CPU for rendering even if a GPU is available.", json_schema_extra=Categories.MemoryPerformance)
    max_cache_size      : Optional[float] = Field(default=None, gt=0, description="Maximum memory amount used by model cache for rapid switching", json_schema_extra=Categories.MemoryPerformance)
    max_vram_cache_size : Optional[float] = Field(default=None, ge=0, description="Amount of VRAM reserved for model storage", json_schema_extra=Categories.MemoryPerformance)
    xformers_enabled    : bool = Field(default=True, description="Enable/disable memory-efficient attention", json_schema_extra=Categories.MemoryPerformance)
    tiled_decode        : bool = Field(default=False, description="Whether to enable tiled VAE decode (reduces memory consumption with some performance penalty)", json_schema_extra=Categories.MemoryPerformance)
    lora_dir            : Optional[Path] = Field(default=None, description='Path to a directory of LoRA/LyCORIS models to be imported on startup.', json_schema_extra=Categories.Paths)
    embedding_dir       : Optional[Path] = Field(default=None, description='Path to a directory of Textual Inversion embeddings to be imported on startup.', json_schema_extra=Categories.Paths)
    controlnet_dir      : Optional[Path] = Field(default=None, description='Path to a directory of ControlNet embeddings to be imported on startup.', json_schema_extra=Categories.Paths)

    # this is not referred to in the source code and can be removed entirely
    #free_gpu_mem        : Optional[bool] = Field(default=None, description="If true, purge model from GPU after each generation.", json_schema_extra=Categories.MemoryPerformance)

    # See InvokeAIAppConfig subclass below for CACHE and DEVICE categories
    # fmt: on

    model_config = SettingsConfigDict(validate_assignment=True, env_prefix="INVOKEAI")

    def parse_args(
        self,
        argv: Optional[list[str]] = None,
        conf: Optional[DictConfig] = None,
        clobber: Optional[bool] = False,
    ) -> None:
        """
        Update settings with contents of init file, environment, and command-line settings.

        :param conf: alternate Omegaconf dictionary object
        :param argv: aternate sys.argv list
        :param clobber: ovewrite any initialization parameters passed during initialization
        """
        # Set the runtime root directory. We parse command-line switches here
        # in order to pick up the --root_dir option.
        super().parse_args(argv)
        loaded_conf = None
        if conf is None:
            try:
                loaded_conf = OmegaConf.load(self.root_dir / INIT_FILE)
            except Exception:
                pass
        if isinstance(loaded_conf, DictConfig):
            InvokeAISettings.initconf = loaded_conf
        else:
            InvokeAISettings.initconf = conf

        # parse args again in order to pick up settings in configuration file
        super().parse_args(argv)

        if self.singleton_init and not clobber:
            # When setting values in this way, set validate_assignment to true if you want to validate the value.
            for k, v in self.singleton_init.items():
                setattr(self, k, v)

    @classmethod
    def get_config(cls, **kwargs: Any) -> InvokeAIAppConfig:
        """Return a singleton InvokeAIAppConfig configuration object."""
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
        """Path to the runtime root directory."""
        if self.root:
            root = Path(self.root).expanduser().absolute()
        else:
            root = self.find_root().expanduser().absolute()
        self.root = root  # insulate ourselves from relative paths that may change
        return root.resolve()

    @property
    def root_dir(self) -> Path:
        """Alias for above."""
        return self.root_path

    def _resolve(self, partial_path: Path) -> Path:
        return (self.root_path / partial_path).resolve()

    @property
    def init_file_path(self) -> Path:
        """Path to invokeai.yaml."""
        resolved_path = self._resolve(INIT_FILE)
        assert resolved_path is not None
        return resolved_path

    @property
    def output_path(self) -> Optional[Path]:
        """Path to defaults outputs directory."""
        return self._resolve(self.outdir)

    @property
    def db_path(self) -> Path:
        """Path to the invokeai.db file."""
        db_dir = self._resolve(self.db_dir)
        assert db_dir is not None
        return db_dir / DB_FILE

    @property
    def model_conf_path(self) -> Path:
        """Path to models configuration file."""
        return self._resolve(self.conf_path)

    @property
    def legacy_conf_path(self) -> Path:
        """Path to directory of legacy configuration files (e.g. v1-inference.yaml)."""
        return self._resolve(self.legacy_conf_dir)

    @property
    def models_path(self) -> Path:
        """Path to the models directory."""
        return self._resolve(self.models_dir)

    @property
    def custom_nodes_path(self) -> Path:
        """Path to the custom nodes directory."""
        custom_nodes_path = self._resolve(self.custom_nodes_dir)
        assert custom_nodes_path is not None
        return custom_nodes_path

    # the following methods support legacy calls leftover from the Globals era
    @property
    def full_precision(self) -> bool:
        """Return true if precision set to float32."""
        return self.precision == "float32"

    @property
    def try_patchmatch(self) -> bool:
        """Return true if patchmatch true."""
        return self.patchmatch

    @property
    def nsfw_checker(self) -> bool:
        """Return value for NSFW checker. The NSFW node is always active and disabled from Web UI."""
        return True

    @property
    def invisible_watermark(self) -> bool:
        """Return value of invisible watermark. It is always active and disabled from Web UI."""
        return True

    @property
    def ram_cache_size(self) -> Union[Literal["auto"], float]:
        """Return the ram cache size using the legacy or modern setting."""
        return self.max_cache_size or self.ram

    @property
    def vram_cache_size(self) -> Union[Literal["auto"], float]:
        """Return the vram cache size using the legacy or modern setting."""
        return self.max_vram_cache_size or self.vram

    @property
    def use_cpu(self) -> bool:
        """Return true if the device is set to CPU or the always_use_cpu flag is set."""
        return self.always_use_cpu or self.device == "cpu"

    @property
    def disable_xformers(self) -> bool:
        """Return true if enable_xformers is false (reversed logic) and attention type is not set to xformers."""
        disabled_in_config = not self.xformers_enabled
        return disabled_in_config and self.attention_type != "xformers"

    @property
    def profiles_path(self) -> Path:
        """Path to the graph profiles directory."""
        return self._resolve(self.profiles_dir)

    @staticmethod
    def find_root() -> Path:
        """Choose the runtime root directory when not specified on command line or init file."""
        return _find_root()


def get_invokeai_config(**kwargs: Any) -> InvokeAIAppConfig:
    """Legacy function which returns InvokeAIAppConfig.get_config()."""
    return InvokeAIAppConfig.get_config(**kwargs)


def _find_root() -> Path:
    venv = Path(os.environ.get("VIRTUAL_ENV") or ".")
    if os.environ.get("INVOKEAI_ROOT"):
        root = Path(os.environ["INVOKEAI_ROOT"])
    elif any((venv.parent / x).exists() for x in [INIT_FILE, LEGACY_INIT_FILE]):
        root = (venv.parent).resolve()
    else:
        root = Path("~/invokeai").expanduser().resolve()
    return root
