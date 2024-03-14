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
        legacy_conf_dir: configs/stable-diffusion
        outdir: outputs
     ...

"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator
from pydantic.config import JsonDict
from pydantic_settings import SettingsConfigDict

from invokeai.backend.model_hash.model_hash import HASHING_ALGORITHMS

from .config_base import InvokeAISettings

INIT_FILE = Path("invokeai.yaml")
DB_FILE = Path("invokeai.db")
LEGACY_INIT_FILE = Path("invokeai.init")
DEFAULT_RAM_CACHE = 10.0
DEFAULT_VRAM_CACHE = 0.25
DEFAULT_CONVERT_CACHE = 20.0


class Categories(object):
    """Category headers for configuration variable groups."""

    WebServer: JsonDict = {"category": "Web Server"}
    Features: JsonDict = {"category": "Features"}
    Paths: JsonDict = {"category": "Paths"}
    Logging: JsonDict = {"category": "Logging"}
    Development: JsonDict = {"category": "Development"}
    CLIArgs: JsonDict = {"category": "CLIArgs"}
    ModelInstall: JsonDict = {"category": "Model Install"}
    ModelCache: JsonDict = {"category": "Model Cache"}
    Device: JsonDict = {"category": "Device"}
    Generation: JsonDict = {"category": "Generation"}
    Queue: JsonDict = {"category": "Queue"}
    Nodes: JsonDict = {"category": "Nodes"}
    MemoryPerformance: JsonDict = {"category": "Memory/Performance"}
    Deprecated: JsonDict = {"category": "Deprecated"}


class URLRegexToken(BaseModel):
    url_regex: str = Field(description="Regular expression to match against the URL")
    token: str = Field(description="Token to use when the URL matches the regex")

    @field_validator("url_regex")
    @classmethod
    def validate_url_regex(cls, v: str) -> str:
        """Validate that the value is a valid regex."""
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid regex: {e}")
        return v


class InvokeAIAppConfig(InvokeAISettings):
    """Invoke App Configuration

    Attributes:
        host: **Web Server**: IP address to bind to. Use `0.0.0.0` to serve to your local network.
        port: **Web Server**: Port to bind to.
        allow_origins: **Web Server**: Allowed CORS origins.
        allow_credentials: **Web Server**: Allow CORS credentials.
        allow_methods: **Web Server**: Methods allowed for CORS.
        allow_headers: **Web Server**: Headers allowed for CORS.
        ssl_certfile: **Web Server**: SSL certificate file for HTTPS.
        ssl_keyfile: **Web Server**: SSL key file for HTTPS.
        esrgan: **Features**: Enables or disables the upscaling code.
        internet_available: **Features**: If true, attempt to download models on the fly; otherwise only use local models.
        log_tokenization: **Features**: Enable logging of parsed prompt tokens.
        patchmatch: **Features**: Enable patchmatch inpaint code.
        ignore_missing_core_models: **Features**: Ignore missing core models on startup. If `True`, the app will attempt to download missing models on startup.
        root: **Paths**: The InvokeAI runtime root directory.
        autoimport_dir: **Paths**: Path to a directory of models files to be imported on startup.
        models_dir: **Paths**: Path to the models directory.
        convert_cache_dir: **Paths**: Path to the converted models cache directory. When loading a non-diffusers model, it will be converted and store on disk at this location.
        legacy_conf_dir: **Paths**: Path to directory of legacy checkpoint config files.
        db_dir: **Paths**: Path to InvokeAI databases directory.
        outdir: **Paths**: Path to directory for outputs.
        custom_nodes_dir: **Paths**: Path to directory for custom nodes.
        from_file: **Paths**: Take command input from the indicated file (command-line client only).
        log_handlers: **Logging**: Log handler. Valid options are "console", "file=<path>", "syslog=path|address:host:port", "http=<url>".
        log_format: **Logging**: Log format. Use "plain" for text-only, "color" for colorized output, "legacy" for 2.3-style logging and "syslog" for syslog-style.
        log_level: **Logging**: Emit logging messages at this level or higher.
        log_sql: **Logging**: Log SQL queries. `log_level` must be `debug` for this to do anything. Extremely verbose.
        use_memory_db: **Development**: Use in-memory database. Useful for development.
        dev_reload: **Development**: Automatically reload when Python sources are changed. Does not reload node definitions.
        profile_graphs: **Development**: Enable graph profiling using `cProfile`.
        profile_prefix: **Development**: An optional prefix for profile output files.
        profiles_dir: **Development**: Path to profiles output directory.
        version: **CLIArgs**: CLI arg - show InvokeAI version and exit.
        hashing_algorithm: **Model Install**: Model hashing algorthim for model installs. 'blake3' is best for SSDs. 'blake3_single' is best for spinning disk HDDs. 'random' disables hashing, instead assigning a UUID to models. Useful when using a memory db to reduce model installation time, or if you don't care about storing stable hashes for models. Alternatively, any other hashlib algorithm is accepted, though these are not nearly as performant as blake3.
        remote_api_tokens: **Model Install**: List of regular expression and token pairs used when downloading models from URLs. The download URL is tested against the regex, and if it matches, the token is provided in as a Bearer token.
        ram: **Model Cache**: Maximum memory amount used by memory model cache for rapid switching (GB).
        vram: **Model Cache**: Amount of VRAM reserved for model storage (GB)
        convert_cache: **Model Cache**: Maximum size of on-disk converted models cache (GB)
        lazy_offload: **Model Cache**: Keep models in VRAM until their space is needed.
        log_memory_usage: **Model Cache**: If True, a memory snapshot will be captured before and after every model cache operation, and the result will be logged (at debug level). There is a time cost to capturing the memory snapshots, so it is recommended to only enable this feature if you are actively inspecting the model cache's behaviour.
        device: **Device**: Preferred execution device. `auto` will choose the device depending on the hardware platform and the installed torch capabilities.
        precision: **Device**: Floating point precision. `float16` will consume half the memory of `float32` but produce slightly lower-quality images. The `auto` setting will guess the proper precision based on your video card and operating system.
        sequential_guidance: **Generation**: Whether to calculate guidance in serial instead of in parallel, lowering memory requirements.
        attention_type: **Generation**: Attention type.
        attention_slice_size: **Generation**: Slice size, valid when attention_type=="sliced".
        force_tiled_decode: **Generation**: Whether to enable tiled VAE decode (reduces memory consumption with some performance penalty).
        png_compress_level: **Generation**: The compress_level setting of PIL.Image.save(), used for PNG encoding. All settings are lossless. 0 = no compression, 1 = fastest with slightly larger filesize, 9 = slowest with smallest filesize. 1 is typically the best setting.
        max_queue_size: **Queue**: Maximum number of items in the session queue.
        allow_nodes: **Nodes**: List of nodes to allow. Omit to allow all.
        deny_nodes: **Nodes**: List of nodes to deny. Omit to deny none.
        node_cache_size: **Nodes**: How many cached nodes to keep in memory.
    """

    singleton_config: ClassVar[Optional[InvokeAIAppConfig]] = None
    singleton_init: ClassVar[Optional[Dict[str, Any]]] = None

    # fmt: off
    type: Literal["InvokeAI"] = "InvokeAI"

    # WEB
    host                : str = Field(default="127.0.0.1", description="IP address to bind to. Use `0.0.0.0` to serve to your local network.", json_schema_extra=Categories.WebServer)
    port                : int = Field(default=9090, description="Port to bind to.", json_schema_extra=Categories.WebServer)
    allow_origins       : List[str] = Field(default=[], description="Allowed CORS origins.", json_schema_extra=Categories.WebServer)
    allow_credentials   : bool = Field(default=True, description="Allow CORS credentials.", json_schema_extra=Categories.WebServer)
    allow_methods       : List[str] = Field(default=["*"], description="Methods allowed for CORS.", json_schema_extra=Categories.WebServer)
    allow_headers       : List[str] = Field(default=["*"], description="Headers allowed for CORS.", json_schema_extra=Categories.WebServer)
    # SSL options correspond to https://www.uvicorn.org/settings/#https
    ssl_certfile        : Optional[Path] = Field(default=None, description="SSL certificate file for HTTPS.", json_schema_extra=Categories.WebServer)
    ssl_keyfile         : Optional[Path] = Field(default=None, description="SSL key file for HTTPS.", json_schema_extra=Categories.WebServer)

    # FEATURES
    esrgan              : bool = Field(default=True, description="Enables or disables the upscaling code.", json_schema_extra=Categories.Features)
    # TODO(psyche): This is not used anywhere.
    internet_available  : bool = Field(default=True, description="If true, attempt to download models on the fly; otherwise only use local models.", json_schema_extra=Categories.Features)
    log_tokenization    : bool = Field(default=False, description="Enable logging of parsed prompt tokens.", json_schema_extra=Categories.Features)
    patchmatch          : bool = Field(default=True, description="Enable patchmatch inpaint code.", json_schema_extra=Categories.Features)
    ignore_missing_core_models : bool = Field(default=False, description='Ignore missing core models on startup. If `True`, the app will attempt to download missing models on startup.', json_schema_extra=Categories.Features)

    # PATHS
    root                : Optional[Path] = Field(default=None, description='The InvokeAI runtime root directory.', json_schema_extra=Categories.Paths)
    autoimport_dir      : Path = Field(default=Path('autoimport'), description='Path to a directory of models files to be imported on startup.', json_schema_extra=Categories.Paths)
    models_dir          : Path = Field(default=Path('models'), description='Path to the models directory.', json_schema_extra=Categories.Paths)
    convert_cache_dir   : Path = Field(default=Path('models/.cache'), description='Path to the converted models cache directory. When loading a non-diffusers model, it will be converted and store on disk at this location.', json_schema_extra=Categories.Paths)
    legacy_conf_dir     : Path = Field(default=Path('configs/stable-diffusion'), description='Path to directory of legacy checkpoint config files.', json_schema_extra=Categories.Paths)
    db_dir              : Path = Field(default=Path('databases'), description='Path to InvokeAI databases directory.', json_schema_extra=Categories.Paths)
    outdir              : Path = Field(default=Path('outputs'), description='Path to directory for outputs.', json_schema_extra=Categories.Paths)
    custom_nodes_dir    : Path = Field(default=Path('nodes'), description='Path to directory for custom nodes.', json_schema_extra=Categories.Paths)
    # TODO(psyche): This is not used anywhere.
    from_file           : Optional[Path] = Field(default=None, description='Take command input from the indicated file (command-line client only).', json_schema_extra=Categories.Paths)

    # LOGGING
    log_handlers        : List[str] = Field(default=["console"], description='Log handler. Valid options are "console", "file=<path>", "syslog=path|address:host:port", "http=<url>".', json_schema_extra=Categories.Logging)
    # note - would be better to read the log_format values from logging.py, but this creates circular dependencies issues
    log_format          : Literal['plain', 'color', 'syslog', 'legacy'] = Field(default="color", description='Log format. Use "plain" for text-only, "color" for colorized output, "legacy" for 2.3-style logging and "syslog" for syslog-style.', json_schema_extra=Categories.Logging)
    log_level           : Literal["debug", "info", "warning", "error", "critical"] = Field(default="info", description="Emit logging messages at this level or higher.", json_schema_extra=Categories.Logging)
    log_sql             : bool = Field(default=False, description="Log SQL queries. `log_level` must be `debug` for this to do anything. Extremely verbose.", json_schema_extra=Categories.Logging)

    # Development
    use_memory_db       : bool = Field(default=False, description='Use in-memory database. Useful for development.', json_schema_extra=Categories.Development)
    dev_reload          : bool = Field(default=False, description="Automatically reload when Python sources are changed. Does not reload node definitions.", json_schema_extra=Categories.Development)
    profile_graphs      : bool = Field(default=False, description="Enable graph profiling using `cProfile`.", json_schema_extra=Categories.Development)
    profile_prefix      : Optional[str] = Field(default=None, description="An optional prefix for profile output files.", json_schema_extra=Categories.Development)
    profiles_dir        : Path = Field(default=Path('profiles'), description="Path to profiles output directory.", json_schema_extra=Categories.Development)

    version             : bool = Field(default=False, description="CLI arg - show InvokeAI version and exit.", json_schema_extra=Categories.CLIArgs)

    # CACHE
    ram                 : float = Field(default=DEFAULT_RAM_CACHE, gt=0, description="Maximum memory amount used by memory model cache for rapid switching (GB).", json_schema_extra=Categories.ModelCache, )
    vram                : float = Field(default=DEFAULT_VRAM_CACHE, ge=0, description="Amount of VRAM reserved for model storage (GB)", json_schema_extra=Categories.ModelCache, )
    convert_cache       : float = Field(default=DEFAULT_CONVERT_CACHE, ge=0, description="Maximum size of on-disk converted models cache (GB)", json_schema_extra=Categories.ModelCache)

    lazy_offload        : bool = Field(default=True, description="Keep models in VRAM until their space is needed.", json_schema_extra=Categories.ModelCache, )
    log_memory_usage    : bool = Field(default=False, description="If True, a memory snapshot will be captured before and after every model cache operation, and the result will be logged (at debug level). There is a time cost to capturing the memory snapshots, so it is recommended to only enable this feature if you are actively inspecting the model cache's behaviour.", json_schema_extra=Categories.ModelCache)

    # DEVICE
    device              : Literal["auto", "cpu", "cuda", "cuda:1", "mps"] = Field(default="auto", description="Preferred execution device. `auto` will choose the device depending on the hardware platform and the installed torch capabilities.", json_schema_extra=Categories.Device)
    precision           : Literal["auto", "float16", "bfloat16", "float32", "autocast"] = Field(default="auto", description="Floating point precision. `float16` will consume half the memory of `float32` but produce slightly lower-quality images. The `auto` setting will guess the proper precision based on your video card and operating system.", json_schema_extra=Categories.Device)

    # GENERATION
    sequential_guidance : bool = Field(default=False, description="Whether to calculate guidance in serial instead of in parallel, lowering memory requirements.", json_schema_extra=Categories.Generation)
    attention_type      : Literal["auto", "normal", "xformers", "sliced", "torch-sdp"] = Field(default="auto", description="Attention type.", json_schema_extra=Categories.Generation)
    attention_slice_size: Literal["auto", "balanced", "max", 1, 2, 3, 4, 5, 6, 7, 8] = Field(default="auto", description='Slice size, valid when attention_type=="sliced".', json_schema_extra=Categories.Generation)
    force_tiled_decode  : bool = Field(default=False, description="Whether to enable tiled VAE decode (reduces memory consumption with some performance penalty).", json_schema_extra=Categories.Generation)
    png_compress_level  : int = Field(default=1, description="The compress_level setting of PIL.Image.save(), used for PNG encoding. All settings are lossless. 0 = no compression, 1 = fastest with slightly larger filesize, 9 = slowest with smallest filesize. 1 is typically the best setting.", json_schema_extra=Categories.Generation)

    # QUEUE
    max_queue_size      : int = Field(default=10000, gt=0, description="Maximum number of items in the session queue.", json_schema_extra=Categories.Queue)

    # NODES
    allow_nodes         : Optional[List[str]] = Field(default=None, description="List of nodes to allow. Omit to allow all.", json_schema_extra=Categories.Nodes)
    deny_nodes          : Optional[List[str]] = Field(default=None, description="List of nodes to deny. Omit to deny none.", json_schema_extra=Categories.Nodes)
    node_cache_size     : int = Field(default=512, description="How many cached nodes to keep in memory.", json_schema_extra=Categories.Nodes)

    # MODEL INSTALL
    hashing_algorithm   : HASHING_ALGORITHMS = Field(default="blake3", description="Model hashing algorthim for model installs. 'blake3' is best for SSDs. 'blake3_single' is best for spinning disk HDDs. 'random' disables hashing, instead assigning a UUID to models. Useful when using a memory db to reduce model installation time, or if you don't care about storing stable hashes for models. Alternatively, any other hashlib algorithm is accepted, though these are not nearly as performant as blake3.", json_schema_extra=Categories.ModelInstall)
    remote_api_tokens   : Optional[list[URLRegexToken]] = Field(
        default=None,
        description="List of regular expression and token pairs used when downloading models from URLs. The download URL is tested against the regex, and if it matches, the token is provided in as a Bearer token.",
        json_schema_extra=Categories.ModelInstall
    )

    # TODO(psyche): Can we just remove these then?
    # DEPRECATED FIELDS - STILL HERE IN ORDER TO OBTAN VALUES FROM PRE-3.1 CONFIG FILES
    always_use_cpu      : bool = Field(default=False, description="If true, use the CPU for rendering even if a GPU is available.", json_schema_extra=Categories.Deprecated)
    max_cache_size      : Optional[float] = Field(default=None, gt=0, description="Maximum memory amount used by model cache for rapid switching", json_schema_extra=Categories.Deprecated)
    max_vram_cache_size : Optional[float] = Field(default=None, ge=0, description="Amount of VRAM reserved for model storage", json_schema_extra=Categories.Deprecated)
    xformers_enabled    : bool = Field(default=True, description="Enable/disable memory-efficient attention", json_schema_extra=Categories.Deprecated)
    tiled_decode        : bool = Field(default=False, description="Whether to enable tiled VAE decode (reduces memory consumption with some performance penalty)", json_schema_extra=Categories.Deprecated)
    lora_dir            : Optional[Path] = Field(default=None, description='Path to a directory of LoRA/LyCORIS models to be imported on startup.', json_schema_extra=Categories.Deprecated)
    embedding_dir       : Optional[Path] = Field(default=None, description='Path to a directory of Textual Inversion embeddings to be imported on startup.', json_schema_extra=Categories.Deprecated)
    controlnet_dir      : Optional[Path] = Field(default=None, description='Path to a directory of ControlNet embeddings to be imported on startup.', json_schema_extra=Categories.Deprecated)
    conf_path           : Path = Field(default=Path('configs/models.yaml'), description='Path to models definition file', json_schema_extra=Categories.Deprecated)

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
    def models_convert_cache_path(self) -> Path:
        """Path to the converted cache models directory."""
        return self._resolve(self.convert_cache_dir)

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
    def ram_cache_size(self) -> float:
        """Return the ram cache size using the legacy or modern setting (GB)."""
        return self.max_cache_size or self.ram

    @property
    def vram_cache_size(self) -> float:
        """Return the vram cache size using the legacy or modern setting (GB)."""
        return self.max_vram_cache_size or self.vram

    @property
    def convert_cache_size(self) -> float:
        """Return the convert cache size on disk (GB)."""
        return self.convert_cache

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

    @staticmethod
    def generate_docstrings() -> str:
        """Helper function for mkdocs. Generates a docstring for the InvokeAIAppConfig class.

        You shouldn't run this manually. Instead, run `scripts/update-config-docstring.py` to update the docstring.
        A makefile target is also available: `make update-config-docstring`.

        See that script for more information about why this is necessary.
        """
        docstring = '    """Invoke App Configuration\n\n'
        docstring += "    Attributes:"

        field_descriptions: dict[str, list[str]] = {}

        for k, v in InvokeAIAppConfig.model_fields.items():
            if not isinstance(v.json_schema_extra, dict):
                # Should never happen
                continue

            category = v.json_schema_extra.get("category", None)
            if not isinstance(category, str) or category == "Deprecated":
                continue
            if not field_descriptions.get(category):
                field_descriptions[category] = []
            field_descriptions[category].append(f"        {k}: **{category}**: {v.description}")

        for c in [
            "Web Server",
            "Features",
            "Paths",
            "Logging",
            "Development",
            "CLIArgs",
            "Model Install",
            "Model Cache",
            "Device",
            "Generation",
            "Queue",
            "Nodes",
        ]:
            docstring += "\n"
            docstring += "\n".join(field_descriptions[c])

        docstring += '\n    """'

        return docstring


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
