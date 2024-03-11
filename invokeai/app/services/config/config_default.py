from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from invokeai.backend.model_hash.model_hash import HASHING_ALGORITHMS
from invokeai.frontend.cli.app_arg_parser import app_arg_parser

INIT_FILE = Path("invokeai.yaml")
DB_FILE = Path("invokeai.db")
LEGACY_INIT_FILE = Path("invokeai.init")
DEFAULT_RAM_CACHE = 10.0
DEFAULT_VRAM_CACHE = 0.25
DEFAULT_CONVERT_CACHE = 20.0
DEVICE = Literal["auto", "cpu", "cuda", "cuda:1", "mps"]
PRECISION = Literal["auto", "float16", "bfloat16", "float32", "autocast"]
ATTENTION_TYPE = Literal["auto", "normal", "xformers", "sliced", "torch-sdp"]
ATTENTION_SLICE_SIZE = Literal["auto", "balanced", "max", 1, 2, 3, 4, 5, 6, 7, 8]
LOG_FORMAT = Literal["plain", "color", "syslog", "legacy"]
LOG_LEVEL = Literal["debug", "info", "warning", "error", "critical"]
CONFIG_SCHEMA_VERSION = 4


class URLRegexTokenPair(BaseModel):
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


class ConfigMeta(BaseModel):
    """Metadata for the config file. This is not stored in the config object."""

    schema_version: int = CONFIG_SCHEMA_VERSION


class InvokeAIAppConfig(BaseSettings):
    """Invoke's global app configuration.

    Typically, you won't need to interact with this class directly. Instead, use the `get_config` function from `invokeai.app.services.config` to get a singleton config object.

    Attributes:
        host: IP address to bind to. Use `0.0.0.0` to serve to your local network.
        port: Port to bind to.
        allow_origins: Allowed CORS origins.
        allow_credentials: Allow CORS credentials.
        allow_methods: Methods allowed for CORS.
        allow_headers: Headers allowed for CORS.
        ssl_certfile: SSL certificate file for HTTPS. See https://www.uvicorn.org/settings/#https.
        ssl_keyfile: SSL key file for HTTPS. See https://www.uvicorn.org/settings/#https.
        log_tokenization: Enable logging of parsed prompt tokens.
        patchmatch: Enable patchmatch inpaint code.
        ignore_missing_core_models: Ignore missing core models on startup. If `True`, the app will attempt to download missing models on startup.
        autoimport_dir: Path to a directory of models files to be imported on startup. WARNING: This may be a relative path. Use `autoimport_path` to get the resolved absolute path.
        models_dir: Path to the models directory. WARNING: This may be a relative path. Use `models_path` to get the resolved absolute path.
        convert_cache_dir: Path to the converted models cache directory. When loading a non-diffusers model, it will be converted and store on disk at this location. WARNING: This may be a relative path. Use `convert_cache_path` to get the resolved absolute path.
        legacy_conf_dir: Path to directory of legacy checkpoint config files. WARNING: This may be a relative path. Use `legacy_conf_path` to get the resolved absolute path.
        db_dir: Path to InvokeAI databases directory. WARNING: This may be a relative path. Use `db_path` to get the resolved absolute path.
        outputs_dir: Path to directory for outputs. WARNING: This may be a relative path. Use `outputs_path` to get the resolved absolute path.
        custom_nodes_dir: Path to directory for custom nodes. WARNING: This may be a relative path. Use `custom_nodes_path` to get the resolved absolute path.
        log_handlers: Log handler. Valid options are "console", "file=<path>", "syslog=path|address:host:port", "http=<url>".
        log_format: Log format. Use "plain" for text-only, "color" for colorized output, "legacy" for 2.3-style logging and "syslog" for syslog-style.
        log_level: Emit logging messages at this level or higher.
        log_sql: Log SQL queries. `log_level` must be `debug` for this to do anything. Extremely verbose.
        use_memory_db: Use in-memory database. Useful for development.
        dev_reload: Automatically reload when Python sources are changed. Does not reload node definitions.
        profile_graphs: Enable graph profiling using `cProfile`.
        profile_prefix: An optional prefix for profile output files.
        profiles_dir: Path to profiles output directory. WARNING: This may be a relative path. Use `profiles_path` to get the resolved absolute path.
        ram: Maximum memory amount used by memory model cache for rapid switching (GB).
        vram: Amount of VRAM reserved for model storage (GB)
        convert_cache: Maximum size of on-disk converted models cache (GB)
        lazy_offload: Keep models in VRAM until their space is needed.
        log_memory_usage: If True, a memory snapshot will be captured before and after every model cache operation, and the result will be logged (at debug level). There is a time cost to capturing the memory snapshots, so it is recommended to only enable this feature if you are actively inspecting the model cache's behaviour.
        device: Preferred execution device. `auto` will choose the device depending on the hardware platform and the installed torch capabilities.
        precision: Floating point precision. `float16` will consume half the memory of `float32` but produce slightly lower-quality images. The `auto` setting will guess the proper precision based on your video card and operating system.
        sequential_guidance: Whether to calculate guidance in serial instead of in parallel, lowering memory requirements.
        attention_type: Attention type.
        attention_slice_size: Slice size, valid when attention_type=="sliced".
        force_tiled_decode: Whether to enable tiled VAE decode (reduces memory consumption with some performance penalty).
        pil_compress_level: The compress_level setting of PIL.Image.save(), used for PNG encoding. All settings are lossless. 0 = no compression, 1 = fastest with slightly larger filesize, 9 = slowest with smallest filesize. 1 is typically the best setting.
        max_queue_size: Maximum number of items in the session queue.
        allow_nodes: List of nodes to allow. Omit to allow all.
        deny_nodes: List of nodes to deny. Omit to deny none.
        node_cache_size: How many cached nodes to keep in memory.
        hashing_algorithm: Model hashing algorthim for model installs. 'blake3' is best for SSDs. 'blake3_single' is best for spinning disk HDDs. 'random' disables hashing, instead assigning a UUID to models. Useful when using a memory db to reduce model installation time, or if you don't care about storing stable hashes for models. Alternatively, any other hashlib algorithm is accepted, though these are not nearly as performant as blake3.
        remote_api_tokens: List of regular expression and token pairs used when downloading models from URLs. The download URL is tested against the regex, and if it matches, the token is provided in as a Bearer token.
    """

    _root: Optional[Path] = PrivateAttr(default=None)

    # fmt: off

    # WEB
    host:                           str = Field(default="127.0.0.1",        description="IP address to bind to. Use `0.0.0.0` to serve to your local network.")
    port:                           int = Field(default=9090,               description="Port to bind to.")
    allow_origins:            list[str] = Field(default=[],                 description="Allowed CORS origins.")
    allow_credentials:             bool = Field(default=True,               description="Allow CORS credentials.")
    allow_methods:            list[str] = Field(default=["*"],              description="Methods allowed for CORS.")
    allow_headers:            list[str] = Field(default=["*"],              description="Headers allowed for CORS.")
    ssl_certfile:        Optional[Path] = Field(default=None,               description="SSL certificate file for HTTPS. See https://www.uvicorn.org/settings/#https.")
    ssl_keyfile:         Optional[Path] = Field(default=None,               description="SSL key file for HTTPS. See https://www.uvicorn.org/settings/#https.")

    # MISC FEATURES
    log_tokenization:              bool = Field(default=False,              description="Enable logging of parsed prompt tokens.")
    patchmatch:                    bool = Field(default=True,               description="Enable patchmatch inpaint code.")
    ignore_missing_core_models:    bool = Field(default=False,              description="Ignore missing core models on startup. If `True`, the app will attempt to download missing models on startup.")

    # PATHS
    autoimport_dir:                Path = Field(default=Path("autoimport"), description="Path to a directory of models files to be imported on startup. WARNING: This may be a relative path. Use `autoimport_path` to get the resolved absolute path.")
    models_dir:                    Path = Field(default=Path("models"),     description="Path to the models directory. WARNING: This may be a relative path. Use `models_path` to get the resolved absolute path.")
    convert_cache_dir:             Path = Field(default=Path("models/.cache"), description="Path to the converted models cache directory. When loading a non-diffusers model, it will be converted and store on disk at this location. WARNING: This may be a relative path. Use `convert_cache_path` to get the resolved absolute path.")
    legacy_conf_dir:               Path = Field(default=Path("configs/stable-diffusion"), description="Path to directory of legacy checkpoint config files. WARNING: This may be a relative path. Use `legacy_conf_path` to get the resolved absolute path.")
    db_dir:                        Path = Field(default=Path("databases"),  description="Path to InvokeAI databases directory. WARNING: This may be a relative path. Use `db_path` to get the resolved absolute path.")
    outputs_dir:                   Path = Field(default=Path("outputs"),    description="Path to directory for outputs. WARNING: This may be a relative path. Use `outputs_path` to get the resolved absolute path.")
    custom_nodes_dir:              Path = Field(default=Path("nodes"),      description="Path to directory for custom nodes. WARNING: This may be a relative path. Use `custom_nodes_path` to get the resolved absolute path.")

    # LOGGING
    log_handlers:             list[str] = Field(default=["console"],        description='Log handler. Valid options are "console", "file=<path>", "syslog=path|address:host:port", "http=<url>".')
    # note - would be better to read the log_format values from logging.py, but this creates circular dependencies issues
    log_format:              LOG_FORMAT = Field(default="color",            description='Log format. Use "plain" for text-only, "color" for colorized output, "legacy" for 2.3-style logging and "syslog" for syslog-style.')
    log_level:                LOG_LEVEL = Field(default="info",             description="Emit logging messages at this level or higher.")
    log_sql:                       bool = Field(default=False,              description="Log SQL queries. `log_level` must be `debug` for this to do anything. Extremely verbose.")

    # Development
    use_memory_db:                 bool = Field(default=False,              description="Use in-memory database. Useful for development.")
    dev_reload:                    bool = Field(default=False,              description="Automatically reload when Python sources are changed. Does not reload node definitions.")
    profile_graphs:                bool = Field(default=False,              description="Enable graph profiling using `cProfile`.")
    profile_prefix:       Optional[str] = Field(default=None,               description="An optional prefix for profile output files.")
    profiles_dir:                  Path = Field(default=Path("profiles"),   description="Path to profiles output directory. WARNING: This may be a relative path. Use `profiles_path` to get the resolved absolute path.")

    # CACHE
    ram:                          float = Field(default=DEFAULT_RAM_CACHE, gt=0, description="Maximum memory amount used by memory model cache for rapid switching (GB).")
    vram:                         float = Field(default=DEFAULT_VRAM_CACHE, ge=0, description="Amount of VRAM reserved for model storage (GB)")
    convert_cache:                float = Field(default=DEFAULT_CONVERT_CACHE, ge=0, description="Maximum size of on-disk converted models cache (GB)")
    lazy_offload:                  bool = Field(default=True,               description="Keep models in VRAM until their space is needed.")
    log_memory_usage:              bool = Field(default=False,              description="If True, a memory snapshot will be captured before and after every model cache operation, and the result will be logged (at debug level). There is a time cost to capturing the memory snapshots, so it is recommended to only enable this feature if you are actively inspecting the model cache's behaviour.")

    # DEVICE
    device:                      DEVICE = Field(default="auto",             description="Preferred execution device. `auto` will choose the device depending on the hardware platform and the installed torch capabilities.")
    precision:                PRECISION = Field(default="auto",             description="Floating point precision. `float16` will consume half the memory of `float32` but produce slightly lower-quality images. The `auto` setting will guess the proper precision based on your video card and operating system.")

    # GENERATION
    sequential_guidance:           bool = Field(default=False,              description="Whether to calculate guidance in serial instead of in parallel, lowering memory requirements.")
    attention_type:      ATTENTION_TYPE = Field(default="auto",             description="Attention type.")
    attention_slice_size: ATTENTION_SLICE_SIZE = Field(default="auto",      description='Slice size, valid when attention_type=="sliced".')
    force_tiled_decode:            bool = Field(default=False,              description="Whether to enable tiled VAE decode (reduces memory consumption with some performance penalty).")
    pil_compress_level:             int = Field(default=1,                  description="The compress_level setting of PIL.Image.save(), used for PNG encoding. All settings are lossless. 0 = no compression, 1 = fastest with slightly larger filesize, 9 = slowest with smallest filesize. 1 is typically the best setting.")
    max_queue_size:                 int = Field(default=10000, gt=0,        description="Maximum number of items in the session queue.")

    # NODES
    allow_nodes:    Optional[list[str]] = Field(default=None,               description="List of nodes to allow. Omit to allow all.")
    deny_nodes:     Optional[list[str]] = Field(default=None,               description="List of nodes to deny. Omit to deny none.")
    node_cache_size:                int = Field(default=512,                description="How many cached nodes to keep in memory.")

    # MODEL INSTALL
    hashing_algorithm: HASHING_ALGORITHMS = Field(default="blake3",         description="Model hashing algorthim for model installs. 'blake3' is best for SSDs. 'blake3_single' is best for spinning disk HDDs. 'random' disables hashing, instead assigning a UUID to models. Useful when using a memory db to reduce model installation time, or if you don't care about storing stable hashes for models. Alternatively, any other hashlib algorithm is accepted, though these are not nearly as performant as blake3.")
    remote_api_tokens: Optional[list[URLRegexTokenPair]] = Field(default=None, description="List of regular expression and token pairs used when downloading models from URLs. The download URL is tested against the regex, and if it matches, the token is provided in as a Bearer token.")

    # fmt: on

    model_config = SettingsConfigDict(env_file_encoding="utf-8", case_sensitive=True, env_prefix="INVOKEAI")

    def update_config(self, config: dict[str, Any] | InvokeAIAppConfig) -> None:
        """Updates the config, overwriting existing values.

        Args:
            config: A dictionary of config settings, or instance of `InvokeAIAppConfig`. If an instance of \
                `InvokeAIAppConfig`, only the explicitly set fields will be merged into the singleton config.
        """

        if isinstance(config, dict):
            new_config = self.model_validate(config)
        else:
            new_config = config

        for field_name in new_config.model_fields_set:
            new_value = getattr(new_config, field_name)
            current_value = getattr(self, field_name)
            if new_value != current_value:
                setattr(self, field_name, new_value)

    def write_file(self, dest_path: Path) -> None:
        """Write the current configuration to file. This will overwrite the existing file.

        A `meta` stanza is added to the top of the file, containing metadata about the config file. This is not stored in the config object.

        Args:
            dest_path: Path to write the config to.
        """
        with open(dest_path, "w") as file:
            meta_dict = {"meta": ConfigMeta().model_dump()}
            config_dict = self.model_dump(mode="json", exclude_unset=True, exclude_defaults=True)
            file.write("# Internal metadata\n")
            file.write(yaml.dump(meta_dict, sort_keys=False))
            file.write("\n")
            file.write("# User settings\n")
            file.write(yaml.dump(config_dict, sort_keys=False))

    def merge_from_file(self, source_path: Optional[Path] = None) -> None:
        """Read the config from the `invokeai.yaml` file, migrating it if necessary and merging it into the singleton config.

        This function will write to the `invokeai.yaml` file if the config is migrated.

        Args:
            source_path: Path to the config file. If not provided, the default path is used.
        """
        config_from_file = load_and_migrate_config(source_path or self.init_file_path)
        self.update_config(config_from_file)

    def parse_args(self) -> None:
        """Parse the CLI args and set the runtime root directory."""
        opt = app_arg_parser.parse_args()
        if root := getattr(opt, "root", None):
            self.set_root(Path(root))

    def set_root(self, root: Path) -> None:
        """Set the runtime root directory. This is typically set using a CLI arg."""
        assert isinstance(root, Path)
        self._root = root

    def _resolve(self, partial_path: Path) -> Path:
        return (self.root_path / partial_path).resolve()

    @property
    def root_path(self) -> Path:
        """Path to the runtime root directory, resolved to an absolute path."""
        if self._root:
            root = Path(self._root).expanduser().absolute()
        else:
            root = self.find_root().expanduser().absolute()
        self._root = root  # insulate ourselves from relative paths that may change
        return root.resolve()

    @property
    def init_file_path(self) -> Path:
        """Path to invokeai.yaml, resolved to an absolute path.."""
        resolved_path = self._resolve(INIT_FILE)
        assert resolved_path is not None
        return resolved_path

    @property
    def autoimport_path(self) -> Path:
        """Path to the autoimports directory, resolved to an absolute path.."""
        return self._resolve(self.autoimport_dir)

    @property
    def outputs_path(self) -> Optional[Path]:
        """Path to the outputs directory, resolved to an absolute path.."""
        return self._resolve(self.outputs_dir)

    @property
    def db_path(self) -> Path:
        """Path to the invokeai.db file, resolved to an absolute path.."""
        db_dir = self._resolve(self.db_dir)
        assert db_dir is not None
        return db_dir / DB_FILE

    @property
    def legacy_conf_path(self) -> Path:
        """Path to directory of legacy configuration files (e.g. v1-inference.yaml), resolved to an absolute path.."""
        return self._resolve(self.legacy_conf_dir)

    @property
    def models_path(self) -> Path:
        """Path to the models directory, resolved to an absolute path.."""
        return self._resolve(self.models_dir)

    @property
    def convert_cache_path(self) -> Path:
        """Path to the converted cache models directory, resolved to an absolute path.."""
        return self._resolve(self.convert_cache_dir)

    @property
    def custom_nodes_path(self) -> Path:
        """Path to the custom nodes directory, resolved to an absolute path.."""
        custom_nodes_path = self._resolve(self.custom_nodes_dir)
        assert custom_nodes_path is not None
        return custom_nodes_path

    @property
    def profiles_path(self) -> Path:
        """Path to the graph profiles directory, resolved to an absolute path.."""
        return self._resolve(self.profiles_dir)

    @staticmethod
    def find_root() -> Path:
        """Choose the runtime root directory when not specified on command line or init file."""
        venv = Path(os.environ.get("VIRTUAL_ENV") or ".")
        if os.environ.get("INVOKEAI_ROOT"):
            root = Path(os.environ["INVOKEAI_ROOT"])
        elif any((venv.parent / x).exists() for x in [INIT_FILE, LEGACY_INIT_FILE]):
            root = (venv.parent).resolve()
        else:
            root = Path("~/invokeai").expanduser().resolve()
        return root


def generate_config_docstrings() -> str:
    """Helper function for mkdocs. Generates a docstring for the InvokeAIAppConfig class.

    You shouldn't run this manually. Instead, run `scripts/update-config-docstring.py` to update the docstring.
    A makefile target is also available: `make update-config-docstring`.

    See that script for more information about why this is necessary.
    """
    docstring = '    """Invoke\'s global app configuration.\n\n'
    docstring += "    Typically, you won't need to interact with this class directly. Instead, use the `get_config` function from `invokeai.app.services.config` to get a singleton config object.\n\n"
    docstring += "    Attributes:\n"

    field_descriptions: list[str] = []

    for k, v in InvokeAIAppConfig.model_fields.items():
        if v.exclude:
            continue
        field_descriptions.append(f"        {k}: {v.description}")

    docstring += "\n".join(field_descriptions)
    docstring += '\n    """'

    return docstring


def migrate_v3_config_dict(config_dict: dict[str, Any]) -> InvokeAIAppConfig:
    """Migrate a v3 config dictionary to the latest version.

    Args:
        config_dict: A dictionary of settings from a v3 config file.

    Returns:
        An instance of `InvokeAIAppConfig` with the migrated settings.

    """
    parsed_config_dict: dict[str, Any] = {}
    for _category_name, category_dict in config_dict["InvokeAI"].items():
        for k, v in category_dict.items():
            # `outdir` was renamed to `outputs_dir` in v4
            if k == "outdir":
                parsed_config_dict["outputs_dir"] = v
            # `max_cache_size` was renamed to `ram` some time in v3, but both names were used
            if k == "max_cache_size" and "ram" not in category_dict:
                parsed_config_dict["ram"] = v
            # `max_vram_cache_size` was renamed to `vram` some time in v3, but both names were used
            if k == "max_vram_cache_size" and "vram" not in category_dict:
                parsed_config_dict["vram"] = v
            elif k in InvokeAIAppConfig.model_fields:
                # skip unknown fields
                parsed_config_dict[k] = v
    return InvokeAIAppConfig.model_validate(parsed_config_dict)


def load_and_migrate_config(config_path: Path) -> InvokeAIAppConfig:
    """Load and migrate a config file to the latest version.

    Args:
        config_path: Path to the config file.

    Returns:
        An instance of `InvokeAIAppConfig` with the loaded and migrated settings.
    """
    assert config_path.suffix == ".yaml"
    with open(config_path) as file:
        loaded_config_dict = yaml.safe_load(file)

    assert isinstance(loaded_config_dict, dict)

    if "InvokeAI" in loaded_config_dict:
        # This is a v3 config file, attempt to migrate it
        try:
            config = migrate_v3_config_dict(loaded_config_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load and migrate v3 config file {config_path}: {e}") from e
        config_path.rename(config_path.with_suffix(".yaml.bak"))
        # By excluding defaults, we ensure that the new config file only contains the settings that were explicitly set
        config.write_file(config_path)
        return config
    else:
        # Attempt to load as a v4 config file
        try:
            # Meta is not included in the model fields, so we need to validate it separately
            config_meta = ConfigMeta.model_validate(loaded_config_dict.pop("meta"))
            assert (
                config_meta.schema_version == CONFIG_SCHEMA_VERSION
            ), f"Invalid schema version, expected {CONFIG_SCHEMA_VERSION}: {config_meta.schema_version}"
            return InvokeAIAppConfig.model_validate(loaded_config_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load config file {config_path}: {e}") from e


@lru_cache(maxsize=1)
def get_config() -> InvokeAIAppConfig:
    """Return the global singleton app config"""
    return InvokeAIAppConfig()
