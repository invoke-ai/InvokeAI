# TODO(psyche): pydantic-settings supports YAML settings sources. If we can figure out a way to integrate the YAML
# migration logic, we could use that for simpler config loading.

from __future__ import annotations

import copy
import filecmp
import locale
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

import invokeai.configs as model_configs
from invokeai.backend.model_hash.model_hash import HASHING_ALGORITHMS
from invokeai.frontend.cli.arg_parser import InvokeAIArgs

INIT_FILE = Path("invokeai.yaml")
DB_FILE = Path("invokeai.db")
LEGACY_INIT_FILE = Path("invokeai.init")
PRECISION = Literal["auto", "float16", "bfloat16", "float32"]
ATTENTION_TYPE = Literal["auto", "normal", "xformers", "sliced", "torch-sdp"]
ATTENTION_SLICE_SIZE = Literal["auto", "balanced", "max", 1, 2, 3, 4, 5, 6, 7, 8]
LOG_FORMAT = Literal["plain", "color", "syslog", "legacy"]
LOG_LEVEL = Literal["debug", "info", "warning", "error", "critical"]
CONFIG_SCHEMA_VERSION = "4.0.2"


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
        models_dir: Path to the models directory.
        convert_cache_dir: Path to the converted models cache directory (DEPRECATED, but do not delete because it is needed for migration from previous versions).
        download_cache_dir: Path to the directory that contains dynamically downloaded models.
        legacy_conf_dir: Path to directory of legacy checkpoint config files.
        db_dir: Path to InvokeAI databases directory.
        outputs_dir: Path to directory for outputs.
        custom_nodes_dir: Path to directory for custom nodes.
        style_presets_dir: Path to directory for style presets.
        workflow_thumbnails_dir: Path to directory for workflow thumbnails.
        log_handlers: Log handler. Valid options are "console", "file=<path>", "syslog=path|address:host:port", "http=<url>".
        log_format: Log format. Use "plain" for text-only, "color" for colorized output, "legacy" for 2.3-style logging and "syslog" for syslog-style.<br>Valid values: `plain`, `color`, `syslog`, `legacy`
        log_level: Emit logging messages at this level or higher.<br>Valid values: `debug`, `info`, `warning`, `error`, `critical`
        log_sql: Log SQL queries. `log_level` must be `debug` for this to do anything. Extremely verbose.
        log_level_network: Log level for network-related messages. 'info' and 'debug' are very verbose.<br>Valid values: `debug`, `info`, `warning`, `error`, `critical`
        use_memory_db: Use in-memory database. Useful for development.
        dev_reload: Automatically reload when Python sources are changed. Does not reload node definitions.
        profile_graphs: Enable graph profiling using `cProfile`.
        profile_prefix: An optional prefix for profile output files.
        profiles_dir: Path to profiles output directory.
        max_cache_ram_gb: The maximum amount of CPU RAM to use for model caching in GB. If unset, the limit will be configured based on the available RAM. In most cases, it is recommended to leave this unset.
        max_cache_vram_gb: The amount of VRAM to use for model caching in GB. If unset, the limit will be configured based on the available VRAM and the device_working_mem_gb. In most cases, it is recommended to leave this unset.
        log_memory_usage: If True, a memory snapshot will be captured before and after every model cache operation, and the result will be logged (at debug level). There is a time cost to capturing the memory snapshots, so it is recommended to only enable this feature if you are actively inspecting the model cache's behaviour.
        device_working_mem_gb: The amount of working memory to keep available on the compute device (in GB). Has no effect if running on CPU. If you are experiencing OOM errors, try increasing this value.
        enable_partial_loading: Enable partial loading of models. This enables models to run with reduced VRAM requirements (at the cost of slower speed) by streaming the model from RAM to VRAM as its used. In some edge cases, partial loading can cause models to run more slowly if they were previously being fully loaded into VRAM.
        keep_ram_copy_of_weights: Whether to keep a full RAM copy of a model's weights when the model is loaded in VRAM. Keeping a RAM copy increases average RAM usage, but speeds up model switching and LoRA patching (assuming there is sufficient RAM). Set this to False if RAM pressure is consistently high.
        ram: DEPRECATED: This setting is no longer used. It has been replaced by `max_cache_ram_gb`, but most users will not need to use this config since automatic cache size limits should work well in most cases. This config setting will be removed once the new model cache behavior is stable.
        vram: DEPRECATED: This setting is no longer used. It has been replaced by `max_cache_vram_gb`, but most users will not need to use this config since automatic cache size limits should work well in most cases. This config setting will be removed once the new model cache behavior is stable.
        lazy_offload: DEPRECATED: This setting is no longer used. Lazy-offloading is enabled by default. This config setting will be removed once the new model cache behavior is stable.
        pytorch_cuda_alloc_conf: Configure the Torch CUDA memory allocator. This will impact peak reserved VRAM usage and performance. Setting to "backend:cudaMallocAsync" works well on many systems. The optimal configuration is highly dependent on the system configuration (device type, VRAM, CUDA driver version, etc.), so must be tuned experimentally.
        device: Preferred execution device. `auto` will choose the device depending on the hardware platform and the installed torch capabilities.<br>Valid values: `auto`, `cpu`, `cuda`, `mps`, `cuda:N` (where N is a device number)
        precision: Floating point precision. `float16` will consume half the memory of `float32` but produce slightly lower-quality images. The `auto` setting will guess the proper precision based on your video card and operating system.<br>Valid values: `auto`, `float16`, `bfloat16`, `float32`
        sequential_guidance: Whether to calculate guidance in serial instead of in parallel, lowering memory requirements.
        attention_type: Attention type.<br>Valid values: `auto`, `normal`, `xformers`, `sliced`, `torch-sdp`
        attention_slice_size: Slice size, valid when attention_type=="sliced".<br>Valid values: `auto`, `balanced`, `max`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`
        force_tiled_decode: Whether to enable tiled VAE decode (reduces memory consumption with some performance penalty).
        pil_compress_level: The compress_level setting of PIL.Image.save(), used for PNG encoding. All settings are lossless. 0 = no compression, 1 = fastest with slightly larger filesize, 9 = slowest with smallest filesize. 1 is typically the best setting.
        max_queue_size: Maximum number of items in the session queue.
        clear_queue_on_startup: Empties session queue on startup.
        allow_nodes: List of nodes to allow. Omit to allow all.
        deny_nodes: List of nodes to deny. Omit to deny none.
        node_cache_size: How many cached nodes to keep in memory.
        hashing_algorithm: Model hashing algorthim for model installs. 'blake3_multi' is best for SSDs. 'blake3_single' is best for spinning disk HDDs. 'random' disables hashing, instead assigning a UUID to models. Useful when using a memory db to reduce model installation time, or if you don't care about storing stable hashes for models. Alternatively, any other hashlib algorithm is accepted, though these are not nearly as performant as blake3.<br>Valid values: `blake3_multi`, `blake3_single`, `random`, `md5`, `sha1`, `sha224`, `sha256`, `sha384`, `sha512`, `blake2b`, `blake2s`, `sha3_224`, `sha3_256`, `sha3_384`, `sha3_512`, `shake_128`, `shake_256`
        remote_api_tokens: List of regular expression and token pairs used when downloading models from URLs. The download URL is tested against the regex, and if it matches, the token is provided in as a Bearer token.
        scan_models_on_startup: Scan the models directory on startup, registering orphaned models. This is typically only used in conjunction with `use_memory_db` for testing purposes.
        unsafe_disable_picklescan: UNSAFE. Disable the picklescan security check during model installation. Recommended only for development and testing purposes. This will allow arbitrary code execution during model installation, so should never be used in production.
    """

    _root: Optional[Path] = PrivateAttr(default=None)
    _config_file: Optional[Path] = PrivateAttr(default=None)

    # fmt: off

    # INTERNAL
    schema_version:                 str = Field(default=CONFIG_SCHEMA_VERSION, description="Schema version of the config file. This is not a user-configurable setting.")
    # This is only used during v3 models.yaml migration
    legacy_models_yaml_path: Optional[Path] = Field(default=None,           description="Path to the legacy models.yaml file. This is not a user-configurable setting.")

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

    # PATHS
    models_dir:                    Path = Field(default=Path("models"),     description="Path to the models directory.")
    convert_cache_dir:             Path = Field(default=Path("models/.convert_cache"), description="Path to the converted models cache directory (DEPRECATED, but do not delete because it is needed for migration from previous versions).")
    download_cache_dir:            Path = Field(default=Path("models/.download_cache"), description="Path to the directory that contains dynamically downloaded models.")
    legacy_conf_dir:               Path = Field(default=Path("configs"), description="Path to directory of legacy checkpoint config files.")
    db_dir:                        Path = Field(default=Path("databases"),  description="Path to InvokeAI databases directory.")
    outputs_dir:                   Path = Field(default=Path("outputs"),    description="Path to directory for outputs.")
    custom_nodes_dir:              Path = Field(default=Path("nodes"),      description="Path to directory for custom nodes.")
    style_presets_dir:      Path = Field(default=Path("style_presets"),      description="Path to directory for style presets.")
    workflow_thumbnails_dir: Path = Field(default=Path("workflow_thumbnails"), description="Path to directory for workflow thumbnails.")

    # LOGGING
    log_handlers:             list[str] = Field(default=["console"],        description='Log handler. Valid options are "console", "file=<path>", "syslog=path|address:host:port", "http=<url>".')
    # note - would be better to read the log_format values from logging.py, but this creates circular dependencies issues
    log_format:              LOG_FORMAT = Field(default="color",            description='Log format. Use "plain" for text-only, "color" for colorized output, "legacy" for 2.3-style logging and "syslog" for syslog-style.')
    log_level:                LOG_LEVEL = Field(default="info",             description="Emit logging messages at this level or higher.")
    log_sql:                       bool = Field(default=False,              description="Log SQL queries. `log_level` must be `debug` for this to do anything. Extremely verbose.")
    log_level_network:        LOG_LEVEL = Field(default='warning',          description="Log level for network-related messages. 'info' and 'debug' are very verbose.")

    # Development
    use_memory_db:                 bool = Field(default=False,              description="Use in-memory database. Useful for development.")
    dev_reload:                    bool = Field(default=False,              description="Automatically reload when Python sources are changed. Does not reload node definitions.")
    profile_graphs:                bool = Field(default=False,              description="Enable graph profiling using `cProfile`.")
    profile_prefix:       Optional[str] = Field(default=None,               description="An optional prefix for profile output files.")
    profiles_dir:                  Path = Field(default=Path("profiles"),   description="Path to profiles output directory.")

    # CACHE
    max_cache_ram_gb:   Optional[float] = Field(default=None, gt=0,         description="The maximum amount of CPU RAM to use for model caching in GB. If unset, the limit will be configured based on the available RAM. In most cases, it is recommended to leave this unset.")
    max_cache_vram_gb:  Optional[float] = Field(default=None, ge=0,         description="The amount of VRAM to use for model caching in GB. If unset, the limit will be configured based on the available VRAM and the device_working_mem_gb. In most cases, it is recommended to leave this unset.")
    log_memory_usage:              bool = Field(default=False,              description="If True, a memory snapshot will be captured before and after every model cache operation, and the result will be logged (at debug level). There is a time cost to capturing the memory snapshots, so it is recommended to only enable this feature if you are actively inspecting the model cache's behaviour.")
    device_working_mem_gb:        float = Field(default=3,                  description="The amount of working memory to keep available on the compute device (in GB). Has no effect if running on CPU. If you are experiencing OOM errors, try increasing this value.")
    enable_partial_loading:        bool = Field(default=False,              description="Enable partial loading of models. This enables models to run with reduced VRAM requirements (at the cost of slower speed) by streaming the model from RAM to VRAM as its used. In some edge cases, partial loading can cause models to run more slowly if they were previously being fully loaded into VRAM.")
    keep_ram_copy_of_weights:      bool = Field(default=True,              description="Whether to keep a full RAM copy of a model's weights when the model is loaded in VRAM. Keeping a RAM copy increases average RAM usage, but speeds up model switching and LoRA patching (assuming there is sufficient RAM). Set this to False if RAM pressure is consistently high.")
    # Deprecated CACHE configs
    ram:                Optional[float] = Field(default=None, gt=0,         description="DEPRECATED: This setting is no longer used. It has been replaced by `max_cache_ram_gb`, but most users will not need to use this config since automatic cache size limits should work well in most cases. This config setting will be removed once the new model cache behavior is stable.")
    vram:               Optional[float] = Field(default=None, ge=0,         description="DEPRECATED: This setting is no longer used. It has been replaced by `max_cache_vram_gb`, but most users will not need to use this config since automatic cache size limits should work well in most cases. This config setting will be removed once the new model cache behavior is stable.")
    lazy_offload:                  bool = Field(default=True,               description="DEPRECATED: This setting is no longer used. Lazy-offloading is enabled by default. This config setting will be removed once the new model cache behavior is stable.")

    # PyTorch Memory Allocator
    pytorch_cuda_alloc_conf: Optional[str] = Field(default=None,            description="Configure the Torch CUDA memory allocator. This will impact peak reserved VRAM usage and performance. Setting to \"backend:cudaMallocAsync\" works well on many systems. The optimal configuration is highly dependent on the system configuration (device type, VRAM, CUDA driver version, etc.), so must be tuned experimentally.")

    # DEVICE
    device:                      str = Field(default="auto",                description="Preferred execution device. `auto` will choose the device depending on the hardware platform and the installed torch capabilities.<br>Valid values: `auto`, `cpu`, `cuda`, `mps`, `cuda:N` (where N is a device number)", pattern=r"^(auto|cpu|mps|cuda(:\d+)?)$")
    precision:                PRECISION = Field(default="auto",             description="Floating point precision. `float16` will consume half the memory of `float32` but produce slightly lower-quality images. The `auto` setting will guess the proper precision based on your video card and operating system.")

    # GENERATION
    sequential_guidance:           bool = Field(default=False,              description="Whether to calculate guidance in serial instead of in parallel, lowering memory requirements.")
    attention_type:      ATTENTION_TYPE = Field(default="auto",             description="Attention type.")
    attention_slice_size: ATTENTION_SLICE_SIZE = Field(default="auto",      description='Slice size, valid when attention_type=="sliced".')
    force_tiled_decode:            bool = Field(default=False,              description="Whether to enable tiled VAE decode (reduces memory consumption with some performance penalty).")
    pil_compress_level:             int = Field(default=1,                  description="The compress_level setting of PIL.Image.save(), used for PNG encoding. All settings are lossless. 0 = no compression, 1 = fastest with slightly larger filesize, 9 = slowest with smallest filesize. 1 is typically the best setting.")
    max_queue_size:                 int = Field(default=10000, gt=0,        description="Maximum number of items in the session queue.")
    clear_queue_on_startup:        bool = Field(default=False,              description="Empties session queue on startup.")

    # NODES
    allow_nodes:    Optional[list[str]] = Field(default=None,               description="List of nodes to allow. Omit to allow all.")
    deny_nodes:     Optional[list[str]] = Field(default=None,               description="List of nodes to deny. Omit to deny none.")
    node_cache_size:                int = Field(default=512,                description="How many cached nodes to keep in memory.")

    # MODEL INSTALL
    hashing_algorithm: HASHING_ALGORITHMS = Field(default="blake3_single",  description="Model hashing algorthim for model installs. 'blake3_multi' is best for SSDs. 'blake3_single' is best for spinning disk HDDs. 'random' disables hashing, instead assigning a UUID to models. Useful when using a memory db to reduce model installation time, or if you don't care about storing stable hashes for models. Alternatively, any other hashlib algorithm is accepted, though these are not nearly as performant as blake3.")
    remote_api_tokens: Optional[list[URLRegexTokenPair]] = Field(default=None, description="List of regular expression and token pairs used when downloading models from URLs. The download URL is tested against the regex, and if it matches, the token is provided in as a Bearer token.")
    scan_models_on_startup:        bool = Field(default=False,              description="Scan the models directory on startup, registering orphaned models. This is typically only used in conjunction with `use_memory_db` for testing purposes.")
    unsafe_disable_picklescan:     bool = Field(default=False,              description="UNSAFE. Disable the picklescan security check during model installation. Recommended only for development and testing purposes. This will allow arbitrary code execution during model installation, so should never be used in production.")

    # fmt: on

    model_config = SettingsConfigDict(env_prefix="INVOKEAI_", env_ignore_empty=True)

    def update_config(self, config: dict[str, Any] | InvokeAIAppConfig, clobber: bool = True) -> None:
        """Updates the config, overwriting existing values.

        Args:
            config: A dictionary of config settings, or instance of `InvokeAIAppConfig`. If an instance of \
                `InvokeAIAppConfig`, only the explicitly set fields will be merged into the singleton config.
            clobber: If `True`, overwrite existing values. If `False`, only update fields that are not already set.
        """

        if isinstance(config, dict):
            new_config = self.model_validate(config)
        else:
            new_config = config

        for field_name in new_config.model_fields_set:
            new_value = getattr(new_config, field_name)
            current_value = getattr(self, field_name)

            if field_name in self.model_fields_set and not clobber:
                continue

            if new_value != current_value:
                setattr(self, field_name, new_value)

    def write_file(self, dest_path: Path, as_example: bool = False) -> None:
        """Write the current configuration to file. This will overwrite the existing file.

        A `meta` stanza is added to the top of the file, containing metadata about the config file. This is not stored in the config object.

        Args:
            dest_path: Path to write the config to.
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w") as file:
            # Meta fields should be written in a separate stanza - skip legacy_models_yaml_path
            meta_dict = self.model_dump(mode="json", include={"schema_version"})

            # User settings
            config_dict = self.model_dump(
                mode="json",
                exclude_unset=False if as_example else True,
                exclude_defaults=False if as_example else True,
                exclude_none=True if as_example else False,
                exclude={"schema_version", "legacy_models_yaml_path"},
            )

            if as_example:
                file.write("# This is an example file with default and example settings.\n")
                file.write("# You should not copy this whole file into your config.\n")
                file.write("# Only add the settings you need to change to your config file.\n\n")
            file.write("# Internal metadata - do not edit:\n")
            file.write(yaml.dump(meta_dict, sort_keys=False))
            file.write("\n")
            file.write("# Put user settings here - see https://invoke-ai.github.io/InvokeAI/configuration/:\n")
            if len(config_dict) > 0:
                file.write(yaml.dump(config_dict, sort_keys=False))

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
    def config_file_path(self) -> Path:
        """Path to invokeai.yaml, resolved to an absolute path.."""
        resolved_path = self._resolve(self._config_file or INIT_FILE)
        assert resolved_path is not None
        return resolved_path

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
    def style_presets_path(self) -> Path:
        """Path to the style presets directory, resolved to an absolute path.."""
        return self._resolve(self.style_presets_dir)

    @property
    def workflow_thumbnails_path(self) -> Path:
        """Path to the workflow thumbnails directory, resolved to an absolute path.."""
        return self._resolve(self.workflow_thumbnails_dir)

    @property
    def convert_cache_path(self) -> Path:
        """Path to the converted cache models directory, resolved to an absolute path.."""
        return self._resolve(self.convert_cache_dir)

    @property
    def download_cache_path(self) -> Path:
        """Path to the downloaded models directory, resolved to an absolute path.."""
        return self._resolve(self.download_cache_dir)

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
        if os.environ.get("INVOKEAI_ROOT"):
            root = Path(os.environ["INVOKEAI_ROOT"])
        elif venv := os.environ.get("VIRTUAL_ENV", None):
            root = Path(venv).parent.resolve()
        else:
            root = Path("~/invokeai").expanduser().resolve()
        return root


class DefaultInvokeAIAppConfig(InvokeAIAppConfig):
    """A version of `InvokeAIAppConfig` that does not automatically parse any settings from environment variables
    or any file.

    This is useful for writing out a default config file.

    Note that init settings are set if provided.
    """

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


def migrate_v3_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Migrate a v3 config dictionary to a v4.0.0.

    Args:
        config_dict: A dictionary of settings from a v3 config file.

    Returns:
        An `InvokeAIAppConfig` config dict.

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
            # autocast was removed in v4.0.1
            if k == "precision" and v == "autocast":
                parsed_config_dict["precision"] = "auto"
            if k == "conf_path":
                parsed_config_dict["legacy_models_yaml_path"] = v
            if k == "legacy_conf_dir":
                # The old default for this was "configs/stable-diffusion" ("configs\stable-diffusion" on Windows).
                if v == "configs/stable-diffusion" or v == "configs\\stable-diffusion":
                    # If if the incoming config has the default value, skip
                    continue
                elif Path(v).name == "stable-diffusion":
                    # Else if the path ends in "stable-diffusion", we assume the parent is the new correct path.
                    parsed_config_dict["legacy_conf_dir"] = str(Path(v).parent)
                else:
                    # Else we do not attempt to migrate this setting
                    parsed_config_dict["legacy_conf_dir"] = v
            elif k in InvokeAIAppConfig.model_fields:
                # skip unknown fields
                parsed_config_dict[k] = v
    parsed_config_dict["schema_version"] = "4.0.0"
    return parsed_config_dict


def migrate_v4_0_0_to_4_0_1_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Migrate v4.0.0 config dictionary to a v4.0.1 config dictionary

    Args:
        config_dict: A dictionary of settings from a v4.0.0 config file.

    Returns:
        A config dict with the settings migrated to v4.0.1.
    """
    parsed_config_dict: dict[str, Any] = copy.deepcopy(config_dict)
    # precision "autocast" was replaced by "auto" in v4.0.1
    if parsed_config_dict.get("precision") == "autocast":
        parsed_config_dict["precision"] = "auto"
    parsed_config_dict["schema_version"] = "4.0.1"
    return parsed_config_dict


def migrate_v4_0_1_to_4_0_2_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Migrate v4.0.1 config dictionary to a v4.0.2 config dictionary.

    Args:
        config_dict: A dictionary of settings from a v4.0.1 config file.

    Returns:
        An config dict with the settings migrated to v4.0.2.
    """
    parsed_config_dict: dict[str, Any] = copy.deepcopy(config_dict)
    # convert_cache was removed in 4.0.2
    parsed_config_dict.pop("convert_cache", None)
    parsed_config_dict["schema_version"] = "4.0.2"
    return parsed_config_dict


def load_and_migrate_config(config_path: Path) -> InvokeAIAppConfig:
    """Load and migrate a config file to the latest version.

    Args:
        config_path: Path to the config file.

    Returns:
        An instance of `InvokeAIAppConfig` with the loaded and migrated settings.
    """
    assert config_path.suffix == ".yaml"
    with open(config_path, "rt", encoding=locale.getpreferredencoding()) as file:
        loaded_config_dict: dict[str, Any] = yaml.safe_load(file)

    assert isinstance(loaded_config_dict, dict)

    migrated = False
    if "InvokeAI" in loaded_config_dict:
        migrated = True
        loaded_config_dict = migrate_v3_config_dict(loaded_config_dict)  # pyright: ignore [reportUnknownArgumentType]
    if loaded_config_dict["schema_version"] == "4.0.0":
        migrated = True
        loaded_config_dict = migrate_v4_0_0_to_4_0_1_config_dict(loaded_config_dict)
    if loaded_config_dict["schema_version"] == "4.0.1":
        migrated = True
        loaded_config_dict = migrate_v4_0_1_to_4_0_2_config_dict(loaded_config_dict)

    if migrated:
        shutil.copy(config_path, config_path.with_suffix(".yaml.bak"))
        try:
            # load and write without environment variables
            migrated_config = DefaultInvokeAIAppConfig.model_validate(loaded_config_dict)
            migrated_config.write_file(config_path)
        except Exception as e:
            shutil.copy(config_path.with_suffix(".yaml.bak"), config_path)
            raise RuntimeError(f"Failed to load and migrate v3 config file {config_path}: {e}") from e

    try:
        # Meta is not included in the model fields, so we need to validate it separately
        config = InvokeAIAppConfig.model_validate(loaded_config_dict)
        assert config.schema_version == CONFIG_SCHEMA_VERSION, (
            f"Invalid schema version, expected {CONFIG_SCHEMA_VERSION}: {config.schema_version}"
        )
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config file {config_path}: {e}") from e


@lru_cache(maxsize=1)
def get_config() -> InvokeAIAppConfig:
    """Get the global singleton app config.

    When first called, this function:
    - Creates a config object. `pydantic-settings` handles merging of settings from environment variables, but not the init file.
    - Retrieves any provided CLI args from the InvokeAIArgs class. It does not _parse_ the CLI args; that is done in the main entrypoint.
    - Sets the root dir, if provided via CLI args.
    - Logs in to HF if there is no valid token already.
    - Copies all legacy configs to the legacy conf dir (needed for conversion from ckpt to diffusers).
    - Reads and merges in settings from the config file if it exists, else writes out a default config file.

    On subsequent calls, the object is returned from the cache.
    """
    # This object includes environment variables, as parsed by pydantic-settings
    config = InvokeAIAppConfig()

    args = InvokeAIArgs.args

    # This flag serves as a proxy for whether the config was retrieved in the context of the full application or not.
    # If it is False, we should just return a default config and not set the root, log in to HF, etc.
    if not InvokeAIArgs.did_parse:
        return config

    # Set CLI args
    if root := getattr(args, "root", None):
        config._root = Path(root)
    if config_file := getattr(args, "config_file", None):
        config._config_file = Path(config_file)

    # Create the example config file, with some extra example values provided
    example_config = DefaultInvokeAIAppConfig()
    example_config.remote_api_tokens = [
        URLRegexTokenPair(url_regex="cool-models.com", token="my_secret_token"),
        URLRegexTokenPair(url_regex="nifty-models.com", token="some_other_token"),
    ]
    example_config.write_file(config.config_file_path.with_suffix(".example.yaml"), as_example=True)

    # Copy all legacy configs only if needed
    # We know `__path__[0]` is correct here
    configs_src = Path(model_configs.__path__[0])  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    dest_path = config.legacy_conf_path

    # Create destination (we don't need to check for existence)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Compare directories recursively
    comparison = filecmp.dircmp(configs_src, dest_path)
    need_copy = any(
        [
            comparison.left_only,  # Files exist only in source
            comparison.diff_files,  # Files that differ
            comparison.common_funny,  # Files that couldn't be compared
        ]
    )

    if need_copy:
        # Get permissions from destination directory
        dest_mode = dest_path.stat().st_mode

        # Copy directory tree
        shutil.copytree(configs_src, dest_path, dirs_exist_ok=True)

        # Set permissions on copied files to match destination directory
        dest_path.chmod(dest_mode)
        for p in dest_path.glob("**/*"):
            p.chmod(dest_mode)

    if config.config_file_path.exists():
        config_from_file = load_and_migrate_config(config.config_file_path)
        # Clobbering here will overwrite any settings that were set via environment variables
        config.update_config(config_from_file, clobber=False)
    else:
        # We should never write env vars to the config file
        default_config = DefaultInvokeAIAppConfig()
        default_config.write_file(config.config_file_path, as_example=False)

    return config
