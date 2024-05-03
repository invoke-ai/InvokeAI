# Copyright 2024 Lincoln D. Stein and the InvokeAI Development Team

"""
Utility class for migrating among versions of the InvokeAI app config schema.
"""

import locale
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, TypeAlias

import yaml
from packaging.version import Version

import invokeai.configs as model_configs
from invokeai.frontend.cli.arg_parser import InvokeAIArgs

from .config_default import CONFIG_SCHEMA_VERSION, DefaultInvokeAIAppConfig, InvokeAIAppConfig, URLRegexTokenPair
from .migrations import AppConfigDict, Migrations, MigrationsBase

MigrationFunction: TypeAlias = Callable[[AppConfigDict], AppConfigDict]


@dataclass
class MigrationEntry:
    """Defines an individual migration."""

    from_version: Version
    to_version: Version
    function: MigrationFunction


class ConfigMigrator:
    """This class allows migrators to register their input and output versions."""

    def __init__(self, migrations: type[MigrationsBase]) -> None:
        self._migrations: List[MigrationEntry] = []
        migrations.load(self)

    def register(
        self,
        from_version: str,
        to_version: str,
    ) -> Callable[[MigrationFunction], MigrationFunction]:
        """Define a decorator which registers the migration between two versions."""

        def decorator(function: MigrationFunction) -> MigrationFunction:
            if any((from_version == m.from_version) or (to_version == m.to_version) for m in self._migrations):
                raise ValueError(
                    f"function {function.__name__} is trying to register a migration for version {str(from_version)}, but this migration has already been registered."
                )
            self._migrations.append(
                MigrationEntry(from_version=Version(from_version), to_version=Version(to_version), function=function)
            )
            return function

        return decorator

    @staticmethod
    def _check_for_discontinuities(migrations: List[MigrationEntry]) -> None:
        current_version = Version("3.0.0")
        for m in migrations:
            if current_version != m.from_version:
                raise ValueError(
                    f"Migration functions are not continuous. Expected from_version={current_version} but got from_version={m.from_version}, for migration function {m.function.__name__}"
                )
            current_version = m.to_version

    def run_migrations(self, config_dict: AppConfigDict) -> AppConfigDict:
        """
        Use the registered migration steps to bring config up to latest version.

        :param config: The original configuration.
        :return: The new configuration, lifted up to the latest version.

        As a side effect, the new configuration will be written to disk.
        If an inconsistency in the registered migration steps' `from_version`
        and `to_version` parameters are identified, this will raise a
        ValueError exception.
        """
        # Sort migrations by version number and raise a ValueError if
        # any version range overlaps are detected.
        sorted_migrations = sorted(self._migrations, key=lambda x: x.from_version)
        self._check_for_discontinuities(sorted_migrations)

        if "InvokeAI" in config_dict:
            version = Version("3.0.0")
        else:
            version = Version(config_dict["schema_version"])

        for migration in sorted_migrations:
            if version == migration.from_version and version < migration.to_version:
                config_dict = migration.function(config_dict)
                version = migration.to_version

        config_dict["schema_version"] = str(version)
        return config_dict


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

    # Copy all legacy configs - We know `__path__[0]` is correct here
    configs_src = Path(model_configs.__path__[0])  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    shutil.copytree(configs_src, config.legacy_conf_path, dirs_exist_ok=True)

    if config.config_file_path.exists():
        config_from_file = load_and_migrate_config(config.config_file_path)
        config_from_file.write_file(config.config_file_path)
        # Clobbering here will overwrite any settings that were set via environment variables
        config.update_config(config_from_file, clobber=False)
    else:
        # We should never write env vars to the config file
        default_config = DefaultInvokeAIAppConfig()
        default_config.write_file(config.config_file_path, as_example=False)

    return config


def load_and_migrate_config(config_path: Path) -> InvokeAIAppConfig:
    """Load and migrate a config file to the latest version.

    Args:
        config_path: Path to the config file.

    Returns:
        An instance of `InvokeAIAppConfig` with the loaded and migrated settings.
    """
    assert config_path.suffix == ".yaml"
    with open(config_path, "rt", encoding=locale.getpreferredencoding()) as file:
        loaded_config_dict = yaml.safe_load(file)

    assert isinstance(loaded_config_dict, dict)

    shutil.copy(config_path, config_path.with_suffix(".yaml.bak"))
    try:
        migrator = ConfigMigrator(Migrations)
        migrated_config_dict = migrator.run_migrations(loaded_config_dict)  # pyright: ignore [reportUnknownArgumentType]
    except Exception as e:
        shutil.copy(config_path.with_suffix(".yaml.bak"), config_path)
        raise RuntimeError(f"Failed to load and migrate config file {config_path}: {e}") from e

    # Attempt to load as a v4 config file
    try:
        config = InvokeAIAppConfig.model_validate(migrated_config_dict)
        assert (
            config.schema_version == CONFIG_SCHEMA_VERSION
        ), f"Invalid schema version, expected {CONFIG_SCHEMA_VERSION} but got {config.schema_version}"
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config file {config_path}: {e}") from e
