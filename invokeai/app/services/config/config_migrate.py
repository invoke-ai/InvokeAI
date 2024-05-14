# Copyright 2024 Lincoln D. Stein and the InvokeAI Development Team

"""
Utility class for migrating among versions of the InvokeAI app config schema.
"""

import locale
import shutil
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import yaml
from packaging.version import Version

import invokeai.configs as model_configs
from invokeai.app.services.config.config_common import AppConfigDict, ConfigMigration
from invokeai.app.services.config.migrations import config_migration_1, config_migration_2
from invokeai.frontend.cli.arg_parser import InvokeAIArgs

from .config_default import CONFIG_SCHEMA_VERSION, DefaultInvokeAIAppConfig, InvokeAIAppConfig, URLRegexTokenPair


class ConfigMigrator:
    """This class allows migrators to register their input and output versions."""

    def __init__(self) -> None:
        self._migrations: set[ConfigMigration] = set()

    def register(self, migration: ConfigMigration) -> None:
        migration_from_already_registered = any(m.from_version == migration.from_version for m in self._migrations)
        migration_to_already_registered = any(m.to_version == migration.to_version for m in self._migrations)
        if migration_from_already_registered or migration_to_already_registered:
            raise ValueError(
                f"A migration from {migration.from_version} or to {migration.to_version} has already been registered."
            )
        self._migrations.add(migration)

    @staticmethod
    def _check_for_discontinuities(migrations: list[ConfigMigration]) -> None:
        current_version = Version("3.0.0")
        for m in migrations:
            if current_version != m.from_version:
                raise ValueError(
                    f"Migration functions are not continuous. Expected from_version={current_version} but got from_version={m.from_version}, for migration function {m.function.__name__}"
                )
            current_version = m.to_version

    def run_migrations(self, original_config: AppConfigDict) -> AppConfigDict:
        """
        Use the registered migrations to bring config up to latest version.

        Args:
            original_config: The original configuration.

        Returns:
            The new configuration, lifted up to the latest version.
        """

        # Sort migrations by version number and raise a ValueError if any version range overlaps are detected.
        sorted_migrations = sorted(self._migrations, key=lambda x: x.from_version)
        self._check_for_discontinuities(sorted_migrations)

        # Do not mutate the incoming dict - we don't know who else may be using it
        migrated_config = deepcopy(original_config)

        # v3.0.0 configs did not have "schema_version", but did have "InvokeAI"
        if "InvokeAI" in migrated_config:
            version = Version("3.0.0")
        else:
            version = Version(migrated_config["schema_version"])

        for migration in sorted_migrations:
            if version == migration.from_version:
                migrated_config = migration.function(migrated_config)
                version = migration.to_version

        # We must end on the latest version
        assert migrated_config["schema_version"] == str(sorted_migrations[-1].to_version)
        return migrated_config


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
        loaded_config_dict: AppConfigDict = yaml.safe_load(file)

    assert isinstance(loaded_config_dict, dict)

    shutil.copy(config_path, config_path.with_suffix(".yaml.bak"))
    try:
        migrator = ConfigMigrator()
        migrator.register(config_migration_1)
        migrator.register(config_migration_2)
        migrated_config_dict = migrator.run_migrations(loaded_config_dict)
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
