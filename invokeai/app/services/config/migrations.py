# Copyright 2024 Lincoln D. Stein and the InvokeAI Development Team

"""
Schema migrations to perform on an InvokeAIAppConfig object.

The Migrations class defined in this module defines a series of
schema version migration steps for the InvokeAIConfig object.

To define a new migration, add a migration function to
Migrations.load_migrations() following the existing examples.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from .config_default import InvokeAIAppConfig

if TYPE_CHECKING:
    from .config_migrate import ConfigMigrator

AppConfigDict: TypeAlias = dict[str, Any]


class MigrationsBase(ABC):
    """Define the config file migration steps to apply, abstract base class."""

    @classmethod
    @abstractmethod
    def load(cls, migrator: "ConfigMigrator") -> None:
        """Use the provided migrator to register the configuration migrations to be run."""


class Migrations(MigrationsBase):
    """Configuration migration steps to apply."""

    @classmethod
    def load(cls, migrator: "ConfigMigrator") -> None:
        """Define migrations to perform."""

        ##################
        # 3.0.0 -> 4.0.0 #
        ##################
        @migrator.register(from_version="3.0.0", to_version="4.0.0")
        def migrate_1(config_dict: dict[str, Any]) -> dict[str, Any]:
            """Migrate a v3 config dictionary to a current config object.

            Args:
                config_dict: A dictionary of settings from a v3 config file.

            Returns:
                A dictionary of settings from a 4.0.0 config file.

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
            return parsed_config_dict

        ##################
        # 4.0.0 -> 4.0.1 #
        ##################
        @migrator.register(from_version="4.0.0", to_version="4.0.1")
        def migrate_2(config_dict: dict[str, Any]) -> dict[str, Any]:
            """Migrate v4.0.0 config dictionary to v4.0.1.

            Args:
                config_dict: A dictionary of settings from a v4.0.0 config file.

            Returns:
                A dictionary of settings from a v4.0.1 config file
            """
            parsed_config_dict: dict[str, Any] = {}
            for k, v in config_dict.items():
                # autocast was removed from precision in v4.0.1
                if k == "precision" and v == "autocast":
                    parsed_config_dict["precision"] = "auto"
                else:
                    parsed_config_dict[k] = v
            return parsed_config_dict
