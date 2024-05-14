# Copyright 2024 Lincoln D. Stein and the InvokeAI Development Team

"""
Schema migrations to perform on an InvokeAIAppConfig object.

The Migrations class defined in this module defines a series of
schema version migration steps for the InvokeAIConfig object.

To define a new migration, add a migration function to
Migrations.load_migrations() following the existing examples.
"""

from pathlib import Path

from packaging.version import Version

from invokeai.app.services.config.config_common import AppConfigDict, ConfigMigration

from .config_default import InvokeAIAppConfig


def migrate_v300_to_v400(original_config: AppConfigDict) -> AppConfigDict:
    """Migrate a v3.0.0 config dict to v4.0.0.

    Changes in this migration:
    - `outdir` was renamed to `outputs_dir`
    - `max_cache_size` was renamed to `ram`
    - `max_vram_cache_size` was renamed to `vram`
    - `conf_path`, which pointed to the old `models.yaml`, was removed - but if need to stash it to migrate the entries
        to the database
    - `legacy_conf_dir` was changed from a path relative to the app root, to a path relative to $INVOKEAI_ROOT/configs

    Args:
        config_dict: The v3.0.0 config dict to migrate.

    Returns:
        The migrated v4.0.0 config dict.
    """
    migrated_config: AppConfigDict = {}
    for _category_name, category_dict in original_config["InvokeAI"].items():
        for k, v in category_dict.items():
            # `outdir` was renamed to `outputs_dir` in v4
            if k == "outdir":
                migrated_config["outputs_dir"] = v
            # `max_cache_size` was renamed to `ram` some time in v3, but both names were used
            if k == "max_cache_size" and "ram" not in category_dict:
                migrated_config["ram"] = v
            # `max_vram_cache_size` was renamed to `vram` some time in v3, but both names were used
            if k == "max_vram_cache_size" and "vram" not in category_dict:
                migrated_config["vram"] = v
            if k == "conf_path":
                migrated_config["legacy_models_yaml_path"] = v
            if k == "legacy_conf_dir":
                # The old default for this was "configs/stable-diffusion" ("configs\stable-diffusion" on Windows).
                if v == "configs/stable-diffusion" or v == "configs\\stable-diffusion":
                    # If if the incoming config has the default value, skip
                    continue
                elif Path(v).name == "stable-diffusion":
                    # Else if the path ends in "stable-diffusion", we assume the parent is the new correct path.
                    migrated_config["legacy_conf_dir"] = str(Path(v).parent)
                else:
                    # Else we do not attempt to migrate this setting
                    migrated_config["legacy_conf_dir"] = v
            elif k in InvokeAIAppConfig.model_fields:
                # skip unknown fields
                migrated_config[k] = v
    migrated_config["schema_version"] = "4.0.0"
    return migrated_config


config_migration_1 = ConfigMigration(
    from_version=Version("3.0.0"), to_version=Version("4.0.0"), function=migrate_v300_to_v400
)


def migrate_v400_to_v401(original_config: AppConfigDict) -> AppConfigDict:
    """Migrate a v4.0.0 config dict to v4.0.1.

    Changes in this migration:
    - `precision: "autocast"` was removed, fall back to "auto"

    Args:
        config_dict: The v4.0.0 config dict to migrate.

    Returns:
        The migrated v4.0.1 config dict.
    """
    migrated_config: AppConfigDict = {}
    for k, v in original_config.items():
        # autocast was removed from precision in v4.0.1
        if k == "precision" and v == "autocast":
            migrated_config["precision"] = "auto"
        # skip unknown fields
        elif k in InvokeAIAppConfig.model_fields:
            migrated_config[k] = v
    migrated_config["schema_version"] = "4.0.1"
    return migrated_config


config_migration_2 = ConfigMigration(
    from_version=Version("4.0.0"), to_version=Version("4.0.1"), function=migrate_v400_to_v401
)
