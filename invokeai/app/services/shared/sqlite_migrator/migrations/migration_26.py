import json
import sqlite3
from logging import Logger
from pathlib import Path
from typing import Any

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, ZImageVariantType


class Migration26Callback:
    def __init__(self, app_config: InvokeAIAppConfig, logger: Logger) -> None:
        self._app_config = app_config
        self._logger = logger

    def _detect_variant_from_scheduler(self, model_path: Path) -> ZImageVariantType:
        """Detect Z-Image variant from scheduler config for Diffusers models.

        Z-Image variants are distinguished by the scheduler shift value:
        - Turbo (distilled): shift = 3.0
        - Base (undistilled): shift = 6.0
        """
        scheduler_config_path = model_path / "scheduler" / "scheduler_config.json"

        if not scheduler_config_path.exists():
            return ZImageVariantType.Turbo

        try:
            with open(scheduler_config_path, "r", encoding="utf-8") as f:
                scheduler_config = json.load(f)

            shift = scheduler_config.get("shift", 3.0)

            # ZBase (undistilled) uses shift = 6.0, Turbo uses shift = 3.0
            if shift >= 5.0:
                return ZImageVariantType.ZBase
            else:
                return ZImageVariantType.Turbo
        except (json.JSONDecodeError, OSError) as e:
            self._logger.warning(f"Could not read scheduler config: {e}, defaulting to Turbo")
            return ZImageVariantType.Turbo

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT id, config FROM models;")
        rows = cursor.fetchall()

        migrated_turbo = 0
        migrated_base = 0

        for model_id, config_json in rows:
            try:
                config_dict: dict[str, Any] = json.loads(config_json)

                # Only migrate Z-Image main models
                if config_dict.get("base") != BaseModelType.ZImage.value:
                    continue

                if config_dict.get("type") != ModelType.Main.value:
                    continue

                # Skip if variant already set
                if "variant" in config_dict:
                    continue

                # Determine variant based on format
                model_format = config_dict.get("format")
                model_path = config_dict.get("path")

                if model_format == ModelFormat.Diffusers.value and model_path:
                    # For Diffusers models, detect from scheduler config
                    variant = self._detect_variant_from_scheduler(Path(model_path))
                else:
                    # For Checkpoint/GGUF, default to Turbo (Base only available as Diffusers)
                    variant = ZImageVariantType.Turbo

                config_dict["variant"] = variant.value

                cursor.execute(
                    "UPDATE models SET config = ? WHERE id = ?;",
                    (json.dumps(config_dict), model_id),
                )

                if variant == ZImageVariantType.ZBase:
                    migrated_base += 1
                else:
                    migrated_turbo += 1

            except json.JSONDecodeError as e:
                self._logger.error("Invalid config JSON for model %s: %s", model_id, e)
                raise

        total = migrated_turbo + migrated_base
        if total > 0:
            self._logger.info(
                f"Migration complete: {total} Z-Image model configs updated "
                f"({migrated_turbo} Turbo, {migrated_base} Base)"
            )
        else:
            self._logger.info("Migration complete: no Z-Image model configs needed migration")


def build_migration_26(app_config: InvokeAIAppConfig, logger: Logger) -> Migration:
    """Builds the migration object for migrating from version 25 to version 26.

    This migration adds the variant field to existing Z-Image main models.
    Models installed before the variant field was added will default to Turbo
    (the only variant available before Z-Image Base support was added).
    """

    return Migration(
        from_version=25,
        to_version=26,
        callback=Migration26Callback(app_config=app_config, logger=logger),
    )
