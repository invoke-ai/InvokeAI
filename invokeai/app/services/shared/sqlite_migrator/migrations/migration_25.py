import json
import sqlite3
from logging import Logger
from typing import Any

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
from invokeai.backend.model_manager.taxonomy import ModelType, Qwen3VariantType


class Migration25Callback:
    def __init__(self, app_config: InvokeAIAppConfig, logger: Logger) -> None:
        self._app_config = app_config
        self._logger = logger

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT id, config FROM models;")
        rows = cursor.fetchall()

        migrated_count = 0

        for model_id, config_json in rows:
            try:
                config_dict: dict[str, Any] = json.loads(config_json)

                if config_dict.get("type") != ModelType.Qwen3Encoder.value:
                    continue

                if "variant" in config_dict:
                    continue

                config_dict["variant"] = Qwen3VariantType.Qwen3_4B.value

                cursor.execute(
                    "UPDATE models SET config = ? WHERE id = ?;",
                    (json.dumps(config_dict), model_id),
                )
                migrated_count += 1

            except json.JSONDecodeError as e:
                self._logger.error("Invalid config JSON for model %s: %s", model_id, e)
                raise

        if migrated_count > 0:
            self._logger.info(f"Migration complete: {migrated_count} Qwen3 encoder configs updated with variant field")
        else:
            self._logger.info("Migration complete: no Qwen3 encoder configs needed migration")


def build_migration_25(app_config: InvokeAIAppConfig, logger: Logger) -> Migration:
    """Builds the migration object for migrating from version 24 to version 25.

    This migration adds the variant field to existing Qwen3 encoder models.
    Models installed before the variant field was added will default to Qwen3_4B (for Z-Image compatibility).
    """

    return Migration(
        from_version=24,
        to_version=25,
        callback=Migration25Callback(app_config=app_config, logger=logger),
    )
