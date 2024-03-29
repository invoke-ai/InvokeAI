import sqlite3
from pathlib import Path

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration8Callback:
    def __init__(self, app_config: InvokeAIAppConfig) -> None:
        self._app_config = app_config

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._drop_model_config_table(cursor)
        self._migrate_abs_models_to_rel(cursor)

    def _drop_model_config_table(self, cursor: sqlite3.Cursor) -> None:
        """Drops the old model_config table. This was missed in a previous migration."""

        cursor.execute("DROP TABLE IF EXISTS model_config;")

    def _migrate_abs_models_to_rel(self, cursor: sqlite3.Cursor) -> None:
        """Check all model paths & legacy config paths to determine if they are inside Invoke-managed directories. If
        they are, update the paths to be relative to the managed directories.

        This migration is a no-op for normal users (their paths will already be relative), but is necessary for users
        who have been testing the RCs with their live databases. The paths were made absolute in the initial RC, but this
        change was reverted. To smooth over the revert for our tests, we can migrate the paths back to relative.
        """

        models_path = self._app_config.models_path
        legacy_conf_path = self._app_config.legacy_conf_path
        legacy_conf_dir = self._app_config.legacy_conf_dir

        stmt = """---sql
        SELECT
            id,
            path,
            json_extract(config, '$.config_path') as config_path
        FROM models;
        """

        all_models = cursor.execute(stmt).fetchall()

        for model_id, model_path, model_config_path in all_models:
            # If the model path is inside the models directory, update it to be relative to the models directory.
            if Path(model_path).is_relative_to(models_path):
                new_path = Path(model_path).relative_to(models_path)
                cursor.execute(
                    """--sql
                    UPDATE models
                    SET config = json_set(config, '$.path', ?)
                    WHERE id = ?;
                    """,
                    (str(new_path), model_id),
                )
            # If the model has a legacy config path and it is inside the legacy conf directory, update it to be
            # relative to the legacy conf directory. This also fixes up cases in which the config path was
            # incorrectly relativized to the root directory. It will now be relativized to the legacy conf directory.
            if model_config_path:
                if Path(model_config_path).is_relative_to(legacy_conf_path):
                    new_config_path = Path(model_config_path).relative_to(legacy_conf_path)
                elif Path(model_config_path).is_relative_to(legacy_conf_dir):
                    new_config_path = Path(*Path(model_config_path).parts[1:])
                else:
                    new_config_path = None
                if new_config_path:
                    cursor.execute(
                        """--sql
                        UPDATE models
                        SET config = json_set(config, '$.config_path', ?)
                        WHERE id = ?;
                        """,
                        (str(new_config_path), model_id),
                    )


def build_migration_8(app_config: InvokeAIAppConfig) -> Migration:
    """
    Build the migration from database version 7 to 8.

    This migration does the following:
    - Removes the `model_config` table.
    - Migrates absolute model & legacy config paths to be relative to the models directory.
    """
    migration_8 = Migration(
        from_version=7,
        to_version=8,
        callback=Migration8Callback(app_config),
    )

    return migration_8
