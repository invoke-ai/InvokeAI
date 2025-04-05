import sqlite3

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk


class Migration19Callback:
    def __init__(self, app_config: InvokeAIAppConfig):
        self.models_path = app_config.models_path

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._populate_size(cursor)
        self._add_size_column(cursor)

    def _add_size_column(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            "ALTER TABLE models ADD COLUMN file_size INTEGER "
            "GENERATED ALWAYS as (json_extract(config, '$.file_size')) VIRTUAL NOT NULL"
        )

    def _populate_size(self, cursor: sqlite3.Cursor) -> None:
        all_models = cursor.execute("SELECT id, path FROM models;").fetchall()

        for model_id, model_path in all_models:
            mod = ModelOnDisk(self.models_path / model_path)
            cursor.execute(
                "UPDATE models SET config = json_set(config, '$.file_size', ?) WHERE id = ?", (mod.size(), model_id)
            )


def build_migration_19(app_config: InvokeAIAppConfig) -> Migration:
    return Migration(
        from_version=18,
        to_version=19,
        callback=Migration19Callback(app_config),
    )
