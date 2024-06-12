import shutil
import sqlite3

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration12Callback:
    def __init__(self, app_config: InvokeAIAppConfig) -> None:
        self._app_config = app_config

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._remove_model_convert_cache_dir()

    def _remove_model_convert_cache_dir(self) -> None:
        """
        Removes unused model convert cache directory
        """
        convert_cache = self._app_config.convert_cache_path
        shutil.rmtree(convert_cache, ignore_errors=True)


def build_migration_12(app_config: InvokeAIAppConfig) -> Migration:
    """
    Build the migration from database version 11 to 12.

    This migration removes the now-unused model convert cache directory.
    """
    migration_12 = Migration(
        from_version=11,
        to_version=12,
        callback=Migration12Callback(app_config),
    )

    return migration_12
