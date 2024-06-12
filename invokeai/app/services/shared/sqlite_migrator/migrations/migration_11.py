import sqlite3
import shutil

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration11Callback:
    def __init__(self, app_config: InvokeAIAppConfig) -> None:
        self._app_config = app_config

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._remove_model_convert_cache_dir()

    def _remove_model_convert_cache_dir(self) -> None:
        """
        Removes unused model convert cache directory
        """
        convert_cache = self._app_config.convert_cache_path
        print(f'DEBUG: convert_cache = {convert_cache}')
        # shutil.rmtree(convert_cache)


def build_migration_11(app_config: InvokeAIAppConfig) -> Migration:
    """
    Build the migration from database version 10 to 11.

    This migration removes the now-unused model convert cache directory.
    """
    migration_11 = Migration(
        from_version=10,
        to_version=11,
        callback=Migration11Callback(app_config),
    )

    return migration_11
