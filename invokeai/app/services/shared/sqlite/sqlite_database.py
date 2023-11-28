import sqlite3
import threading
from logging import Logger

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite.sqlite_common import sqlite_memory
from invokeai.app.services.shared.sqlite.sqlite_migrator import MigrationSet, SQLiteMigrator


class SqliteDatabase:
    def __init__(self, config: InvokeAIAppConfig, logger: Logger):
        self._logger = logger
        self._config = config

        if self._config.use_memory_db:
            location = sqlite_memory
            self._logger.info("Using in-memory database")
        else:
            db_path = self._config.db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
            location = db_path
            self._logger.info(f"Using database at {location}")

        self.conn = sqlite3.connect(location, check_same_thread=False)
        self.lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row

        if self._config.log_sql:
            self.conn.set_trace_callback(self._logger.debug)

        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._migrator = SQLiteMigrator(db_path=location, lock=self.lock, logger=self._logger)

    def clean(self) -> None:
        try:
            self.lock.acquire()
            self.conn.execute("VACUUM;")
            self.conn.commit()
            self._logger.info("Cleaned database")
        except sqlite3.Error as e:
            self._logger.error(f"Error cleaning database: {e}")
            raise
        finally:
            self.lock.release()

    def register_migration_set(self, migration_set: MigrationSet) -> None:
        self._migrator.register_migration_set(migration_set)

    def run_migrations(self) -> None:
        self._migrator.run_migrations()
