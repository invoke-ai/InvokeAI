import sqlite3
import threading
from logging import Logger
from pathlib import Path

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite.migrations.migration_1 import migration_1
from invokeai.app.services.shared.sqlite.migrations.migration_2 import migration_2
from invokeai.app.services.shared.sqlite.sqlite_common import sqlite_memory
from invokeai.app.services.shared.sqlite.sqlite_migrator import SQLiteMigrator


class SqliteDatabase:
    database: Path | str

    def __init__(self, config: InvokeAIAppConfig, logger: Logger):
        self._logger = logger
        self._config = config
        if self._config.use_memory_db:
            self.database = sqlite_memory
            logger.info("Using in-memory database")
        else:
            self.database = self._config.db_path
            self.database.parent.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Using database at {self.database}")

        self.conn = sqlite3.connect(database=self.database, check_same_thread=False)
        self.lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row

        if self._config.log_sql:
            self.conn.set_trace_callback(self._logger.debug)

        self.conn.execute("PRAGMA foreign_keys = ON;")

        migrator = SQLiteMigrator(conn=self.conn, database=self.database, lock=self.lock, logger=self._logger)
        migrator.register_migration(migration_1)
        migrator.register_migration(migration_2)
        migrator.run_migrations()

    def clean(self) -> None:
        with self.lock:
            try:
                if self.database == sqlite_memory:
                    return
                initial_db_size = Path(self.database).stat().st_size
                self.conn.execute("VACUUM;")
                self.conn.commit()
                final_db_size = Path(self.database).stat().st_size
                freed_space_in_mb = round((initial_db_size - final_db_size) / 1024 / 1024, 2)
                if freed_space_in_mb > 0:
                    self._logger.info(f"Cleaned database (freed {freed_space_in_mb}MB)")
            except Exception as e:
                self._logger.error(f"Error cleaning database: {e}")
                raise
