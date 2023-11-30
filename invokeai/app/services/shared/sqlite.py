import sqlite3
import threading
from logging import Logger
from pathlib import Path

from invokeai.app.services.config import InvokeAIAppConfig

sqlite_memory = ":memory:"


class SqliteDatabase:
    def __init__(self, config: InvokeAIAppConfig, logger: Logger):
        self._logger = logger
        self._config = config

        if self._config.use_memory_db:
            self.db_path = sqlite_memory
            logger.info("Using in-memory database")
        else:
            db_path = self._config.db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_path)
            self._logger.info(f"Using database at {self.db_path}")

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row

        if self._config.log_sql:
            self.conn.set_trace_callback(self._logger.debug)

        self.conn.execute("PRAGMA foreign_keys = ON;")

    def clean(self) -> None:
        try:
            if self.db_path == sqlite_memory:
                return
            initial_db_size = Path(self.db_path).stat().st_size
            self.lock.acquire()
            self.conn.execute("VACUUM;")
            self.conn.commit()
            final_db_size = Path(self.db_path).stat().st_size
            freed_space_in_mb = round((initial_db_size - final_db_size) / 1024 / 1024, 2)
            if freed_space_in_mb > 0:
                self._logger.info(f"Cleaned database (freed {freed_space_in_mb}MB)")
        except Exception as e:
            self._logger.error(f"Error cleaning database: {e}")
            raise e
        finally:
            self.lock.release()
