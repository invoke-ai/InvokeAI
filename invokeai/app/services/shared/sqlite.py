import sqlite3
import threading
from logging import Logger

from invokeai.app.services.config import InvokeAIAppConfig

sqlite_memory = ":memory:"


class SqliteDatabase:
    conn: sqlite3.Connection
    lock: threading.RLock
    _logger: Logger
    _config: InvokeAIAppConfig

    def __init__(self, config: InvokeAIAppConfig, logger: Logger):
        self._logger = logger
        self._config = config

        if self._config.use_memory_db:
            location = sqlite_memory
            logger.info("Using in-memory database")
        else:
            db_path = self._config.db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
            location = str(db_path)
            self._logger.info(f"Using database at {location}")

        self.conn = sqlite3.connect(location, check_same_thread=False)
        self.lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row

        if self._config.log_sql:
            self.conn.set_trace_callback(self._logger.debug)

        self.conn.execute("PRAGMA foreign_keys = ON;")

    def clean(self) -> None:
        try:
            self.lock.acquire()
            self.conn.execute("VACUUM;")
            self.conn.commit()
            self._logger.info("Cleaned database")
        except Exception as e:
            self._logger.error(f"Error cleaning database: {e}")
            raise e
        finally:
            self.lock.release()
