# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Team

from __future__ import annotations

import sqlite3
import threading
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

from invokeai.backend.model_manager import ModelConfigBase, ModelType, SubModelType

from invokeai.backend.model_manager.storage import (
    ModelConfigStore,
    ModelConfigStoreSQL,
    ModelConfigStoreYAML,
    UnknownModelException,
)
from invokeai.backend.util.logging import InvokeAILogger

from .config import InvokeAIAppConfig


class ModelRecordServiceBase(ModelConfigStore):
    """
    Responsible for managing model configuration records.

    This is an ABC that is simply a subclassing of the ModelConfigStore ABC
    in the backend.
    """

    @classmethod
    @abstractmethod
    def from_db_file(cls, db_file: Path) -> ModelRecordServiceBase:
        """
        Initialize a new object from a database file.

        If the path does not exist, a new sqlite3 db will be initialized.

        :param db_file: Path to the database file
        """
        pass

    @classmethod
    def get_impl(
        cls, config: InvokeAIAppConfig, conn: Optional[sqlite3.Connection] = None, lock: Optional[threading.Lock] = None
    ) -> Union[ModelRecordServiceSQL, ModelRecordServiceFile]:
        """
        Choose either a ModelConfigStoreSQL or a ModelConfigStoreFile backend.

        Logic is as follows:
        1. if config.model_config_db contains a Path, then
           a. if the path looks like a .db file, open a new sqlite3 connection and return a ModelRecordServiceSQL
           b. if the path looks like a .yaml file, return a new ModelRecordServiceFile
           c. otherwise bail
        2. if config.model_config_db is the literal 'auto', then use the passed sqlite3 connection and thread lock.
           a. if either of these is missing, then we create our own connection to the invokeai.db file, which *should*
              be a safe thing to do - sqlite3 will use file-level locking.
        3. if config.model_config_db is None, then fall back to config.conf_path, using a yaml file
        """
        logger = InvokeAILogger.get_logger()
        db = config.model_config_db
        if db is None:
            return ModelRecordServiceFile.from_db_file(config.model_conf_path)
        if str(db) == "auto":
            logger.info("Model config storage = main InvokeAI database")
            return (
                ModelRecordServiceSQL.from_connection(conn, lock)
                if (conn and lock)
                else ModelRecordServiceSQL.from_db_file(config.db_path)
            )
        assert isinstance(db, Path)
        suffix = db.suffix
        if suffix == ".yaml":
            logger.info(f"Model config storage = {str(db)}")
            return ModelRecordServiceFile.from_db_file(config.root_path / db)
        elif suffix == ".db":
            logger.info(f"Model config storage = {str(db)}")
            return ModelRecordServiceSQL.from_db_file(config.root_path / db)
        else:
            raise ValueError(
                f'Unrecognized model config record db file type {db} in "model_config_db" configuration variable.'
            )


class ModelRecordServiceSQL(ModelConfigStoreSQL):
    """
    ModelRecordService that uses Sqlite for its backend.
    Please see invokeai/backend/model_manager/storage/sql.py for
    the implementation.
    """

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection, lock: threading.Lock) -> ModelRecordServiceSQL:
        """
        Initialize a new object from preexisting sqlite3 connection and threading lock objects.

        This is the same as the default __init__() constructor.

        :param conn: sqlite3 connection object
        :param lock: threading Lock object
        """
        return cls(conn, lock)

    @classmethod
    def from_db_file(cls, db_file: Path) -> ModelRecordServiceSQL:  # noqa D102 - docstring in ABC
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_file, check_same_thread=False)
        lock = threading.Lock()
        return cls(conn, lock)


class ModelRecordServiceFile(ModelConfigStoreYAML):
    """
    ModelRecordService that uses a YAML file for its backend.

    Please see invokeai/backend/model_manager/storage/yaml.py for
    the implementation.
    """

    @classmethod
    def from_db_file(cls, db_file: Path) -> ModelRecordServiceFile:  # noqa D102 - docstring in ABC
        return cls(db_file)
