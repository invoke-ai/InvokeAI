# Copyright (c) 2023 Lincoln D. Stein
"""Migrate from the InvokeAI v2 models.yaml format to the v3 sqlite format."""

from hashlib import sha1
from logging import Logger

from omegaconf import DictConfig, OmegaConf
from pydantic import TypeAdapter

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import (
    DuplicateModelException,
    ModelRecordServiceSQL,
    UnknownModelException,
)
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    ModelType,
)
from invokeai.backend.model_manager.hash import FastModelHash
from invokeai.backend.util.logging import InvokeAILogger

ModelsValidator = TypeAdapter(AnyModelConfig)


class MigrateModelYamlToDb:
    """
    Migrate the InvokeAI models.yaml format (VERSION 3.0.0) to SQL3 database format (VERSION 3.2.0)

    The class has one externally useful method, migrate(), which scans the
    currently models.yaml file and imports all its entries into invokeai.db.

    Use this way:

      from invokeai.backend.model_manager/migrate_to_db import MigrateModelYamlToDb
      MigrateModelYamlToDb().migrate()

    """

    config: InvokeAIAppConfig
    logger: Logger

    def __init__(self) -> None:
        self.config = InvokeAIAppConfig.get_config()
        self.config.parse_args()
        self.logger = InvokeAILogger.get_logger()

    def get_db(self) -> ModelRecordServiceSQL:
        """Fetch the sqlite3 database for this installation."""
        db_path = None if self.config.use_memory_db else self.config.db_path
        db = SqliteDatabase(db_path=db_path, logger=self.logger, verbose=self.config.log_sql)
        return ModelRecordServiceSQL(db)

    def get_yaml(self) -> DictConfig:
        """Fetch the models.yaml DictConfig for this installation."""
        yaml_path = self.config.model_conf_path
        omegaconf = OmegaConf.load(yaml_path)
        assert isinstance(omegaconf, DictConfig)
        return omegaconf

    def migrate(self) -> None:
        """Do the migration from models.yaml to invokeai.db."""
        db = self.get_db()
        yaml = self.get_yaml()

        for model_key, stanza in yaml.items():
            if model_key == "__metadata__":
                assert (
                    stanza["version"] == "3.0.0"
                ), f"This script works on version 3.0.0 yaml files, but your configuration points to a {stanza['version']} version"
                continue

            base_type, model_type, model_name = str(model_key).split("/")
            hash = FastModelHash.hash(self.config.models_path / stanza.path)
            assert isinstance(model_key, str)
            new_key = sha1(model_key.encode("utf-8")).hexdigest()

            stanza["base"] = BaseModelType(base_type)
            stanza["type"] = ModelType(model_type)
            stanza["name"] = model_name
            stanza["original_hash"] = hash
            stanza["current_hash"] = hash

            new_config: AnyModelConfig = ModelsValidator.validate_python(stanza)  # type: ignore # see https://github.com/pydantic/pydantic/discussions/7094

            try:
                if original_record := db.search_by_path(stanza.path):
                    key = original_record[0].key
                    self.logger.info(f"Updating model {model_name} with information from models.yaml using key {key}")
                    db.update_model(key, new_config)
                else:
                    self.logger.info(f"Adding model {model_name} with key {model_key}")
                    db.add_model(new_key, new_config)
            except DuplicateModelException:
                self.logger.warning(f"Model {model_name} is already in the database")
            except UnknownModelException:
                self.logger.warning(f"Model at {stanza.path} could not be found in database")


def main():
    MigrateModelYamlToDb().migrate()


if __name__ == "__main__":
    main()
