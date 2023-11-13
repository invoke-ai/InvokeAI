# Copyright (c) 2023 Lincoln D. Stein
"""Migrate from the InvokeAI v2 models.yaml format to the v3 sqlite format."""

from hashlib import sha1

from omegaconf import DictConfig, OmegaConf
from pydantic import TypeAdapter

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import (
    DuplicateModelException,
    ModelRecordServiceSQL,
)
from invokeai.app.services.shared.sqlite import SqliteDatabase
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
    logger: InvokeAILogger

    def __init__(self):
        self.config = InvokeAIAppConfig.get_config()
        self.config.parse_args()
        self.logger = InvokeAILogger.get_logger()

    def get_db(self) -> ModelRecordServiceSQL:
        """Fetch the sqlite3 database for this installation."""
        db = SqliteDatabase(self.config, self.logger)
        return ModelRecordServiceSQL(db)

    def get_yaml(self) -> DictConfig:
        """Fetch the models.yaml DictConfig for this installation."""
        yaml_path = self.config.model_conf_path
        return OmegaConf.load(yaml_path)

    def migrate(self):
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
            new_key = sha1(model_key.encode("utf-8")).hexdigest()

            stanza["base"] = BaseModelType(base_type)
            stanza["type"] = ModelType(model_type)
            stanza["name"] = model_name
            stanza["original_hash"] = hash
            stanza["current_hash"] = hash

            new_config = ModelsValidator.validate_python(stanza)
            self.logger.info(f"Adding model {model_name} with key {model_key}")
            try:
                db.add_model(new_key, new_config)
            except DuplicateModelException:
                self.logger.warning(f"Model {model_name} is already in the database")


def main():
    MigrateModelYamlToDb().migrate()


if __name__ == "__main__":
    main()
