# Copyright (c) 2023 Lincoln D. Stein
"""Migrate from the InvokeAI v2 models.yaml format to the v3 sqlite format."""

import json
import sqlite3
from logging import Logger
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from pydantic import TypeAdapter

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import (
    DuplicateModelException,
    UnknownModelException,
)
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    ModelConfigFactory,
    ModelType,
)
from invokeai.backend.model_manager.hash import ModelHash

ModelsValidator = TypeAdapter(AnyModelConfig)


class MigrateModelYamlToDb1:
    """
    Migrate the InvokeAI models.yaml format (VERSION 3.0.0) to SQL3 database format (VERSION 3.5.0).

    The class has one externally useful method, migrate(), which scans the
    currently models.yaml file and imports all its entries into invokeai.db.

    Use this way:

      from invokeai.backend.model_manager/migrate_to_db import MigrateModelYamlToDb
      MigrateModelYamlToDb().migrate()

    """

    config: InvokeAIAppConfig
    logger: Logger
    cursor: sqlite3.Cursor

    def __init__(self, config: InvokeAIAppConfig, logger: Logger, cursor: sqlite3.Cursor = None) -> None:
        self.config = config
        self.logger = logger
        self.cursor = cursor

    def get_yaml(self) -> DictConfig:
        """Fetch the models.yaml DictConfig for this installation."""
        yaml_path = self.config.model_conf_path
        omegaconf = OmegaConf.load(yaml_path)
        assert isinstance(omegaconf, DictConfig)
        return omegaconf

    def migrate(self) -> None:
        """Do the migration from models.yaml to invokeai.db."""
        try:
            yaml = self.get_yaml()
        except OSError:
            return

        for model_key, stanza in yaml.items():
            if model_key == "__metadata__":
                assert (
                    stanza["version"] == "3.0.0"
                ), f"This script works on version 3.0.0 yaml files, but your configuration points to a {stanza['version']} version"
                continue

            base_type, model_type, model_name = str(model_key).split("/")
            try:
                hash = ModelHash().hash(self.config.models_path / stanza.path)
            except OSError:
                self.logger.warning(f"The model at {stanza.path} is not a valid file or directory. Skipping migration.")
                continue

            stanza["base"] = BaseModelType(base_type)
            stanza["type"] = ModelType(model_type)
            stanza["name"] = model_name
            stanza["hash"] = hash
            new_key = hash  # deterministic key assignment

            # special case for ip adapters, which need the new `image_encoder_model_id` field
            if stanza["type"] == ModelType.IPAdapter:
                try:
                    stanza["image_encoder_model_id"] = self._get_image_encoder_model_id(
                        self.config.models_path / stanza.path
                    )
                except OSError:
                    self.logger.warning(f"Could not determine image encoder for {stanza.path}. Skipping.")
                    continue

            new_config: AnyModelConfig = ModelsValidator.validate_python(stanza)  # type: ignore # see https://github.com/pydantic/pydantic/discussions/7094

            try:
                if original_record := self._search_by_path(stanza.path):
                    key = original_record.key
                    self.logger.info(f"Updating model {model_name} with information from models.yaml using key {key}")
                    self._update_model(key, new_config)
                else:
                    self.logger.info(f"Adding model {model_name} with key {new_key}")
                    self._add_model(new_key, new_config)
            except DuplicateModelException:
                self.logger.warning(f"Model {model_name} is already in the database")
            except UnknownModelException:
                self.logger.warning(f"Model at {stanza.path} could not be found in database")

    def _search_by_path(self, path: Path) -> Optional[AnyModelConfig]:
        self.cursor.execute(
            """--sql
            SELECT config FROM models
            WHERE path=?;
            """,
            (str(path),),
        )
        results = [ModelConfigFactory.make_config(json.loads(x[0])) for x in self.cursor.fetchall()]
        return results[0] if results else None

    def _update_model(self, key: str, config: AnyModelConfig) -> None:
        record = ModelConfigFactory.make_config(config, key=key)  # ensure it is a valid config obect
        json_serialized = record.model_dump_json()  # and turn it into a json string.
        self.cursor.execute(
            """--sql
            UPDATE models
            SET
                config=?
            WHERE id=?;
            """,
            (json_serialized, key),
        )
        if self.cursor.rowcount == 0:
            raise UnknownModelException("model not found")

    def _add_model(self, key: str, config: AnyModelConfig) -> None:
        record = ModelConfigFactory.make_config(config, key=key)  # ensure it is a valid config obect.
        json_serialized = record.model_dump_json()  # and turn it into a json string.
        try:
            self.cursor.execute(
                """--sql
                INSERT INTO models (
                   id,
                   config
                  )
                VALUES (?,?,?);
                """,
                (
                    key,
                    json_serialized,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise DuplicateModelException(f"{record.name}: model is already in database") from exc

    def _get_image_encoder_model_id(self, model_path: Path) -> str:
        with open(model_path / "image_encoder.txt") as f:
            encoder = f.read()
        return encoder.strip()
