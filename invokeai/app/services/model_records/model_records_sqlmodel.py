"""SQLModel implementation of ModelRecordServiceBase."""

import json
import logging
from math import ceil
from pathlib import Path
from typing import List, Optional, Union

import pydantic
from sqlalchemy import func, literal_column
from sqlmodel import select

from invokeai.app.services.model_records.model_records_base import (
    DuplicateModelException,
    ModelRecordChanges,
    ModelRecordOrderBy,
    ModelRecordServiceBase,
    ModelSummary,
    UnknownModelException,
)
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.models import ModelTable
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.model_manager.configs.factory import AnyModelConfig, ModelConfigFactory
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType

# Mapping from ModelRecordOrderBy to column expressions
_ORDER_COLS = {
    ModelRecordOrderBy.Default: "type, base, name, format",
    ModelRecordOrderBy.Type: "type",
    ModelRecordOrderBy.Base: "base",
    ModelRecordOrderBy.Name: "name",
    ModelRecordOrderBy.Format: "format",
}


class ModelRecordServiceSqlModel(ModelRecordServiceBase):
    """SQLModel implementation of ModelConfigStore."""

    def __init__(self, db: SqliteDatabase, logger: logging.Logger):
        super().__init__()
        self._db = db
        self._logger = logger

    def add_model(self, config: AnyModelConfig) -> AnyModelConfig:
        row = ModelTable(id=config.key, config=config.model_dump_json())
        try:
            with self._db.get_session() as session:
                session.add(row)
        except Exception as e:
            err_str = str(e)
            if "UNIQUE constraint failed" in err_str:
                if "models.path" in err_str:
                    msg = f"A model with path '{config.path}' is already installed"
                elif "models.name" in err_str:
                    msg = f"A model with name='{config.name}', type='{config.type}', base='{config.base}' is already installed"
                else:
                    msg = f"A model with key '{config.key}' is already installed"
                raise DuplicateModelException(msg) from e
            raise
        return self.get_model(config.key)

    def del_model(self, key: str) -> None:
        with self._db.get_session() as session:
            row = session.get(ModelTable, key)
            if row is None:
                raise UnknownModelException("model not found")
            session.delete(row)

    def update_model(self, key: str, changes: ModelRecordChanges, allow_class_change: bool = False) -> AnyModelConfig:
        record = self.get_model(key)

        if allow_class_change:
            record_as_dict = record.model_dump()
            for field_name in changes.model_fields_set:
                record_as_dict[field_name] = getattr(changes, field_name)
            record = ModelConfigFactory.from_dict(record_as_dict)
        else:
            for field_name in changes.model_fields_set:
                setattr(record, field_name, getattr(changes, field_name))

        json_serialized = record.model_dump_json()

        with self._db.get_session() as session:
            row = session.get(ModelTable, key)
            if row is None:
                raise UnknownModelException("model not found")
            row.config = json_serialized
            session.add(row)

        return self.get_model(key)

    def replace_model(self, key: str, new_config: AnyModelConfig) -> AnyModelConfig:
        if key != new_config.key:
            raise ValueError("key does not match new_config.key")
        with self._db.get_session() as session:
            row = session.get(ModelTable, key)
            if row is None:
                raise UnknownModelException("model not found")
            row.config = new_config.model_dump_json()
            session.add(row)
        return self.get_model(key)

    def get_model(self, key: str) -> AnyModelConfig:
        with self._db.get_readonly_session() as session:
            row = session.get(ModelTable, key)
            if row is None:
                raise UnknownModelException("model not found")
            return ModelConfigFactory.from_dict(json.loads(row.config))

    def get_model_by_hash(self, hash: str) -> AnyModelConfig:
        with self._db.get_readonly_session() as session:
            stmt = select(ModelTable).where(literal_column("hash") == hash)
            row = session.exec(stmt).first()
            if row is None:
                raise UnknownModelException("model not found")
            return ModelConfigFactory.from_dict(json.loads(row.config))

    def exists(self, key: str) -> bool:
        with self._db.get_readonly_session() as session:
            row = session.get(ModelTable, key)
        return row is not None

    def search_by_attr(
        self,
        model_name: Optional[str] = None,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None,
        model_format: Optional[ModelFormat] = None,
        order_by: ModelRecordOrderBy = ModelRecordOrderBy.Default,
    ) -> List[AnyModelConfig]:
        with self._db.get_readonly_session() as session:
            stmt = select(ModelTable)

            if model_name:
                stmt = stmt.where(literal_column("name") == model_name)
            if base_model:
                stmt = stmt.where(literal_column("base") == base_model)
            if model_type:
                stmt = stmt.where(literal_column("type") == model_type)
            if model_format:
                stmt = stmt.where(literal_column("format") == model_format)

            # Apply ordering via the generated columns
            if order_by == ModelRecordOrderBy.Default:
                stmt = stmt.order_by(
                    literal_column("type"),
                    literal_column("base"),
                    literal_column("name"),
                    literal_column("format"),
                )
            elif order_by == ModelRecordOrderBy.Type:
                stmt = stmt.order_by(literal_column("type"))
            elif order_by == ModelRecordOrderBy.Base:
                stmt = stmt.order_by(literal_column("base"))
            elif order_by == ModelRecordOrderBy.Name:
                stmt = stmt.order_by(literal_column("name"))
            elif order_by == ModelRecordOrderBy.Format:
                stmt = stmt.order_by(literal_column("format"))

            rows = session.exec(stmt).all()
            # Extract config strings while still in the session
            config_strings = [row.config for row in rows]

        results: list[AnyModelConfig] = []
        for config_str in config_strings:
            try:
                model_config = ModelConfigFactory.from_dict(json.loads(config_str))
            except pydantic.ValidationError as e:
                config_preview = f"{config_str[:64]}..." if len(config_str) > 64 else config_str
                try:
                    name = json.loads(config_str).get("name", "<unknown>")
                except Exception:
                    name = "<unknown>"
                self._logger.warning(
                    f"Skipping invalid model config in the database with name {name}. ({config_preview})"
                )
                self._logger.warning(f"Validation error: {e}")
            else:
                results.append(model_config)

        return results

    def search_by_path(self, path: Union[str, Path]) -> List[AnyModelConfig]:
        with self._db.get_readonly_session() as session:
            stmt = select(ModelTable).where(literal_column("path") == str(path))
            rows = session.exec(stmt).all()
            configs = [r.config for r in rows]
        return [ModelConfigFactory.from_dict(json.loads(c)) for c in configs]

    def search_by_hash(self, hash: str) -> List[AnyModelConfig]:
        with self._db.get_readonly_session() as session:
            stmt = select(ModelTable).where(literal_column("hash") == hash)
            rows = session.exec(stmt).all()
            configs = [r.config for r in rows]
        return [ModelConfigFactory.from_dict(json.loads(c)) for c in configs]

    def list_models(
        self, page: int = 0, per_page: int = 10, order_by: ModelRecordOrderBy = ModelRecordOrderBy.Default
    ) -> PaginatedResults[ModelSummary]:
        with self._db.get_readonly_session() as session:
            # Total count
            count_stmt = select(func.count()).select_from(ModelTable)
            total = session.exec(count_stmt).one()

            # Data query
            stmt = select(ModelTable)
            if order_by == ModelRecordOrderBy.Default:
                stmt = stmt.order_by(
                    literal_column("type"),
                    literal_column("base"),
                    literal_column("name"),
                    literal_column("format"),
                )
            elif order_by == ModelRecordOrderBy.Type:
                stmt = stmt.order_by(literal_column("type"))
            elif order_by == ModelRecordOrderBy.Base:
                stmt = stmt.order_by(literal_column("base"))
            elif order_by == ModelRecordOrderBy.Name:
                stmt = stmt.order_by(literal_column("name"))
            elif order_by == ModelRecordOrderBy.Format:
                stmt = stmt.order_by(literal_column("format"))

            stmt = stmt.limit(per_page).offset(page * per_page)
            rows = session.exec(stmt).all()
            configs = [r.config for r in rows]

        items = [ModelSummary.model_validate({"config": c}) for c in configs]
        return PaginatedResults(
            page=page,
            pages=ceil(total / per_page),
            per_page=per_page,
            total=total,
            items=items,
        )
