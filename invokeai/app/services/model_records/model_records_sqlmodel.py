"""SQLModel implementation of ModelRecordServiceBase."""

import logging
from pathlib import Path
from typing import List, Optional, Union

from invokeai.app.services.model_records.model_records_base import (
    ModelRecordChanges,
    ModelRecordOrderBy,
    ModelRecordServiceBase,
    ModelSummary,
)
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.model_manager.configs.factory import AnyModelConfig, ModelConfigFactory
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


class ModelRecordServiceSqlModel(ModelRecordServiceBase):
    """SQLModel implementation of ModelConfigStore."""

    def __init__(self, db: SqliteDatabase, logger: logging.Logger):
        super().__init__()
        self._db = db
        self._q = db.queries
        self._logger = logger

    def add_model(self, config: AnyModelConfig) -> AnyModelConfig:
        self._q.models_insert(config)
        return self.get_model(config.key)

    def del_model(self, key: str) -> None:
        self._q.models_delete(key)

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

        self._q.models_update_config_json(key, record.model_dump_json())
        return self.get_model(key)

    def replace_model(self, key: str, new_config: AnyModelConfig) -> AnyModelConfig:
        if key != new_config.key:
            raise ValueError("key does not match new_config.key")
        self._q.models_update_config_json(key, new_config.model_dump_json())
        return self.get_model(key)

    def get_model(self, key: str) -> AnyModelConfig:
        return self._q.models_get(key)

    def get_model_by_hash(self, hash: str) -> AnyModelConfig:
        return self._q.models_get_by_hash(hash)

    def exists(self, key: str) -> bool:
        return self._q.models_exists(key)

    def search_by_attr(
        self,
        model_name: Optional[str] = None,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None,
        model_format: Optional[ModelFormat] = None,
        order_by: ModelRecordOrderBy = ModelRecordOrderBy.Default,
        direction: SQLiteDirection = SQLiteDirection.Ascending,
    ) -> List[AnyModelConfig]:
        return self._q.models_search_by_attr(
            model_name=model_name,
            base_model=base_model,
            model_type=model_type,
            model_format=model_format,
            order_by=order_by,
            direction=direction,
        )

    def search_by_path(self, path: Union[str, Path]) -> List[AnyModelConfig]:
        return self._q.models_search_by_path(path)

    def search_by_hash(self, hash: str) -> List[AnyModelConfig]:
        return self._q.models_search_by_hash(hash)

    def list_models(
        self,
        page: int = 0,
        per_page: int = 10,
        order_by: ModelRecordOrderBy = ModelRecordOrderBy.Default,
        direction: SQLiteDirection = SQLiteDirection.Ascending,
    ) -> PaginatedResults[ModelSummary]:
        return self._q.models_list_summaries(page=page, per_page=per_page, order_by=order_by, direction=direction)
