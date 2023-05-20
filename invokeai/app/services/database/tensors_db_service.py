from abc import ABC, abstractmethod
from enum import Enum
import enum
import sqlite3
import threading
from typing import Any, Optional, Type, TypeVar, Union
from pydantic import BaseModel, Field
from torch import Tensor
from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.models.resources import TensorKind

from invokeai.app.services.item_storage import PaginatedResults


class GeneratedTensorEntity(BaseModel):
    """Deserialized generated (eg result or intermediate) tensors DB entity."""

    id: str = Field(description="The unique identifier for the tensor.")
    session_id: str = Field(description="The session ID.")
    node_id: str = Field(description="The node ID.")
    metadata: GeneratedImageOrLatentsMetadata = Field(
        description="The metadata for the tensor."
    )


class UploadedTensorEntity(BaseModel):
    """Deserialized uploaded tensors DB entity."""

    id: str = Field(description="The unique identifier for the tensor.")
    metadata: UploadedImageOrLatentsMetadata = Field(
        description="The metadata for the tensor."
    )


class TensorsDbServiceBase(ABC):
    """Responsible for interfacing with `tensors` store."""

    @abstractmethod
    def get(self, id: str) -> Union[GeneratedTensorEntity, UploadedTensorEntity, None]:
        """Gets an tensor from the `tensors` store."""
        pass

    @abstractmethod
    def get_many(
        self, tensor_kind: TensorKind, page: int = 0, per_page: int = 10
    ) -> PaginatedResults[Union[GeneratedTensorEntity, UploadedTensorEntity]]:
        """Gets a page of tensors from the `tensors` store."""
        pass

    @abstractmethod
    def delete(self, id: str) -> None:
        """Deletes an tensor from the `tensors` store."""
        pass

    @abstractmethod
    def set(
        self,
        id: str,
        tensor_kind: TensorKind,
        session_id: Optional[str],
        node_id: Optional[str],
        metadata: GeneratedImageOrLatentsMetadata | UploadedImageOrLatentsMetadata,
    ) -> None:
        """Sets an tensor in the `tensors` store."""
        pass
