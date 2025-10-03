from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)


class Unknown_Config(Config_Base):
    """Model config for unknown models, used as a fallback when we cannot identify a model."""

    base: Literal[BaseModelType.Unknown] = Field(default=BaseModelType.Unknown)
    type: Literal[ModelType.Unknown] = Field(default=ModelType.Unknown)
    format: Literal[ModelFormat.Unknown] = Field(default=ModelFormat.Unknown)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        raise NotAMatchError("unknown model config cannot match any model")
