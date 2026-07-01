from copy import deepcopy
from typing import Any, Literal, Self

from pydantic import Field

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)

app_config = get_config()


class Unknown_Config(Config_Base):
    """Model config for unknown models, used as a fallback when we cannot positively identify a model."""

    base: Literal[BaseModelType.Unknown] = Field(default=BaseModelType.Unknown)
    type: Literal[ModelType.Unknown] = Field(default=ModelType.Unknown)
    format: Literal[ModelFormat.Unknown] = Field(default=ModelFormat.Unknown)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        """Create an Unknown_Config for models that couldn't be positively identified.

        Note: Basic path validation (file extensions, directory structure) is already
        performed by ModelConfigFactory before this method is called.
        """

        cloned_override_fields = deepcopy(override_fields)
        cloned_override_fields.pop("base", None)
        cloned_override_fields.pop("type", None)
        cloned_override_fields.pop("format", None)

        return cls(
            **cloned_override_fields,
            # Override the type/format/base to ensure it's marked as unknown.
            base=BaseModelType.Unknown,
            type=ModelType.Unknown,
            format=ModelFormat.Unknown,
        )
