from typing import (
    Literal,
    Self,
)

from pydantic import Field
from typing_extensions import Any

from invokeai.backend.model_manager.configs.base import Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    common_config_paths,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelType,
)


class LlavaOnevision_Diffusers_Config(Diffusers_Config_Base, Config_Base):
    """Model config for Llava Onevision models."""

    type: Literal[ModelType.LlavaOnevision] = Field(default=ModelType.LlavaOnevision)
    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "LlavaOnevisionForConditionalGeneration",
            },
        )

        return cls(**override_fields)
