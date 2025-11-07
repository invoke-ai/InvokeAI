from typing import (
    Literal,
    Self,
)

from pydantic import Field
from typing_extensions import Any

from invokeai.backend.model_manager.configs.base import Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    get_class_name_from_config_dict_or_raise,
    get_config_dict_or_raise,
    raise_for_override_fields,
    raise_if_not_dir,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)


class CLIPVision_Diffusers_Config(Diffusers_Config_Base, Config_Base):
    """Model config for CLIPVision."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.CLIPVision] = Field(default=ModelType.CLIPVision)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        cls.raise_if_config_doesnt_look_like_clip_vision(mod)

        return cls(**override_fields)

    @classmethod
    def raise_if_config_doesnt_look_like_clip_vision(cls, mod: ModelOnDisk) -> None:
        config_dict = get_config_dict_or_raise(mod.path / "config.json")
        class_name = get_class_name_from_config_dict_or_raise(config_dict)

        if class_name == "CLIPVisionModelWithProjection":
            looks_like_clip_vision = True
        elif class_name == "CLIPModel" and "vision_config" in config_dict:
            looks_like_clip_vision = True
        else:
            looks_like_clip_vision = False

        if not looks_like_clip_vision:
            raise NotAMatchError(
                f"config class name is {class_name}, not CLIPVisionModelWithProjection or CLIPModel with vision_config"
            )
