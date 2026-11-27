from typing import (
    Literal,
    Self,
)

from pydantic import Field
from typing_extensions import Any

from invokeai.backend.model_manager.configs.base import Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    common_config_paths,
    get_config_dict_or_raise,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)


class SigLIP_Diffusers_Config(Diffusers_Config_Base, Config_Base):
    """Model config for SigLIP."""

    type: Literal[ModelType.SigLIP] = Field(default=ModelType.SigLIP)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)
    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        config = get_config_dict_or_raise(common_config_paths(mod.path))
        if "_class_name" in config or "architectures" in config:
            raise_for_class_name(config, {"SiglipModel"})
        # SigLIP2's fixed-resolution checkpoints (e.g. google/siglip2-base-patch16-512) name no class, only the
        # model type; the vision tower is what the loader builds.
        elif config.get("model_type") != "siglip" or not isinstance(config.get("vision_config"), dict):
            raise NotAMatchError("config is not a SigLIP model config")

        return cls(**override_fields)
