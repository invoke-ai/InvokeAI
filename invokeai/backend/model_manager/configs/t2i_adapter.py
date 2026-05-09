from typing import (
    Literal,
    Self,
)

from pydantic import Field
from typing_extensions import Any

from invokeai.backend.model_manager.configs.base import Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.controlnet import ControlAdapterDefaultSettings
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


class T2IAdapter_Diffusers_Config_Base(Diffusers_Config_Base):
    """Model config for T2I."""

    type: Literal[ModelType.T2IAdapter] = Field(default=ModelType.T2IAdapter)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)
    default_settings: ControlAdapterDefaultSettings | None = Field(None)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "T2IAdapter",
            },
        )

        cls._validate_base(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        config_dict = get_config_dict_or_raise(common_config_paths(mod.path))

        adapter_type = config_dict.get("adapter_type")

        match adapter_type:
            case "full_adapter_xl":
                return BaseModelType.StableDiffusionXL
            case "full_adapter" | "light_adapter":
                return BaseModelType.StableDiffusion1
            case _:
                raise NotAMatchError(f"unrecognized adapter_type '{adapter_type}'")


class T2IAdapter_Diffusers_SD1_Config(T2IAdapter_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class T2IAdapter_Diffusers_SDXL_Config(T2IAdapter_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)
