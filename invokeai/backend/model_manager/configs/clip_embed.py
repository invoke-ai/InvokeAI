from typing import (
    Literal,
    Self,
)

from pydantic import Field
from typing_extensions import Any

from invokeai.backend.model_manager.configs.base import Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    get_config_dict_or_raise,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ClipVariantType,
    ModelFormat,
    ModelType,
)


def get_clip_variant_type_from_config(config: dict[str, Any]) -> ClipVariantType | None:
    try:
        hidden_size = config.get("hidden_size")
        match hidden_size:
            case 1280:
                return ClipVariantType.G
            case 768:
                return ClipVariantType.L
            case _:
                return None
    except Exception:
        return None


class CLIPEmbed_Diffusers_Config_Base(Diffusers_Config_Base):
    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.CLIPEmbed] = Field(default=ModelType.CLIPEmbed)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        raise_for_class_name(
            {
                mod.path / "config.json",
                mod.path / "text_encoder" / "config.json",
            },
            {
                "CLIPModel",
                "CLIPTextModel",
                "CLIPTextModelWithProjection",
            },
        )

        cls._validate_variant(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_variant(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model variant does not match this config class."""
        expected_variant = cls.model_fields["variant"].default
        config = get_config_dict_or_raise(
            {
                mod.path / "config.json",
                mod.path / "text_encoder" / "config.json",
            },
        )
        recognized_variant = get_clip_variant_type_from_config(config)

        if recognized_variant is None:
            raise NotAMatchError("unable to determine CLIP variant from config")

        if expected_variant is not recognized_variant:
            raise NotAMatchError(f"variant is {recognized_variant}, not {expected_variant}")


class CLIPEmbed_Diffusers_G_Config(CLIPEmbed_Diffusers_Config_Base, Config_Base):
    variant: Literal[ClipVariantType.G] = Field(default=ClipVariantType.G)


class CLIPEmbed_Diffusers_L_Config(CLIPEmbed_Diffusers_Config_Base, Config_Base):
    variant: Literal[ClipVariantType.L] = Field(default=ClipVariantType.L)
