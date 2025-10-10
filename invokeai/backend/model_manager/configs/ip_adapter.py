from abc import ABC
from typing import (
    Literal,
    Self,
)

from pydantic import BaseModel, Field
from typing_extensions import Any

from invokeai.backend.flux.ip_adapter.state_dict_utils import is_state_dict_xlabs_ip_adapter
from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
    state_dict_has_any_keys_starting_with,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)


class IPAdapter_Config_Base(ABC, BaseModel):
    type: Literal[ModelType.IPAdapter] = Field(default=ModelType.IPAdapter)


class IPAdapter_InvokeAI_Config_Base(IPAdapter_Config_Base):
    """Model config for IP Adapter diffusers format models."""

    format: Literal[ModelFormat.InvokeAI] = Field(default=ModelFormat.InvokeAI)

    # TODO(ryand): Should we deprecate this field? From what I can tell, it hasn't been probed correctly for a long
    # time. Need to go through the history to make sure I'm understanding this fully.
    image_encoder_model_id: str = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_has_weights_file(mod)

        cls._validate_has_image_encoder_metadata_file(mod)

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
    def _validate_has_weights_file(cls, mod: ModelOnDisk) -> None:
        weights_file = mod.path / "ip_adapter.bin"
        if not weights_file.exists():
            raise NotAMatchError("missing ip_adapter.bin weights file")

    @classmethod
    def _validate_has_image_encoder_metadata_file(cls, mod: ModelOnDisk) -> None:
        image_encoder_metadata_file = mod.path / "image_encoder.txt"
        if not image_encoder_metadata_file.exists():
            raise NotAMatchError("missing image_encoder.txt metadata file")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        state_dict = mod.load_state_dict()

        try:
            cross_attention_dim = state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[-1]
        except Exception as e:
            raise NotAMatchError(f"unable to determine cross attention dimension: {e}") from e

        match cross_attention_dim:
            case 768:
                return BaseModelType.StableDiffusion1
            case 1024:
                return BaseModelType.StableDiffusion2
            case 2048:
                return BaseModelType.StableDiffusionXL
            case _:
                raise NotAMatchError(f"unrecognized cross attention dimension {cross_attention_dim}")


class IPAdapter_InvokeAI_SD1_Config(IPAdapter_InvokeAI_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class IPAdapter_InvokeAI_SD2_Config(IPAdapter_InvokeAI_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class IPAdapter_InvokeAI_SDXL_Config(IPAdapter_InvokeAI_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class IPAdapter_Checkpoint_Config_Base(IPAdapter_Config_Base):
    """Model config for IP Adapter checkpoint format models."""

    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_ip_adapter(mod)

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
    def _validate_looks_like_ip_adapter(cls, mod: ModelOnDisk) -> None:
        if not state_dict_has_any_keys_starting_with(
            mod.load_state_dict(),
            {
                "image_proj.",
                "ip_adapter.",
                # XLabs FLUX IP-Adapter models have keys startinh with "ip_adapter_proj_model.".
                "ip_adapter_proj_model.",
            },
        ):
            raise NotAMatchError("model does not match Checkpoint IP Adapter heuristics")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        state_dict = mod.load_state_dict()

        if is_state_dict_xlabs_ip_adapter(state_dict):
            return BaseModelType.Flux

        try:
            cross_attention_dim = state_dict["ip_adapter.1.to_k_ip.weight"].shape[-1]
        except Exception as e:
            raise NotAMatchError(f"unable to determine cross attention dimension: {e}") from e

        match cross_attention_dim:
            case 768:
                return BaseModelType.StableDiffusion1
            case 1024:
                return BaseModelType.StableDiffusion2
            case 2048:
                return BaseModelType.StableDiffusionXL
            case _:
                raise NotAMatchError(f"unrecognized cross attention dimension {cross_attention_dim}")


class IPAdapter_Checkpoint_SD1_Config(IPAdapter_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class IPAdapter_Checkpoint_SD2_Config(IPAdapter_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class IPAdapter_Checkpoint_SDXL_Config(IPAdapter_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class IPAdapter_Checkpoint_FLUX_Config(IPAdapter_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
