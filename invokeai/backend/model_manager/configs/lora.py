from abc import ABC
from pathlib import Path
from typing import (
    Any,
    Literal,
    Self,
)

from pydantic import BaseModel, ConfigDict, Field

from invokeai.backend.model_manager.configs.base import (
    Config_Base,
)
from invokeai.backend.model_manager.configs.controlnet import ControlAdapterDefaultSettings
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
    state_dict_has_any_keys_ending_with,
    state_dict_has_any_keys_starting_with,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.omi import flux_dev_1_lora, stable_diffusion_xl_1_lora
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    FluxLoRAFormat,
    ModelFormat,
    ModelType,
)
from invokeai.backend.model_manager.util.model_util import lora_token_vector_length
from invokeai.backend.patches.lora_conversions.flux_control_lora_utils import is_state_dict_likely_flux_control


class LoraModelDefaultSettings(BaseModel):
    weight: float | None = Field(default=None, ge=-1, le=2, description="Default weight for this model")
    model_config = ConfigDict(extra="forbid")


class LoRA_Config_Base(ABC, BaseModel):
    """Base class for LoRA models."""

    type: Literal[ModelType.LoRA] = Field(default=ModelType.LoRA)
    trigger_phrases: set[str] | None = Field(
        default=None,
        description="Set of trigger phrases for this model",
    )
    default_settings: LoraModelDefaultSettings | None = Field(
        default=None,
        description="Default settings for this model",
    )


def _get_flux_lora_format(mod: ModelOnDisk) -> FluxLoRAFormat | None:
    # TODO(psyche): Moving this import to the function to avoid circular imports. Refactor later.
    from invokeai.backend.patches.lora_conversions.formats import flux_format_from_state_dict

    state_dict = mod.load_state_dict()
    value = flux_format_from_state_dict(state_dict, mod.metadata())
    return value


class LoRA_OMI_Config_Base(LoRA_Config_Base):
    format: Literal[ModelFormat.OMI] = Field(default=ModelFormat.OMI)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_omi_lora(mod)

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
    def _validate_looks_like_omi_lora(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model metadata does not look like an OMI LoRA."""
        flux_format = _get_flux_lora_format(mod)
        if flux_format in [FluxLoRAFormat.Control, FluxLoRAFormat.Diffusers]:
            raise NotAMatchError("model looks like ControlLoRA or Diffusers LoRA")

        metadata = mod.metadata()

        metadata_looks_like_omi_lora = (
            bool(metadata.get("modelspec.sai_model_spec"))
            and metadata.get("ot_branch") == "omi_format"
            and metadata.get("modelspec.architecture", "").split("/")[1].lower() == "lora"
        )

        if not metadata_looks_like_omi_lora:
            raise NotAMatchError("metadata does not look like OMI LoRA")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> Literal[BaseModelType.Flux, BaseModelType.StableDiffusionXL]:
        metadata = mod.metadata()
        architecture = metadata["modelspec.architecture"]

        if architecture == stable_diffusion_xl_1_lora:
            return BaseModelType.StableDiffusionXL
        elif architecture == flux_dev_1_lora:
            return BaseModelType.Flux
        else:
            raise NotAMatchError(f"unrecognised/unsupported architecture for OMI LoRA: {architecture}")


class LoRA_OMI_SDXL_Config(LoRA_OMI_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class LoRA_OMI_FLUX_Config(LoRA_OMI_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class LoRA_LyCORIS_Config_Base(LoRA_Config_Base):
    """Model config for LoRA/Lycoris models."""

    type: Literal[ModelType.LoRA] = Field(default=ModelType.LoRA)
    format: Literal[ModelFormat.LyCORIS] = Field(default=ModelFormat.LyCORIS)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_lora(mod)

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
    def _validate_looks_like_lora(cls, mod: ModelOnDisk) -> None:
        # First rule out ControlLoRA and Diffusers LoRA
        flux_format = _get_flux_lora_format(mod)
        if flux_format in [FluxLoRAFormat.Control]:
            raise NotAMatchError("model looks like Control LoRA")

        # Note: Existence of these key prefixes/suffixes does not guarantee that this is a LoRA.
        # Some main models have these keys, likely due to the creator merging in a LoRA.
        has_key_with_lora_prefix = state_dict_has_any_keys_starting_with(
            mod.load_state_dict(),
            {
                "lora_te_",
                "lora_unet_",
                "lora_te1_",
                "lora_te2_",
                "lora_transformer_",
            },
        )

        has_key_with_lora_suffix = state_dict_has_any_keys_ending_with(
            mod.load_state_dict(),
            {
                "to_k_lora.up.weight",
                "to_q_lora.down.weight",
                "lora_A.weight",
                "lora_B.weight",
            },
        )

        if not has_key_with_lora_prefix and not has_key_with_lora_suffix:
            raise NotAMatchError("model does not match LyCORIS LoRA heuristics")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        if _get_flux_lora_format(mod):
            return BaseModelType.Flux

        state_dict = mod.load_state_dict()
        # If we've gotten here, we assume that the model is a Stable Diffusion model
        token_vector_length = lora_token_vector_length(state_dict)
        if token_vector_length == 768:
            return BaseModelType.StableDiffusion1
        elif token_vector_length == 1024:
            return BaseModelType.StableDiffusion2
        elif token_vector_length == 1280:
            return BaseModelType.StableDiffusionXL  # recognizes format at https://civitai.com/models/224641
        elif token_vector_length == 2048:
            return BaseModelType.StableDiffusionXL
        else:
            raise NotAMatchError(f"unrecognized token vector length {token_vector_length}")


class LoRA_LyCORIS_SD1_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class LoRA_LyCORIS_SD2_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class LoRA_LyCORIS_SDXL_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class LoRA_LyCORIS_FLUX_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class ControlAdapter_Config_Base(ABC, BaseModel):
    default_settings: ControlAdapterDefaultSettings | None = Field(None)


class ControlLoRA_LyCORIS_FLUX_Config(ControlAdapter_Config_Base, Config_Base):
    """Model config for Control LoRA models."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    type: Literal[ModelType.ControlLoRa] = Field(default=ModelType.ControlLoRa)
    format: Literal[ModelFormat.LyCORIS] = Field(default=ModelFormat.LyCORIS)

    trigger_phrases: set[str] | None = Field(None)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_control_lora(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_looks_like_control_lora(cls, mod: ModelOnDisk) -> None:
        state_dict = mod.load_state_dict()

        if not is_state_dict_likely_flux_control(state_dict):
            raise NotAMatchError("model state dict does not look like a Flux Control LoRA")


class LoRA_Diffusers_Config_Base(LoRA_Config_Base):
    """Model config for LoRA/Diffusers models."""

    # TODO(psyche): Needs base handling. For FLUX, the Diffusers format does not indicate a folder model; it indicates
    # the weights format. FLUX Diffusers LoRAs are single files.

    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

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
        if _get_flux_lora_format(mod):
            return BaseModelType.Flux

        # If we've gotten here, we assume that the LoRA is a Stable Diffusion LoRA
        path_to_weight_file = cls._get_weight_file_or_raise(mod)
        state_dict = mod.load_state_dict(path_to_weight_file)
        token_vector_length = lora_token_vector_length(state_dict)

        match token_vector_length:
            case 768:
                return BaseModelType.StableDiffusion1
            case 1024:
                return BaseModelType.StableDiffusion2
            case 1280:
                return BaseModelType.StableDiffusionXL  # recognizes format at https://civitai.com/models/224641
            case 2048:
                return BaseModelType.StableDiffusionXL
            case _:
                raise NotAMatchError(f"unrecognized token vector length {token_vector_length}")

    @classmethod
    def _get_weight_file_or_raise(cls, mod: ModelOnDisk) -> Path:
        suffixes = ["bin", "safetensors"]
        weight_files = [mod.path / f"pytorch_lora_weights.{sfx}" for sfx in suffixes]
        for wf in weight_files:
            if wf.exists():
                return wf
        raise NotAMatchError("missing pytorch_lora_weights.bin or pytorch_lora_weights.safetensors")


class LoRA_Diffusers_SD1_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class LoRA_Diffusers_SD2_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class LoRA_Diffusers_SDXL_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class LoRA_Diffusers_FLUX_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
