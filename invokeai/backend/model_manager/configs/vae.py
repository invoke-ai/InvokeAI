import re
from typing import (
    Literal,
    Self,
)

from pydantic import Field
from typing_extensions import Any

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    common_config_paths,
    get_config_dict_or_raise,
    raise_for_class_name,
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

REGEX_TO_BASE: dict[str, BaseModelType] = {
    r"xl": BaseModelType.StableDiffusionXL,
    r"sd2": BaseModelType.StableDiffusion2,
    r"vae": BaseModelType.StableDiffusion1,
    r"FLUX.1-schnell_ae": BaseModelType.Flux,
}


class VAE_Checkpoint_Config_Base(Checkpoint_Config_Base):
    """Model config for standalone VAE models."""

    type: Literal[ModelType.VAE] = Field(default=ModelType.VAE)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_vae(mod)

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
    def _validate_looks_like_vae(cls, mod: ModelOnDisk) -> None:
        if not state_dict_has_any_keys_starting_with(
            mod.load_state_dict(),
            {
                "encoder.conv_in",
                "decoder.conv_in",
            },
        ):
            raise NotAMatchError("model does not match Checkpoint VAE heuristics")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        # Heuristic: VAEs of all architectures have a similar structure; the best we can do is guess based on name
        for regexp, base in REGEX_TO_BASE.items():
            if re.search(regexp, mod.path.name, re.IGNORECASE):
                return base

        raise NotAMatchError("cannot determine base type")


class VAE_Checkpoint_SD1_Config(VAE_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class VAE_Checkpoint_SD2_Config(VAE_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class VAE_Checkpoint_SDXL_Config(VAE_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class VAE_Checkpoint_FLUX_Config(VAE_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class VAE_Diffusers_Config_Base(Diffusers_Config_Base):
    """Model config for standalone VAE models (diffusers version)."""

    type: Literal[ModelType.VAE] = Field(default=ModelType.VAE)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "AutoencoderKL",
                "AutoencoderTiny",
            },
        )

        # Unfortunately it is difficult to distinguish SD1 and SDXL VAEs by config alone, so we may need to
        # guess based on name if the config is inconclusive.
        override_name = override_fields.get("name")
        cls._validate_base(mod, override_name)

        return cls(**override_fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk, override_name: str | None = None) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod, override_name)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _config_looks_like_sdxl(cls, config: dict[str, Any]) -> bool:
        # Heuristic: These config values that distinguish Stability's SD 1.x VAE from their SDXL VAE.
        return config.get("scaling_factor", 0) == 0.13025 and config.get("sample_size") in [512, 1024]

    @classmethod
    def _name_looks_like_sdxl(cls, mod: ModelOnDisk, override_name: str | None = None) -> bool:
        # Heuristic: SD and SDXL VAE are the same shape (3-channel RGB to 4-channel float scaled down
        # by a factor of 8), so we can't necessarily tell them apart by config hyperparameters. Best
        # we can do is guess based on name.
        return bool(re.search(r"xl\b", override_name or mod.path.name, re.IGNORECASE))

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk, override_name: str | None = None) -> BaseModelType:
        config_dict = get_config_dict_or_raise(common_config_paths(mod.path))
        if cls._config_looks_like_sdxl(config_dict):
            return BaseModelType.StableDiffusionXL
        elif cls._name_looks_like_sdxl(mod, override_name):
            return BaseModelType.StableDiffusionXL
        else:
            # TODO(psyche): Figure out how to positively identify SD1 here, and raise if we can't. Until then, YOLO.
            return BaseModelType.StableDiffusion1


class VAE_Diffusers_SD1_Config(VAE_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class VAE_Diffusers_SDXL_Config(VAE_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)
