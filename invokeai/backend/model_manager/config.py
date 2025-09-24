# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Configuration definitions for image generation models.

Typical usage:

  from invokeai.backend.model_manager import ModelConfigFactory
  raw = dict(path='models/sd-1/main/foo.ckpt',
             name='foo',
             base='sd-1',
             type='main',
             config='configs/stable-diffusion/v1-inference.yaml',
             variant='normal',
             format='checkpoint'
            )
  config = ModelConfigFactory.make_config(raw)
  print(config.name)

Validation errors will raise an InvalidModelConfigException error.

"""

# pyright: reportIncompatibleVariableOverride=false
import json
import logging
import re
import time
from abc import ABC
from enum import Enum
from inspect import isabstract
from pathlib import Path
from typing import (
    ClassVar,
    Literal,
    Optional,
    Self,
    Type,
    TypeAlias,
    Union,
)

import torch
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, TypeAdapter, ValidationError
from typing_extensions import Annotated, Any, Dict

from invokeai.app.services.config.config_default import get_config
from invokeai.app.util.misc import uuid_string
from invokeai.backend.model_hash.hash_validator import validate_hash
from invokeai.backend.model_hash.model_hash import HASHING_ALGORITHMS
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.omi import flux_dev_1_lora, stable_diffusion_xl_1_lora
from invokeai.backend.model_manager.taxonomy import (
    AnyVariant,
    BaseModelType,
    ClipVariantType,
    FluxLoRAFormat,
    FluxVariantType,
    ModelFormat,
    ModelRepoVariant,
    ModelSourceType,
    ModelType,
    ModelVariantType,
    SchedulerPredictionType,
    SubModelType,
    variant_type_adapter,
)
from invokeai.backend.model_manager.util.model_util import lora_token_vector_length
from invokeai.backend.spandrel_image_to_image_model import SpandrelImageToImageModel
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES

logger = logging.getLogger(__name__)
app_config = get_config()


class InvalidModelConfigException(Exception):
    """Exception for when config parser doesn't recognize this combination of model type and format."""

    pass


class NotAMatch(Exception):
    """Exception for when a model does not match a config class.

    Args:
        config_class: The config class that was being tested.
        reason: The reason why the model did not match.
    """

    def __init__(
        self,
        config_class: type,
        reason: str,
    ):
        super().__init__(f"{config_class.__name__}: {reason}")


DEFAULTS_PRECISION = Literal["fp16", "fp32"]

# These utility functions are tightly coupled to the config classes below in order to make the process of raising
# NotAMatch exceptions as easy and consistent as possible.


def _get_config_or_raise(
    config_class: type,
    config_path: Path,
) -> dict[str, Any]:
    """Load the config file at the given path, or raise NotAMatch if it cannot be loaded."""
    if not config_path.exists():
        raise NotAMatch(config_class, f"missing config file: {config_path}")

    try:
        config = load_json(config_path)
        return config
    except Exception as e:
        raise NotAMatch(config_class, f"unable to load config file: {config_path}") from e


def _validate_class_names(
    config_class: type,
    config_path: Path,
    valid_class_names: set[str],
) -> None:
    """Raise NotAMatch if the config file is missing or does not contain a valid class name."""

    config = _get_config_or_raise(config_class, config_path)

    try:
        if "_class_name" in config:
            config_class_name = config["_class_name"]
        elif "architectures" in config:
            config_class_name = config["architectures"][0]
        else:
            raise ValueError("missing _class_name or architectures field")
    except Exception as e:
        raise NotAMatch(config_class, f"unable to determine class name from config file: {config_path}") from e

    if config_class_name not in valid_class_names:
        raise NotAMatch(config_class, f"model class is not one of {valid_class_names}, got {config_class_name}")


def _validate_overrides(
    config_class: type,
    provided_overrides: dict[str, Any],
    valid_overrides: dict[str, Any],
) -> bool:
    """Check if the provided overrides match the valid overrides for this config class.

    Args:
        config_class: The config class that is being tested.
        provided_overrides: The overrides provided by the user.
        valid_overrides: The overrides that are valid for this config class.

    Returns:
        True if all provided overrides match the valid overrides, False if some valid overrides are missing.

    Raises:
        NotAMatch if any override does not match the allowed value.
    """
    is_perfect_match = True
    for key, value in valid_overrides.items():
        if key not in provided_overrides:
            is_perfect_match = False
            continue
        if provided_overrides[key] != value:
            raise NotAMatch(
                config_class,
                f"override {key}={provided_overrides[key]} does not match required value {key}={value}",
            )

    return is_perfect_match


class SubmodelDefinition(BaseModel):
    path_or_prefix: str
    model_type: ModelType
    variant: AnyVariant | None = None

    model_config = ConfigDict(protected_namespaces=())


class MainModelDefaultSettings(BaseModel):
    vae: str | None = Field(default=None, description="Default VAE for this model (model key)")
    vae_precision: DEFAULTS_PRECISION | None = Field(default=None, description="Default VAE precision for this model")
    scheduler: SCHEDULER_NAME_VALUES | None = Field(default=None, description="Default scheduler for this model")
    steps: int | None = Field(default=None, gt=0, description="Default number of steps for this model")
    cfg_scale: float | None = Field(default=None, ge=1, description="Default CFG Scale for this model")
    cfg_rescale_multiplier: float | None = Field(
        default=None, ge=0, lt=1, description="Default CFG Rescale Multiplier for this model"
    )
    width: int | None = Field(default=None, multiple_of=8, ge=64, description="Default width for this model")
    height: int | None = Field(default=None, multiple_of=8, ge=64, description="Default height for this model")
    guidance: float | None = Field(default=None, ge=1, description="Default Guidance for this model")

    model_config = ConfigDict(extra="forbid")


class LoraModelDefaultSettings(BaseModel):
    weight: float | None = Field(default=None, ge=-1, le=2, description="Default weight for this model")
    model_config = ConfigDict(extra="forbid")


class ControlAdapterDefaultSettings(BaseModel):
    # This could be narrowed to controlnet processor nodes, but they change. Leaving this a string is safer.
    preprocessor: str | None
    model_config = ConfigDict(extra="forbid")


class LegacyProbeMixin:
    """Mixin for classes using the legacy probe for model classification."""

    @classmethod
    def matches(cls, *args, **kwargs):
        raise NotImplementedError(f"Method 'matches' not implemented for {cls.__name__}")

    @classmethod
    def parse(cls, *args, **kwargs):
        raise NotImplementedError(f"Method 'parse' not implemented for {cls.__name__}")


class ModelConfigBase(ABC, BaseModel):
    """
    Abstract Base class for model configurations.

    To create a new config type, inherit from this class and implement its interface:
    - (mandatory) override methods 'matches' and 'parse'
    - (mandatory) define fields 'type' and 'format' as class attributes

    - (optional) override method 'get_tag'
    - (optional) override field _MATCH_SPEED

    See MinimalConfigExample in test_model_probe.py for an example implementation.
    """

    @staticmethod
    def json_schema_extra(schema: dict[str, Any]) -> None:
        schema["required"].extend(["key", "base", "type", "format"])

    model_config = ConfigDict(validate_assignment=True, json_schema_extra=json_schema_extra)

    key: str = Field(description="A unique key for this model.", default_factory=uuid_string)
    hash: str = Field(description="The hash of the model file(s).")
    path: str = Field(
        description="Path to the model on the filesystem. Relative paths are relative to the Invoke root directory."
    )
    file_size: int = Field(description="The size of the model in bytes.")
    name: str = Field(description="Name of the model.")
    type: ModelType = Field(description="Model type")
    format: ModelFormat = Field(description="Model format")
    base: BaseModelType = Field(description="The base model.")
    source: str = Field(description="The original source of the model (path, URL or repo_id).")
    source_type: ModelSourceType = Field(description="The type of source")

    description: Optional[str] = Field(description="Model description", default=None)
    source_api_response: Optional[str] = Field(
        description="The original API response from the source, as stringified JSON.", default=None
    )
    cover_image: Optional[str] = Field(description="Url for image to preview model", default=None)
    submodels: Optional[Dict[SubModelType, SubmodelDefinition]] = Field(
        description="Loadable submodels in this model", default=None
    )
    usage_info: Optional[str] = Field(default=None, description="Usage information for this model")

    USING_LEGACY_PROBE: ClassVar[set[Type["AnyModelConfig"]]] = set()
    USING_CLASSIFY_API: ClassVar[set[Type["AnyModelConfig"]]] = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if issubclass(cls, LegacyProbeMixin):
            ModelConfigBase.USING_LEGACY_PROBE.add(cls)
        else:
            ModelConfigBase.USING_CLASSIFY_API.add(cls)

    @staticmethod
    def all_config_classes():
        subclasses = ModelConfigBase.USING_LEGACY_PROBE | ModelConfigBase.USING_CLASSIFY_API
        concrete = {cls for cls in subclasses if not isabstract(cls)}
        return concrete

    @classmethod
    def get_tag(cls) -> Tag:
        type = cls.model_fields["type"].default.value
        format = cls.model_fields["format"].default.value
        return Tag(f"{type}.{format}")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        """Given the model on disk and any overrides, return an instance of this config class.

        Implementations should raise NotAMatch if the model does not match this config class."""
        raise NotImplementedError(f"from_model_on_disk not implemented for {cls.__name__}")


class UnknownModelConfig(ModelConfigBase):
    base: Literal[BaseModelType.Unknown] = BaseModelType.Unknown
    type: Literal[ModelType.Unknown] = ModelType.Unknown
    format: Literal[ModelFormat.Unknown] = ModelFormat.Unknown

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        raise NotAMatch(cls, "unknown model config cannot match any model")


class CheckpointConfigBase(ABC, BaseModel):
    """Base class for checkpoint-style models."""

    format: Literal[ModelFormat.Checkpoint, ModelFormat.BnbQuantizednf4b, ModelFormat.GGUFQuantized] = Field(
        description="Format of the provided checkpoint model",
        default=ModelFormat.Checkpoint,
    )
    config_path: str | None = Field(
        description="path to the checkpoint model config file",
        default=None,
    )
    converted_at: float | None = Field(
        description="When this model was last converted to diffusers",
        default_factory=time.time,
    )


class DiffusersConfigBase(ABC, BaseModel):
    """Base class for diffusers-style models."""

    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers
    repo_variant: Optional[ModelRepoVariant] = ModelRepoVariant.Default


class LoRAConfigBase(ABC, BaseModel):
    """Base class for LoRA models."""

    type: Literal[ModelType.LoRA] = ModelType.LoRA
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)
    default_settings: Optional[LoraModelDefaultSettings] = Field(
        description="Default settings for this model", default=None
    )

    @classmethod
    def flux_lora_format(cls, mod: ModelOnDisk):
        key = "FLUX_LORA_FORMAT"
        if key in mod.cache:
            return mod.cache[key]

        from invokeai.backend.patches.lora_conversions.formats import flux_format_from_state_dict

        sd = mod.load_state_dict(mod.path)
        value = flux_format_from_state_dict(sd, mod.metadata())
        mod.cache[key] = value
        return value

    @classmethod
    def base_model(cls, mod: ModelOnDisk) -> BaseModelType:
        if cls.flux_lora_format(mod):
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
            raise InvalidModelConfigException("Unknown LoRA type")


class T5EncoderConfigBase(ABC, BaseModel):
    """Base class for diffusers-style models."""

    base: Literal[BaseModelType.Any] = BaseModelType.Any
    type: Literal[ModelType.T5Encoder] = ModelType.T5Encoder


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as file:
        return json.load(file)


class T5EncoderConfig(T5EncoderConfigBase, ModelConfigBase):
    format: Literal[ModelFormat.T5Encoder] = ModelFormat.T5Encoder

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.T5Encoder,
        "format": ModelFormat.T5Encoder,
    }

    VALID_CLASS_NAMES: ClassVar = {
        "T5EncoderModel",
    }

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        if mod.path.is_file():
            raise NotAMatch(cls, "model path is a file, not a directory")

        _validate_class_names(
            config_class=cls,
            config_path=mod.path / "text_encoder_2" / "config.json",
            valid_class_names=cls.VALID_CLASS_NAMES,
        )

        # Heuristic: Look for the presence of the unquantized config file (not present for bnb-quantized models)
        has_unquantized_config = (mod.path / "text_encoder_2" / "model.safetensors.index.json").exists()

        if not has_unquantized_config:
            raise NotAMatch(cls, "missing text_encoder_2/model.safetensors.index.json")

        return cls(**fields)


class T5EncoderBnbQuantizedLlmInt8bConfig(T5EncoderConfigBase, ModelConfigBase):
    format: Literal[ModelFormat.BnbQuantizedLlmInt8b] = ModelFormat.BnbQuantizedLlmInt8b

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.T5Encoder,
        "format": ModelFormat.BnbQuantizedLlmInt8b,
    }

    VALID_CLASS_NAMES: ClassVar = {
        "T5EncoderModel",
    }

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        # Heuristic: Look for the T5EncoderModel class name in the config
        _validate_class_names(
            config_class=cls,
            config_path=mod.path / "text_encoder_2" / "config.json",
            valid_class_names=cls.VALID_CLASS_NAMES,
        )

        # Heuristic: look for the quantization in the filename name
        filename_looks_like_bnb = any(x for x in mod.weight_files() if "llm_int8" in x.as_posix())

        # Heuristic: Look for the presence of "SCB" suffixes in state dict keys
        has_scb_key_suffix = mod.has_keys_ending_with("SCB")

        if not filename_looks_like_bnb and not has_scb_key_suffix:
            raise NotAMatch(cls, "missing bnb quantization indicators")

        return cls(**fields)


class LoRAOmiConfig(LoRAConfigBase, ModelConfigBase):
    format: Literal[ModelFormat.OMI] = ModelFormat.OMI

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.LoRA,
        "format": ModelFormat.OMI,
    }

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        # Heuristic: OMI LoRAs are always files, never directories
        if mod.path.is_dir():
            raise NotAMatch(cls, "model path is a directory, not a file")

        # Heuristic: differential diagnosis vs ControlLoRA and Diffusers
        if cls.flux_lora_format(mod) in [FluxLoRAFormat.Control, FluxLoRAFormat.Diffusers]:
            raise NotAMatch(cls, "model is a ControlLoRA or Diffusers LoRA")

        # Heuristic: Look for OMI LoRA metadata
        metadata = mod.metadata()
        is_omi_lora_heuristic = (
            bool(metadata.get("modelspec.sai_model_spec"))
            and metadata.get("ot_branch") == "omi_format"
            and metadata.get("modelspec.architecture", "").split("/")[1].lower() == "lora"
        )

        if not is_omi_lora_heuristic:
            raise NotAMatch(cls, "model does not match OMI LoRA heuristics")

        base = fields.get("base") or cls._get_base_or_raise(mod)

        return cls(**fields, base=base)

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        metadata = mod.metadata()
        architecture = metadata["modelspec.architecture"]

        if architecture == stable_diffusion_xl_1_lora:
            return BaseModelType.StableDiffusionXL
        elif architecture == flux_dev_1_lora:
            return BaseModelType.Flux
        else:
            raise NotAMatch(cls, f"unrecognised/unsupported architecture for OMI LoRA: {architecture}")


class LoRALyCORISConfig(LoRAConfigBase, ModelConfigBase):
    """Model config for LoRA/Lycoris models."""

    format: Literal[ModelFormat.LyCORIS] = ModelFormat.LyCORIS

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.LoRA,
        "format": ModelFormat.LyCORIS,
    }

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        # Heuristic: LyCORIS LoRAs are always files, never directories
        if mod.path.is_dir():
            raise NotAMatch(cls, "model path is a directory, not a file")

        # Heuristic: differential diagnosis vs ControlLoRA and Diffusers
        if cls.flux_lora_format(mod) in [FluxLoRAFormat.Control, FluxLoRAFormat.Diffusers]:
            raise NotAMatch(cls, "model is a ControlLoRA or Diffusers LoRA")

        # Note: Existence of these key prefixes/suffixes does not guarantee that this is a LoRA.
        # Some main models have these keys, likely due to the creator merging in a LoRA.
        has_key_with_lora_prefix = mod.has_keys_starting_with(
            {
                "lora_te_",
                "lora_unet_",
                "lora_te1_",
                "lora_te2_",
                "lora_transformer_",
            }
        )

        has_key_with_lora_suffix = mod.has_keys_ending_with(
            {
                "to_k_lora.up.weight",
                "to_q_lora.down.weight",
                "lora_A.weight",
                "lora_B.weight",
            }
        )

        if not has_key_with_lora_prefix and not has_key_with_lora_suffix:
            raise NotAMatch(cls, "model does not match LyCORIS LoRA heuristics")

        return cls(**fields)


class ControlAdapterConfigBase(ABC, BaseModel):
    default_settings: Optional[ControlAdapterDefaultSettings] = Field(
        description="Default settings for this model", default=None
    )


class ControlLoRALyCORISConfig(ControlAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for Control LoRA models."""

    type: Literal[ModelType.ControlLoRa] = ModelType.ControlLoRa
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)
    format: Literal[ModelFormat.LyCORIS] = ModelFormat.LyCORIS


class ControlLoRADiffusersConfig(ControlAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for Control LoRA models."""

    type: Literal[ModelType.ControlLoRa] = ModelType.ControlLoRa
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class LoRADiffusersConfig(LoRAConfigBase, ModelConfigBase):
    """Model config for LoRA/Diffusers models."""

    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.LoRA,
        "format": ModelFormat.Diffusers,
    }

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        # Heuristic: Diffusers LoRAs are always directories, never files
        if mod.path.is_file():
            raise NotAMatch(cls, "model path is a file, not a directory")

        is_flux_lora_diffusers = cls.flux_lora_format(mod) == FluxLoRAFormat.Diffusers

        suffixes = ["bin", "safetensors"]
        weight_files = [mod.path / f"pytorch_lora_weights.{sfx}" for sfx in suffixes]
        has_lora_weight_file = any(wf.exists() for wf in weight_files)

        if not is_flux_lora_diffusers and not has_lora_weight_file:
            raise NotAMatch(cls, "model does not match Diffusers LoRA heuristics")

        return cls(**fields)


class VAEConfigBase(ABC, BaseModel):
    type: Literal[ModelType.VAE] = ModelType.VAE


class VAECheckpointConfig(VAEConfigBase, CheckpointConfigBase, ModelConfigBase):
    """Model config for standalone VAE models."""

    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.VAE,
        "format": ModelFormat.Checkpoint,
    }

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        if mod.path.is_dir():
            raise NotAMatch(cls, "model path is a directory, not a file")

        if not mod.has_keys_starting_with({"encoder.conv_in", "decoder.conv_in"}):
            raise NotAMatch(cls, "model does not match Checkpoint VAE heuristics")

        base = fields.get("base") or cls._get_base_or_raise(mod)
        return cls(**fields, base=base)

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        # Heuristic: VAEs of all architectures have a similar structure; the best we can do is guess based on name
        for regexp, basetype in [
            (r"xl", BaseModelType.StableDiffusionXL),
            (r"sd2", BaseModelType.StableDiffusion2),
            (r"vae", BaseModelType.StableDiffusion1),
            (r"FLUX.1-schnell_ae", BaseModelType.Flux),
        ]:
            if re.search(regexp, mod.path.name, re.IGNORECASE):
                return basetype

        raise NotAMatch(cls, "cannot determine base type")


class VAEDiffusersConfig(VAEConfigBase, DiffusersConfigBase, ModelConfigBase):
    """Model config for standalone VAE models (diffusers version)."""

    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.VAE,
        "format": ModelFormat.Diffusers,
    }
    VALID_CLASS_NAMES: ClassVar = {
        "AutoencoderKL",
        "AutoencoderTiny",
    }

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        if mod.path.is_file():
            raise NotAMatch(cls, "model path is a file, not a directory")

        _validate_class_names(
            config_class=cls,
            config_path=mod.path / "config.json",
            valid_class_names=cls.VALID_CLASS_NAMES,
        )

        base = fields.get("base") or cls._get_base_or_raise(mod)
        return cls(**fields, base=base)

    @classmethod
    def _config_looks_like_sdxl(cls, config: dict[str, Any]) -> bool:
        # Heuristic: These config values that distinguish Stability's SD 1.x VAE from their SDXL VAE.
        return config.get("scaling_factor", 0) == 0.13025 and config.get("sample_size") in [512, 1024]

    @classmethod
    def _name_looks_like_sdxl(cls, mod: ModelOnDisk) -> bool:
        # Heuristic: SD and SDXL VAE are the same shape (3-channel RGB to 4-channel float scaled down
        # by a factor of 8), so we can't necessarily tell them apart by config hyperparameters. Best
        # we can do is guess based on name.
        return bool(re.search(r"xl\b", cls._guess_name(mod), re.IGNORECASE))

    @classmethod
    def _guess_name(cls, mod: ModelOnDisk) -> str:
        name = mod.path.name
        if name == "vae":
            name = mod.path.parent.name
        return name

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        config = _get_config_or_raise(cls, mod.path / "config.json")
        if cls._config_looks_like_sdxl(config):
            return BaseModelType.StableDiffusionXL
        elif cls._name_looks_like_sdxl(mod):
            return BaseModelType.StableDiffusionXL
        else:
            # TODO(psyche): Figure out how to positively identify SD1 here, and raise if we can't. Until then, YOLO.
            return BaseModelType.StableDiffusion1


class ControlNetDiffusersConfig(DiffusersConfigBase, ControlAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = ModelType.ControlNet
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class ControlNetCheckpointConfig(CheckpointConfigBase, ControlAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = ModelType.ControlNet


class TextualInversionConfigBase(ABC, BaseModel):
    type: Literal[ModelType.TextualInversion] = ModelType.TextualInversion

    KNOWN_KEYS: ClassVar = {"string_to_param", "emb_params", "clip_g"}

    @classmethod
    def _file_looks_like_embedding(cls, mod: ModelOnDisk, path: Path | None = None) -> bool:
        try:
            p = path or mod.path

            if not p.exists():
                return False

            if p.is_dir():
                return False

            if p.name in [f"learned_embeds.{s}" for s in mod.weight_files()]:
                return True

            state_dict = mod.load_state_dict(p)

            # Heuristic: textual inversion embeddings have these keys
            if any(key in cls.KNOWN_KEYS for key in state_dict.keys()):
                return True

            # Heuristic: small state dict with all tensor values
            if (len(state_dict)) < 10 and all(isinstance(v, torch.Tensor) for v in state_dict.values()):
                return True

            return False
        except Exception:
            return False

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk, path: Path | None = None) -> BaseModelType:
        p = path or mod.path

        try:
            state_dict = mod.load_state_dict(p)
        except Exception as e:
            raise NotAMatch(cls, f"unable to load state dict from {p}: {e}") from e

        try:
            if "string_to_token" in state_dict:
                token_dim = list(state_dict["string_to_param"].values())[0].shape[-1]
            elif "emb_params" in state_dict:
                token_dim = state_dict["emb_params"].shape[-1]
            elif "clip_g" in state_dict:
                token_dim = state_dict["clip_g"].shape[-1]
            else:
                token_dim = list(state_dict.values())[0].shape[0]
        except Exception as e:
            raise NotAMatch(cls, f"unable to determine token dimension from state dict in {p}: {e}") from e

        match token_dim:
            case 768:
                return BaseModelType.StableDiffusion1
            case 1024:
                return BaseModelType.StableDiffusion2
            case 1280:
                return BaseModelType.StableDiffusionXL
            case _:
                raise NotAMatch(cls, f"unrecognized token dimension {token_dim}")


class TextualInversionFileConfig(TextualInversionConfigBase, ModelConfigBase):
    """Model config for textual inversion embeddings."""

    format: Literal[ModelFormat.EmbeddingFile] = ModelFormat.EmbeddingFile

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.TextualInversion,
        "format": ModelFormat.EmbeddingFile,
    }

    @classmethod
    def get_tag(cls) -> Tag:
        return Tag(f"{ModelType.TextualInversion.value}.{ModelFormat.EmbeddingFile.value}")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        if mod.path.is_dir():
            raise NotAMatch(cls, "model path is a directory, not a file")

        if not cls._file_looks_like_embedding(mod):
            raise NotAMatch(cls, "model does not look like a textual inversion embedding file")

        base = fields.get("base") or cls._get_base_or_raise(mod)
        return cls(**fields, base=base)


class TextualInversionFolderConfig(TextualInversionConfigBase, ModelConfigBase):
    """Model config for textual inversion embeddings."""

    format: Literal[ModelFormat.EmbeddingFolder] = ModelFormat.EmbeddingFolder

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.TextualInversion,
        "format": ModelFormat.EmbeddingFolder,
    }

    @classmethod
    def get_tag(cls) -> Tag:
        return Tag(f"{ModelType.TextualInversion.value}.{ModelFormat.EmbeddingFolder.value}")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        if mod.path.is_file():
            raise NotAMatch(cls, "model path is a file, not a directory")

        for p in mod.weight_files():
            if cls._file_looks_like_embedding(mod, p):
                base = fields.get("base") or cls._get_base_or_raise(mod, p)
                return cls(**fields, base=base)

        raise NotAMatch(cls, "model does not look like a textual inversion embedding folder")


class MainConfigBase(ABC, BaseModel):
    type: Literal[ModelType.Main] = ModelType.Main
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)
    default_settings: Optional[MainModelDefaultSettings] = Field(
        description="Default settings for this model", default=None
    )
    variant: ModelVariantType | FluxVariantType = ModelVariantType.Normal


class VideoConfigBase(ABC, BaseModel):
    type: Literal[ModelType.Video] = ModelType.Video
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)
    default_settings: Optional[MainModelDefaultSettings] = Field(
        description="Default settings for this model", default=None
    )
    variant: ModelVariantType = ModelVariantType.Normal


class MainCheckpointConfig(CheckpointConfigBase, MainConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for main checkpoint models."""

    prediction_type: SchedulerPredictionType = SchedulerPredictionType.Epsilon
    upcast_attention: bool = False


class MainBnbQuantized4bCheckpointConfig(CheckpointConfigBase, MainConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for main checkpoint models."""

    format: Literal[ModelFormat.BnbQuantizednf4b] = ModelFormat.BnbQuantizednf4b
    prediction_type: SchedulerPredictionType = SchedulerPredictionType.Epsilon
    upcast_attention: bool = False


class MainGGUFCheckpointConfig(CheckpointConfigBase, MainConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for main checkpoint models."""

    format: Literal[ModelFormat.GGUFQuantized] = ModelFormat.GGUFQuantized
    prediction_type: SchedulerPredictionType = SchedulerPredictionType.Epsilon
    upcast_attention: bool = False


class MainDiffusersConfig(DiffusersConfigBase, MainConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for main diffusers models."""

    pass


class IPAdapterConfigBase(ABC, BaseModel):
    type: Literal[ModelType.IPAdapter] = ModelType.IPAdapter


class IPAdapterInvokeAIConfig(IPAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for IP Adapter diffusers format models."""

    # TODO(ryand): Should we deprecate this field? From what I can tell, it hasn't been probed correctly for a long
    # time. Need to go through the history to make sure I'm understanding this fully.
    image_encoder_model_id: str
    format: Literal[ModelFormat.InvokeAI] = ModelFormat.InvokeAI


class IPAdapterCheckpointConfig(IPAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for IP Adapter checkpoint format models."""

    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint


class CLIPEmbedDiffusersConfig(DiffusersConfigBase):
    """Model config for Clip Embeddings."""

    variant: ClipVariantType = Field(...)
    type: Literal[ModelType.CLIPEmbed] = ModelType.CLIPEmbed
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers
    base: Literal[BaseModelType.Any] = BaseModelType.Any

    VALID_CLASS_NAMES: ClassVar = {
        "CLIPModel",
        "CLIPTextModel",
        "CLIPTextModelWithProjection",
    }

    @classmethod
    def get_clip_variant_type(cls, config: dict[str, Any]) -> ClipVariantType | None:
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


class CLIPGEmbedDiffusersConfig(CLIPEmbedDiffusersConfig, ModelConfigBase):
    """Model config for CLIP-G Embeddings."""

    variant: Literal[ClipVariantType.G] = ClipVariantType.G

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.CLIPEmbed,
        "format": ModelFormat.Diffusers,
        "variant": ClipVariantType.G,
    }

    @classmethod
    def get_tag(cls) -> Tag:
        return Tag(f"{ModelType.CLIPEmbed.value}.{ModelFormat.Diffusers.value}.{ClipVariantType.G.value}")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        if mod.path.is_file():
            raise NotAMatch(cls, "model path is a file, not a directory")

        config_path = mod.path / "config.json"

        _validate_class_names(
            config_class=cls,
            config_path=config_path,
            valid_class_names=cls.VALID_CLASS_NAMES,
        )

        config = _get_config_or_raise(cls, config_path)

        clip_variant = cls.get_clip_variant_type(config)

        if clip_variant is not ClipVariantType.G:
            raise NotAMatch(cls, "model does not match CLIP-G heuristics")

        return cls(**fields)


class CLIPLEmbedDiffusersConfig(CLIPEmbedDiffusersConfig, ModelConfigBase):
    """Model config for CLIP-L Embeddings."""

    variant: Literal[ClipVariantType.L] = ClipVariantType.L

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.CLIPEmbed,
        "format": ModelFormat.Diffusers,
        "variant": ClipVariantType.L,
    }

    @classmethod
    def get_tag(cls) -> Tag:
        return Tag(f"{ModelType.CLIPEmbed.value}.{ModelFormat.Diffusers.value}.{ClipVariantType.L.value}")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        if mod.path.is_file():
            raise NotAMatch(cls, "model path is a file, not a directory")

        config_path = mod.path / "config.json"

        _validate_class_names(
            config_class=cls,
            config_path=config_path,
            valid_class_names=cls.VALID_CLASS_NAMES,
        )

        config = _get_config_or_raise(cls, config_path)
        clip_variant = cls.get_clip_variant_type(config)

        if clip_variant is not ClipVariantType.L:
            raise NotAMatch(cls, "model does not match CLIP-L heuristics")

        return cls(**fields)


class CLIPVisionDiffusersConfig(DiffusersConfigBase, ModelConfigBase):
    """Model config for CLIPVision."""

    type: Literal[ModelType.CLIPVision] = ModelType.CLIPVision
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.CLIPVision,
        "format": ModelFormat.Diffusers,
    }

    VALID_CLASS_NAMES: ClassVar = {
        "CLIPVisionModelWithProjection",
    }

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        if mod.path.is_file():
            raise NotAMatch(cls, "model path is a file, not a directory")

        config_path = mod.path / "config.json"

        _validate_class_names(
            config_class=cls,
            config_path=config_path,
            valid_class_names=cls.VALID_CLASS_NAMES,
        )

        return cls(**fields)


class T2IAdapterConfig(DiffusersConfigBase, ControlAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for T2I."""

    type: Literal[ModelType.T2IAdapter] = ModelType.T2IAdapter
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class SpandrelImageToImageConfig(ModelConfigBase):
    """Model config for Spandrel Image to Image models."""

    base: Literal[BaseModelType.Any] = BaseModelType.Any
    type: Literal[ModelType.SpandrelImageToImage] = ModelType.SpandrelImageToImage
    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.SpandrelImageToImage,
        "format": ModelFormat.Checkpoint,
        "base": BaseModelType.Any,
    }

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        if not mod.path.is_file():
            raise NotAMatch(cls, "model path is a directory, not a file")

        try:
            # It would be nice to avoid having to load the Spandrel model from disk here. A couple of options were
            # explored to avoid this:
            # 1. Call `SpandrelImageToImageModel.load_from_state_dict(ckpt)`, where `ckpt` is a state_dict on the meta
            #    device. Unfortunately, some Spandrel models perform operations during initialization that are not
            #    supported on meta tensors.
            # 2. Spandrel has internal logic to determine a model's type from its state_dict before loading the model.
            #    This logic is not exposed in spandrel's public API. We could copy the logic here, but then we have to
            #    maintain it, and the risk of false positive detections is higher.
            SpandrelImageToImageModel.load_from_file(mod.path)
            base = fields.get("base") or BaseModelType.Any
            return cls(**fields, base=base)
        except Exception as e:
            raise NotAMatch(cls, "model does not match SpandrelImageToImage heuristics") from e


class SigLIPConfig(DiffusersConfigBase, ModelConfigBase):
    """Model config for SigLIP."""

    type: Literal[ModelType.SigLIP] = ModelType.SigLIP
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.SigLIP,
        "format": ModelFormat.Diffusers,
    }

    VALID_CLASS_NAMES: ClassVar = {
        "SiglipModel",
    }

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        if mod.path.is_file():
            raise NotAMatch(cls, "model path is a file, not a directory")

        config_path = mod.path / "config.json"

        _validate_class_names(
            config_class=cls,
            config_path=config_path,
            valid_class_names=cls.VALID_CLASS_NAMES,
        )

        return cls(**fields)


class FluxReduxConfig(LegacyProbeMixin, ModelConfigBase):
    """Model config for FLUX Tools Redux model."""

    type: Literal[ModelType.FluxRedux] = ModelType.FluxRedux
    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint


class LlavaOnevisionConfig(DiffusersConfigBase, ModelConfigBase):
    """Model config for Llava Onevision models."""

    type: Literal[ModelType.LlavaOnevision] = ModelType.LlavaOnevision
    base: Literal[BaseModelType.Any] = BaseModelType.Any
    variant: Literal[ModelVariantType.Normal] = ModelVariantType.Normal

    VALID_OVERRIDES: ClassVar = {
        "type": ModelType.LlavaOnevision,
        "format": ModelFormat.Diffusers,
    }

    VALID_CLASS_NAMES: ClassVar = {
        "LlavaOnevisionForConditionalGeneration",
    }

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        if _validate_overrides(
            config_class=cls,
            provided_overrides=fields,
            valid_overrides=cls.VALID_OVERRIDES,
        ):
            return cls(**fields)

        if mod.path.is_file():
            raise NotAMatch(cls, "model path is a file, not a directory")

        config_path = mod.path / "config.json"

        _validate_class_names(
            config_class=cls,
            config_path=config_path,
            valid_class_names=cls.VALID_CLASS_NAMES,
        )

        return cls(**fields)


class ApiModelConfig(MainConfigBase, ModelConfigBase):
    """Model config for API-based models."""

    format: Literal[ModelFormat.Api] = ModelFormat.Api

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        raise NotAMatch(cls, "API models cannot be built from disk")


class VideoApiModelConfig(VideoConfigBase, ModelConfigBase):
    """Model config for API-based video models."""

    format: Literal[ModelFormat.Api] = ModelFormat.Api

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        raise NotAMatch(cls, "API models cannot be built from disk")


def get_model_discriminator_value(v: Any) -> str:
    """
    Computes the discriminator value for a model config.
    https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions-with-callable-discriminator
    """
    format_ = type_ = variant_ = None

    if isinstance(v, dict):
        format_ = v.get("format")
        if isinstance(format_, Enum):
            format_ = format_.value

        type_ = v.get("type")
        if isinstance(type_, Enum):
            type_ = type_.value

        variant_ = v.get("variant")
        if isinstance(variant_, Enum):
            variant_ = variant_.value
    else:
        format_ = v.format.value
        type_ = v.type.value
        variant_ = getattr(v, "variant", None)
        if variant_:
            variant_ = variant_.value

    # Ideally, each config would be uniquely identified with a combination of fields
    # i.e. (type, format, variant) without any special cases. Alas...

    # Previously, CLIPEmbed did not have any variants, meaning older database entries lack a variant field.
    # To maintain compatibility, we default to ClipVariantType.L in this case.
    if type_ == ModelType.CLIPEmbed.value:
        return f"{type_}.{format_}.{variant_}"
    return f"{type_}.{format_}"


# The types are listed explicitly because IDEs/LSPs can't identify the correct types
# when AnyModelConfig is constructed dynamically using ModelConfigBase.all_config_classes
AnyModelConfig = Annotated[
    Union[
        Annotated[MainDiffusersConfig, MainDiffusersConfig.get_tag()],
        Annotated[MainCheckpointConfig, MainCheckpointConfig.get_tag()],
        Annotated[MainBnbQuantized4bCheckpointConfig, MainBnbQuantized4bCheckpointConfig.get_tag()],
        Annotated[MainGGUFCheckpointConfig, MainGGUFCheckpointConfig.get_tag()],
        Annotated[VAEDiffusersConfig, VAEDiffusersConfig.get_tag()],
        Annotated[VAECheckpointConfig, VAECheckpointConfig.get_tag()],
        Annotated[ControlNetDiffusersConfig, ControlNetDiffusersConfig.get_tag()],
        Annotated[ControlNetCheckpointConfig, ControlNetCheckpointConfig.get_tag()],
        Annotated[LoRALyCORISConfig, LoRALyCORISConfig.get_tag()],
        Annotated[LoRAOmiConfig, LoRAOmiConfig.get_tag()],
        Annotated[ControlLoRALyCORISConfig, ControlLoRALyCORISConfig.get_tag()],
        Annotated[ControlLoRADiffusersConfig, ControlLoRADiffusersConfig.get_tag()],
        Annotated[LoRADiffusersConfig, LoRADiffusersConfig.get_tag()],
        Annotated[T5EncoderConfig, T5EncoderConfig.get_tag()],
        Annotated[T5EncoderBnbQuantizedLlmInt8bConfig, T5EncoderBnbQuantizedLlmInt8bConfig.get_tag()],
        Annotated[TextualInversionFileConfig, TextualInversionFileConfig.get_tag()],
        Annotated[TextualInversionFolderConfig, TextualInversionFolderConfig.get_tag()],
        Annotated[IPAdapterInvokeAIConfig, IPAdapterInvokeAIConfig.get_tag()],
        Annotated[IPAdapterCheckpointConfig, IPAdapterCheckpointConfig.get_tag()],
        Annotated[T2IAdapterConfig, T2IAdapterConfig.get_tag()],
        Annotated[SpandrelImageToImageConfig, SpandrelImageToImageConfig.get_tag()],
        Annotated[CLIPVisionDiffusersConfig, CLIPVisionDiffusersConfig.get_tag()],
        Annotated[CLIPLEmbedDiffusersConfig, CLIPLEmbedDiffusersConfig.get_tag()],
        Annotated[CLIPGEmbedDiffusersConfig, CLIPGEmbedDiffusersConfig.get_tag()],
        Annotated[SigLIPConfig, SigLIPConfig.get_tag()],
        Annotated[FluxReduxConfig, FluxReduxConfig.get_tag()],
        Annotated[LlavaOnevisionConfig, LlavaOnevisionConfig.get_tag()],
        Annotated[ApiModelConfig, ApiModelConfig.get_tag()],
        Annotated[VideoApiModelConfig, VideoApiModelConfig.get_tag()],
        Annotated[UnknownModelConfig, UnknownModelConfig.get_tag()],
    ],
    Discriminator(get_model_discriminator_value),
]

AnyModelConfigValidator = TypeAdapter[AnyModelConfig](AnyModelConfig)
AnyDefaultSettings: TypeAlias = Union[MainModelDefaultSettings, LoraModelDefaultSettings, ControlAdapterDefaultSettings]


class ModelConfigFactory:
    @staticmethod
    def make_config(model_data: Dict[str, Any], timestamp: Optional[float] = None) -> AnyModelConfig:
        """Return the appropriate config object from raw dict values."""
        model = AnyModelConfigValidator.validate_python(model_data)
        if isinstance(model, CheckpointConfigBase) and timestamp:
            model.converted_at = timestamp
        validate_hash(model.hash)
        return model

    @staticmethod
    def build_common_fields(
        mod: ModelOnDisk,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Builds the common fields for all model configs.

        Args:
            mod: The model on disk to extract fields from.
            overrides: A optional dictionary of fields to override. These fields will take precedence over the values
                extracted from the model on disk.

        - Casts string fields to their Enum types.
        - Does not validate the fields against the model config schema.
        """

        _overrides: dict[str, Any] = overrides or {}
        fields: dict[str, Any] = {}

        if "type" in _overrides:
            fields["type"] = ModelType(_overrides["type"])

        if "format" in _overrides:
            fields["format"] = ModelFormat(_overrides["format"])

        if "base" in _overrides:
            fields["base"] = BaseModelType(_overrides["base"])

        if "source_type" in _overrides:
            fields["source_type"] = ModelSourceType(_overrides["source_type"])

        if "variant" in _overrides:
            fields["variant"] = variant_type_adapter.validate_strings(_overrides["variant"])

        fields["path"] = mod.path.as_posix()
        fields["source"] = _overrides.get("source") or fields["path"]
        fields["source_type"] = _overrides.get("source_type") or ModelSourceType.Path
        fields["name"] = _overrides.get("name") or mod.name
        fields["hash"] = _overrides.get("hash") or mod.hash()
        fields["key"] = _overrides.get("key") or uuid_string()
        fields["description"] = _overrides.get("description")
        fields["repo_variant"] = _overrides.get("repo_variant") or mod.repo_variant()
        fields["file_size"] = _overrides.get("file_size") or mod.size()

        return fields

    @staticmethod
    def from_model_on_disk(
        mod: str | Path | ModelOnDisk,
        overrides: dict[str, Any] | None = None,
        hash_algo: HASHING_ALGORITHMS = "blake3_single",
    ) -> AnyModelConfig:
        """
        Returns the best matching ModelConfig instance from a model's file/folder path.
        Raises InvalidModelConfigException if no valid configuration is found.
        Created to deprecate ModelProbe.probe
        """
        if isinstance(mod, Path | str):
            mod = ModelOnDisk(Path(mod), hash_algo)

        # We will always need these fields to build any model config.
        fields = ModelConfigFactory.build_common_fields(mod, overrides)

        # Store results as a mapping of config class to either an instance of that class or an exception
        # that was raised when trying to build it.
        results: dict[str, AnyModelConfig | Exception] = {}

        # Try to build an instance of each model config class that uses the classify API.
        # Each class will either return an instance of itself or raise NotAMatch if it doesn't match.
        # Other exceptions may be raised if something unexpected happens during matching or building.
        for config_class in ModelConfigBase.USING_CLASSIFY_API:
            class_name = config_class.__name__
            try:
                instance = config_class.from_model_on_disk(mod, fields)
                results[class_name] = instance
            except NotAMatch as e:
                results[class_name] = e
                logger.debug(f"No match for {config_class.__name__} on model {mod.name}")
            except ValidationError as e:
                # This means the model matched, but we couldn't create the pydantic model instance for the config.
                # Maybe invalid overrides were provided?
                results[class_name] = e
                logger.warning(f"Schema validation error for {config_class.__name__} on model {mod.name}: {e}")
            except Exception as e:
                results[class_name] = e
                logger.warning(f"Unexpected exception while matching {mod.name} to {config_class.__name__}: {e}")

        matches = [r for r in results.values() if isinstance(r, ModelConfigBase)]

        if not matches and app_config.allow_unknown_models:
            logger.warning(f"Unable to identify model {mod.name}, classifying as UnknownModelConfig")
            logger.debug(f"Model matching results: {results}")
            return UnknownModelConfig(**fields)

        instance = next(iter(matches))
        if len(matches) > 1:
            # TODO(psyche): When we get multiple matches, at most only 1 will be correct. We should disambiguate the
            # matches, probably on a case-by-case basis.
            #
            # One known case is certain SD main (pipeline) models can look like a LoRA. This could happen if the model
            # contains merged in LoRA weights.
            logger.warning(
                f"Multiple model config classes matched for model {mod.name}: {[type(m).__name__ for m in matches]}. Using {type(instance).__name__}."
            )
        logger.info(f"Model {mod.name} classified as {type(instance).__name__}")
        return instance
