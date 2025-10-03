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

import json
import logging
import re
import time
from abc import ABC
from enum import Enum
from functools import cache
from inspect import isabstract
from pathlib import Path
from typing import (
    ClassVar,
    Literal,
    Optional,
    Self,
    Type,
    Union,
)

import torch
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, TypeAdapter, ValidationError
from pydantic_core import CoreSchema, PydanticUndefined, SchemaValidator
from typing_extensions import Annotated, Any, Dict

from invokeai.app.services.config.config_default import get_config
from invokeai.app.util.misc import uuid_string
from invokeai.backend.flux.controlnet.state_dict_utils import (
    is_state_dict_instantx_controlnet,
    is_state_dict_xlabs_controlnet,
)
from invokeai.backend.flux.ip_adapter.state_dict_utils import is_state_dict_xlabs_ip_adapter
from invokeai.backend.flux.redux.flux_redux_state_dict_utils import is_state_dict_likely_flux_redux
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
from invokeai.backend.patches.lora_conversions.flux_control_lora_utils import is_state_dict_likely_flux_control
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
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


class FieldValidator:
    """Utility class for validating individual fields of a Pydantic model without instantiating the whole model.

    See: https://github.com/pydantic/pydantic/discussions/7367#discussioncomment-14213144
    """

    @staticmethod
    def find_field_schema(model: type[BaseModel], field_name: str) -> CoreSchema:
        """Find the Pydantic core schema for a specific field in a model."""
        schema: CoreSchema = model.__pydantic_core_schema__.copy()
        # we shallow copied, be careful not to mutate the original schema!

        assert schema["type"] in ["definitions", "model"]

        # find the field schema
        field_schema = schema["schema"]  # type: ignore
        while "fields" not in field_schema:
            field_schema = field_schema["schema"]  # type: ignore

        field_schema = field_schema["fields"][field_name]["schema"]  # type: ignore

        # if the original schema is a definition schema, replace the model schema with the field schema
        if schema["type"] == "definitions":
            schema["schema"] = field_schema
            return schema
        else:
            return field_schema

    @cache
    @staticmethod
    def get_validator(model: type[BaseModel], field_name: str) -> SchemaValidator:
        """Get a SchemaValidator for a specific field in a model."""
        return SchemaValidator(FieldValidator.find_field_schema(model, field_name))

    @staticmethod
    def validate_field(model: type[BaseModel], field_name: str, value: Any) -> Any:
        """Validate a value for a specific field in a model."""
        return FieldValidator.get_validator(model, field_name).validate_python(value)


def has_any_keys(state_dict: dict[str | int, Any], keys: str | set[str]) -> bool:
    """Returns true if the state dict has any of the specified keys."""
    _keys = {keys} if isinstance(keys, str) else keys
    return any(key in state_dict for key in _keys)


def has_any_keys_starting_with(state_dict: dict[str | int, Any], prefixes: str | set[str]) -> bool:
    """Returns true if the state dict has any keys starting with any of the specified prefixes."""
    _prefixes = {prefixes} if isinstance(prefixes, str) else prefixes
    return any(any(key.startswith(prefix) for prefix in _prefixes) for key in state_dict.keys() if isinstance(key, str))


def has_any_keys_ending_with(state_dict: dict[str | int, Any], suffixes: str | set[str]) -> bool:
    """Returns true if the state dict has any keys ending with any of the specified suffixes."""
    _suffixes = {suffixes} if isinstance(suffixes, str) else suffixes
    return any(any(key.endswith(suffix) for suffix in _suffixes) for key in state_dict.keys() if isinstance(key, str))


def common_config_paths(path: Path) -> set[Path]:
    """Returns common config file paths for models stored in directories."""
    return {path / "config.json", path / "model_index.json"}


# These utility functions are tightly coupled to the config classes below in order to make the process of raising
# NotAMatch exceptions as easy and consistent as possible.


def _get_config_or_raise(
    config_class: type,
    config_path: Path | set[Path],
) -> dict[str, Any]:
    """Load the config file at the given path, or raise NotAMatch if it cannot be loaded."""
    paths_to_check = config_path if isinstance(config_path, set) else {config_path}

    problems: dict[Path, str] = {}

    for p in paths_to_check:
        if not p.exists():
            problems[p] = "file does not exist"
            continue

        try:
            with open(p, "r") as file:
                config = json.load(file)

            return config
        except Exception as e:
            problems[p] = str(e)
            continue

    raise NotAMatch(config_class, f"unable to load config file(s): {problems}")


def _get_class_name_from_config(
    config_class: type,
    config_path: Path | set[Path],
) -> str:
    """Load the config file and return the class name.

    Raises:
        NotAMatch if the config file is missing or does not contain a valid class name.
    """

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

    if not isinstance(config_class_name, str):
        raise NotAMatch(config_class, f"_class_name or architectures field is not a string: {config_class_name}")

    return config_class_name


def _validate_class_name(config_class: type[BaseModel], config_path: Path | set[Path], expected: set[str]) -> None:
    """Check if the class name in the config file matches the expected class names.

    Args:
        config_class: The config class that is being tested.
        config_path: The path to the config file.
        expected: The expected class names."""

    class_name = _get_class_name_from_config(config_class, config_path)
    if class_name not in expected:
        raise NotAMatch(config_class, f"invalid class name from config: {class_name}")


def _validate_override_fields(
    config_class: type[BaseModel],
    override_fields: dict[str, Any],
) -> None:
    """Check if the provided override fields are valid for the config class.

    Args:
        config_class: The config class that is being tested.
        override_fields: The override fields provided by the user.

    Raises:
        NotAMatch if any override field is invalid for the config.
    """
    for field_name, override_value in override_fields.items():
        if field_name not in config_class.model_fields:
            raise NotAMatch(config_class, f"unknown override field: {field_name}")
        try:
            FieldValidator.validate_field(config_class, field_name, override_value)
        except ValidationError as e:
            raise NotAMatch(config_class, f"invalid override for field '{field_name}': {e}") from e


def _validate_is_file(
    config_class: type,
    mod: ModelOnDisk,
) -> None:
    """Raise NotAMatch if the model path is not a file."""
    if not mod.path.is_file():
        raise NotAMatch(config_class, "model path is not a file")


def _validate_is_dir(
    config_class: type,
    mod: ModelOnDisk,
) -> None:
    """Raise NotAMatch if the model path is not a directory."""
    if not mod.path.is_dir():
        raise NotAMatch(config_class, "model path is not a directory")


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

    pass


class Config_Base(ABC, BaseModel):
    """
    Abstract base class for model configurations. A model config describes a specific combination of model base, type and
    format, along with other metadata about the model. For example, a Stable Diffusion 1.x main model in checkpoint format
    would have base=sd-1, type=main, format=checkpoint.

    To create a new config type, inherit from this class and implement its interface:
    - Define method 'from_model_on_disk' that returns an instance of the class or raises NotAMatch. This method will be
        called during model installation to determine the correct config class for a model.
    - Define fields 'type', 'base' and 'format' as pydantic fields. These should be Literals with a single value. A
        default must be provided for each of these fields.

    If multiple combinations of base, type and format need to be supported, create a separate subclass for each.

    See MinimalConfigExample in test_model_probe.py for an example implementation.
    """

    key: str = Field(
        description="A unique key for this model.",
        default_factory=uuid_string,
    )
    hash: str = Field(
        description="The hash of the model file(s).",
    )
    path: str = Field(
        description="Path to the model on the filesystem. Relative paths are relative to the Invoke root directory.",
    )
    file_size: int = Field(
        description="The size of the model in bytes.",
    )
    name: str = Field(
        description="Name of the model.",
    )
    description: str | None = Field(
        description="Model description",
        default=None,
    )
    source: str = Field(
        description="The original source of the model (path, URL or repo_id).",
    )
    source_type: ModelSourceType = Field(
        description="The type of source",
    )
    source_api_response: str | None = Field(
        description="The original API response from the source, as stringified JSON.",
        default=None,
    )
    cover_image: str | None = Field(
        description="Url for image to preview model",
        default=None,
    )
    submodels: dict[SubModelType, SubmodelDefinition] | None = Field(
        description="Loadable submodels in this model",
        default=None,
    )
    usage_info: str | None = Field(
        default=None,
        description="Usage information for this model",
    )

    CONFIG_CLASSES: ClassVar[set[Type["AnyModelConfig"]]] = set()

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_serialization_defaults_required=True,
        json_schema_mode_override="serialization",
    )

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register non-abstract subclasses so we can iterate over them later during model probing.
        if not isabstract(cls):
            cls.CONFIG_CLASSES.add(cls)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        # Ensure that subclasses define 'base', 'type' and 'format' fields and provide defaults for them. Each subclass
        # is expected to represent a single combination of base, type and format.
        for name in ("type", "base", "format"):
            assert name in cls.model_fields, f"{cls.__name__} must define a '{name}' field"
            assert cls.model_fields[name].default is not PydanticUndefined, (
                f"{cls.__name__} must define a default for the '{name}' field"
            )

    @classmethod
    def get_tag(cls) -> Tag:
        """Constructs a pydantic discriminated union tag for this model config class. When a config is deserialized,
        pydantic uses the tag to determine which subclass to instantiate.

        The tag is a dot-separated string of the type, format, base and variant (if applicable).
        """
        tag_strings: list[str] = []
        for name in ("type", "format", "base", "variant"):
            if field := cls.model_fields.get(name):
                if field.default is not PydanticUndefined:
                    # We expect each of these fields has an Enum for its default; we want the value of the enum.
                    tag_strings.append(field.default.value)
        return Tag(".".join(tag_strings))

    @staticmethod
    def get_model_discriminator_value(v: Any) -> str:
        """
        Computes the discriminator value for a model config.
        https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions-with-callable-discriminator
        """
        if isinstance(v, Config_Base):
            # We have an instance of a ModelConfigBase subclass - use its tag directly.
            return v.get_tag().tag
        if isinstance(v, dict):
            # We have a dict - compute the tag from its fields.
            tag_strings: list[str] = []
            if type_ := v.get("type"):
                if isinstance(type_, Enum):
                    type_ = type_.value
                tag_strings.append(type_)

            if format_ := v.get("format"):
                if isinstance(format_, Enum):
                    format_ = format_.value
                tag_strings.append(format_)

            if base_ := v.get("base"):
                if isinstance(base_, Enum):
                    base_ = base_.value
                tag_strings.append(base_)

            # Special case: CLIP Embed models also need the variant to distinguish them.
            if (
                type_ == ModelType.CLIPEmbed.value
                and format_ == ModelFormat.Diffusers.value
                and base_ == BaseModelType.Any.value
            ):
                if variant_value := v.get("variant"):
                    if isinstance(variant_value, Enum):
                        variant_value = variant_value.value
                    tag_strings.append(variant_value)
                else:
                    raise ValueError("CLIP Embed model config dict must include a 'variant' field")

            return ".".join(tag_strings)
        else:
            raise TypeError("Model config discriminator value must be computed from a dict or ModelConfigBase instance")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        """Given the model on disk and any overrides, return an instance of this config class.

        Implementations should raise NotAMatch if the model does not match this config class."""
        raise NotImplementedError(f"from_model_on_disk not implemented for {cls.__name__}")


class Unknown_Config(Config_Base):
    """Model config for unknown models, used as a fallback when we cannot identify a model."""

    base: Literal[BaseModelType.Unknown] = Field(default=BaseModelType.Unknown)
    type: Literal[ModelType.Unknown] = Field(default=ModelType.Unknown)
    format: Literal[ModelFormat.Unknown] = Field(default=ModelFormat.Unknown)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        raise NotAMatch(cls, "unknown model config cannot match any model")


class Checkpoint_Config_Base(ABC, BaseModel):
    """Base class for checkpoint-style models."""

    config_path: str | None = Field(
        description="Path to the config for this model, if any.",
        default=None,
    )
    converted_at: float | None = Field(
        description="When this model was last converted to diffusers",
        default_factory=time.time,
    )


class Diffusers_Config_Base(ABC, BaseModel):
    """Base class for diffusers-style models."""

    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)
    repo_variant: Optional[ModelRepoVariant] = Field(ModelRepoVariant.Default)

    @classmethod
    def _get_repo_variant_or_raise(cls, mod: ModelOnDisk) -> ModelRepoVariant:
        # get all files ending in .bin or .safetensors
        weight_files = list(mod.path.glob("**/*.safetensors"))
        weight_files.extend(list(mod.path.glob("**/*.bin")))
        for x in weight_files:
            if ".fp16" in x.suffixes:
                return ModelRepoVariant.FP16
            if "openvino_model" in x.name:
                return ModelRepoVariant.OpenVINO
            if "flax_model" in x.name:
                return ModelRepoVariant.Flax
            if x.suffix == ".onnx":
                return ModelRepoVariant.ONNX
        return ModelRepoVariant.Default


class T5Encoder_T5Encoder_Config(Config_Base):
    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.T5Encoder] = Field(default=ModelType.T5Encoder)
    format: Literal[ModelFormat.T5Encoder] = Field(default=ModelFormat.T5Encoder)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        _validate_class_name(
            cls,
            common_config_paths(mod.path),
            {
                "T5EncoderModel",
            },
        )

        cls._validate_has_unquantized_config_file(mod)

        return cls(**fields)

    @classmethod
    def _validate_has_unquantized_config_file(cls, mod: ModelOnDisk) -> None:
        has_unquantized_config = (mod.path / "text_encoder_2" / "model.safetensors.index.json").exists()

        if not has_unquantized_config:
            raise NotAMatch(cls, "missing text_encoder_2/model.safetensors.index.json")


class T5Encoder_BnBLLMint8_Config(Config_Base):
    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.T5Encoder] = Field(default=ModelType.T5Encoder)
    format: Literal[ModelFormat.BnbQuantizedLlmInt8b] = Field(default=ModelFormat.BnbQuantizedLlmInt8b)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        _validate_class_name(
            cls,
            common_config_paths(mod.path),
            {
                "T5EncoderModel",
            },
        )

        cls._validate_filename_looks_like_bnb_quantized(mod)

        cls._validate_model_looks_like_bnb_quantized(mod)

        return cls(**fields)

    @classmethod
    def _validate_filename_looks_like_bnb_quantized(cls, mod: ModelOnDisk) -> None:
        filename_looks_like_bnb = any(x for x in mod.weight_files() if "llm_int8" in x.as_posix())
        if not filename_looks_like_bnb:
            raise NotAMatch(cls, "filename does not look like bnb quantized llm_int8")

    @classmethod
    def _validate_model_looks_like_bnb_quantized(cls, mod: ModelOnDisk) -> None:
        has_scb_key_suffix = has_any_keys_ending_with(mod.load_state_dict(), "SCB")
        if not has_scb_key_suffix:
            raise NotAMatch(cls, "state dict does not look like bnb quantized llm_int8")


class LoRA_Config_Base(ABC, BaseModel):
    """Base class for LoRA models."""

    type: Literal[ModelType.LoRA] = Field(default=ModelType.LoRA)
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)
    default_settings: Optional[LoraModelDefaultSettings] = Field(
        description="Default settings for this model", default=None
    )


def _get_flux_lora_format(mod: ModelOnDisk) -> FluxLoRAFormat | None:
    # TODO(psyche): Moving this import to the function to avoid circular imports. Refactor later.
    from invokeai.backend.patches.lora_conversions.formats import flux_format_from_state_dict

    state_dict = mod.load_state_dict(mod.path)
    value = flux_format_from_state_dict(state_dict, mod.metadata())
    return value


class LoRA_OMI_Config_Base(LoRA_Config_Base):
    format: Literal[ModelFormat.OMI] = Field(default=ModelFormat.OMI)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_file(cls, mod)

        _validate_override_fields(cls, fields)

        cls._validate_looks_like_omi_lora(mod)

        cls._validate_base(mod)

        return cls(**fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatch(cls, f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _validate_looks_like_omi_lora(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model metadata does not look like an OMI LoRA."""
        flux_format = _get_flux_lora_format(mod)
        if flux_format in [FluxLoRAFormat.Control, FluxLoRAFormat.Diffusers]:
            raise NotAMatch(cls, "model looks like ControlLoRA or Diffusers LoRA")

        metadata = mod.metadata()

        metadata_looks_like_omi_lora = (
            bool(metadata.get("modelspec.sai_model_spec"))
            and metadata.get("ot_branch") == "omi_format"
            and metadata.get("modelspec.architecture", "").split("/")[1].lower() == "lora"
        )

        if not metadata_looks_like_omi_lora:
            raise NotAMatch(cls, "metadata does not look like OMI LoRA")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> Literal[BaseModelType.Flux, BaseModelType.StableDiffusionXL]:
        metadata = mod.metadata()
        architecture = metadata["modelspec.architecture"]

        if architecture == stable_diffusion_xl_1_lora:
            return BaseModelType.StableDiffusionXL
        elif architecture == flux_dev_1_lora:
            return BaseModelType.Flux
        else:
            raise NotAMatch(cls, f"unrecognised/unsupported architecture for OMI LoRA: {architecture}")


class LoRA_OMI_SDXL_Config(LoRA_OMI_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class LoRA_OMI_FLUX_Config(LoRA_OMI_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class LoRA_LyCORIS_Config_Base(LoRA_Config_Base):
    """Model config for LoRA/Lycoris models."""

    type: Literal[ModelType.LoRA] = Field(default=ModelType.LoRA)
    format: Literal[ModelFormat.LyCORIS] = Field(default=ModelFormat.LyCORIS)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_file(cls, mod)

        _validate_override_fields(cls, fields)

        cls._validate_looks_like_lora(mod)

        cls._validate_base(mod)

        return cls(**fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatch(cls, f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _validate_looks_like_lora(cls, mod: ModelOnDisk) -> None:
        # First rule out ControlLoRA and Diffusers LoRA
        flux_format = _get_flux_lora_format(mod)
        if flux_format in [FluxLoRAFormat.Control, FluxLoRAFormat.Diffusers]:
            raise NotAMatch(cls, "model looks like ControlLoRA or Diffusers LoRA")

        # Note: Existence of these key prefixes/suffixes does not guarantee that this is a LoRA.
        # Some main models have these keys, likely due to the creator merging in a LoRA.
        has_key_with_lora_prefix = has_any_keys_starting_with(
            mod.load_state_dict(),
            {
                "lora_te_",
                "lora_unet_",
                "lora_te1_",
                "lora_te2_",
                "lora_transformer_",
            },
        )

        has_key_with_lora_suffix = has_any_keys_ending_with(
            mod.load_state_dict(),
            {
                "to_k_lora.up.weight",
                "to_q_lora.down.weight",
                "lora_A.weight",
                "lora_B.weight",
            },
        )

        if not has_key_with_lora_prefix and not has_key_with_lora_suffix:
            raise NotAMatch(cls, "model does not match LyCORIS LoRA heuristics")

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
            raise NotAMatch(cls, f"unrecognized token vector length {token_vector_length}")


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
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_file(cls, mod)

        _validate_override_fields(cls, fields)

        cls._validate_looks_like_control_lora(mod)

        return cls(**fields)

    @classmethod
    def _validate_looks_like_control_lora(cls, mod: ModelOnDisk) -> None:
        state_dict = mod.load_state_dict()

        if not is_state_dict_likely_flux_control(state_dict):
            raise NotAMatch(cls, "model state dict does not look like a Flux Control LoRA")


class LoRA_Diffusers_Config_Base(LoRA_Config_Base):
    """Model config for LoRA/Diffusers models."""

    # TODO(psyche): Needs base handling. For FLUX, the Diffusers format does not indicate a folder model; it indicates
    # the weights format. FLUX Diffusers LoRAs are single files.

    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        cls._validate_base(mod)

        return cls(**fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatch(cls, f"base is {recognized_base}, not {expected_base}")

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
                raise NotAMatch(cls, f"unrecognized token vector length {token_vector_length}")

    @classmethod
    def _get_weight_file_or_raise(cls, mod: ModelOnDisk) -> Path:
        suffixes = ["bin", "safetensors"]
        weight_files = [mod.path / f"pytorch_lora_weights.{sfx}" for sfx in suffixes]
        for wf in weight_files:
            if wf.exists():
                return wf
        raise NotAMatch(cls, "missing pytorch_lora_weights.bin or pytorch_lora_weights.safetensors")


class LoRA_Diffusers_SD1_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class LoRA_Diffusers_SD2_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class LoRA_Diffusers_SDXL_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class LoRA_Diffusers_FLUX_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class VAE_Checkpoint_Config_Base(Checkpoint_Config_Base):
    """Model config for standalone VAE models."""

    type: Literal[ModelType.VAE] = Field(default=ModelType.VAE)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    REGEX_TO_BASE: ClassVar[dict[str, BaseModelType]] = {
        r"xl": BaseModelType.StableDiffusionXL,
        r"sd2": BaseModelType.StableDiffusion2,
        r"vae": BaseModelType.StableDiffusion1,
        r"FLUX.1-schnell_ae": BaseModelType.Flux,
    }

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_file(cls, mod)

        _validate_override_fields(cls, fields)

        cls._validate_looks_like_vae(mod)

        cls._validate_base(mod)

        return cls(**fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatch(cls, f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _validate_looks_like_vae(cls, mod: ModelOnDisk) -> None:
        if not has_any_keys_starting_with(
            mod.load_state_dict(),
            {
                "encoder.conv_in",
                "decoder.conv_in",
            },
        ):
            raise NotAMatch(cls, "model does not match Checkpoint VAE heuristics")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        # Heuristic: VAEs of all architectures have a similar structure; the best we can do is guess based on name
        for regexp, base in cls.REGEX_TO_BASE.items():
            if re.search(regexp, mod.path.name, re.IGNORECASE):
                return base

        raise NotAMatch(cls, "cannot determine base type")


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
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        _validate_class_name(
            cls,
            common_config_paths(mod.path),
            {
                "AutoencoderKL",
                "AutoencoderTiny",
            },
        )

        cls._validate_base(mod)

        return cls(**fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatch(cls, f"base is {recognized_base}, not {expected_base}")

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
        config = _get_config_or_raise(cls, common_config_paths(mod.path))
        if cls._config_looks_like_sdxl(config):
            return BaseModelType.StableDiffusionXL
        elif cls._name_looks_like_sdxl(mod):
            return BaseModelType.StableDiffusionXL
        else:
            # TODO(psyche): Figure out how to positively identify SD1 here, and raise if we can't. Until then, YOLO.
            return BaseModelType.StableDiffusion1


class VAE_Diffusers_SD1_Config(VAE_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class VAE_Diffusers_SDXL_Config(VAE_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class ControlNet_Diffusers_Config_Base(Diffusers_Config_Base, ControlAdapter_Config_Base):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = Field(default=ModelType.ControlNet)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        _validate_class_name(
            cls,
            common_config_paths(mod.path),
            {
                "ControlNetModel",
                "FluxControlNetModel",
            },
        )

        cls._validate_base(mod)

        return cls(**fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatch(cls, f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        config = _get_config_or_raise(cls, common_config_paths(mod.path))

        if config.get("_class_name") == "FluxControlNetModel":
            return BaseModelType.Flux

        dimension = config.get("cross_attention_dim")

        match dimension:
            case 768:
                return BaseModelType.StableDiffusion1
            case 1024:
                # No obvious way to distinguish between sd2-base and sd2-768, but we don't really differentiate them
                # anyway.
                return BaseModelType.StableDiffusion2
            case 2048:
                return BaseModelType.StableDiffusionXL
            case _:
                raise NotAMatch(cls, f"unrecognized cross_attention_dim {dimension}")


class ControlNet_Diffusers_SD1_Config(ControlNet_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class ControlNet_Diffusers_SD2_Config(ControlNet_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class ControlNet_Diffusers_SDXL_Config(ControlNet_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class ControlNet_Diffusers_FLUX_Config(ControlNet_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class ControlNet_Checkpoint_Config_Base(Checkpoint_Config_Base, ControlAdapter_Config_Base):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = Field(default=ModelType.ControlNet)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_file(cls, mod)

        _validate_override_fields(cls, fields)

        cls._validate_looks_like_controlnet(mod)

        cls._validate_base(mod)

        return cls(**fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatch(cls, f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _validate_looks_like_controlnet(cls, mod: ModelOnDisk) -> None:
        if has_any_keys_starting_with(
            mod.load_state_dict(),
            {
                "controlnet",
                "control_model",
                "input_blocks",
                # XLabs FLUX ControlNet models have keys starting with "controlnet_blocks."
                # For example: https://huggingface.co/XLabs-AI/flux-controlnet-collections/blob/86ab1e915a389d5857135c00e0d350e9e38a9048/flux-canny-controlnet_v2.safetensors
                # TODO(ryand): This is very fragile. XLabs FLUX ControlNet models also contain keys starting with
                # "double_blocks.", which we check for above. But, I'm afraid to modify this logic because it is so
                # delicate.
                "controlnet_blocks",
            },
        ):
            raise NotAMatch(cls, "state dict does not look like a ControlNet checkpoint")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        state_dict = mod.load_state_dict()

        if is_state_dict_xlabs_controlnet(state_dict) or is_state_dict_instantx_controlnet(state_dict):
            # TODO(ryand): Should I distinguish between XLabs, InstantX and other ControlNet models by implementing
            # get_format()?
            return BaseModelType.Flux

        for key in (
            "control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
            "controlnet_mid_block.bias",
            "input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight",
        ):
            if key not in state_dict:
                continue
            width = state_dict[key].shape[-1]
            match width:
                case 768:
                    return BaseModelType.StableDiffusion1
                case 1024:
                    return BaseModelType.StableDiffusion2
                case 2048:
                    return BaseModelType.StableDiffusionXL
                case 1280:
                    return BaseModelType.StableDiffusionXL
                case _:
                    pass

        raise NotAMatch(cls, "unable to determine base type from state dict")


class ControlNet_Checkpoint_SD1_Config(ControlNet_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class ControlNet_Checkpoint_SD2_Config(ControlNet_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class ControlNet_Checkpoint_SDXL_Config(ControlNet_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class ControlNet_Checkpoint_FLUX_Config(ControlNet_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class TI_Config_Base(ABC, BaseModel):
    type: Literal[ModelType.TextualInversion] = Field(default=ModelType.TextualInversion)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk, path: Path | None = None) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod, path)
        if expected_base is not recognized_base:
            raise NotAMatch(cls, f"base is {recognized_base}, not {expected_base}")

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
            if any(key in {"string_to_param", "emb_params", "clip_g"} for key in state_dict.keys()):
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


class TI_File_Config_Base(TI_Config_Base):
    """Model config for textual inversion embeddings."""

    format: Literal[ModelFormat.EmbeddingFile] = Field(default=ModelFormat.EmbeddingFile)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_file(cls, mod)

        _validate_override_fields(cls, fields)

        if not cls._file_looks_like_embedding(mod):
            raise NotAMatch(cls, "model does not look like a textual inversion embedding file")

        cls._validate_base(mod)

        return cls(**fields)


class TI_File_SD1_Config(TI_File_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class TI_File_SD2_Config(TI_File_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class TI_File_SDXL_Config(TI_File_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class TI_Folder_Config_Base(TI_Config_Base):
    """Model config for textual inversion embeddings."""

    format: Literal[ModelFormat.EmbeddingFolder] = Field(default=ModelFormat.EmbeddingFolder)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        for p in mod.weight_files():
            if cls._file_looks_like_embedding(mod, p):
                cls._validate_base(mod, p)
                return cls(**fields)

        raise NotAMatch(cls, "model does not look like a textual inversion embedding folder")


class TI_Folder_SD1_Config(TI_Folder_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class TI_Folder_SD2_Config(TI_Folder_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class TI_Folder_SDXL_Config(TI_Folder_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class Main_Config_Base(ABC, BaseModel):
    type: Literal[ModelType.Main] = Field(default=ModelType.Main)
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)
    default_settings: Optional[MainModelDefaultSettings] = Field(
        description="Default settings for this model", default=None
    )


def _has_bnb_nf4_keys(state_dict: dict[str | int, Any]) -> bool:
    bnb_nf4_keys = {
        "double_blocks.0.img_attn.proj.weight.quant_state.bitsandbytes__nf4",
        "model.diffusion_model.double_blocks.0.img_attn.proj.weight.quant_state.bitsandbytes__nf4",
    }
    return any(key in state_dict for key in bnb_nf4_keys)


def _has_ggml_tensors(state_dict: dict[str | int, Any]) -> bool:
    return any(isinstance(v, GGMLTensor) for v in state_dict.values())


def _has_main_keys(state_dict: dict[str | int, Any]) -> bool:
    for key in state_dict.keys():
        if isinstance(key, int):
            continue
        elif key.startswith(
            (
                "cond_stage_model.",
                "first_stage_model.",
                "model.diffusion_model.",
                # Some FLUX checkpoint files contain transformer keys prefixed with "model.diffusion_model".
                # This prefix is typically used to distinguish between multiple models bundled in a single file.
                "model.diffusion_model.double_blocks.",
            )
        ):
            return True
        elif key.startswith("double_blocks.") and "ip_adapter" not in key:
            # FLUX models in the official BFL format contain keys with the "double_blocks." prefix, but we must be
            # careful to avoid false positives on XLabs FLUX IP-Adapter models.
            return True
    return False


class Main_Checkpoint_Config_Base(Checkpoint_Config_Base, Main_Config_Base):
    """Model config for main checkpoint models."""

    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    prediction_type: SchedulerPredictionType = Field()
    variant: ModelVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_file(cls, mod)

        _validate_override_fields(cls, fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_base(mod)

        prediction_type = fields.get("prediction_type") or cls._get_scheduler_prediction_type_or_raise(mod)

        variant = fields.get("variant") or cls._get_variant_or_raise(mod)

        return cls(**fields, prediction_type=prediction_type, variant=variant)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatch(cls, f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        state_dict = mod.load_state_dict()

        key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in state_dict and state_dict[key_name].shape[-1] == 768:
            return BaseModelType.StableDiffusion1
        if key_name in state_dict and state_dict[key_name].shape[-1] == 1024:
            return BaseModelType.StableDiffusion2

        key_name = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in state_dict and state_dict[key_name].shape[-1] == 2048:
            return BaseModelType.StableDiffusionXL
        elif key_name in state_dict and state_dict[key_name].shape[-1] == 1280:
            return BaseModelType.StableDiffusionXLRefiner

        raise NotAMatch(cls, "unable to determine base type from state dict")

    @classmethod
    def _get_scheduler_prediction_type_or_raise(cls, mod: ModelOnDisk) -> SchedulerPredictionType:
        base = cls.model_fields["base"].default

        if base is BaseModelType.StableDiffusion2:
            state_dict = mod.load_state_dict()
            key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
            if key_name in state_dict and state_dict[key_name].shape[-1] == 1024:
                if "global_step" in state_dict:
                    if state_dict["global_step"] == 220000:
                        return SchedulerPredictionType.Epsilon
                    elif state_dict["global_step"] == 110000:
                        return SchedulerPredictionType.VPrediction
            return SchedulerPredictionType.VPrediction
        else:
            return SchedulerPredictionType.Epsilon

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> ModelVariantType:
        base = cls.model_fields["base"].default

        state_dict = mod.load_state_dict()
        key_name = "model.diffusion_model.input_blocks.0.0.weight"

        if key_name not in state_dict:
            raise NotAMatch(cls, "unable to determine model variant from state dict")

        in_channels = state_dict["model.diffusion_model.input_blocks.0.0.weight"].shape[1]

        match in_channels:
            case 4:
                return ModelVariantType.Normal
            case 5:
                # Only SD2 has a depth variant
                assert base is BaseModelType.StableDiffusion2, f"unexpected unet in_channels 5 for base '{base}'"
                return ModelVariantType.Depth
            case 9:
                return ModelVariantType.Inpaint
            case _:
                raise NotAMatch(cls, f"unrecognized unet in_channels {in_channels} for base '{base}'")

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatch(cls, "state dict does not look like a main model")


class Main_Checkpoint_SD1_Config(Main_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class Main_Checkpoint_SD2_Config(Main_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class Main_Checkpoint_SDXL_Config(Main_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class Main_Checkpoint_SDXLRefiner_Config(Main_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXLRefiner] = Field(default=BaseModelType.StableDiffusionXLRefiner)


def _get_flux_variant(state_dict: dict[str | int, Any]) -> FluxVariantType | None:
    # FLUX Model variant types are distinguished by input channels and the presence of certain keys.

    # Input channels are derived from the shape of either "img_in.weight" or "model.diffusion_model.img_in.weight".
    #
    # Known models that use the latter key:
    # - https://civitai.com/models/885098?modelVersionId=990775
    # - https://civitai.com/models/1018060?modelVersionId=1596255
    # - https://civitai.com/models/978314/ultrareal-fine-tune?modelVersionId=1413133
    #
    # Input channels for known FLUX models:
    # - Unquantized Dev and Schnell have in_channels=64
    # - BNB-NF4 Dev and Schnell have in_channels=1
    # - FLUX Fill has in_channels=384
    # - Unsure of quantized FLUX Fill models
    # - Unsure of GGUF-quantized models

    in_channels = None
    for key in {"img_in.weight", "model.diffusion_model.img_in.weight"}:
        if key in state_dict:
            in_channels = state_dict[key].shape[1]
            break

    if in_channels is None:
        # TODO(psyche): Should we have a graceful fallback here? Previously we fell back to the "normal" variant,
        # but this variant is no longer used for FLUX models. If we get here, but the model is definitely a FLUX
        # model, we should figure out a good fallback value.
        return None

    # Because FLUX Dev and Schnell models have the same in_channels, we need to check for the presence of
    # certain keys to distinguish between them.
    is_flux_dev = (
        "guidance_in.out_layer.weight" in state_dict
        or "model.diffusion_model.guidance_in.out_layer.weight" in state_dict
    )

    if is_flux_dev and in_channels == 384:
        return FluxVariantType.DevFill
    elif is_flux_dev:
        return FluxVariantType.Dev
    else:
        # Must be a Schnell model...?
        return FluxVariantType.Schnell


class Main_Checkpoint_FLUX_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for main checkpoint models."""

    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)

    variant: FluxVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_file(cls, mod)

        _validate_override_fields(cls, fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_is_flux(mod)

        cls._validate_does_not_look_like_bnb_quantized(mod)

        cls._validate_does_not_look_like_gguf_quantized(mod)

        variant = fields.get("variant") or cls._get_variant_or_raise(mod)

        return cls(**fields, variant=variant)

    @classmethod
    def _validate_is_flux(cls, mod: ModelOnDisk) -> None:
        if not has_any_keys(
            mod.load_state_dict(),
            {
                "double_blocks.0.img_attn.norm.key_norm.scale",
                "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale",
            },
        ):
            raise NotAMatch(cls, "state dict does not look like a FLUX checkpoint")

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> FluxVariantType:
        # FLUX Model variant types are distinguished by input channels and the presence of certain keys.
        state_dict = mod.load_state_dict()
        variant = _get_flux_variant(state_dict)

        if variant is None:
            # TODO(psyche): Should we have a graceful fallback here? Previously we fell back to the "normal" variant,
            # but this variant is no longer used for FLUX models. If we get here, but the model is definitely a FLUX
            # model, we should figure out a good fallback value.
            raise NotAMatch(cls, "unable to determine model variant from state dict")

        return variant

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatch(cls, "state dict does not look like a main model")

    @classmethod
    def _validate_does_not_look_like_bnb_quantized(cls, mod: ModelOnDisk) -> None:
        has_bnb_nf4_keys = _has_bnb_nf4_keys(mod.load_state_dict())
        if has_bnb_nf4_keys:
            raise NotAMatch(cls, "state dict looks like bnb quantized nf4")

    @classmethod
    def _validate_does_not_look_like_gguf_quantized(cls, mod: ModelOnDisk):
        has_ggml_tensors = _has_ggml_tensors(mod.load_state_dict())
        if has_ggml_tensors:
            raise NotAMatch(cls, "state dict looks like GGUF quantized")


class Main_BnBNF4_FLUX_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for main checkpoint models."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    format: Literal[ModelFormat.BnbQuantizednf4b] = Field(default=ModelFormat.BnbQuantizednf4b)

    variant: FluxVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_file(cls, mod)

        _validate_override_fields(cls, fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_model_looks_like_bnb_quantized(mod)

        variant = fields.get("variant") or cls._get_variant_or_raise(mod)

        return cls(**fields, variant=variant)

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> FluxVariantType:
        # FLUX Model variant types are distinguished by input channels and the presence of certain keys.
        state_dict = mod.load_state_dict()
        variant = _get_flux_variant(state_dict)

        if variant is None:
            # TODO(psyche): Should we have a graceful fallback here? Previously we fell back to the "normal" variant,
            # but this variant is no longer used for FLUX models. If we get here, but the model is definitely a FLUX
            # model, we should figure out a good fallback value.
            raise NotAMatch(cls, "unable to determine model variant from state dict")

        return variant

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatch(cls, "state dict does not look like a main model")

    @classmethod
    def _validate_model_looks_like_bnb_quantized(cls, mod: ModelOnDisk) -> None:
        has_bnb_nf4_keys = _has_bnb_nf4_keys(mod.load_state_dict())
        if not has_bnb_nf4_keys:
            raise NotAMatch(cls, "state dict does not look like bnb quantized nf4")


class Main_GGUF_FLUX_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for main checkpoint models."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)

    variant: FluxVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_file(cls, mod)

        _validate_override_fields(cls, fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_looks_like_gguf_quantized(mod)

        variant = fields.get("variant") or cls._get_variant_or_raise(mod)

        return cls(**fields, variant=variant)

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> FluxVariantType:
        # FLUX Model variant types are distinguished by input channels and the presence of certain keys.
        state_dict = mod.load_state_dict()
        variant = _get_flux_variant(state_dict)

        if variant is None:
            # TODO(psyche): Should we have a graceful fallback here? Previously we fell back to the "normal" variant,
            # but this variant is no longer used for FLUX models. If we get here, but the model is definitely a FLUX
            # model, we should figure out a good fallback value.
            raise NotAMatch(cls, "unable to determine model variant from state dict")

        return variant

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatch(cls, "state dict does not look like a main model")

    @classmethod
    def _validate_looks_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        has_ggml_tensors = _has_ggml_tensors(mod.load_state_dict())
        if not has_ggml_tensors:
            raise NotAMatch(cls, "state dict does not look like GGUF quantized")


class Main_Diffusers_Config_Base(Diffusers_Config_Base, Main_Config_Base):
    prediction_type: SchedulerPredictionType = Field()
    variant: ModelVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        _validate_class_name(
            cls,
            common_config_paths(mod.path),
            {
                # SD 1.x and 2.x
                "StableDiffusionPipeline",
                "StableDiffusionInpaintPipeline",
                # SDXL
                "StableDiffusionXLPipeline",
                "StableDiffusionXLInpaintPipeline",
                # SDXL Refiner
                "StableDiffusionXLImg2ImgPipeline",
                # TODO(psyche): Do we actually support LCM models? I don't see using this class anywhere in the codebase.
                "LatentConsistencyModelPipeline",
            },
        )

        cls._validate_base(mod)

        variant = fields.get("variant") or cls._get_variant_or_raise(mod)

        prediction_type = fields.get("prediction_type") or cls._get_scheduler_prediction_type_or_raise(mod)

        repo_variant = fields.get("repo_variant") or cls._get_repo_variant_or_raise(mod)

        return cls(
            **fields,
            variant=variant,
            prediction_type=prediction_type,
            repo_variant=repo_variant,
        )

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatch(cls, f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        # Handle pipelines with a UNet (i.e SD 1.x, SD2.x, SDXL).
        unet_config_path = mod.path / "unet" / "config.json"
        if unet_config_path.exists():
            with open(unet_config_path) as file:
                unet_conf = json.load(file)
            cross_attention_dim = unet_conf.get("cross_attention_dim")
            match cross_attention_dim:
                case 768:
                    return BaseModelType.StableDiffusion1
                case 1024:
                    return BaseModelType.StableDiffusion2
                case 1280:
                    return BaseModelType.StableDiffusionXLRefiner
                case 2048:
                    return BaseModelType.StableDiffusionXL
                case _:
                    raise NotAMatch(cls, f"unrecognized cross_attention_dim {cross_attention_dim}")

        raise NotAMatch(cls, "unable to determine base type")

    @classmethod
    def _get_scheduler_prediction_type_or_raise(cls, mod: ModelOnDisk) -> SchedulerPredictionType:
        scheduler_conf = _get_config_or_raise(cls, mod.path / "scheduler" / "scheduler_config.json")

        # TODO(psyche): Is epsilon the right default or should we raise if it's not present?
        prediction_type = scheduler_conf.get("prediction_type", "epsilon")

        match prediction_type:
            case "v_prediction":
                return SchedulerPredictionType.VPrediction
            case "epsilon":
                return SchedulerPredictionType.Epsilon
            case _:
                raise NotAMatch(cls, f"unrecognized scheduler prediction_type {prediction_type}")

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> ModelVariantType:
        base = cls.model_fields["base"].default
        unet_config = _get_config_or_raise(cls, mod.path / "unet" / "config.json")
        in_channels = unet_config.get("in_channels")

        match in_channels:
            case 4:
                return ModelVariantType.Normal
            case 5:
                # Only SD2 has a depth variant
                assert base is BaseModelType.StableDiffusion2, f"unexpected unet in_channels 5 for base '{base}'"
                return ModelVariantType.Depth
            case 9:
                return ModelVariantType.Inpaint
            case _:
                raise NotAMatch(cls, f"unrecognized unet in_channels {in_channels} for base '{base}'")


class Main_Diffusers_SD1_Config(Main_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(BaseModelType.StableDiffusion1)


class Main_Diffusers_SD2_Config(Main_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(BaseModelType.StableDiffusion2)


class Main_Diffusers_SDXL_Config(Main_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(BaseModelType.StableDiffusionXL)


class Main_Diffusers_SDXLRefiner_Config(Main_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXLRefiner] = Field(BaseModelType.StableDiffusionXLRefiner)


class Main_Diffusers_SD3_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion3] = Field(BaseModelType.StableDiffusion3)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        # This check implies the base type - no further validation needed.
        _validate_class_name(
            cls,
            common_config_paths(mod.path),
            {
                "StableDiffusion3Pipeline",
                "SD3Transformer2DModel",
            },
        )

        submodels = fields.get("submodels") or cls._get_submodels_or_raise(mod)

        repo_variant = fields.get("repo_variant") or cls._get_repo_variant_or_raise(mod)

        return cls(
            **fields,
            submodels=submodels,
            repo_variant=repo_variant,
        )

    @classmethod
    def _get_submodels_or_raise(cls, mod: ModelOnDisk) -> dict[SubModelType, SubmodelDefinition]:
        # Example: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/model_index.json
        config = _get_config_or_raise(cls, common_config_paths(mod.path))

        submodels: dict[SubModelType, SubmodelDefinition] = {}

        for key, value in config.items():
            # Anything that starts with an underscore is top-level metadata, not a submodel
            if key.startswith("_") or not (isinstance(value, list) and len(value) == 2):
                continue
            # The key is something like "transformer" and is a submodel - it will be in a dir of the same name.
            # The value value is something like ["diffusers", "SD3Transformer2DModel"]
            _library_name, class_name = value

            match class_name:
                case "CLIPTextModelWithProjection":
                    model_type = ModelType.CLIPEmbed
                    path_or_prefix = (mod.path / key).resolve().as_posix()

                    # We need to read the config to determine the variant of the CLIP model.
                    clip_embed_config = _get_config_or_raise(
                        cls, {mod.path / key / "config.json", mod.path / key / "model_index.json"}
                    )
                    variant = _get_clip_variant_type_from_config(clip_embed_config)
                    submodels[SubModelType(key)] = SubmodelDefinition(
                        path_or_prefix=path_or_prefix,
                        model_type=model_type,
                        variant=variant,
                    )
                case "SD3Transformer2DModel":
                    model_type = ModelType.Main
                    path_or_prefix = (mod.path / key).resolve().as_posix()
                    variant = None
                    submodels[SubModelType(key)] = SubmodelDefinition(
                        path_or_prefix=path_or_prefix,
                        model_type=model_type,
                        variant=variant,
                    )
                case _:
                    pass

        return submodels


class Main_Diffusers_CogView4_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    base: Literal[BaseModelType.CogView4] = Field(BaseModelType.CogView4)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        # This check implies the base type - no further validation needed.
        _validate_class_name(
            cls,
            common_config_paths(mod.path),
            {
                "CogView4Pipeline",
            },
        )

        repo_variant = fields.get("repo_variant") or cls._get_repo_variant_or_raise(mod)

        return cls(
            **fields,
            repo_variant=repo_variant,
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
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        cls._validate_has_weights_file(mod)

        cls._validate_has_image_encoder_metadata_file(mod)

        cls._validate_base(mod)

        return cls(**fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatch(cls, f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _validate_has_weights_file(cls, mod: ModelOnDisk) -> None:
        weights_file = mod.path / "ip_adapter.bin"
        if not weights_file.exists():
            raise NotAMatch(cls, "missing ip_adapter.bin weights file")

    @classmethod
    def _validate_has_image_encoder_metadata_file(cls, mod: ModelOnDisk) -> None:
        image_encoder_metadata_file = mod.path / "image_encoder.txt"
        if not image_encoder_metadata_file.exists():
            raise NotAMatch(cls, "missing image_encoder.txt metadata file")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        state_dict = mod.load_state_dict()

        try:
            cross_attention_dim = state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[-1]
        except Exception as e:
            raise NotAMatch(cls, f"unable to determine cross attention dimension: {e}") from e

        match cross_attention_dim:
            case 1280:
                return BaseModelType.StableDiffusionXL
            case 768:
                return BaseModelType.StableDiffusion1
            case 1024:
                return BaseModelType.StableDiffusion2
            case _:
                raise NotAMatch(cls, f"unrecognized cross attention dimension {cross_attention_dim}")


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
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_file(cls, mod)

        _validate_override_fields(cls, fields)

        cls._validate_looks_like_ip_adapter(mod)

        cls._validate_base(mod)

        return cls(**fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatch(cls, f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _validate_looks_like_ip_adapter(cls, mod: ModelOnDisk) -> None:
        if not has_any_keys_starting_with(
            mod.load_state_dict(),
            {
                "image_proj.",
                "ip_adapter.",
                # XLabs FLUX IP-Adapter models have keys startinh with "ip_adapter_proj_model.".
                "ip_adapter_proj_model.",
            },
        ):
            raise NotAMatch(cls, "model does not match Checkpoint IP Adapter heuristics")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        state_dict = mod.load_state_dict()

        if is_state_dict_xlabs_ip_adapter(state_dict):
            return BaseModelType.Flux

        try:
            cross_attention_dim = state_dict["ip_adapter.1.to_k_ip.weight"].shape[-1]
        except Exception as e:
            raise NotAMatch(cls, f"unable to determine cross attention dimension: {e}") from e

        match cross_attention_dim:
            case 1280:
                return BaseModelType.StableDiffusionXL
            case 768:
                return BaseModelType.StableDiffusion1
            case 1024:
                return BaseModelType.StableDiffusion2
            case _:
                raise NotAMatch(cls, f"unrecognized cross attention dimension {cross_attention_dim}")


class IPAdapter_Checkpoint_SD1_Config(IPAdapter_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class IPAdapter_Checkpoint_SD2_Config(IPAdapter_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class IPAdapter_Checkpoint_SDXL_Config(IPAdapter_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class IPAdapter_Checkpoint_FLUX_Config(IPAdapter_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


def _get_clip_variant_type_from_config(config: dict[str, Any]) -> ClipVariantType | None:
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
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        _validate_class_name(
            cls,
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

        return cls(**fields)

    @classmethod
    def _validate_variant(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model variant does not match this config class."""
        expected_variant = cls.model_fields["variant"].default
        config = _get_config_or_raise(
            cls,
            {
                mod.path / "config.json",
                mod.path / "text_encoder" / "config.json",
            },
        )
        recognized_variant = _get_clip_variant_type_from_config(config)

        if recognized_variant is None:
            raise NotAMatch(cls, "unable to determine CLIP variant from config")

        if expected_variant is not recognized_variant:
            raise NotAMatch(cls, f"variant is {recognized_variant}, not {expected_variant}")


class CLIPEmbed_Diffusers_G_Config(CLIPEmbed_Diffusers_Config_Base, Config_Base):
    variant: Literal[ClipVariantType.G] = Field(default=ClipVariantType.G)


class CLIPEmbed_Diffusers_L_Config(CLIPEmbed_Diffusers_Config_Base, Config_Base):
    variant: Literal[ClipVariantType.L] = Field(default=ClipVariantType.L)


class CLIPVision_Diffusers_Config(Diffusers_Config_Base, Config_Base):
    """Model config for CLIPVision."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.CLIPVision] = Field(default=ModelType.CLIPVision)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        _validate_class_name(
            cls,
            common_config_paths(mod.path),
            {
                "CLIPVisionModelWithProjection",
            },
        )

        return cls(**fields)


class T2IAdapter_Diffusers_Config_Base(Diffusers_Config_Base, ControlAdapter_Config_Base):
    """Model config for T2I."""

    type: Literal[ModelType.T2IAdapter] = Field(default=ModelType.T2IAdapter)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        _validate_class_name(
            cls,
            common_config_paths(mod.path),
            {
                "T2IAdapter",
            },
        )

        cls._validate_base(mod)

        return cls(**fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatch(cls, f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        config = _get_config_or_raise(cls, common_config_paths(mod.path))

        adapter_type = config.get("adapter_type")

        match adapter_type:
            case "full_adapter_xl":
                return BaseModelType.StableDiffusionXL
            case "full_adapter" | "light_adapter":
                return BaseModelType.StableDiffusion1
            case _:
                raise NotAMatch(cls, f"unrecognized adapter_type '{adapter_type}'")


class T2IAdapter_Diffusers_SD1_Config(T2IAdapter_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class T2IAdapter_Diffusers_SDXL_Config(T2IAdapter_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class Spandrel_Checkpoint_Config(Config_Base):
    """Model config for Spandrel Image to Image models."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.SpandrelImageToImage] = Field(default=ModelType.SpandrelImageToImage)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_file(cls, mod)

        _validate_override_fields(cls, fields)

        cls._validate_spandrel_loads_model(mod)

        return cls(**fields)

    @classmethod
    def _validate_spandrel_loads_model(cls, mod: ModelOnDisk) -> None:
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
        except Exception as e:
            raise NotAMatch(cls, "model does not match SpandrelImageToImage heuristics") from e


class SigLIP_Diffusers_Config(Diffusers_Config_Base, Config_Base):
    """Model config for SigLIP."""

    type: Literal[ModelType.SigLIP] = Field(default=ModelType.SigLIP)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)
    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        _validate_class_name(
            cls,
            common_config_paths(mod.path),
            {
                "SiglipModel",
            },
        )

        return cls(**fields)


class FLUXRedux_Checkpoint_Config(Config_Base):
    """Model config for FLUX Tools Redux model."""

    type: Literal[ModelType.FluxRedux] = Field(default=ModelType.FluxRedux)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_file(cls, mod)

        _validate_override_fields(cls, fields)

        if not is_state_dict_likely_flux_redux(mod.load_state_dict()):
            raise NotAMatch(cls, "model does not match FLUX Tools Redux heuristics")

        return cls(**fields)


class LlavaOnevision_Diffusers_Config(Diffusers_Config_Base, Config_Base):
    """Model config for Llava Onevision models."""

    type: Literal[ModelType.LlavaOnevision] = Field(default=ModelType.LlavaOnevision)
    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    variant: Literal[ModelVariantType.Normal] = Field(default=ModelVariantType.Normal)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        _validate_is_dir(cls, mod)

        _validate_override_fields(cls, fields)

        _validate_class_name(
            cls,
            common_config_paths(mod.path),
            {
                "LlavaOnevisionForConditionalGeneration",
            },
        )

        return cls(**fields)


class ExternalAPI_Config_Base(ABC, BaseModel):
    """Model config for API-based models."""

    format: Literal[ModelFormat.Api] = Field(default=ModelFormat.Api)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, fields: dict[str, Any]) -> Self:
        raise NotAMatch(cls, "External API models cannot be built from disk")


class ExternalAPI_ChatGPT4o_Config(ExternalAPI_Config_Base, Main_Config_Base, Config_Base):
    base: Literal[BaseModelType.ChatGPT4o] = Field(default=BaseModelType.ChatGPT4o)


class ExternalAPI_Gemini2_5_Config(ExternalAPI_Config_Base, Main_Config_Base, Config_Base):
    base: Literal[BaseModelType.Gemini2_5] = Field(default=BaseModelType.Gemini2_5)


class ExternalAPI_Imagen3_Config(ExternalAPI_Config_Base, Main_Config_Base, Config_Base):
    base: Literal[BaseModelType.Imagen3] = Field(default=BaseModelType.Imagen3)


class ExternalAPI_Imagen4_Config(ExternalAPI_Config_Base, Main_Config_Base, Config_Base):
    base: Literal[BaseModelType.Imagen4] = Field(default=BaseModelType.Imagen4)


class ExternalAPI_FluxKontext_Config(ExternalAPI_Config_Base, Main_Config_Base, Config_Base):
    base: Literal[BaseModelType.FluxKontext] = Field(default=BaseModelType.FluxKontext)


class VideoConfigBase(ABC, BaseModel):
    type: Literal[ModelType.Video] = Field(default=ModelType.Video)
    trigger_phrases: set[str] | None = Field(description="Set of trigger phrases for this model", default=None)
    default_settings: MainModelDefaultSettings | None = Field(
        description="Default settings for this model", default=None
    )


class ExternalAPI_Veo3_Config(ExternalAPI_Config_Base, VideoConfigBase, Config_Base):
    base: Literal[BaseModelType.FluxKontext] = Field(default=BaseModelType.FluxKontext)


class ExternalAPI_Runway_Config(ExternalAPI_Config_Base, VideoConfigBase, Config_Base):
    base: Literal[BaseModelType.FluxKontext] = Field(default=BaseModelType.FluxKontext)


# The types are listed explicitly because IDEs/LSPs can't identify the correct types
# when AnyModelConfig is constructed dynamically using ModelConfigBase.all_config_classes
AnyModelConfig = Annotated[
    Union[
        # Main (Pipeline) - diffusers format
        Annotated[Main_Diffusers_SD1_Config, Main_Diffusers_SD1_Config.get_tag()],
        Annotated[Main_Diffusers_SD2_Config, Main_Diffusers_SD2_Config.get_tag()],
        Annotated[Main_Diffusers_SDXL_Config, Main_Diffusers_SDXL_Config.get_tag()],
        Annotated[Main_Diffusers_SDXLRefiner_Config, Main_Diffusers_SDXLRefiner_Config.get_tag()],
        Annotated[Main_Diffusers_SD3_Config, Main_Diffusers_SD3_Config.get_tag()],
        Annotated[Main_Diffusers_CogView4_Config, Main_Diffusers_CogView4_Config.get_tag()],
        # Main (Pipeline) - checkpoint format
        Annotated[Main_Checkpoint_SD1_Config, Main_Checkpoint_SD1_Config.get_tag()],
        Annotated[Main_Checkpoint_SD2_Config, Main_Checkpoint_SD2_Config.get_tag()],
        Annotated[Main_Checkpoint_SDXL_Config, Main_Checkpoint_SDXL_Config.get_tag()],
        Annotated[Main_Checkpoint_SDXLRefiner_Config, Main_Checkpoint_SDXLRefiner_Config.get_tag()],
        Annotated[Main_Checkpoint_FLUX_Config, Main_Checkpoint_FLUX_Config.get_tag()],
        # Main (Pipeline) - quantized formats
        Annotated[Main_BnBNF4_FLUX_Config, Main_BnBNF4_FLUX_Config.get_tag()],
        Annotated[Main_GGUF_FLUX_Config, Main_GGUF_FLUX_Config.get_tag()],
        # VAE - checkpoint format
        Annotated[VAE_Checkpoint_SD1_Config, VAE_Checkpoint_SD1_Config.get_tag()],
        Annotated[VAE_Checkpoint_SD2_Config, VAE_Checkpoint_SD2_Config.get_tag()],
        Annotated[VAE_Checkpoint_SDXL_Config, VAE_Checkpoint_SDXL_Config.get_tag()],
        Annotated[VAE_Checkpoint_FLUX_Config, VAE_Checkpoint_FLUX_Config.get_tag()],
        # VAE - diffusers format
        Annotated[VAE_Diffusers_SD1_Config, VAE_Diffusers_SD1_Config.get_tag()],
        Annotated[VAE_Diffusers_SDXL_Config, VAE_Diffusers_SDXL_Config.get_tag()],
        # ControlNet - checkpoint format
        Annotated[ControlNet_Checkpoint_SD1_Config, ControlNet_Checkpoint_SD1_Config.get_tag()],
        Annotated[ControlNet_Checkpoint_SD2_Config, ControlNet_Checkpoint_SD2_Config.get_tag()],
        Annotated[ControlNet_Checkpoint_SDXL_Config, ControlNet_Checkpoint_SDXL_Config.get_tag()],
        Annotated[ControlNet_Checkpoint_FLUX_Config, ControlNet_Checkpoint_FLUX_Config.get_tag()],
        # ControlNet - diffusers format
        Annotated[ControlNet_Diffusers_SD1_Config, ControlNet_Diffusers_SD1_Config.get_tag()],
        Annotated[ControlNet_Diffusers_SD2_Config, ControlNet_Diffusers_SD2_Config.get_tag()],
        Annotated[ControlNet_Diffusers_SDXL_Config, ControlNet_Diffusers_SDXL_Config.get_tag()],
        Annotated[ControlNet_Diffusers_FLUX_Config, ControlNet_Diffusers_FLUX_Config.get_tag()],
        # LoRA - LyCORIS format
        Annotated[LoRA_LyCORIS_SD1_Config, LoRA_LyCORIS_SD1_Config.get_tag()],
        Annotated[LoRA_LyCORIS_SD2_Config, LoRA_LyCORIS_SD2_Config.get_tag()],
        Annotated[LoRA_LyCORIS_SDXL_Config, LoRA_LyCORIS_SDXL_Config.get_tag()],
        Annotated[LoRA_LyCORIS_FLUX_Config, LoRA_LyCORIS_FLUX_Config.get_tag()],
        # LoRA - OMI format
        Annotated[LoRA_OMI_SDXL_Config, LoRA_OMI_SDXL_Config.get_tag()],
        Annotated[LoRA_OMI_FLUX_Config, LoRA_OMI_FLUX_Config.get_tag()],
        # LoRA - diffusers format
        Annotated[LoRA_Diffusers_SD1_Config, LoRA_Diffusers_SD1_Config.get_tag()],
        Annotated[LoRA_Diffusers_SD2_Config, LoRA_Diffusers_SD2_Config.get_tag()],
        Annotated[LoRA_Diffusers_SDXL_Config, LoRA_Diffusers_SDXL_Config.get_tag()],
        Annotated[LoRA_Diffusers_FLUX_Config, LoRA_Diffusers_FLUX_Config.get_tag()],
        # ControlLoRA - diffusers format
        Annotated[ControlLoRA_LyCORIS_FLUX_Config, ControlLoRA_LyCORIS_FLUX_Config.get_tag()],
        # T5 Encoder - all formats
        Annotated[T5Encoder_T5Encoder_Config, T5Encoder_T5Encoder_Config.get_tag()],
        Annotated[T5Encoder_BnBLLMint8_Config, T5Encoder_BnBLLMint8_Config.get_tag()],
        # TI - file format
        Annotated[TI_File_SD1_Config, TI_File_SD1_Config.get_tag()],
        Annotated[TI_File_SD2_Config, TI_File_SD2_Config.get_tag()],
        Annotated[TI_File_SDXL_Config, TI_File_SDXL_Config.get_tag()],
        # TI - folder format
        Annotated[TI_Folder_SD1_Config, TI_Folder_SD1_Config.get_tag()],
        Annotated[TI_Folder_SD2_Config, TI_Folder_SD2_Config.get_tag()],
        Annotated[TI_Folder_SDXL_Config, TI_Folder_SDXL_Config.get_tag()],
        # IP Adapter - InvokeAI format
        Annotated[IPAdapter_InvokeAI_SD1_Config, IPAdapter_InvokeAI_SD1_Config.get_tag()],
        Annotated[IPAdapter_InvokeAI_SD2_Config, IPAdapter_InvokeAI_SD2_Config.get_tag()],
        Annotated[IPAdapter_InvokeAI_SDXL_Config, IPAdapter_InvokeAI_SDXL_Config.get_tag()],
        # IP Adapter - checkpoint format
        Annotated[IPAdapter_Checkpoint_SD1_Config, IPAdapter_Checkpoint_SD1_Config.get_tag()],
        Annotated[IPAdapter_Checkpoint_SD2_Config, IPAdapter_Checkpoint_SD2_Config.get_tag()],
        Annotated[IPAdapter_Checkpoint_SDXL_Config, IPAdapter_Checkpoint_SDXL_Config.get_tag()],
        Annotated[IPAdapter_Checkpoint_FLUX_Config, IPAdapter_Checkpoint_FLUX_Config.get_tag()],
        # T2I Adapter - diffusers format
        Annotated[T2IAdapter_Diffusers_SD1_Config, T2IAdapter_Diffusers_SD1_Config.get_tag()],
        Annotated[T2IAdapter_Diffusers_SDXL_Config, T2IAdapter_Diffusers_SDXL_Config.get_tag()],
        # Misc models
        Annotated[Spandrel_Checkpoint_Config, Spandrel_Checkpoint_Config.get_tag()],
        Annotated[CLIPEmbed_Diffusers_G_Config, CLIPEmbed_Diffusers_G_Config.get_tag()],
        Annotated[CLIPEmbed_Diffusers_L_Config, CLIPEmbed_Diffusers_L_Config.get_tag()],
        Annotated[CLIPVision_Diffusers_Config, CLIPVision_Diffusers_Config.get_tag()],
        Annotated[SigLIP_Diffusers_Config, SigLIP_Diffusers_Config.get_tag()],
        Annotated[FLUXRedux_Checkpoint_Config, FLUXRedux_Checkpoint_Config.get_tag()],
        Annotated[LlavaOnevision_Diffusers_Config, LlavaOnevision_Diffusers_Config.get_tag()],
        # API models
        Annotated[ExternalAPI_ChatGPT4o_Config, ExternalAPI_ChatGPT4o_Config.get_tag()],
        Annotated[ExternalAPI_Gemini2_5_Config, ExternalAPI_Gemini2_5_Config.get_tag()],
        Annotated[ExternalAPI_Imagen3_Config, ExternalAPI_Imagen3_Config.get_tag()],
        Annotated[ExternalAPI_Imagen4_Config, ExternalAPI_Imagen4_Config.get_tag()],
        Annotated[ExternalAPI_FluxKontext_Config, ExternalAPI_FluxKontext_Config.get_tag()],
        Annotated[ExternalAPI_Veo3_Config, ExternalAPI_Veo3_Config.get_tag()],
        Annotated[ExternalAPI_Runway_Config, ExternalAPI_Runway_Config.get_tag()],
        # Unknown model (fallback)
        Annotated[Unknown_Config, Unknown_Config.get_tag()],
    ],
    Discriminator(Config_Base.get_model_discriminator_value),
]

AnyModelConfigValidator = TypeAdapter[AnyModelConfig](AnyModelConfig)


class ModelConfigFactory:
    @staticmethod
    def make_config(model_data: Dict[str, Any], timestamp: Optional[float] = None) -> AnyModelConfig:
        """Return the appropriate config object from raw dict values."""
        model = AnyModelConfigValidator.validate_python(model_data)
        if isinstance(model, Checkpoint_Config_Base) and timestamp:
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
        for config_class in Config_Base.CONFIG_CLASSES:
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

        matches = [r for r in results.values() if isinstance(r, Config_Base)]

        if not matches and app_config.allow_unknown_models:
            logger.warning(f"Unable to identify model {mod.name}, falling back to Unknown_Config")
            return Unknown_Config(**fields)

        if len(matches) > 1:
            # We have multiple matches, in which case at most 1 is correct. We need to pick one.
            #
            # Known cases:
            # - SD main models can look like a LoRA when they have merged in LoRA weights. Prefer the main model.
            # - SD main models in diffusers format can look like a CLIP Embed; they have a text_encoder folder with
            #   a config.json file. Prefer the main model.

            # Sort the matching according to known special cases.
            def sort_key(m: AnyModelConfig) -> int:
                match m.type:
                    case ModelType.Main:
                        return 0
                    case ModelType.LoRA:
                        return 1
                    case ModelType.CLIPEmbed:
                        return 2
                    case _:
                        return 3

            matches.sort(key=sort_key)
            logger.warning(
                f"Multiple model config classes matched for model {mod.name}: {[type(m).__name__ for m in matches]}. Using {type(matches[0]).__name__}."
            )

        instance = matches[0]
        logger.info(f"Model {mod.name} classified as {type(instance).__name__}")
        return instance
