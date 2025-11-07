from abc import ABC, abstractmethod
from enum import Enum
from inspect import isabstract
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Self,
    Type,
)

from pydantic import BaseModel, ConfigDict, Field, Tag
from pydantic_core import PydanticUndefined

from invokeai.app.util.misc import uuid_string
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    AnyVariant,
    BaseModelType,
    ModelFormat,
    ModelRepoVariant,
    ModelSourceType,
    ModelType,
)

if TYPE_CHECKING:
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

    # These fields are common to all model configs.

    key: str = Field(
        default_factory=uuid_string,
        description="A unique key for this model.",
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
        default=None,
        description="Model description",
    )
    source: str = Field(
        description="The original source of the model (path, URL or repo_id).",
    )
    source_type: ModelSourceType = Field(
        description="The type of source",
    )
    source_api_response: str | None = Field(
        default=None,
        description="The original API response from the source, as stringified JSON.",
    )
    cover_image: str | None = Field(
        default=None,
        description="Url for image to preview model",
    )

    CONFIG_CLASSES: ClassVar[set[Type["Config_Base"]]] = set()
    """Set of all non-abstract subclasses of Config_Base, for use during model probing. In other words, this is the set
    of all known model config types."""

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_serialization_defaults_required=True,
        json_schema_mode_override="serialization",
    )

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register non-abstract subclasses so we can iterate over them later during model probing. Note that
        # isabstract() will return False if the class does not have any abstract methods, even if it inherits from ABC.
        # We must check for ABC lest we unintentionally register some abstract model config classes.
        if not isabstract(cls) and ABC not in cls.__bases__:
            cls.CONFIG_CLASSES.add(cls)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        # Ensure that model configs define 'base', 'type' and 'format' fields and provide defaults for them. Each
        # subclass is expected to represent a single combination of base, type and format.
        #
        # This pydantic dunder method is called after the pydantic model for a class is created. The normal
        # __init_subclass__ is too early to do this check.
        for name in ("type", "base", "format"):
            if name not in cls.model_fields:
                raise NotImplementedError(f"{cls.__name__} must define a '{name}' field")
            if cls.model_fields[name].default is PydanticUndefined:
                raise NotImplementedError(f"{cls.__name__} must define a default for the '{name}' field")

    @classmethod
    def get_tag(cls) -> Tag:
        """Constructs a pydantic discriminated union tag for this model config class. When a config is deserialized,
        pydantic uses the tag to determine which subclass to instantiate.

        The tag is a dot-separated string of the type, format, base and variant (if applicable).
        """
        tag_strings: list[str] = []
        for name in ("type", "format", "base", "variant"):
            if field := cls.model_fields.get(name):
                # The check in __pydantic_init_subclass__ ensures that type, format and base are always present with
                # defaults. variant does not require a default, but if it has one, we need to add it to the tag. We can
                # check for the presence of a default by seeing if it's not PydanticUndefined, a sentinel value used by
                # pydantic to indicate that no default was provided.
                if field.default is not PydanticUndefined:
                    # We expect each of these fields has an Enum for its default; we want the value of the enum.
                    tag_strings.append(field.default.value)
        return Tag(".".join(tag_strings))

    @staticmethod
    def get_model_discriminator_value(v: Any) -> str:
        """Computes the discriminator value for a model config discriminated union."""
        # This is called by pydantic during deserialization and serialization to determine which model the data
        # represents. It can get either a dict (during deserialization) or an instance of a Config_Base subclass
        # (during serialization).
        #
        # See: https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions-with-callable-discriminator
        if isinstance(v, Config_Base):
            # We have an instance of a ModelConfigBase subclass - use its tag directly.
            return v.get_tag().tag
        if isinstance(v, dict):
            # We have a dict - attempt to compute a tag from its fields.
            tag_strings: list[str] = []
            if type_ := v.get("type"):
                if isinstance(type_, Enum):
                    type_ = str(type_.value)
                elif not isinstance(type_, str):
                    raise ValueError("Model config dict 'type' field must be a string or Enum")
                tag_strings.append(type_)

            if format_ := v.get("format"):
                if isinstance(format_, Enum):
                    format_ = str(format_.value)
                elif not isinstance(format_, str):
                    raise ValueError("Model config dict 'format' field must be a string or Enum")
                tag_strings.append(format_)

            if base_ := v.get("base"):
                if isinstance(base_, Enum):
                    base_ = str(base_.value)
                elif not isinstance(base_, str):
                    raise ValueError("Model config dict 'base' field must be a string or Enum")
                tag_strings.append(base_)

            # Special case: CLIP Embed models also need the variant to distinguish them.
            if (
                type_ == ModelType.CLIPEmbed.value
                and format_ == ModelFormat.Diffusers.value
                and base_ == BaseModelType.Any.value
            ):
                if variant_ := v.get("variant"):
                    if isinstance(variant_, Enum):
                        variant_ = variant_.value
                    elif not isinstance(variant_, str):
                        raise ValueError("Model config dict 'variant' field must be a string or Enum")
                    tag_strings.append(variant_)
                else:
                    raise ValueError("CLIP Embed model config dict must include a 'variant' field")

            return ".".join(tag_strings)
        else:
            raise ValueError(
                "Model config discriminator value must be computed from a dict or ModelConfigBase instance"
            )

    @classmethod
    @abstractmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        """Given the model on disk and any override fields, attempt to construct an instance of this config class.

        This method serves to identify whether the model on disk matches this config class, and if so, to extract any
        additional metadata needed to instantiate the config.

        Implementations should raise a NotAMatchError if the model does not match this config class."""
        raise NotImplementedError(f"from_model_on_disk not implemented for {cls.__name__}")


class Checkpoint_Config_Base(ABC, BaseModel):
    """Base class for checkpoint-style models."""

    config_path: str | None = Field(
        description="Path to the config for this model, if any.",
        default=None,
    )


class Diffusers_Config_Base(ABC, BaseModel):
    """Base class for diffusers-style models."""

    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)
    repo_variant: ModelRepoVariant = Field(ModelRepoVariant.Default)

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


class SubmodelDefinition(BaseModel):
    path_or_prefix: str
    model_type: ModelType
    variant: AnyVariant | None = None

    model_config = ConfigDict(protected_namespaces=())
