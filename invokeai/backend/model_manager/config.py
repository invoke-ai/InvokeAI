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

import time
from enum import Enum
from typing import Literal, Optional, Type, TypeAlias, Union

import torch
from diffusers.models.modeling_utils import ModelMixin
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, TypeAdapter
from typing_extensions import Annotated, Any, Dict

from invokeai.app.invocations.constants import SCHEDULER_NAME_VALUES
from invokeai.app.util.misc import uuid_string

from ..raw_model import RawModel

# ModelMixin is the base class for all diffusers and transformers models
# RawModel is the InvokeAI wrapper class for ip_adapters, loras, textual_inversion and onnx runtime
AnyModel = Union[ModelMixin, RawModel, torch.nn.Module]


class InvalidModelConfigException(Exception):
    """Exception for when config parser doesn't recognized this combination of model type and format."""


class BaseModelType(str, Enum):
    """Base model type."""

    Any = "any"
    StableDiffusion1 = "sd-1"
    StableDiffusion2 = "sd-2"
    StableDiffusionXL = "sdxl"
    StableDiffusionXLRefiner = "sdxl-refiner"
    # Kandinsky2_1 = "kandinsky-2.1"


class ModelType(str, Enum):
    """Model type."""

    ONNX = "onnx"
    Main = "main"
    VAE = "vae"
    LoRA = "lora"
    ControlNet = "controlnet"  # used by model_probe
    TextualInversion = "embedding"
    IPAdapter = "ip_adapter"
    CLIPVision = "clip_vision"
    T2IAdapter = "t2i_adapter"


class SubModelType(str, Enum):
    """Submodel type."""

    UNet = "unet"
    TextEncoder = "text_encoder"
    TextEncoder2 = "text_encoder_2"
    Tokenizer = "tokenizer"
    Tokenizer2 = "tokenizer_2"
    VAE = "vae"
    VAEDecoder = "vae_decoder"
    VAEEncoder = "vae_encoder"
    Scheduler = "scheduler"
    SafetyChecker = "safety_checker"


class ModelVariantType(str, Enum):
    """Variant type."""

    Normal = "normal"
    Inpaint = "inpaint"
    Depth = "depth"


class ModelFormat(str, Enum):
    """Storage format of model."""

    Diffusers = "diffusers"
    Checkpoint = "checkpoint"
    LyCORIS = "lycoris"
    ONNX = "onnx"
    Olive = "olive"
    EmbeddingFile = "embedding_file"
    EmbeddingFolder = "embedding_folder"
    InvokeAI = "invokeai"


class SchedulerPredictionType(str, Enum):
    """Scheduler prediction type."""

    Epsilon = "epsilon"
    VPrediction = "v_prediction"
    Sample = "sample"


class ModelRepoVariant(str, Enum):
    """Various hugging face variants on the diffusers format."""

    Default = ""  # model files without "fp16" or other qualifier - empty str
    FP16 = "fp16"
    FP32 = "fp32"
    ONNX = "onnx"
    OpenVINO = "openvino"
    Flax = "flax"


class ModelSourceType(str, Enum):
    """Model source type."""

    Path = "path"
    Url = "url"
    HFRepoID = "hf_repo_id"


DEFAULTS_PRECISION = Literal["fp16", "fp32"]


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

    model_config = ConfigDict(extra="forbid")


class ControlAdapterDefaultSettings(BaseModel):
    # This could be narrowed to controlnet processor nodes, but they change. Leaving this a string is safer.
    preprocessor: str | None

    model_config = ConfigDict(extra="forbid")


class ModelConfigBase(BaseModel):
    """Base class for model configuration information."""

    key: str = Field(description="A unique key for this model.", default_factory=uuid_string)
    hash: str = Field(description="The hash of the model file(s).")
    path: str = Field(
        description="Path to the model on the filesystem. Relative paths are relative to the Invoke root directory."
    )
    name: str = Field(description="Name of the model.")
    base: BaseModelType = Field(description="The base model.")
    description: Optional[str] = Field(description="Model description", default=None)
    source: str = Field(description="The original source of the model (path, URL or repo_id).")
    source_type: ModelSourceType = Field(description="The type of source")
    source_api_response: Optional[str] = Field(
        description="The original API response from the source, as stringified JSON.", default=None
    )
    cover_image: Optional[str] = Field(description="Url for image to preview model", default=None)

    @staticmethod
    def json_schema_extra(schema: dict[str, Any], model_class: Type[BaseModel]) -> None:
        schema["required"].extend(["key", "type", "format"])

    model_config = ConfigDict(validate_assignment=True, json_schema_extra=json_schema_extra)


class CheckpointConfigBase(ModelConfigBase):
    """Model config for checkpoint-style models."""

    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint
    config_path: str = Field(description="path to the checkpoint model config file")
    converted_at: Optional[float] = Field(
        description="When this model was last converted to diffusers", default_factory=time.time
    )


class DiffusersConfigBase(ModelConfigBase):
    """Model config for diffusers-style models."""

    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers
    repo_variant: Optional[ModelRepoVariant] = ModelRepoVariant.Default


class LoRAConfigBase(ModelConfigBase):
    type: Literal[ModelType.LoRA] = ModelType.LoRA
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)


class LoRALyCORISConfig(LoRAConfigBase):
    """Model config for LoRA/Lycoris models."""

    format: Literal[ModelFormat.LyCORIS] = ModelFormat.LyCORIS

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.LoRA.value}.{ModelFormat.LyCORIS.value}")


class LoRADiffusersConfig(LoRAConfigBase):
    """Model config for LoRA/Diffusers models."""

    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.LoRA.value}.{ModelFormat.Diffusers.value}")


class VAECheckpointConfig(CheckpointConfigBase):
    """Model config for standalone VAE models."""

    type: Literal[ModelType.VAE] = ModelType.VAE
    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.VAE.value}.{ModelFormat.Checkpoint.value}")


class VAEDiffusersConfig(ModelConfigBase):
    """Model config for standalone VAE models (diffusers version)."""

    type: Literal[ModelType.VAE] = ModelType.VAE
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.VAE.value}.{ModelFormat.Diffusers.value}")


class ControlAdapterConfigBase(BaseModel):
    default_settings: Optional[ControlAdapterDefaultSettings] = Field(
        description="Default settings for this model", default=None
    )


class ControlNetDiffusersConfig(DiffusersConfigBase, ControlAdapterConfigBase):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = ModelType.ControlNet
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.ControlNet.value}.{ModelFormat.Diffusers.value}")


class ControlNetCheckpointConfig(CheckpointConfigBase, ControlAdapterConfigBase):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = ModelType.ControlNet
    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.ControlNet.value}.{ModelFormat.Checkpoint.value}")


class TextualInversionFileConfig(ModelConfigBase):
    """Model config for textual inversion embeddings."""

    type: Literal[ModelType.TextualInversion] = ModelType.TextualInversion
    format: Literal[ModelFormat.EmbeddingFile] = ModelFormat.EmbeddingFile

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.TextualInversion.value}.{ModelFormat.EmbeddingFile.value}")


class TextualInversionFolderConfig(ModelConfigBase):
    """Model config for textual inversion embeddings."""

    type: Literal[ModelType.TextualInversion] = ModelType.TextualInversion
    format: Literal[ModelFormat.EmbeddingFolder] = ModelFormat.EmbeddingFolder

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.TextualInversion.value}.{ModelFormat.EmbeddingFolder.value}")


class MainConfigBase(ModelConfigBase):
    type: Literal[ModelType.Main] = ModelType.Main
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)
    default_settings: Optional[MainModelDefaultSettings] = Field(
        description="Default settings for this model", default=None
    )


class MainCheckpointConfig(CheckpointConfigBase, MainConfigBase):
    """Model config for main checkpoint models."""

    variant: ModelVariantType = ModelVariantType.Normal
    prediction_type: SchedulerPredictionType = SchedulerPredictionType.Epsilon
    upcast_attention: bool = False

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.Main.value}.{ModelFormat.Checkpoint.value}")


class MainDiffusersConfig(DiffusersConfigBase, MainConfigBase):
    """Model config for main diffusers models."""

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.Main.value}.{ModelFormat.Diffusers.value}")


class IPAdapterBaseConfig(ModelConfigBase):
    type: Literal[ModelType.IPAdapter] = ModelType.IPAdapter


class IPAdapterInvokeAIConfig(IPAdapterBaseConfig):
    """Model config for IP Adapter diffusers format models."""

    image_encoder_model_id: str
    format: Literal[ModelFormat.InvokeAI] = ModelFormat.InvokeAI

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.IPAdapter.value}.{ModelFormat.InvokeAI.value}")


class IPAdapterCheckpointConfig(IPAdapterBaseConfig):
    """Model config for IP Adapter checkpoint format models."""

    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.IPAdapter.value}.{ModelFormat.Checkpoint.value}")


class CLIPVisionBaseConfig(ModelConfigBase):
    type: Literal[ModelType.CLIPVision] = ModelType.CLIPVision


class CLIPVisionCheckpointConfig(CLIPVisionBaseConfig):
    """Model config for CLIPVision."""

    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.CLIPVision.value}.{ModelFormat.Checkpoint.value}")


class CLIPVisionDiffusersConfig(CLIPVisionBaseConfig):
    """Model config for CLIPVision."""

    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.CLIPVision.value}.{ModelFormat.Diffusers.value}")


class T2IAdapterConfig(DiffusersConfigBase, ControlAdapterConfigBase):
    """Model config for T2I."""

    type: Literal[ModelType.T2IAdapter] = ModelType.T2IAdapter
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.T2IAdapter.value}.{ModelFormat.Diffusers.value}")


def get_model_discriminator_value(v: Any) -> str:
    """
    Computes the discriminator value for a model config.
    https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions-with-callable-discriminator
    """
    format_ = None
    type_ = None
    if isinstance(v, dict):
        format_ = v.get("format")
        if isinstance(format_, Enum):
            format_ = format_.value
        type_ = v.get("type")
        if isinstance(type_, Enum):
            type_ = type_.value
    else:
        format_ = v.format.value
        type_ = v.type.value
    v = f"{type_}.{format_}"
    return v


AnyModelConfig = Annotated[
    Union[
        Annotated[MainDiffusersConfig, MainDiffusersConfig.get_tag()],
        Annotated[MainCheckpointConfig, MainCheckpointConfig.get_tag()],
        Annotated[VAEDiffusersConfig, VAEDiffusersConfig.get_tag()],
        Annotated[VAECheckpointConfig, VAECheckpointConfig.get_tag()],
        Annotated[ControlNetDiffusersConfig, ControlNetDiffusersConfig.get_tag()],
        Annotated[ControlNetCheckpointConfig, ControlNetCheckpointConfig.get_tag()],
        Annotated[LoRALyCORISConfig, LoRALyCORISConfig.get_tag()],
        Annotated[LoRADiffusersConfig, LoRADiffusersConfig.get_tag()],
        Annotated[TextualInversionFileConfig, TextualInversionFileConfig.get_tag()],
        Annotated[TextualInversionFolderConfig, TextualInversionFolderConfig.get_tag()],
        Annotated[IPAdapterInvokeAIConfig, IPAdapterInvokeAIConfig.get_tag()],
        Annotated[IPAdapterCheckpointConfig, IPAdapterCheckpointConfig.get_tag()],
        Annotated[T2IAdapterConfig, T2IAdapterConfig.get_tag()],
        Annotated[CLIPVisionDiffusersConfig, CLIPVisionDiffusersConfig.get_tag()],
        Annotated[CLIPVisionCheckpointConfig, CLIPVisionCheckpointConfig.get_tag()],
    ],
    Discriminator(get_model_discriminator_value),
]

AnyModelConfigValidator = TypeAdapter(AnyModelConfig)
AnyDefaultSettings: TypeAlias = Union[MainModelDefaultSettings, ControlAdapterDefaultSettings]


class ModelConfigFactory(object):
    """Class for parsing config dicts into StableDiffusion Config obects."""

    @classmethod
    def make_config(
        cls,
        model_data: Union[Dict[str, Any], AnyModelConfig],
        key: Optional[str] = None,
        dest_class: Optional[Type[ModelConfigBase]] = None,
        timestamp: Optional[float] = None,
    ) -> AnyModelConfig:
        """
        Return the appropriate config object from raw dict values.

        :param model_data: A raw dict corresponding the obect fields to be
        parsed into a ModelConfigBase obect (or descendent), or a ModelConfigBase
        object, which will be passed through unchanged.
        :param dest_class: The config class to be returned. If not provided, will
        be selected automatically.
        """
        model: Optional[ModelConfigBase] = None
        if isinstance(model_data, ModelConfigBase):
            model = model_data
        elif dest_class:
            model = dest_class.model_validate(model_data)
        else:
            # mypy doesn't typecheck TypeAdapters well?
            model = AnyModelConfigValidator.validate_python(model_data)  # type: ignore
        assert model is not None
        if key:
            model.key = key
        if isinstance(model, CheckpointConfigBase) and timestamp is not None:
            model.converted_at = timestamp
        return model  # type: ignore
