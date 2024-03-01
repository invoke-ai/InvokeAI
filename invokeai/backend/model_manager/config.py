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
from typing import Literal, Optional, Type, Union

import torch
from diffusers import ModelMixin
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, TypeAdapter
from typing_extensions import Annotated, Any, Dict

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
    Vae = "vae"
    Lora = "lora"
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
    Vae = "vae"
    VaeDecoder = "vae_decoder"
    VaeEncoder = "vae_encoder"
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
    Lycoris = "lycoris"
    Onnx = "onnx"
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

    DEFAULT = ""  # model files without "fp16" or other qualifier - empty str
    FP16 = "fp16"
    FP32 = "fp32"
    ONNX = "onnx"
    OPENVINO = "openvino"
    FLAX = "flax"


class ModelConfigBase(BaseModel):
    """Base class for model configuration information."""

    path: str = Field(description="filesystem path to the model file or directory")
    name: str = Field(description="model name")
    base: BaseModelType = Field(description="base model")
    key: str = Field(description="unique key for model", default="<NOKEY>")
    original_hash: Optional[str] = Field(
        description="original fasthash of model contents", default=None
    )  # this is assigned at install time and will not change
    current_hash: Optional[str] = Field(
        description="current fasthash of model contents", default=None
    )  # if model is converted or otherwise modified, this will hold updated hash
    description: Optional[str] = Field(description="human readable description of the model", default=None)
    source: Optional[str] = Field(description="model original source (path, URL or repo_id)", default=None)
    last_modified: Optional[float] = Field(description="timestamp for modification time", default_factory=time.time)

    @staticmethod
    def json_schema_extra(schema: dict[str, Any], model_class: Type[BaseModel]) -> None:
        schema["required"].extend(
            ["key", "base", "type", "format", "original_hash", "current_hash", "source", "last_modified"]
        )

    model_config = ConfigDict(
        use_enum_values=False,
        validate_assignment=True,
        json_schema_extra=json_schema_extra,
    )

    def update(self, attributes: Dict[str, Any]) -> None:
        """Update the object with fields in dict."""
        for key, value in attributes.items():
            setattr(self, key, value)  # may raise a validation error


class _CheckpointConfig(ModelConfigBase):
    """Model config for checkpoint-style models."""

    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint
    config: str = Field(description="path to the checkpoint model config file")


class _DiffusersConfig(ModelConfigBase):
    """Model config for diffusers-style models."""

    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers
    repo_variant: Optional[ModelRepoVariant] = ModelRepoVariant.DEFAULT


class LoRALycorisConfig(ModelConfigBase):
    """Model config for LoRA/Lycoris models."""

    type: Literal[ModelType.Lora] = ModelType.Lora
    format: Literal[ModelFormat.Lycoris] = ModelFormat.Lycoris

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.Lora}.{ModelFormat.Lycoris}")


class LoRADiffusersConfig(ModelConfigBase):
    """Model config for LoRA/Diffusers models."""

    type: Literal[ModelType.Lora] = ModelType.Lora
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.Lora}.{ModelFormat.Diffusers}")


class VaeCheckpointConfig(ModelConfigBase):
    """Model config for standalone VAE models."""

    type: Literal[ModelType.Vae] = ModelType.Vae
    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.Vae}.{ModelFormat.Checkpoint}")


class VaeDiffusersConfig(ModelConfigBase):
    """Model config for standalone VAE models (diffusers version)."""

    type: Literal[ModelType.Vae] = ModelType.Vae
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.Vae}.{ModelFormat.Diffusers}")


class ControlNetDiffusersConfig(_DiffusersConfig):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = ModelType.ControlNet
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.ControlNet}.{ModelFormat.Diffusers}")


class ControlNetCheckpointConfig(_CheckpointConfig):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = ModelType.ControlNet
    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.ControlNet}.{ModelFormat.Checkpoint}")


class TextualInversionFileConfig(ModelConfigBase):
    """Model config for textual inversion embeddings."""

    type: Literal[ModelType.TextualInversion] = ModelType.TextualInversion
    format: Literal[ModelFormat.EmbeddingFile] = ModelFormat.EmbeddingFile

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.TextualInversion}.{ModelFormat.EmbeddingFile}")


class TextualInversionFolderConfig(ModelConfigBase):
    """Model config for textual inversion embeddings."""

    type: Literal[ModelType.TextualInversion] = ModelType.TextualInversion
    format: Literal[ModelFormat.EmbeddingFolder] = ModelFormat.EmbeddingFolder

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.TextualInversion}.{ModelFormat.EmbeddingFolder}")


class _MainConfig(ModelConfigBase):
    """Model config for main models."""

    variant: ModelVariantType = ModelVariantType.Normal
    prediction_type: SchedulerPredictionType = SchedulerPredictionType.Epsilon
    upcast_attention: bool = False
    ztsnr_training: bool = False


class MainCheckpointConfig(_CheckpointConfig, _MainConfig):
    """Model config for main checkpoint models."""

    type: Literal[ModelType.Main] = ModelType.Main

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.Main}.{ModelFormat.Checkpoint}")


class MainDiffusersConfig(_DiffusersConfig, _MainConfig):
    """Model config for main diffusers models."""

    type: Literal[ModelType.Main] = ModelType.Main

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.Main}.{ModelFormat.Diffusers}")


class IPAdapterConfig(ModelConfigBase):
    """Model config for IP Adaptor format models."""

    type: Literal[ModelType.IPAdapter] = ModelType.IPAdapter
    image_encoder_model_id: str
    format: Literal[ModelFormat.InvokeAI]

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.IPAdapter}.{ModelFormat.InvokeAI}")


class CLIPVisionDiffusersConfig(ModelConfigBase):
    """Model config for ClipVision."""

    type: Literal[ModelType.CLIPVision] = ModelType.CLIPVision
    format: Literal[ModelFormat.Diffusers]

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.CLIPVision}.{ModelFormat.Diffusers}")


class T2IAdapterConfig(ModelConfigBase):
    """Model config for T2I."""

    type: Literal[ModelType.T2IAdapter] = ModelType.T2IAdapter
    format: Literal[ModelFormat.Diffusers]

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.T2IAdapter}.{ModelFormat.Diffusers}")


def get_model_discriminator_value(v: Any) -> str:
    """
    Computes the discriminator value for a model config.
    https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions-with-callable-discriminator
    """
    if isinstance(v, dict):
        return f"{v.get('type')}.{v.get('format')}"  # pyright: ignore [reportUnknownMemberType]
    return f"{v.getattr('type')}.{v.getattr('format')}"


AnyModelConfig = Annotated[
    Union[
        Annotated[MainDiffusersConfig, MainDiffusersConfig.get_tag()],
        Annotated[MainCheckpointConfig, MainCheckpointConfig.get_tag()],
        Annotated[VaeDiffusersConfig, VaeDiffusersConfig.get_tag()],
        Annotated[VaeCheckpointConfig, VaeCheckpointConfig.get_tag()],
        Annotated[ControlNetDiffusersConfig, ControlNetDiffusersConfig.get_tag()],
        Annotated[ControlNetCheckpointConfig, ControlNetCheckpointConfig.get_tag()],
        Annotated[LoRALycorisConfig, LoRALycorisConfig.get_tag()],
        Annotated[LoRADiffusersConfig, LoRADiffusersConfig.get_tag()],
        Annotated[TextualInversionFileConfig, TextualInversionFileConfig.get_tag()],
        Annotated[TextualInversionFolderConfig, TextualInversionFolderConfig.get_tag()],
        Annotated[IPAdapterConfig, IPAdapterConfig.get_tag()],
        Annotated[T2IAdapterConfig, T2IAdapterConfig.get_tag()],
        Annotated[CLIPVisionDiffusersConfig, CLIPVisionDiffusersConfig.get_tag()],
    ],
    Discriminator(get_model_discriminator_value),
]

AnyModelConfigValidator = TypeAdapter(AnyModelConfig)

# IMPLEMENTATION NOTE:
# The preferred alternative to the above is a discriminated Union as shown
# below. However, it breaks FastAPI when used as the input Body parameter in a route.
# This is a known issue. Please see:
#   https://github.com/tiangolo/fastapi/discussions/9761 and
#   https://github.com/tiangolo/fastapi/discussions/9287
# AnyModelConfig = Annotated[
#     Union[
#         _MainModelConfig,
#         _ONNXConfig,
#         _VaeConfig,
#         _ControlNetConfig,
#         LoRAConfig,
#         TextualInversionConfig,
#         IPAdapterConfig,
#         CLIPVisionDiffusersConfig,
#         T2IConfig,
#     ],
#     Field(discriminator="type"),
# ]


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
        if timestamp:
            model.last_modified = timestamp
        return model  # type: ignore
