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
from datetime import datetime
from enum import Enum
from typing import Literal, Optional, Type, Union

import torch
from diffusers.models.modeling_utils import ModelMixin
from pydantic import BaseModel, Discriminator, Field, JsonValue, Tag, TypeAdapter
from typing_extensions import Annotated, Any, Dict

from invokeai.backend.model_manager.hash import ALGORITHM, ModelHash
from invokeai.backend.raw_model import RawModel

# ModelMixin is the base class for all diffusers and transformers models
# RawModel is the InvokeAI wrapper class for ip_adapters, loras, textual_inversion and onnx runtime
AnyModel = Union[ModelMixin, RawModel, torch.nn.Module]


class InvalidModelConfigException(Exception):
    """Exception for when config parser doesn't recognized this combination of model type and format."""


class ModelSourceType(str, Enum):
    """The source of the model."""

    HF_REPO_ID = "hf_repo_id"
    CIVITAI = "civitai"
    URL = "url"
    PATH = "path"


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


class _ModelConfigBase(BaseModel):
    """The configuration of a model."""

    id: str = Field(description="The unique identifier of the model")  # Primary Key
    hash: str = Field(description="The BLAKE3 hash of the model.", frozen=True)
    base: BaseModelType = Field(description="The base of the model")
    path: str = Field(description="The path of the model")
    name: str = Field(description="The name of the model")
    description: Optional[str] = Field(description="The description of the model", default=None)

    def compute_hash(self, algorithm: ALGORITHM = "blake3") -> str:
        """Compute the hash of the model."""
        return ModelHash(algorithm).hash(self.path)


class _CheckpointConfig(_ModelConfigBase):
    """Model config for checkpoint-style models."""

    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint
    config_path: str = Field(description="Path to the checkpoint model config file")


class _DiffusersConfig(_ModelConfigBase):
    """Model config for diffusers-style models."""

    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers
    repo_variant: Optional[ModelRepoVariant] = ModelRepoVariant.DEFAULT


class LoRALycorisConfig(_ModelConfigBase):
    """Model config for LoRA/Lycoris models."""

    type: Literal[ModelType.Lora] = ModelType.Lora
    format: Literal[ModelFormat.Lycoris] = ModelFormat.Lycoris

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.Lora}.{ModelFormat.Lycoris}")


class LoRADiffusersConfig(_ModelConfigBase):
    """Model config for LoRA/Diffusers models."""

    type: Literal[ModelType.Lora] = ModelType.Lora
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.Lora}.{ModelFormat.Diffusers}")


class VaeCheckpointConfig(_ModelConfigBase):
    """Model config for standalone VAE models."""

    type: Literal[ModelType.Vae] = ModelType.Vae
    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.Vae}.{ModelFormat.Checkpoint}")


class VaeDiffusersConfig(_ModelConfigBase):
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


class TextualInversionFileConfig(_ModelConfigBase):
    """Model config for textual inversion embeddings."""

    type: Literal[ModelType.TextualInversion] = ModelType.TextualInversion
    format: Literal[ModelFormat.EmbeddingFile] = ModelFormat.EmbeddingFile

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.TextualInversion}.{ModelFormat.EmbeddingFile}")


class TextualInversionFolderConfig(_ModelConfigBase):
    """Model config for textual inversion embeddings."""

    type: Literal[ModelType.TextualInversion] = ModelType.TextualInversion
    format: Literal[ModelFormat.EmbeddingFolder] = ModelFormat.EmbeddingFolder

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.TextualInversion}.{ModelFormat.EmbeddingFolder}")


class _MainConfig(_ModelConfigBase):
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


class IPAdapterConfig(_ModelConfigBase):
    """Model config for IP Adaptor format models."""

    type: Literal[ModelType.IPAdapter] = ModelType.IPAdapter
    image_encoder_model_id: str
    format: Literal[ModelFormat.InvokeAI]

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.IPAdapter}.{ModelFormat.InvokeAI}")


class CLIPVisionDiffusersConfig(_ModelConfigBase):
    """Model config for ClipVision."""

    type: Literal[ModelType.CLIPVision] = ModelType.CLIPVision
    format: Literal[ModelFormat.Diffusers]

    @staticmethod
    def get_tag() -> Tag:
        return Tag(f"{ModelType.CLIPVision}.{ModelFormat.Diffusers}")


class T2IAdapterConfig(_ModelConfigBase):
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


class ModelRecord(BaseModel):
    """A record of a model in the database."""

    # Internal DB/record data
    id: str = Field(description="The unique identifier of the model")  # Primary Key
    config: AnyModelConfig = Field(description="The configuration of the model")
    source: str = Field(
        description="The original source of the model (path, URL or repo_id)",
        frozen=True,  # This field is immutable
    )
    source_type: ModelSourceType = Field(
        description="The type of the source of the model",
        frozen=True,  # This field is immutable
    )
    source_api_response: Optional[JsonValue] = Field(
        description="The original API response from which the model was installed.",
        default=None,
        frozen=True,  # This field is immutable
    )
    created_at: datetime | str = Field(description="When the model was created")
    updated_at: datetime | str = Field(description="When the model was last updated")


class ModelConfigFactory(object):
    """Class for parsing config dicts into StableDiffusion Config obects."""

    @classmethod
    def make_config(
        cls,
        model_data: Union[Dict[str, Any], AnyModelConfig],
        key: Optional[str] = None,
        dest_class: Optional[Type[_ModelConfigBase]] = None,
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
        model: Optional[_ModelConfigBase] = None
        if isinstance(model_data, _ModelConfigBase):
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
