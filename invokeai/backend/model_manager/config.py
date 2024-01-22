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
from enum import Enum
from typing import Literal, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing_extensions import Annotated, Any, Dict


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

    DEFAULT = "default"  # model files without "fp16" or other qualifier
    FP16 = "fp16"
    FP32 = "fp32"
    ONNX = "onnx"
    OPENVINO = "openvino"
    FLAX = "flax"


class ModelConfigBase(BaseModel):
    """Base class for model configuration information."""

    path: str
    name: str
    base: BaseModelType
    type: ModelType
    format: ModelFormat
    key: str = Field(description="unique key for model", default="<NOKEY>")
    original_hash: Optional[str] = Field(
        description="original fasthash of model contents", default=None
    )  # this is assigned at install time and will not change
    current_hash: Optional[str] = Field(
        description="current fasthash of model contents", default=None
    )  # if model is converted or otherwise modified, this will hold updated hash
    description: Optional[str] = Field(default=None)
    source: Optional[str] = Field(description="Model download source (URL or repo_id)", default=None)

    model_config = ConfigDict(
        use_enum_values=False,
        validate_assignment=True,
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

class LoRAConfig(ModelConfigBase):
    """Model config for LoRA/Lycoris models."""

    type: Literal[ModelType.Lora] = ModelType.Lora
    format: Literal[ModelFormat.Lycoris, ModelFormat.Diffusers]


class VaeCheckpointConfig(ModelConfigBase):
    """Model config for standalone VAE models."""

    type: Literal[ModelType.Vae] = ModelType.Vae
    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint


class VaeDiffusersConfig(ModelConfigBase):
    """Model config for standalone VAE models (diffusers version)."""

    type: Literal[ModelType.Vae] = ModelType.Vae
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class ControlNetDiffusersConfig(_DiffusersConfig):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = ModelType.ControlNet
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers

class ControlNetCheckpointConfig(_CheckpointConfig):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = ModelType.ControlNet
    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint


class TextualInversionConfig(ModelConfigBase):
    """Model config for textual inversion embeddings."""

    type: Literal[ModelType.TextualInversion] = ModelType.TextualInversion
    format: Literal[ModelFormat.EmbeddingFile, ModelFormat.EmbeddingFolder]


class _MainConfig(ModelConfigBase):
    """Model config for main models."""

    vae: Optional[str] = Field(default=None)
    variant: ModelVariantType = ModelVariantType.Normal
    ztsnr_training: bool = False


class MainCheckpointConfig(_CheckpointConfig, _MainConfig):
    """Model config for main checkpoint models."""

    type: Literal[ModelType.Main] = ModelType.Main


class MainDiffusersConfig(_DiffusersConfig, _MainConfig):
    """Model config for main diffusers models."""

    type: Literal[ModelType.Main] = ModelType.Main
    prediction_type: SchedulerPredictionType = SchedulerPredictionType.Epsilon
    upcast_attention: bool = False

class ONNXSD1Config(_MainConfig):
    """Model config for ONNX format models based on sd-1."""

    type: Literal[ModelType.ONNX] = ModelType.ONNX
    format: Literal[ModelFormat.Onnx, ModelFormat.Olive]
    base: Literal[BaseModelType.StableDiffusion1] = BaseModelType.StableDiffusion1
    prediction_type: SchedulerPredictionType = SchedulerPredictionType.Epsilon
    upcast_attention: bool = False


class ONNXSD2Config(_MainConfig):
    """Model config for ONNX format models based on sd-2."""

    type: Literal[ModelType.ONNX] = ModelType.ONNX
    format: Literal[ModelFormat.Onnx, ModelFormat.Olive]
    # No yaml config file for ONNX, so these are part of config
    base: Literal[BaseModelType.StableDiffusion2] = BaseModelType.StableDiffusion2
    prediction_type: SchedulerPredictionType = SchedulerPredictionType.VPrediction
    upcast_attention: bool = True


class IPAdapterConfig(ModelConfigBase):
    """Model config for IP Adaptor format models."""

    type: Literal[ModelType.IPAdapter] = ModelType.IPAdapter
    format: Literal[ModelFormat.InvokeAI]


class CLIPVisionDiffusersConfig(ModelConfigBase):
    """Model config for ClipVision."""

    type: Literal[ModelType.CLIPVision] = ModelType.CLIPVision
    format: Literal[ModelFormat.Diffusers]


class T2IConfig(ModelConfigBase):
    """Model config for T2I."""

    type: Literal[ModelType.T2IAdapter] = ModelType.T2IAdapter
    format: Literal[ModelFormat.Diffusers]


_ONNXConfig = Annotated[Union[ONNXSD1Config, ONNXSD2Config], Field(discriminator="base")]
_ControlNetConfig = Annotated[
    Union[ControlNetDiffusersConfig, ControlNetCheckpointConfig],
    Field(discriminator="format"),
]
_VaeConfig = Annotated[Union[VaeDiffusersConfig, VaeCheckpointConfig], Field(discriminator="format")]
_MainModelConfig = Annotated[Union[MainDiffusersConfig, MainCheckpointConfig], Field(discriminator="format")]

AnyModelConfig = Union[
    _MainModelConfig,
    _ONNXConfig,
    _VaeConfig,
    _ControlNetConfig,
    LoRAConfig,
    TextualInversionConfig,
    IPAdapterConfig,
    CLIPVisionDiffusersConfig,
    T2IConfig,
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
        model_data: Union[dict, AnyModelConfig],
        key: Optional[str] = None,
        dest_class: Optional[Type] = None,
    ) -> AnyModelConfig:
        """
        Return the appropriate config object from raw dict values.

        :param model_data: A raw dict corresponding the obect fields to be
        parsed into a ModelConfigBase obect (or descendent), or a ModelConfigBase
        object, which will be passed through unchanged.
        :param dest_class: The config class to be returned. If not provided, will
        be selected automatically.
        """
        if isinstance(model_data, ModelConfigBase):
            model = model_data
        elif dest_class:
            model = dest_class.validate_python(model_data)
        else:
            model = AnyModelConfigValidator.validate_python(model_data)
        if key:
            model.key = key
        return model
