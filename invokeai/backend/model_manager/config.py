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

from pydantic import BaseModel, ConfigDict, Field, ValidationError


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
    description: Optional[str] = Field(None)
    source: Optional[str] = Field(description="Model download source (URL or repo_id)", default=None)

    model_config = ConfigDict(
        use_enum_values=False,
        validate_assignment=True,
    )

    def update(self, attributes: dict):
        """Update the object with fields in dict."""
        for key, value in attributes.items():
            setattr(self, key, value)  # may raise a validation error


class CheckpointConfig(ModelConfigBase):
    """Model config for checkpoint-style models."""

    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint
    config: str = Field(description="path to the checkpoint model config file")


class DiffusersConfig(ModelConfigBase):
    """Model config for diffusers-style models."""

    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class LoRAConfig(ModelConfigBase):
    """Model config for LoRA/Lycoris models."""

    format: Literal[ModelFormat.Lycoris, ModelFormat.Diffusers]


class VaeCheckpointConfig(ModelConfigBase):
    """Model config for standalone VAE models."""

    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint


class VaeDiffusersConfig(ModelConfigBase):
    """Model config for standalone VAE models (diffusers version)."""

    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class ControlNetDiffusersConfig(DiffusersConfig):
    """Model config for ControlNet models (diffusers version)."""

    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class ControlNetCheckpointConfig(CheckpointConfig):
    """Model config for ControlNet models (diffusers version)."""

    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint


class TextualInversionConfig(ModelConfigBase):
    """Model config for textual inversion embeddings."""

    format: Literal[ModelFormat.EmbeddingFile, ModelFormat.EmbeddingFolder]


class MainConfig(ModelConfigBase):
    """Model config for main models."""

    vae: Optional[str] = Field(None)
    variant: ModelVariantType = ModelVariantType.Normal
    ztsnr_training: bool = False


class MainCheckpointConfig(CheckpointConfig, MainConfig):
    """Model config for main checkpoint models."""

    # Note that we do not need prediction_type or upcast_attention here
    # because they are provided in the checkpoint's own config file.


class MainDiffusersConfig(DiffusersConfig, MainConfig):
    """Model config for main diffusers models."""

    prediction_type: SchedulerPredictionType = SchedulerPredictionType.Epsilon
    upcast_attention: bool = False


class ONNXSD1Config(MainConfig):
    """Model config for ONNX format models based on sd-1."""

    format: Literal[ModelFormat.Onnx, ModelFormat.Olive]
    prediction_type: SchedulerPredictionType = SchedulerPredictionType.Epsilon
    upcast_attention: bool = False


class ONNXSD2Config(MainConfig):
    """Model config for ONNX format models based on sd-2."""

    format: Literal[ModelFormat.Onnx, ModelFormat.Olive]
    # No yaml config file for ONNX, so these are part of config
    prediction_type: SchedulerPredictionType = SchedulerPredictionType.VPrediction
    upcast_attention: bool = True


class IPAdapterConfig(ModelConfigBase):
    """Model config for IP Adaptor format models."""

    format: Literal[ModelFormat.InvokeAI]


class CLIPVisionDiffusersConfig(ModelConfigBase):
    """Model config for ClipVision."""

    format: Literal[ModelFormat.Diffusers]


class T2IConfig(ModelConfigBase):
    """Model config for T2I."""

    format: Literal[ModelFormat.Diffusers]


AnyModelConfig = Union[
    MainCheckpointConfig,
    MainDiffusersConfig,
    LoRAConfig,
    TextualInversionConfig,
    ONNXSD1Config,
    ONNXSD2Config,
    VaeCheckpointConfig,
    VaeDiffusersConfig,
    ControlNetDiffusersConfig,
    ControlNetCheckpointConfig,
    IPAdapterConfig,
    CLIPVisionDiffusersConfig,
    T2IConfig,
]


class ModelConfigFactory(object):
    """Class for parsing config dicts into StableDiffusion Config obects."""

    _class_map: dict = {
        ModelFormat.Checkpoint: {
            ModelType.Main: MainCheckpointConfig,
            ModelType.Vae: VaeCheckpointConfig,
        },
        ModelFormat.Diffusers: {
            ModelType.Main: MainDiffusersConfig,
            ModelType.Lora: LoRAConfig,
            ModelType.Vae: VaeDiffusersConfig,
            ModelType.ControlNet: ControlNetDiffusersConfig,
            ModelType.CLIPVision: CLIPVisionDiffusersConfig,
        },
        ModelFormat.Lycoris: {
            ModelType.Lora: LoRAConfig,
        },
        ModelFormat.Onnx: {
            ModelType.ONNX: {
                BaseModelType.StableDiffusion1: ONNXSD1Config,
                BaseModelType.StableDiffusion2: ONNXSD2Config,
            },
        },
        ModelFormat.Olive: {
            ModelType.ONNX: {
                BaseModelType.StableDiffusion1: ONNXSD1Config,
                BaseModelType.StableDiffusion2: ONNXSD2Config,
            },
        },
        ModelFormat.EmbeddingFile: {
            ModelType.TextualInversion: TextualInversionConfig,
        },
        ModelFormat.EmbeddingFolder: {
            ModelType.TextualInversion: TextualInversionConfig,
        },
        ModelFormat.InvokeAI: {
            ModelType.IPAdapter: IPAdapterConfig,
        },
    }

    @classmethod
    def make_config(
        cls,
        model_data: Union[dict, ModelConfigBase],
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
            if key:
                model_data.key = key
            return model_data
        try:
            format = model_data.get("format")
            type = model_data.get("type")
            model_base = model_data.get("base")
            class_to_return = dest_class or cls._class_map[format][type]
            if isinstance(class_to_return, dict):  # additional level allowed
                class_to_return = class_to_return[model_base]
            model = class_to_return.model_validate(model_data)
            if key:
                model.key = key  # ensure consistency
            return model
        except KeyError as exc:
            raise InvalidModelConfigException(f"Unknown combination of format '{format}' and type '{type}'") from exc
        except ValidationError as exc:
            raise InvalidModelConfigException(f"Invalid model configuration passed: {str(exc)}") from exc
