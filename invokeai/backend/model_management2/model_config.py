# Copyright (c) 2023 Lincoln D. Stein and the InvokeAI Development Team
"""
Configuration definitions for image generation models.

Typical usage:

  from invokeai.backend.model_management2.model_config import ModelConfig
  raw = dict(path='models/sd-1/main/foo.ckpt',
             name='foo',
             base_model='sd-1',
             model_type='main',
             config='configs/stable-diffusion/v1-inference.yaml',
             model_variant='normal',
             model_format='checkpoint'
            )
  config = ModelConfig.parse_obj(raw)

Validation errors will raise an InvalidModelConfigException error.

"""
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, Extra
from typing import Optional, Literal, Union
from pydantic.error_wrappers import ValidationError


class InvalidModelConfigException(Exception):
    """Exception raised when the config parser doesn't recognize the passed
    combination of model type and format."""

    pass


class BaseModelType(str, Enum):
    """Base model type."""

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


class SchedulerPredictionType(str, Enum):
    """Scheduler prediction type."""

    Epsilon = "epsilon"
    VPrediction = "v_prediction"
    Sample = "sample"


class ModelConfigBase(BaseModel):
    """Base class for model configuration information."""

    path: Path
    name: str
    base_model: BaseModelType
    model_type: ModelType
    model_format: ModelFormat
    description: Optional[str] = Field(None)
    author: Optional[str] = Field(description="Model author")
    thumbnail_url: Optional[str] = Field(description="URL of thumbnail image")
    license_url: Optional[str] = Field(description="URL of license")
    source_url: Optional[str] = Field(description="Model download source")

    class Config:
        """Pydantic configuration hint."""

        use_enum_values = True
        extra = Extra.forbid
        validate_assignment = True


class CheckpointConfig(ModelConfigBase):
    """Model config for checkpoint-style models."""

    model_format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint
    config: Path = Field(description="path to the checkpoint model config file")


class DiffusersConfig(ModelConfigBase):
    """Model config for diffusers-style models."""

    model_format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class LoRAConfig(ModelConfigBase):
    """Model config for LoRA/Lycoris models."""

    model_format: Literal[ModelFormat.Lycoris, ModelFormat.Diffusers]


class TextualInversionConfig(ModelConfigBase):
    """Model config for textual inversion embeddings."""

    model_format: Literal[ModelFormat.EmbeddingFile, ModelFormat.EmbeddingFolder]


class MainConfig(ModelConfigBase):
    """Model config for main models."""

    vae: Optional[Path] = Field(None)
    model_variant: ModelVariantType


class MainCheckpointConfig(CheckpointConfig, MainConfig):
    """Model config for main checkpoint models."""

    pass


class MainDiffusersConfig(DiffusersConfig, MainConfig):
    """Model config for main diffusers models."""

    pass


class ONNXSD1Config(MainConfig):
    """Model config for ONNX format models based on sd-1."""

    model_format: Literal[ModelFormat.Onnx, ModelFormat.Olive]


class ONNXSD2Config(MainConfig):
    """Model config for ONNX format models based on sd-2."""

    model_format: Literal[ModelFormat.Onnx, ModelFormat.Olive]
    # No yaml config file for ONNX, so these are part of config
    prediction_type: SchedulerPredictionType
    upcast_attention: bool


class ModelConfig(object):
    """Class for parsing config dicts into StableDiffusion*Config obects."""

    _class_map: dict = {
        ModelFormat.Checkpoint: {
            ModelType.Main: MainCheckpointConfig,
        },
        ModelFormat.Diffusers: {
            ModelType.Main: MainDiffusersConfig,
            ModelType.Lora: LoRAConfig,
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
    }

    @classmethod
    def parse_obj(
        cls, raw_data: dict
    ) -> Union[
        MainCheckpointConfig,
        MainDiffusersConfig,
        LoRAConfig,
        TextualInversionConfig,
        ONNXSD1Config,
        ONNXSD2Config,
    ]:
        """Return the appropriate config object from raw dict values."""
        try:
            model_format = raw_data.get("model_format")
            model_type = raw_data.get("model_type")
            model_base = raw_data.get("base_model")
            class_to_return = cls._class_map[model_format][model_type]
            if isinstance(class_to_return, dict):  # additional level allowed
                class_to_return = class_to_return[model_base]
            return class_to_return.parse_obj(raw_data)
        except KeyError:
            raise InvalidModelConfigException(
                f"Unknown combination of model_format '{model_format}' and model_type '{model_type}'"
            )
        except ValidationError as e:
            raise InvalidModelConfigException(f"Invalid model configuration passed: {str(e)}") from e
