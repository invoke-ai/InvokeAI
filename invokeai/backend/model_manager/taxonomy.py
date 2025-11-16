from enum import Enum
from typing import Dict, TypeAlias, Union

import onnxruntime as ort
import torch
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from pydantic import TypeAdapter

from invokeai.backend.raw_model import RawModel

# ModelMixin is the base class for all diffusers and transformers models
# RawModel is the InvokeAI wrapper class for ip_adapters, loras, textual_inversion and onnx runtime
AnyModel: TypeAlias = Union[
    ModelMixin,
    RawModel,
    torch.nn.Module,
    Dict[str, torch.Tensor],
    DiffusionPipeline,
    ort.InferenceSession,
]
"""Type alias for any kind of runtime, in-memory model representation. For example, a torch module or diffusers pipeline."""


class BaseModelType(str, Enum):
    """An enumeration of base model architectures. For example, Stable Diffusion 1.x, Stable Diffusion 2.x, FLUX, etc.

    Every model config must have a base architecture type.

    Not all models are associated with a base architecture. For example, CLIP models are their own thing, not related
    to any particular model architecture. To simplify internal APIs and make it easier to work with models, we use a
    fallback/null value `BaseModelType.Any` for these models, instead of making the model base optional."""

    Any = "any"
    """`Any` is essentially a fallback/null value for models with no base architecture association.
    For example, CLIP models are not related to Stable Diffusion, FLUX, or any other model arch."""
    StableDiffusion1 = "sd-1"
    """Indicates the model is associated with the Stable Diffusion 1.x model architecture, including 1.4 and 1.5."""
    StableDiffusion2 = "sd-2"
    """Indicates the model is associated with the Stable Diffusion 2.x model architecture, including 2.0 and 2.1."""
    StableDiffusion3 = "sd-3"
    """Indicates the model is associated with the Stable Diffusion 3.5 model architecture."""
    StableDiffusionXL = "sdxl"
    """Indicates the model is associated with the Stable Diffusion XL model architecture."""
    StableDiffusionXLRefiner = "sdxl-refiner"
    """Indicates the model is associated with the Stable Diffusion XL Refiner model architecture."""
    Flux = "flux"
    """Indicates the model is associated with FLUX.1 model architecture, including FLUX Dev, Schnell and Fill."""
    CogView4 = "cogview4"
    """Indicates the model is associated with CogView 4 model architecture."""
    Unknown = "unknown"
    """Indicates the model's base architecture is unknown."""


class ModelType(str, Enum):
    """Model type."""

    ONNX = "onnx"
    Main = "main"
    VAE = "vae"
    LoRA = "lora"
    ControlLoRa = "control_lora"
    ControlNet = "controlnet"  # used by model_probe
    TextualInversion = "embedding"
    IPAdapter = "ip_adapter"
    CLIPVision = "clip_vision"
    CLIPEmbed = "clip_embed"
    T2IAdapter = "t2i_adapter"
    T5Encoder = "t5_encoder"
    SpandrelImageToImage = "spandrel_image_to_image"
    SigLIP = "siglip"
    FluxRedux = "flux_redux"
    LlavaOnevision = "llava_onevision"
    Unknown = "unknown"


class SubModelType(str, Enum):
    """Submodel type."""

    UNet = "unet"
    Transformer = "transformer"
    TextEncoder = "text_encoder"
    TextEncoder2 = "text_encoder_2"
    TextEncoder3 = "text_encoder_3"
    Tokenizer = "tokenizer"
    Tokenizer2 = "tokenizer_2"
    Tokenizer3 = "tokenizer_3"
    VAE = "vae"
    VAEDecoder = "vae_decoder"
    VAEEncoder = "vae_encoder"
    Scheduler = "scheduler"
    SafetyChecker = "safety_checker"


class ClipVariantType(str, Enum):
    """Variant type."""

    L = "large"
    G = "gigantic"


class ModelVariantType(str, Enum):
    """Variant type."""

    Normal = "normal"
    Inpaint = "inpaint"
    Depth = "depth"


class FluxVariantType(str, Enum):
    Schnell = "schnell"
    Dev = "dev"
    DevFill = "dev_fill"


class ModelFormat(str, Enum):
    """Storage format of model."""

    OMI = "omi"
    Diffusers = "diffusers"
    Checkpoint = "checkpoint"
    LyCORIS = "lycoris"
    ONNX = "onnx"
    Olive = "olive"
    EmbeddingFile = "embedding_file"
    EmbeddingFolder = "embedding_folder"
    InvokeAI = "invokeai"
    T5Encoder = "t5_encoder"
    BnbQuantizedLlmInt8b = "bnb_quantized_int8b"
    BnbQuantizednf4b = "bnb_quantized_nf4b"
    GGUFQuantized = "gguf_quantized"
    Unknown = "unknown"


class SchedulerPredictionType(str, Enum):
    """Scheduler prediction type."""

    Epsilon = "epsilon"
    VPrediction = "v_prediction"
    Sample = "sample"


class ModelRepoVariant(str, Enum):
    """Various hugging face variants on the diffusers format."""

    Default = ""  # model files without "fp16" or other qualifier
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


class FluxLoRAFormat(str, Enum):
    """Flux LoRA formats."""

    Diffusers = "flux.diffusers"
    Kohya = "flux.kohya"
    OneTrainer = "flux.onetrainer"
    Control = "flux.control"
    AIToolkit = "flux.aitoolkit"


AnyVariant: TypeAlias = Union[ModelVariantType, ClipVariantType, FluxVariantType]
variant_type_adapter = TypeAdapter[ModelVariantType | ClipVariantType | FluxVariantType](
    ModelVariantType | ClipVariantType | FluxVariantType
)
