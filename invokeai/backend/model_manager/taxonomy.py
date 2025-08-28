from enum import Enum
from typing import Dict, TypeAlias, Union

import diffusers
import onnxruntime as ort
import torch
from diffusers import ModelMixin

from invokeai.backend.raw_model import RawModel

# ModelMixin is the base class for all diffusers and transformers models
# RawModel is the InvokeAI wrapper class for ip_adapters, loras, textual_inversion and onnx runtime
AnyModel = Union[
    ModelMixin, RawModel, torch.nn.Module, Dict[str, torch.Tensor], diffusers.DiffusionPipeline, ort.InferenceSession
]


class BaseModelType(str, Enum):
    """Base model type."""

    Any = "any"
    StableDiffusion1 = "sd-1"
    StableDiffusion2 = "sd-2"
    StableDiffusion3 = "sd-3"
    StableDiffusionXL = "sdxl"
    StableDiffusionXLRefiner = "sdxl-refiner"
    Flux = "flux"
    CogView4 = "cogview4"
    Imagen3 = "imagen3"
    Imagen4 = "imagen4"
    Gemini2_5 = "gemini-2.5"
    ChatGPT4o = "chatgpt-4o"
    FluxKontext = "flux-kontext"


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
    Api = "api"


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


AnyVariant: TypeAlias = Union[ModelVariantType, ClipVariantType, None]
