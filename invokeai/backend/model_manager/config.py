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

# pyright: reportIncompatibleVariableOverride=false
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from inspect import isabstract
from pathlib import Path
from typing import ClassVar, Literal, Optional, TypeAlias, Union

import diffusers
import onnxruntime as ort
import safetensors.torch
import torch
from diffusers.models.modeling_utils import ModelMixin
from picklescan.scanner import scan_file_path
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, TypeAdapter
from typing_extensions import Annotated, Any, Dict

from invokeai.app.util.misc import uuid_string
from invokeai.backend.model_hash.hash_validator import validate_hash
from invokeai.backend.model_hash.model_hash import HASHING_ALGORITHMS, ModelHash
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.raw_model import RawModel
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES
from invokeai.backend.util.silence_warnings import SilenceWarnings

logger = logging.getLogger(__name__)

# ModelMixin is the base class for all diffusers and transformers models
# RawModel is the InvokeAI wrapper class for ip_adapters, loras, textual_inversion and onnx runtime
AnyModel = Union[
    ModelMixin, RawModel, torch.nn.Module, Dict[str, torch.Tensor], diffusers.DiffusionPipeline, ort.InferenceSession
]


class InvalidModelConfigException(Exception):
    """Exception for when config parser doesn't recognize this combination of model type and format."""


class BaseModelType(str, Enum):
    """Base model type."""

    Any = "any"
    StableDiffusion1 = "sd-1"
    StableDiffusion2 = "sd-2"
    StableDiffusion3 = "sd-3"
    StableDiffusionXL = "sdxl"
    StableDiffusionXLRefiner = "sdxl-refiner"
    Flux = "flux"
    # Kandinsky2_1 = "kandinsky-2.1"


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


DEFAULTS_PRECISION = Literal["fp16", "fp32"]


AnyVariant: TypeAlias = Union[ModelVariantType, ClipVariantType, None]


class SubmodelDefinition(BaseModel):
    path_or_prefix: str
    model_type: ModelType
    variant: AnyVariant = None

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


class ControlAdapterDefaultSettings(BaseModel):
    # This could be narrowed to controlnet processor nodes, but they change. Leaving this a string is safer.
    preprocessor: str | None
    model_config = ConfigDict(extra="forbid")


class ModelOnDisk:
    """A utility class representing a model stored on disk."""

    def __init__(self, path: Path, hash_algo: HASHING_ALGORITHMS = "blake3_single"):
        self.path = path
        self.format_type = ModelFormat.Diffusers if path.is_dir() else ModelFormat.Checkpoint
        if self.path.suffix in {".safetensors", ".bin", ".pt", ".ckpt"}:
            self.name = path.stem
        else:
            self.name = path.name
        self.hash_algo = hash_algo

    def hash(self):
        return ModelHash(algorithm=self.hash_algo).hash(self.path)

    def size(self):
        if self.format_type == ModelFormat.Checkpoint:
            return self.path.stat().st_size
        return sum(file.stat().st_size for file in self.path.rglob("*"))

    def component_paths(self):
        if self.format_type == ModelFormat.Checkpoint:
            return {self.path}
        extensions = {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".gguf"}
        return {f for f in self.path.rglob("*") if f.suffix in extensions}

    @staticmethod
    def load_state_dict(path: Path):
        with SilenceWarnings():
            if path.suffix.endswith((".ckpt", ".pt", ".pth", ".bin")):
                scan_result = scan_file_path(path)
                if scan_result.infected_files != 0 or scan_result.scan_err:
                    raise RuntimeError(f"The model {path.stem} is potentially infected by malware. Aborting import.")
                checkpoint = torch.load(path, map_location="cpu")
            elif path.suffix.endswith(".gguf"):
                checkpoint = gguf_sd_loader(path, compute_dtype=torch.float32)
            elif path.suffix.endswith(".safetensors"):
                checkpoint = safetensors.torch.load_file(path)
            else:
                raise ValueError(f"Unrecognized model extension: {path.suffix}")

        state_dict = checkpoint.get("state_dict", checkpoint)
        return state_dict


class MatchSpeed(int, Enum):
    """Represents the estimated runtime speed of a config's 'matches' method."""

    FAST = 0
    MED = 1
    SLOW = 2


class ModelConfigBase(ABC, BaseModel):
    """
    Abstract Base class for model configurations.

    To create a new config type, inherit from this class and implement its interface:
    - (mandatory) override methods 'matches' and 'parse'
    - (mandatory) define fields 'type' and 'format' as class attributes

    - (optional) override method 'get_tag'
    - (optional) override field _MATCH_SPEED

    See MinimalConfigExample in test_model_probe.py for an example implementation.
    """

    @staticmethod
    def json_schema_extra(schema: dict[str, Any]) -> None:
        schema["required"].extend(["key", "type", "format"])

    model_config = ConfigDict(validate_assignment=True, json_schema_extra=json_schema_extra)

    key: str = Field(description="A unique key for this model.", default_factory=uuid_string)
    hash: str = Field(description="The hash of the model file(s).")
    path: str = Field(
        description="Path to the model on the filesystem. Relative paths are relative to the Invoke root directory."
    )
    name: str = Field(description="Name of the model.")
    type: ModelType = Field(description="Model type")
    format: ModelFormat = Field(description="Model format")
    base: BaseModelType = Field(description="The base model.")
    source: str = Field(description="The original source of the model (path, URL or repo_id).")
    source_type: ModelSourceType = Field(description="The type of source")

    description: Optional[str] = Field(description="Model description", default=None)
    source_api_response: Optional[str] = Field(
        description="The original API response from the source, as stringified JSON.", default=None
    )
    cover_image: Optional[str] = Field(description="Url for image to preview model", default=None)
    submodels: Optional[Dict[SubModelType, SubmodelDefinition]] = Field(
        description="Loadable submodels in this model", default=None
    )

    _USING_LEGACY_PROBE: ClassVar[set] = set()
    _USING_CLASSIFY_API: ClassVar[set] = set()
    _MATCH_SPEED: ClassVar[MatchSpeed] = MatchSpeed.MED

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if issubclass(cls, LegacyProbeMixin):
            ModelConfigBase._USING_LEGACY_PROBE.add(cls)
        else:
            ModelConfigBase._USING_CLASSIFY_API.add(cls)

    @staticmethod
    def all_config_classes():
        subclasses = ModelConfigBase._USING_LEGACY_PROBE | ModelConfigBase._USING_CLASSIFY_API
        concrete = {cls for cls in subclasses if not isabstract(cls)}
        return concrete

    @staticmethod
    def classify(model_path: Path, hash_algo: HASHING_ALGORITHMS = "blake3_single", **overrides):
        """
        Returns the best matching ModelConfig instance from a model's file/folder path.
        Raises InvalidModelConfigException if no valid configuration is found.
        Created to deprecate ModelProbe.probe
        """
        candidates = ModelConfigBase._USING_CLASSIFY_API
        sorted_by_match_speed = sorted(candidates, key=lambda cls: cls._MATCH_SPEED)
        mod = ModelOnDisk(model_path, hash_algo)

        for config_cls in sorted_by_match_speed:
            try:
                return config_cls.from_model_on_disk(mod, **overrides)
            except InvalidModelConfigException:
                logger.debug(f"ModelConfig '{config_cls.__name__}' failed to parse '{mod.path}', trying next config")
            except Exception as e:
                logger.error(f"Unexpected exception while parsing '{config_cls.__name__}': {e}, trying next config")

        raise InvalidModelConfigException("No valid config found")

    @classmethod
    def get_tag(cls) -> Tag:
        type = cls.model_fields["type"].default.value
        format = cls.model_fields["format"].default.value
        return Tag(f"{type}.{format}")

    @classmethod
    @abstractmethod
    def parse(cls, mod: ModelOnDisk) -> dict[str, Any]:
        """Returns a dictionary with the fields needed to construct the model.
        Raises InvalidModelConfigException if the model is invalid.
        """
        pass

    @classmethod
    @abstractmethod
    def matches(cls, mod: ModelOnDisk) -> bool:
        """Performs a quick check to determine if the config matches the model.
        This doesn't need to be a perfect test - the aim is to eliminate unlikely matches quickly before parsing."""
        pass

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, **overrides):
        """Creates an instance of this config or raises InvalidModelConfigException."""
        if not cls.matches(mod):
            raise InvalidModelConfigException(f"Path {mod.path} does not match {cls.__name__} format")

        fields = cls.parse(mod)

        fields["path"] = mod.path.as_posix()
        fields["source"] = fields.get("source") or fields["path"]
        fields["source_type"] = fields.get("source_type") or ModelSourceType.Path
        fields["name"] = mod.name
        fields["hash"] = fields.get("hash") or mod.hash()

        fields.update(overrides)
        return cls(**fields)


class LegacyProbeMixin:
    """Mixin for classes using the legacy probe for model classification."""

    @classmethod
    def matches(cls, *args, **kwargs):
        raise NotImplementedError(f"Method 'matches' not implemented for {cls.__name__}")

    @classmethod
    def parse(cls, *args, **kwargs):
        raise NotImplementedError(f"Method 'parse' not implemented for {cls.__name__}")


class CheckpointConfigBase(ABC, BaseModel):
    """Base class for checkpoint-style models."""

    format: Literal[ModelFormat.Checkpoint, ModelFormat.BnbQuantizednf4b, ModelFormat.GGUFQuantized] = Field(
        description="Format of the provided checkpoint model", default=ModelFormat.Checkpoint
    )
    config_path: str = Field(description="path to the checkpoint model config file")
    converted_at: Optional[float] = Field(
        description="When this model was last converted to diffusers", default_factory=time.time
    )


class DiffusersConfigBase(ABC, BaseModel):
    """Base class for diffusers-style models."""

    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers
    repo_variant: Optional[ModelRepoVariant] = ModelRepoVariant.Default


class LoRAConfigBase(ABC, BaseModel):
    """Base class for LoRA models."""

    type: Literal[ModelType.LoRA] = ModelType.LoRA
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)


class T5EncoderConfigBase(ABC, BaseModel):
    """Base class for diffusers-style models."""

    type: Literal[ModelType.T5Encoder] = ModelType.T5Encoder


class T5EncoderConfig(T5EncoderConfigBase, LegacyProbeMixin, ModelConfigBase):
    format: Literal[ModelFormat.T5Encoder] = ModelFormat.T5Encoder


class T5EncoderBnbQuantizedLlmInt8bConfig(T5EncoderConfigBase, LegacyProbeMixin, ModelConfigBase):
    format: Literal[ModelFormat.BnbQuantizedLlmInt8b] = ModelFormat.BnbQuantizedLlmInt8b


class LoRALyCORISConfig(LoRAConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for LoRA/Lycoris models."""

    format: Literal[ModelFormat.LyCORIS] = ModelFormat.LyCORIS


class ControlAdapterConfigBase(ABC, BaseModel):
    default_settings: Optional[ControlAdapterDefaultSettings] = Field(
        description="Default settings for this model", default=None
    )


class ControlLoRALyCORISConfig(ControlAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for Control LoRA models."""

    type: Literal[ModelType.ControlLoRa] = ModelType.ControlLoRa
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)
    format: Literal[ModelFormat.LyCORIS] = ModelFormat.LyCORIS


class ControlLoRADiffusersConfig(ControlAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for Control LoRA models."""

    type: Literal[ModelType.ControlLoRa] = ModelType.ControlLoRa
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class LoRADiffusersConfig(LoRAConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for LoRA/Diffusers models."""

    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class VAECheckpointConfig(CheckpointConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for standalone VAE models."""

    type: Literal[ModelType.VAE] = ModelType.VAE


class VAEDiffusersConfig(LegacyProbeMixin, ModelConfigBase):
    """Model config for standalone VAE models (diffusers version)."""

    type: Literal[ModelType.VAE] = ModelType.VAE
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class ControlNetDiffusersConfig(DiffusersConfigBase, ControlAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = ModelType.ControlNet
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class ControlNetCheckpointConfig(CheckpointConfigBase, ControlAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = ModelType.ControlNet


class TextualInversionFileConfig(LegacyProbeMixin, ModelConfigBase):
    """Model config for textual inversion embeddings."""

    type: Literal[ModelType.TextualInversion] = ModelType.TextualInversion
    format: Literal[ModelFormat.EmbeddingFile] = ModelFormat.EmbeddingFile


class TextualInversionFolderConfig(LegacyProbeMixin, ModelConfigBase):
    """Model config for textual inversion embeddings."""

    type: Literal[ModelType.TextualInversion] = ModelType.TextualInversion
    format: Literal[ModelFormat.EmbeddingFolder] = ModelFormat.EmbeddingFolder


class MainConfigBase(ABC, BaseModel):
    type: Literal[ModelType.Main] = ModelType.Main
    trigger_phrases: Optional[set[str]] = Field(description="Set of trigger phrases for this model", default=None)
    default_settings: Optional[MainModelDefaultSettings] = Field(
        description="Default settings for this model", default=None
    )
    variant: AnyVariant = ModelVariantType.Normal


class MainCheckpointConfig(CheckpointConfigBase, MainConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for main checkpoint models."""

    prediction_type: SchedulerPredictionType = SchedulerPredictionType.Epsilon
    upcast_attention: bool = False


class MainBnbQuantized4bCheckpointConfig(CheckpointConfigBase, MainConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for main checkpoint models."""

    format: Literal[ModelFormat.BnbQuantizednf4b] = ModelFormat.BnbQuantizednf4b
    prediction_type: SchedulerPredictionType = SchedulerPredictionType.Epsilon
    upcast_attention: bool = False


class MainGGUFCheckpointConfig(CheckpointConfigBase, MainConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for main checkpoint models."""

    format: Literal[ModelFormat.GGUFQuantized] = ModelFormat.GGUFQuantized
    prediction_type: SchedulerPredictionType = SchedulerPredictionType.Epsilon
    upcast_attention: bool = False


class MainDiffusersConfig(DiffusersConfigBase, MainConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for main diffusers models."""

    pass


class IPAdapterConfigBase(ABC, BaseModel):
    type: Literal[ModelType.IPAdapter] = ModelType.IPAdapter


class IPAdapterInvokeAIConfig(IPAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for IP Adapter diffusers format models."""

    # TODO(ryand): Should we deprecate this field? From what I can tell, it hasn't been probed correctly for a long
    # time. Need to go through the history to make sure I'm understanding this fully.
    image_encoder_model_id: str
    format: Literal[ModelFormat.InvokeAI] = ModelFormat.InvokeAI


class IPAdapterCheckpointConfig(IPAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for IP Adapter checkpoint format models."""

    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint


class CLIPEmbedDiffusersConfig(DiffusersConfigBase):
    """Model config for Clip Embeddings."""

    variant: ClipVariantType = Field(description="Clip variant for this model")
    type: Literal[ModelType.CLIPEmbed] = ModelType.CLIPEmbed
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class CLIPGEmbedDiffusersConfig(CLIPEmbedDiffusersConfig, LegacyProbeMixin, ModelConfigBase):
    """Model config for CLIP-G Embeddings."""

    variant: Literal[ClipVariantType.G] = ClipVariantType.G

    @classmethod
    def get_tag(cls) -> Tag:
        return Tag(f"{ModelType.CLIPEmbed.value}.{ModelFormat.Diffusers.value}.{ClipVariantType.G.value}")


class CLIPLEmbedDiffusersConfig(CLIPEmbedDiffusersConfig, LegacyProbeMixin, ModelConfigBase):
    """Model config for CLIP-L Embeddings."""

    variant: Literal[ClipVariantType.L] = ClipVariantType.L

    @classmethod
    def get_tag(cls) -> Tag:
        return Tag(f"{ModelType.CLIPEmbed.value}.{ModelFormat.Diffusers.value}.{ClipVariantType.L.value}")


class CLIPVisionDiffusersConfig(DiffusersConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for CLIPVision."""

    type: Literal[ModelType.CLIPVision] = ModelType.CLIPVision
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class T2IAdapterConfig(DiffusersConfigBase, ControlAdapterConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for T2I."""

    type: Literal[ModelType.T2IAdapter] = ModelType.T2IAdapter
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class SpandrelImageToImageConfig(LegacyProbeMixin, ModelConfigBase):
    """Model config for Spandrel Image to Image models."""

    _MATCH_SPEED: ClassVar[MatchSpeed] = MatchSpeed.SLOW  # requires loading the model from disk

    type: Literal[ModelType.SpandrelImageToImage] = ModelType.SpandrelImageToImage
    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint


class SigLIPConfig(DiffusersConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for SigLIP."""

    type: Literal[ModelType.SigLIP] = ModelType.SigLIP
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


class FluxReduxConfig(LegacyProbeMixin, ModelConfigBase):
    """Model config for FLUX Tools Redux model."""

    type: Literal[ModelType.FluxRedux] = ModelType.FluxRedux
    format: Literal[ModelFormat.Checkpoint] = ModelFormat.Checkpoint


class LlavaOnevisionConfig(DiffusersConfigBase, LegacyProbeMixin, ModelConfigBase):
    """Model config for Llava Onevision models."""

    type: Literal[ModelType.LlavaOnevision] = ModelType.LlavaOnevision
    format: Literal[ModelFormat.Diffusers] = ModelFormat.Diffusers


def get_model_discriminator_value(v: Any) -> str:
    """
    Computes the discriminator value for a model config.
    https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions-with-callable-discriminator
    """
    format_ = type_ = variant_ = None

    if isinstance(v, dict):
        format_ = v.get("format")
        if isinstance(format_, Enum):
            format_ = format_.value

        type_ = v.get("type")
        if isinstance(type_, Enum):
            type_ = type_.value

        variant_ = v.get("variant")
        if isinstance(variant_, Enum):
            variant_ = variant_.value
    else:
        format_ = v.format.value
        type_ = v.type.value
        variant_ = getattr(v, "variant", None)
        if variant_:
            variant_ = variant_.value

    # Ideally, each config would be uniquely identified with a combination of fields
    # i.e. (type, format, variant) without any special cases. Alas...

    # Previously, CLIPEmbed did not have any variants, meaning older database entries lack a variant field.
    # To maintain compatibility, we default to ClipVariantType.L in this case.
    if type_ == ModelType.CLIPEmbed.value and format_ == ModelFormat.Diffusers.value:
        variant_ = variant_ or ClipVariantType.L.value
        return f"{type_}.{format_}.{variant_}"
    return f"{type_}.{format_}"


# The types are listed explicitly because IDEs/LSPs can't identify the correct types
# when AnyModelConfig is constructed dynamically using ModelConfigBase.all_config_classes
AnyModelConfig = Annotated[
    Union[
        Annotated[MainDiffusersConfig, MainDiffusersConfig.get_tag()],
        Annotated[MainCheckpointConfig, MainCheckpointConfig.get_tag()],
        Annotated[MainBnbQuantized4bCheckpointConfig, MainBnbQuantized4bCheckpointConfig.get_tag()],
        Annotated[MainGGUFCheckpointConfig, MainGGUFCheckpointConfig.get_tag()],
        Annotated[VAEDiffusersConfig, VAEDiffusersConfig.get_tag()],
        Annotated[VAECheckpointConfig, VAECheckpointConfig.get_tag()],
        Annotated[ControlNetDiffusersConfig, ControlNetDiffusersConfig.get_tag()],
        Annotated[ControlNetCheckpointConfig, ControlNetCheckpointConfig.get_tag()],
        Annotated[LoRALyCORISConfig, LoRALyCORISConfig.get_tag()],
        Annotated[ControlLoRALyCORISConfig, ControlLoRALyCORISConfig.get_tag()],
        Annotated[ControlLoRADiffusersConfig, ControlLoRADiffusersConfig.get_tag()],
        Annotated[LoRADiffusersConfig, LoRADiffusersConfig.get_tag()],
        Annotated[T5EncoderConfig, T5EncoderConfig.get_tag()],
        Annotated[T5EncoderBnbQuantizedLlmInt8bConfig, T5EncoderBnbQuantizedLlmInt8bConfig.get_tag()],
        Annotated[TextualInversionFileConfig, TextualInversionFileConfig.get_tag()],
        Annotated[TextualInversionFolderConfig, TextualInversionFolderConfig.get_tag()],
        Annotated[IPAdapterInvokeAIConfig, IPAdapterInvokeAIConfig.get_tag()],
        Annotated[IPAdapterCheckpointConfig, IPAdapterCheckpointConfig.get_tag()],
        Annotated[T2IAdapterConfig, T2IAdapterConfig.get_tag()],
        Annotated[SpandrelImageToImageConfig, SpandrelImageToImageConfig.get_tag()],
        Annotated[CLIPVisionDiffusersConfig, CLIPVisionDiffusersConfig.get_tag()],
        Annotated[CLIPLEmbedDiffusersConfig, CLIPLEmbedDiffusersConfig.get_tag()],
        Annotated[CLIPGEmbedDiffusersConfig, CLIPGEmbedDiffusersConfig.get_tag()],
        Annotated[SigLIPConfig, SigLIPConfig.get_tag()],
        Annotated[FluxReduxConfig, FluxReduxConfig.get_tag()],
        Annotated[LlavaOnevisionConfig, LlavaOnevisionConfig.get_tag()],
    ],
    Discriminator(get_model_discriminator_value),
]

AnyModelConfigValidator = TypeAdapter(AnyModelConfig)
AnyDefaultSettings: TypeAlias = Union[MainModelDefaultSettings, ControlAdapterDefaultSettings]


class ModelConfigFactory:
    @staticmethod
    def make_config(model_data: Dict[str, Any], timestamp: Optional[float] = None) -> AnyModelConfig:
        """Return the appropriate config object from raw dict values."""
        model = AnyModelConfigValidator.validate_python(model_data)  # type: ignore
        if isinstance(model, CheckpointConfigBase) and timestamp:
            model.converted_at = timestamp
        validate_hash(model.hash)
        return model  # type: ignore
