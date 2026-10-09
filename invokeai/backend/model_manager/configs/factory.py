import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TypeVar,
    Union,
)

from pydantic import BaseModel, Discriminator, TypeAdapter, ValidationError
from typing_extensions import Annotated, Any

from invokeai.app.services.config.config_default import get_config
from invokeai.app.util.misc import uuid_string
from invokeai.backend.architectures import resolve_default_settings
from invokeai.backend.model_hash.model_hash import HASHING_ALGORITHMS
from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.clip_embed import CLIPEmbed_Diffusers_G_Config, CLIPEmbed_Diffusers_L_Config
from invokeai.backend.model_manager.configs.clip_vision import CLIPVision_Diffusers_Config
from invokeai.backend.model_manager.configs.controlnet import (
    ControlAdapterDefaultSettings,
    ControlNet_Checkpoint_Anima_Config,
    ControlNet_Checkpoint_FLUX_Config,
    ControlNet_Checkpoint_SD1_Config,
    ControlNet_Checkpoint_SD2_Config,
    ControlNet_Checkpoint_SDXL_Config,
    ControlNet_Checkpoint_ZImage_Config,
    ControlNet_Diffusers_FLUX_Config,
    ControlNet_Diffusers_SD1_Config,
    ControlNet_Diffusers_SD2_Config,
    ControlNet_Diffusers_SDXL_Config,
)
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig
from invokeai.backend.model_manager.configs.flux_redux import FLUXRedux_Checkpoint_Config
from invokeai.backend.model_manager.configs.gemma2_encoder import (
    Gemma2Encoder_Gemma2Encoder_Config,
    Gemma2Encoder_GGUF_Config,
)
from invokeai.backend.model_manager.configs.gemma4_encoder import Gemma4Encoder_Gemma4Encoder_LTX2_Config
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.ip_adapter import (
    IPAdapter_Checkpoint_FLUX_Config,
    IPAdapter_Checkpoint_SD1_Config,
    IPAdapter_Checkpoint_SD2_Config,
    IPAdapter_Checkpoint_SDXL_Config,
    IPAdapter_InvokeAI_SD1_Config,
    IPAdapter_InvokeAI_SD2_Config,
    IPAdapter_InvokeAI_SDXL_Config,
)
from invokeai.backend.model_manager.configs.llava_onevision import LlavaOnevision_Diffusers_Config
from invokeai.backend.model_manager.configs.lora import (
    ControlLoRA_LyCORIS_FLUX_Config,
    LoRA_Diffusers_Flux2_Config,
    LoRA_Diffusers_FLUX_Config,
    LoRA_Diffusers_SD1_Config,
    LoRA_Diffusers_SD2_Config,
    LoRA_Diffusers_SDXL_Config,
    LoRA_Diffusers_ZImage_Config,
    LoRA_LyCORIS_Anima_Config,
    LoRA_LyCORIS_Flux2_Config,
    LoRA_LyCORIS_FLUX_Config,
    LoRA_LyCORIS_Krea2_Config,
    LoRA_LyCORIS_LTX2_Config,
    LoRA_LyCORIS_MiniMaxH3_Config,
    LoRA_LyCORIS_QwenImage_Config,
    LoRA_LyCORIS_SD1_Config,
    LoRA_LyCORIS_SD2_Config,
    LoRA_LyCORIS_SDXL_Config,
    LoRA_LyCORIS_Wan_Config,
    LoRA_LyCORIS_ZImage_Config,
    LoRA_OMI_FLUX_Config,
    LoRA_OMI_SDXL_Config,
    LoraModelDefaultSettings,
)
from invokeai.backend.model_manager.configs.ltx2_duration_head import LTX2DurationHead_Checkpoint_Config
from invokeai.backend.model_manager.configs.main import (
    Main_BnBNF4_FLUX_Config,
    Main_Checkpoint_Anima_Config,
    Main_Checkpoint_ErnieImage_Config,
    Main_Checkpoint_Flux2_Config,
    Main_Checkpoint_FLUX_Config,
    Main_Checkpoint_Ideogram4_Config,
    Main_Checkpoint_Krea2_Config,
    Main_Checkpoint_LTX2_Config,
    Main_Checkpoint_MiniMaxH3_Config,
    Main_Checkpoint_QwenImage_Config,
    Main_Checkpoint_SD1_Config,
    Main_Checkpoint_SD2_Config,
    Main_Checkpoint_SDXL_Config,
    Main_Checkpoint_SDXLRefiner_Config,
    Main_Checkpoint_Wan_Config,
    Main_Checkpoint_ZImage_Config,
    Main_Diffusers_CogView4_Config,
    Main_Diffusers_ErnieImage_Config,
    Main_Diffusers_Flux2_Config,
    Main_Diffusers_FLUX_Config,
    Main_Diffusers_Ideogram4_Config,
    Main_Diffusers_Krea2_Config,
    Main_Diffusers_LTX2_Config,
    Main_Diffusers_MiniMaxH3_Config,
    Main_Diffusers_QwenImage_Config,
    Main_Diffusers_SD1_Config,
    Main_Diffusers_SD2_Config,
    Main_Diffusers_SD3_Config,
    Main_Diffusers_SDXL_Config,
    Main_Diffusers_SDXLRefiner_Config,
    Main_Diffusers_Wan_Config,
    Main_Diffusers_ZImage_Config,
    Main_GGUF_Flux2_Config,
    Main_GGUF_FLUX_Config,
    Main_GGUF_Ideogram4_Config,
    Main_GGUF_Krea2_Config,
    Main_GGUF_QwenImage_Config,
    Main_GGUF_Wan_Config,
    Main_GGUF_ZImage_Config,
    Main_SDNQ_Diffusers_Flux2_Config,
    Main_SDNQ_Diffusers_FLUX_Config,
    Main_SDNQ_Diffusers_ZImage_Config,
    Main_SDNQ_Flux2_Config,
    Main_SDNQ_FLUX_Config,
    Main_SDNQ_ZImage_Config,
)
from invokeai.backend.model_manager.configs.mistral_encoder import (
    MistralEncoder_Checkpoint_Config,
    MistralEncoder_Diffusers_Config,
    MistralEncoder_GGUF_Config,
)
from invokeai.backend.model_manager.configs.pid_decoder import (
    PiDDecoder_Checkpoint_Flux2_Config,
    PiDDecoder_Checkpoint_FLUX_Config,
    PiDDecoder_Checkpoint_QwenImage_Config,
    PiDDecoder_Checkpoint_SD3_Config,
    PiDDecoder_Checkpoint_SDXL_Config,
)
from invokeai.backend.model_manager.configs.qwen3_5_encoder import Qwen35Encoder_Checkpoint_Config
from invokeai.backend.model_manager.configs.qwen3_encoder import (
    Qwen3Encoder_Checkpoint_Config,
    Qwen3Encoder_GGUF_Config,
    Qwen3Encoder_Qwen3Encoder_Config,
    Qwen3Encoder_SDNQ_Config,
    Qwen3Encoder_SDNQ_Folder_Config,
)
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import (
    Qwen3VLEncoder_Checkpoint_Config,
    Qwen3VLEncoder_Checkpoint_MiniMaxH3_Config,
    Qwen3VLEncoder_GGUF_Config,
    Qwen3VLEncoder_Qwen3VLEncoder_Config,
)
from invokeai.backend.model_manager.configs.qwen_vl_encoder import (
    QwenVLEncoder_Checkpoint_Config,
    QwenVLEncoder_Diffusers_Config,
)
from invokeai.backend.model_manager.configs.siglip import SigLIP_Diffusers_Config
from invokeai.backend.model_manager.configs.spandrel import Spandrel_Checkpoint_Config
from invokeai.backend.model_manager.configs.t2i_adapter import (
    T2IAdapter_Diffusers_SD1_Config,
    T2IAdapter_Diffusers_SDXL_Config,
)
from invokeai.backend.model_manager.configs.t5_encoder import (
    T5Encoder_BnBLLMint8_Config,
    T5Encoder_GGUF_Config,
    T5Encoder_SDNQ_Config,
    T5Encoder_T5Encoder_Config,
)
from invokeai.backend.model_manager.configs.text_llm import TextLLM_Diffusers_Config
from invokeai.backend.model_manager.configs.textual_inversion import (
    TI_File_SD1_Config,
    TI_File_SD2_Config,
    TI_File_SDXL_Config,
    TI_Folder_SD1_Config,
    TI_Folder_SD2_Config,
    TI_Folder_SDXL_Config,
)
from invokeai.backend.model_manager.configs.unknown import Unknown_Config
from invokeai.backend.model_manager.configs.vae import (
    VAE_Checkpoint_Anima_Config,
    VAE_Checkpoint_Flux2_Config,
    VAE_Checkpoint_FLUX_Config,
    VAE_Checkpoint_QwenImage_Config,
    VAE_Checkpoint_SD1_Config,
    VAE_Checkpoint_SD2_Config,
    VAE_Checkpoint_SD3_Config,
    VAE_Checkpoint_SDXL_Config,
    VAE_Checkpoint_Wan_Config,
    VAE_Diffusers_Flux2_Config,
    VAE_Diffusers_FLUX_Config,
    VAE_Diffusers_SD1_Config,
    VAE_Diffusers_SD3_Config,
    VAE_Diffusers_SDXL_Config,
    VAE_Diffusers_Wan_Config,
)
from invokeai.backend.model_manager.configs.wan_t5_encoder import WanT5Encoder_WanT5Encoder_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk, read_safetensors_header
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelSourceType,
    ModelType,
    variant_type_adapter,
)
from invokeai.backend.quantization.gguf.loaders import parse_q8_cr_markers

logger = logging.getLogger(__name__)

_FP8_SAFETENSORS_DTYPES = frozenset({"F8_E4M3", "F8_E5M2"})

# What single-file checkpoints bundle beside their denoiser. Float8 weights there say nothing about the denoiser.
_BUNDLED_COMPONENT_KEY_PREFIXES = ("cond_stage_model.", "conditioner.", "first_stage_model.", "text_encoders.", "vae.")

# Where a diffusers folder keeps its denoiser. A single-component folder (a ControlNet) keeps it at the root.
_DENOISER_SUBFOLDERS = frozenset({"transformer", "unet"})

SettingsT = TypeVar("SettingsT", bound=BaseModel)


def _denoiser_stores_float8_weights(mod: ModelOnDisk) -> bool:
    """Whether the denoiser's weights are stored in a float8 the storage cast can reproduce, judged by the safetensors
    headers and never by a file name.

    Only headers are read, a few KB each. A header that cannot be read is no evidence of float8 weights: identification
    did not need that file, and picking a default setting is no reason to fail an install.

    Scaled fp8 checkpoints count too. They briefly did not: FP8 Storage used to fold their per-tensor scales away and
    let the layerwise cast re-encode the result as *unscaled* fp8, which on `flux-2-klein-4b-fp8` flushed 3.4% of the
    weights to zero for the byte count the file already had. The loaders now keep such a file in its own scaled form
    instead, so switching the setting on is lossless and worth doing by default.
    """
    if mod.path.is_file():
        candidates = [mod.path]
    else:
        candidates = [
            path
            for path in mod.weight_files()
            if path.parent == mod.path or path.relative_to(mod.path).parts[0] in _DENOISER_SUBFOLDERS
        ]

    stores_float8 = False
    for path in candidates:
        if path.suffix != ".safetensors":
            continue
        try:
            header = read_safetensors_header(path)
        except (OSError, ValueError) as e:
            logger.debug(f"Could not read the safetensors header of {path} to look for float8 weights: {e}")
            continue
        denoiser = {key: info for key, info in header.items() if not key.startswith(_BUNDLED_COMPONENT_KEY_PREFIXES)}
        stores_float8 = stores_float8 or any(
            isinstance(info, dict) and info.get("dtype") in _FP8_SAFETENSORS_DTYPES for info in denoiser.values()
        )
    return stores_float8


def _identified_default_settings(
    recommended: SettingsT | None,
    settings_cls: type[SettingsT],
    mod: ModelOnDisk,
    override_fields: dict[str, Any] | None,
    # Quoted: `AnyModelConfig` is the union of the concrete config classes and is assembled further down
    # this module. `Config_Base` would not do — it declares none of base/type/format, which only a concrete
    # class does (enforced by `__pydantic_init_subclass__`).
    config: "AnyModelConfig",
) -> SettingsT | None:
    """Layer the settings sent with an install, and FP8 Storage for a float8 denoiser, over the recommended defaults.

    An install setting wins over detection in both directions. Fields the settings class does not have (a main-model
    field sent along with a LoRA install) are dropped, since nothing would read them.

    FP8 Storage is enabled only where the model's own loader implements it. It used to be enabled for any float8
    denoiser whose format was not already quantized, which wrote `fp8_storage: true` onto records whose loader ignores
    it -- a Wan fp8 install was told its weights were halved and then loaded at bf16 size. `fp8_storage_verdict` is the
    same answer the loader gate and the API row give; see `load/fp8_capability.py`.

    A request sent with the install is dropped for such a model rather than stored. The detail panel does not offer the
    control where the loader ignores it, so a stored `true` would be a value nobody can see or clear again -- and the
    Add Models flow has one FP8 Storage checkbox for whatever is being installed, so it is easy to send for a model
    that cannot use it.
    """
    requested = (override_fields or {}).get("default_settings") or {}
    update = {
        name: value for name, value in requested.items() if name in settings_cls.model_fields and value is not None
    }

    # Imported here, not at module scope: `model_loader_registry` imports this module, so the reverse edge would close
    # a cycle. Identification runs long after import time, and the import is a `sys.modules` hit after the first.
    from invokeai.backend.model_manager.load.fp8_capability import fp8_storage_verdict

    if not fp8_storage_verdict(config.base, config.type, config.format).supported:
        update.pop("fp8_storage", None)
    elif "fp8_storage" not in update and _denoiser_stores_float8_weights(mod):
        logger.info(f"{mod.name}: denoiser weights are stored in float8, enabling FP8 Storage by default")
        update["fp8_storage"] = True

    if not update:
        return recommended
    current = recommended.model_dump() if recommended is not None else {}
    return settings_cls.model_validate({**current, **update})


app_config = get_config()

# Known model file extensions for sanity checking
_MODEL_EXTENSIONS = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".gguf",
    ".onnx",
}

# Known config file names for diffusers/transformers models
_CONFIG_FILES = {
    "model_index.json",
    "modular_model_index.json",
    "config.json",
}

# Maximum number of files in a directory to be considered a model
_MAX_FILES_IN_MODEL_DIR = 50

# Maximum depth to search for model files in directories
_MAX_SEARCH_DEPTH = 2

# Classes introduced by the versions pinned by this checkout may not exist in the interpreter used by
# lightweight config tests. Keep explicit markers only for those newly supported classes; established
# classes are resolved from the installed Diffusers/Transformers exports below.
_PINNED_MODEL_CLASS_MARKERS = {
    "Krea2Pipeline",
    # MiniMax H3 classes exist only in an unreleased diffusers branch (vendored under
    # invokeai/backend/minimax_h3); the installed diffusers cannot resolve them.
    "MiniMaxH3ModularPipeline",
    "MiniMaxH3Transformer3DModel",
    "AutoencoderKLMiniMaxH3",
    "AutoencoderKLMiniMaxH3Audio",
}


def _is_known_model_marker(config_name: str, config: Any) -> bool:
    """Return whether a root config names a class/model type provided by our model libraries."""
    import diffusers
    import transformers
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    if not isinstance(config, dict):
        return False

    def has_model_export(module: Any, name: Any, expected_bases: tuple[type, ...]) -> bool:
        if not isinstance(name, str) or not name:
            return False
        try:
            exported = getattr(module, name, None)
            return (
                isinstance(exported, type) and exported not in expected_bases and issubclass(exported, expected_bases)
            )
        except Exception:
            return False

    if config_name in ("model_index.json", "modular_model_index.json"):
        class_name = config.get("_class_name")
        # Non-str _class_name (e.g. a list) must read as "not a marker", not TypeError on the set lookup.
        return (isinstance(class_name, str) and class_name in _PINNED_MODEL_CLASS_MARKERS) or has_model_export(
            diffusers, class_name, (DiffusionPipeline,)
        )

    class_name = config.get("_class_name")
    if (
        (isinstance(class_name, str) and class_name in _PINNED_MODEL_CLASS_MARKERS)
        or has_model_export(diffusers, class_name, (ModelMixin, DiffusionPipeline))
        or has_model_export(transformers, class_name, (PreTrainedModel, PretrainedConfig))
    ):
        return True
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type in CONFIG_MAPPING_NAMES:
        return True
    architectures = config.get("architectures")
    return isinstance(architectures, list) and any(
        has_model_export(transformers, name, (PreTrainedModel,)) for name in architectures
    )


# The types are listed explicitly because IDEs/LSPs can't identify the correct types
# when AnyModelConfig is constructed dynamically using ModelConfigBase.all_config_classes
AnyModelConfig = Annotated[
    Union[
        # Main (Pipeline) - diffusers format
        Annotated[Main_Diffusers_SD1_Config, Main_Diffusers_SD1_Config.get_tag()],
        Annotated[Main_Diffusers_SD2_Config, Main_Diffusers_SD2_Config.get_tag()],
        Annotated[Main_Diffusers_SDXL_Config, Main_Diffusers_SDXL_Config.get_tag()],
        Annotated[Main_Diffusers_SDXLRefiner_Config, Main_Diffusers_SDXLRefiner_Config.get_tag()],
        Annotated[Main_Diffusers_SD3_Config, Main_Diffusers_SD3_Config.get_tag()],
        Annotated[Main_Diffusers_FLUX_Config, Main_Diffusers_FLUX_Config.get_tag()],
        Annotated[Main_Diffusers_Flux2_Config, Main_Diffusers_Flux2_Config.get_tag()],
        Annotated[Main_Diffusers_CogView4_Config, Main_Diffusers_CogView4_Config.get_tag()],
        Annotated[Main_Diffusers_QwenImage_Config, Main_Diffusers_QwenImage_Config.get_tag()],
        Annotated[Main_Diffusers_Wan_Config, Main_Diffusers_Wan_Config.get_tag()],
        Annotated[Main_Diffusers_ZImage_Config, Main_Diffusers_ZImage_Config.get_tag()],
        Annotated[Main_Diffusers_ErnieImage_Config, Main_Diffusers_ErnieImage_Config.get_tag()],
        Annotated[Main_Diffusers_Ideogram4_Config, Main_Diffusers_Ideogram4_Config.get_tag()],
        Annotated[Main_Diffusers_Krea2_Config, Main_Diffusers_Krea2_Config.get_tag()],
        Annotated[Main_Diffusers_MiniMaxH3_Config, Main_Diffusers_MiniMaxH3_Config.get_tag()],
        Annotated[Main_Diffusers_LTX2_Config, Main_Diffusers_LTX2_Config.get_tag()],
        # Main (Pipeline) - checkpoint format
        # IMPORTANT: FLUX.2 must be checked BEFORE FLUX.1 because FLUX.2 has specific validation
        # that will reject FLUX.1 models, but FLUX.1 validation may incorrectly match FLUX.2 models
        Annotated[Main_Checkpoint_SD1_Config, Main_Checkpoint_SD1_Config.get_tag()],
        Annotated[Main_Checkpoint_SD2_Config, Main_Checkpoint_SD2_Config.get_tag()],
        Annotated[Main_Checkpoint_SDXL_Config, Main_Checkpoint_SDXL_Config.get_tag()],
        Annotated[Main_Checkpoint_SDXLRefiner_Config, Main_Checkpoint_SDXLRefiner_Config.get_tag()],
        Annotated[Main_Checkpoint_Flux2_Config, Main_Checkpoint_Flux2_Config.get_tag()],
        Annotated[Main_Checkpoint_FLUX_Config, Main_Checkpoint_FLUX_Config.get_tag()],
        Annotated[Main_Checkpoint_QwenImage_Config, Main_Checkpoint_QwenImage_Config.get_tag()],
        Annotated[Main_Checkpoint_Wan_Config, Main_Checkpoint_Wan_Config.get_tag()],
        Annotated[Main_Checkpoint_ZImage_Config, Main_Checkpoint_ZImage_Config.get_tag()],
        Annotated[Main_Checkpoint_ErnieImage_Config, Main_Checkpoint_ErnieImage_Config.get_tag()],
        Annotated[Main_Checkpoint_Ideogram4_Config, Main_Checkpoint_Ideogram4_Config.get_tag()],
        Annotated[Main_Checkpoint_Krea2_Config, Main_Checkpoint_Krea2_Config.get_tag()],
        Annotated[Main_Checkpoint_Anima_Config, Main_Checkpoint_Anima_Config.get_tag()],
        Annotated[Main_Checkpoint_MiniMaxH3_Config, Main_Checkpoint_MiniMaxH3_Config.get_tag()],
        Annotated[Main_Checkpoint_LTX2_Config, Main_Checkpoint_LTX2_Config.get_tag()],
        # Main (Pipeline) - quantized formats
        # IMPORTANT: FLUX.2 must be checked BEFORE FLUX.1 because FLUX.2 has specific validation
        # that will reject FLUX.1 models, but FLUX.1 validation may incorrectly match FLUX.2 models
        Annotated[Main_BnBNF4_FLUX_Config, Main_BnBNF4_FLUX_Config.get_tag()],
        Annotated[Main_GGUF_Flux2_Config, Main_GGUF_Flux2_Config.get_tag()],
        Annotated[Main_GGUF_FLUX_Config, Main_GGUF_FLUX_Config.get_tag()],
        Annotated[Main_GGUF_QwenImage_Config, Main_GGUF_QwenImage_Config.get_tag()],
        Annotated[Main_GGUF_Wan_Config, Main_GGUF_Wan_Config.get_tag()],
        Annotated[Main_GGUF_ZImage_Config, Main_GGUF_ZImage_Config.get_tag()],
        Annotated[Main_GGUF_Krea2_Config, Main_GGUF_Krea2_Config.get_tag()],
        Annotated[Main_GGUF_Ideogram4_Config, Main_GGUF_Ideogram4_Config.get_tag()],
        # IMPORTANT: FLUX.2 must be listed BEFORE FLUX.1 here. An ambiguous SDNQ transformer
        # checkpoint (prefixed FLUX.2 keys) can look like a FLUX.1 main model, so FLUX.2 must get
        # first refusal. Main_SDNQ_FLUX_Config additionally rejects FLUX.2 state dicts to keep the
        # two mutually exclusive regardless of iteration order.
        Annotated[Main_SDNQ_Flux2_Config, Main_SDNQ_Flux2_Config.get_tag()],
        Annotated[Main_SDNQ_Diffusers_Flux2_Config, Main_SDNQ_Diffusers_Flux2_Config.get_tag()],
        Annotated[Main_SDNQ_FLUX_Config, Main_SDNQ_FLUX_Config.get_tag()],
        Annotated[Main_SDNQ_Diffusers_FLUX_Config, Main_SDNQ_Diffusers_FLUX_Config.get_tag()],
        Annotated[Main_SDNQ_ZImage_Config, Main_SDNQ_ZImage_Config.get_tag()],
        Annotated[Main_SDNQ_Diffusers_ZImage_Config, Main_SDNQ_Diffusers_ZImage_Config.get_tag()],
        # VAE - checkpoint format
        Annotated[VAE_Checkpoint_SD1_Config, VAE_Checkpoint_SD1_Config.get_tag()],
        Annotated[VAE_Checkpoint_SD2_Config, VAE_Checkpoint_SD2_Config.get_tag()],
        Annotated[VAE_Checkpoint_SDXL_Config, VAE_Checkpoint_SDXL_Config.get_tag()],
        Annotated[VAE_Checkpoint_FLUX_Config, VAE_Checkpoint_FLUX_Config.get_tag()],
        Annotated[VAE_Checkpoint_SD3_Config, VAE_Checkpoint_SD3_Config.get_tag()],
        Annotated[VAE_Checkpoint_Flux2_Config, VAE_Checkpoint_Flux2_Config.get_tag()],
        # Wan and Qwen-Image share the 16-channel AutoencoderKLWan layout. The order here decides
        # nothing (see "Configs must exclude each other" in new-model-integration.mdx): both configs
        # defer on `_filename_suggests_wan`, which is what hands a file named for Wan to Wan.
        Annotated[VAE_Checkpoint_Wan_Config, VAE_Checkpoint_Wan_Config.get_tag()],
        Annotated[VAE_Checkpoint_QwenImage_Config, VAE_Checkpoint_QwenImage_Config.get_tag()],
        Annotated[VAE_Checkpoint_Anima_Config, VAE_Checkpoint_Anima_Config.get_tag()],
        # VAE - diffusers format
        Annotated[VAE_Diffusers_SD1_Config, VAE_Diffusers_SD1_Config.get_tag()],
        Annotated[VAE_Diffusers_SDXL_Config, VAE_Diffusers_SDXL_Config.get_tag()],
        Annotated[VAE_Diffusers_FLUX_Config, VAE_Diffusers_FLUX_Config.get_tag()],
        Annotated[VAE_Diffusers_SD3_Config, VAE_Diffusers_SD3_Config.get_tag()],
        Annotated[VAE_Diffusers_Flux2_Config, VAE_Diffusers_Flux2_Config.get_tag()],
        Annotated[VAE_Diffusers_Wan_Config, VAE_Diffusers_Wan_Config.get_tag()],
        # PiD Decoder - checkpoint format
        Annotated[PiDDecoder_Checkpoint_FLUX_Config, PiDDecoder_Checkpoint_FLUX_Config.get_tag()],
        Annotated[PiDDecoder_Checkpoint_Flux2_Config, PiDDecoder_Checkpoint_Flux2_Config.get_tag()],
        Annotated[PiDDecoder_Checkpoint_SD3_Config, PiDDecoder_Checkpoint_SD3_Config.get_tag()],
        Annotated[PiDDecoder_Checkpoint_SDXL_Config, PiDDecoder_Checkpoint_SDXL_Config.get_tag()],
        Annotated[PiDDecoder_Checkpoint_QwenImage_Config, PiDDecoder_Checkpoint_QwenImage_Config.get_tag()],
        # ControlNet - checkpoint format
        Annotated[ControlNet_Checkpoint_SD1_Config, ControlNet_Checkpoint_SD1_Config.get_tag()],
        Annotated[ControlNet_Checkpoint_SD2_Config, ControlNet_Checkpoint_SD2_Config.get_tag()],
        Annotated[ControlNet_Checkpoint_SDXL_Config, ControlNet_Checkpoint_SDXL_Config.get_tag()],
        Annotated[ControlNet_Checkpoint_FLUX_Config, ControlNet_Checkpoint_FLUX_Config.get_tag()],
        Annotated[ControlNet_Checkpoint_ZImage_Config, ControlNet_Checkpoint_ZImage_Config.get_tag()],
        Annotated[ControlNet_Checkpoint_Anima_Config, ControlNet_Checkpoint_Anima_Config.get_tag()],
        # ControlNet - diffusers format
        Annotated[ControlNet_Diffusers_SD1_Config, ControlNet_Diffusers_SD1_Config.get_tag()],
        Annotated[ControlNet_Diffusers_SD2_Config, ControlNet_Diffusers_SD2_Config.get_tag()],
        Annotated[ControlNet_Diffusers_SDXL_Config, ControlNet_Diffusers_SDXL_Config.get_tag()],
        Annotated[ControlNet_Diffusers_FLUX_Config, ControlNet_Diffusers_FLUX_Config.get_tag()],
        # LoRA - LyCORIS format
        # IMPORTANT: FLUX.2 must be checked BEFORE FLUX.1 because FLUX.2 has specific validation
        # that will reject FLUX.1 models, but FLUX.1 validation may incorrectly match FLUX.2 models
        Annotated[LoRA_LyCORIS_SD1_Config, LoRA_LyCORIS_SD1_Config.get_tag()],
        Annotated[LoRA_LyCORIS_SD2_Config, LoRA_LyCORIS_SD2_Config.get_tag()],
        Annotated[LoRA_LyCORIS_SDXL_Config, LoRA_LyCORIS_SDXL_Config.get_tag()],
        Annotated[LoRA_LyCORIS_Flux2_Config, LoRA_LyCORIS_Flux2_Config.get_tag()],
        Annotated[LoRA_LyCORIS_FLUX_Config, LoRA_LyCORIS_FLUX_Config.get_tag()],
        Annotated[LoRA_LyCORIS_ZImage_Config, LoRA_LyCORIS_ZImage_Config.get_tag()],
        Annotated[LoRA_LyCORIS_Krea2_Config, LoRA_LyCORIS_Krea2_Config.get_tag()],
        Annotated[LoRA_LyCORIS_QwenImage_Config, LoRA_LyCORIS_QwenImage_Config.get_tag()],
        # MiniMax H3 keys on H3-exclusive submodules (fused ``attn.qkv_proj``,
        # ``adaln_proj.linear``) and rejects other architectures' signatures, so it
        # is mutually exclusive with Wan/Anima regardless of order (locked in by
        # ``test_minimax_h3_lora_probe_independence.py``).
        Annotated[LoRA_LyCORIS_LTX2_Config, LoRA_LyCORIS_LTX2_Config.get_tag()],
        Annotated[LoRA_LyCORIS_MiniMaxH3_Config, LoRA_LyCORIS_MiniMaxH3_Config.get_tag()],
        # Wan and Anima both target ``blocks.X`` shapes; their LoRA probes are
        # mutually exclusive — Wan rejects Anima's ``_proj``/``mlp``/
        # ``adaln_modulation`` markers, Anima requires at least one of those
        # markers (see ``has_cosmos_dit_*_keys_strict``). Order between these
        # two doesn't affect correctness; mutual exclusivity is locked in by
        # ``test_wan_lora_probe_independence.py``.
        Annotated[LoRA_LyCORIS_Wan_Config, LoRA_LyCORIS_Wan_Config.get_tag()],
        Annotated[LoRA_LyCORIS_Anima_Config, LoRA_LyCORIS_Anima_Config.get_tag()],
        # LoRA - OMI format
        Annotated[LoRA_OMI_SDXL_Config, LoRA_OMI_SDXL_Config.get_tag()],
        Annotated[LoRA_OMI_FLUX_Config, LoRA_OMI_FLUX_Config.get_tag()],
        # LoRA - diffusers format
        # IMPORTANT: FLUX.2 must be checked BEFORE FLUX.1 because FLUX.2 has specific validation
        # that will reject FLUX.1 models, but FLUX.1 validation may incorrectly match FLUX.2 models
        Annotated[LoRA_Diffusers_SD1_Config, LoRA_Diffusers_SD1_Config.get_tag()],
        Annotated[LoRA_Diffusers_SD2_Config, LoRA_Diffusers_SD2_Config.get_tag()],
        Annotated[LoRA_Diffusers_SDXL_Config, LoRA_Diffusers_SDXL_Config.get_tag()],
        Annotated[LoRA_Diffusers_Flux2_Config, LoRA_Diffusers_Flux2_Config.get_tag()],
        Annotated[LoRA_Diffusers_FLUX_Config, LoRA_Diffusers_FLUX_Config.get_tag()],
        Annotated[LoRA_Diffusers_ZImage_Config, LoRA_Diffusers_ZImage_Config.get_tag()],
        # ControlLoRA - diffusers format
        Annotated[ControlLoRA_LyCORIS_FLUX_Config, ControlLoRA_LyCORIS_FLUX_Config.get_tag()],
        # T5 Encoder - all formats
        Annotated[T5Encoder_T5Encoder_Config, T5Encoder_T5Encoder_Config.get_tag()],
        Annotated[T5Encoder_BnBLLMint8_Config, T5Encoder_BnBLLMint8_Config.get_tag()],
        Annotated[T5Encoder_SDNQ_Config, T5Encoder_SDNQ_Config.get_tag()],
        Annotated[T5Encoder_GGUF_Config, T5Encoder_GGUF_Config.get_tag()],
        # Qwen3-VL Encoder (Qwen3-VL multimodal encoder for Krea-2) - checked BEFORE the text-only Qwen3
        # encoder so single-file VL checkpoints (which also carry generic model.layers.* keys) are not
        # misclassified as the Z-Image Qwen3 encoder. The VL probe requires the visual tower.
        # MiniMax H3's truncated 32B conditioning encoder goes first: it matches on explicit
        # safetensors metadata without reading tensors, and the Krea-2 config below is locked to
        # the 4B shape so neither can claim the other's files.
        Annotated[Qwen3VLEncoder_Checkpoint_MiniMaxH3_Config, Qwen3VLEncoder_Checkpoint_MiniMaxH3_Config.get_tag()],
        Annotated[Qwen3VLEncoder_Checkpoint_Config, Qwen3VLEncoder_Checkpoint_Config.get_tag()],
        # Kept mutually exclusive with Qwen3Encoder_GGUF_Config by an architecture-metadata check on
        # both sides, NOT by position in this list: identification iterates `Config_Base.CONFIG_CLASSES`
        # (a set) and `matches_sort_key` puts both encoders in the same bucket, so a double match would
        # be resolved by arbitrary set-iteration order. The check is load-bearing because llama.cpp
        # keeps the visual tower in a separate mmproj file -- a Qwen3-VL GGUF has none to probe for and
        # satisfies the text-only Qwen3 GGUF heuristic in full.
        Annotated[Qwen3VLEncoder_GGUF_Config, Qwen3VLEncoder_GGUF_Config.get_tag()],
        Annotated[Qwen3VLEncoder_Qwen3VLEncoder_Config, Qwen3VLEncoder_Qwen3VLEncoder_Config.get_tag()],
        Annotated[Qwen35Encoder_Checkpoint_Config, Qwen35Encoder_Checkpoint_Config.get_tag()],
        # Qwen3 Encoder
        Annotated[Qwen3Encoder_Qwen3Encoder_Config, Qwen3Encoder_Qwen3Encoder_Config.get_tag()],
        Annotated[Qwen3Encoder_Checkpoint_Config, Qwen3Encoder_Checkpoint_Config.get_tag()],
        Annotated[Qwen3Encoder_GGUF_Config, Qwen3Encoder_GGUF_Config.get_tag()],
        Annotated[Qwen3Encoder_SDNQ_Config, Qwen3Encoder_SDNQ_Config.get_tag()],
        Annotated[Qwen3Encoder_SDNQ_Folder_Config, Qwen3Encoder_SDNQ_Folder_Config.get_tag()],
        # Mistral Encoder (used by FLUX.2 [dev])
        Annotated[MistralEncoder_Diffusers_Config, MistralEncoder_Diffusers_Config.get_tag()],
        Annotated[MistralEncoder_Checkpoint_Config, MistralEncoder_Checkpoint_Config.get_tag()],
        Annotated[MistralEncoder_GGUF_Config, MistralEncoder_GGUF_Config.get_tag()],
        # Gemma 2 Encoder (used by PiD)
        Annotated[Gemma2Encoder_Gemma2Encoder_Config, Gemma2Encoder_Gemma2Encoder_Config.get_tag()],
        Annotated[Gemma4Encoder_Gemma4Encoder_LTX2_Config, Gemma4Encoder_Gemma4Encoder_LTX2_Config.get_tag()],
        Annotated[Gemma2Encoder_GGUF_Config, Gemma2Encoder_GGUF_Config.get_tag()],
        # Qwen VL Encoder (Qwen2.5-VL multimodal encoder for Qwen Image)
        Annotated[QwenVLEncoder_Diffusers_Config, QwenVLEncoder_Diffusers_Config.get_tag()],
        Annotated[QwenVLEncoder_Checkpoint_Config, QwenVLEncoder_Checkpoint_Config.get_tag()],
        # Wan T5 Encoder (UMT5-XXL for Wan 2.2)
        Annotated[WanT5Encoder_WanT5Encoder_Config, WanT5Encoder_WanT5Encoder_Config.get_tag()],
        # TI - file format
        Annotated[TI_File_SD1_Config, TI_File_SD1_Config.get_tag()],
        Annotated[TI_File_SD2_Config, TI_File_SD2_Config.get_tag()],
        Annotated[TI_File_SDXL_Config, TI_File_SDXL_Config.get_tag()],
        # TI - folder format
        Annotated[TI_Folder_SD1_Config, TI_Folder_SD1_Config.get_tag()],
        Annotated[TI_Folder_SD2_Config, TI_Folder_SD2_Config.get_tag()],
        Annotated[TI_Folder_SDXL_Config, TI_Folder_SDXL_Config.get_tag()],
        # IP Adapter - InvokeAI format
        Annotated[IPAdapter_InvokeAI_SD1_Config, IPAdapter_InvokeAI_SD1_Config.get_tag()],
        Annotated[IPAdapter_InvokeAI_SD2_Config, IPAdapter_InvokeAI_SD2_Config.get_tag()],
        Annotated[IPAdapter_InvokeAI_SDXL_Config, IPAdapter_InvokeAI_SDXL_Config.get_tag()],
        # IP Adapter - checkpoint format
        Annotated[IPAdapter_Checkpoint_SD1_Config, IPAdapter_Checkpoint_SD1_Config.get_tag()],
        Annotated[IPAdapter_Checkpoint_SD2_Config, IPAdapter_Checkpoint_SD2_Config.get_tag()],
        Annotated[IPAdapter_Checkpoint_SDXL_Config, IPAdapter_Checkpoint_SDXL_Config.get_tag()],
        Annotated[IPAdapter_Checkpoint_FLUX_Config, IPAdapter_Checkpoint_FLUX_Config.get_tag()],
        # T2I Adapter - diffusers format
        Annotated[T2IAdapter_Diffusers_SD1_Config, T2IAdapter_Diffusers_SD1_Config.get_tag()],
        Annotated[T2IAdapter_Diffusers_SDXL_Config, T2IAdapter_Diffusers_SDXL_Config.get_tag()],
        # Misc models
        Annotated[Spandrel_Checkpoint_Config, Spandrel_Checkpoint_Config.get_tag()],
        Annotated[CLIPEmbed_Diffusers_G_Config, CLIPEmbed_Diffusers_G_Config.get_tag()],
        Annotated[CLIPEmbed_Diffusers_L_Config, CLIPEmbed_Diffusers_L_Config.get_tag()],
        Annotated[CLIPVision_Diffusers_Config, CLIPVision_Diffusers_Config.get_tag()],
        Annotated[SigLIP_Diffusers_Config, SigLIP_Diffusers_Config.get_tag()],
        Annotated[FLUXRedux_Checkpoint_Config, FLUXRedux_Checkpoint_Config.get_tag()],
        Annotated[LTX2DurationHead_Checkpoint_Config, LTX2DurationHead_Checkpoint_Config.get_tag()],
        Annotated[LlavaOnevision_Diffusers_Config, LlavaOnevision_Diffusers_Config.get_tag()],
        Annotated[TextLLM_Diffusers_Config, TextLLM_Diffusers_Config.get_tag()],
        Annotated[ExternalApiModelConfig, ExternalApiModelConfig.get_tag()],
        # Unknown model (fallback)
        Annotated[Unknown_Config, Unknown_Config.get_tag()],
    ],
    Discriminator(Config_Base.get_model_discriminator_value),
]

AnyModelConfigValidator = TypeAdapter[AnyModelConfig](AnyModelConfig)
"""Pydantic TypeAdapter for the AnyModelConfig union, used for parsing and validation.

If you need to parse/validate a dict or JSON into an AnyModelConfig, you should probably use
ModelConfigFactory.from_dict or ModelConfigFactory.from_json instead as they may implement
additional logic in the future.
"""


@dataclass
class ModelClassificationResult:
    """Result of attempting to classify a model on disk into a specific model config.

    Attributes:
        match: The best matching model config, or None if no match was found.
        results: A mapping of model config class names to either an instance of that class (if it matched)
            or an Exception (if it didn't match or an error occurred during matching).
    """

    config: AnyModelConfig | None
    details: dict[str, AnyModelConfig | Exception]

    @property
    def all_matches(self) -> list[AnyModelConfig]:
        """Returns a list of all matching model configs found."""
        return [r for r in self.details.values() if isinstance(r, Config_Base)]

    @property
    def match_count(self) -> int:
        """Returns the number of matching model configs found."""
        return len(self.all_matches)

    @property
    def invalid_matches(self) -> list[InvalidMatchError]:
        """Rejections from config classes that recognised the model but found it unusable.

        Non-empty means `config` is None because the file is broken, not because it is unidentifiable
        — callers can report the specific reason instead of a generic "could not identify".
        """
        return [r for r in self.details.values() if isinstance(r, InvalidMatchError)]


class ModelConfigFactory:
    @staticmethod
    def _detach_traceback(e: Exception) -> Exception:
        """Drop the traceback and exception chain from a failed match before it is stored.

        A stored traceback pins the frames of the config class that raised it, and those frames hold the
        model's whole state dict in their locals. Exception/traceback/frame is a reference cycle, so the
        state dict survives until the cyclic collector runs - and that collector is driven by object
        counts, while a state dict is a few thousand objects holding gigabytes. Installing a queue of
        models therefore accumulates every probed checkpoint in RAM (and, for safetensors, keeps the file
        mapped, which is why moving the file afterwards needs retries on Windows).

        Nothing reads these tracebacks: `details` is only ever consumed for the exception's type and
        message.
        """
        e.__traceback__ = None
        e.__context__ = None
        e.__cause__ = None
        return e

    @staticmethod
    def _raise_for_unsupported_gguf_quantization(mod: ModelOnDisk, config: Config_Base) -> None:
        """Refuse a GGUF whose ComfyUI-GGUF quantization the matched config's loader cannot decode.

        Here rather than in each GGUF config: the question is the same for all of them, and a GGUF config
        added later is refused by default instead of installing a file it fails on at the first render.
        `InvalidMatchError`, so the installer shows the reason rather than registering an unknown model.
        Read from the cached metadata, so a large file is not re-read once per candidate.
        """
        if getattr(config, "format", None) is not ModelFormat.GGUFQuantized:
            return
        try:
            markers = parse_q8_cr_markers(mod.metadata(), mod.path.name)
        except ValueError as e:
            raise InvalidMatchError(str(e)) from None
        if markers and not type(config).DECODES_GGUF_Q8_CR:
            raise InvalidMatchError(
                f"{mod.path.name} is a ComfyUI-GGUF Q8_CR (int8 convrot) file, which this model type cannot load "
                "yet. Use a GGML-quantized (Q8_0, Q4_K, ...) or safetensors build instead."
            )

    @staticmethod
    def from_dict(fields: dict[str, Any]) -> AnyModelConfig:
        """Return the appropriate config object from raw dict values."""
        model = AnyModelConfigValidator.validate_python(fields)
        return model

    @staticmethod
    def from_json(json: str | bytes | bytearray) -> AnyModelConfig:
        """Return the appropriate config object from json."""
        model = AnyModelConfigValidator.validate_json(json)
        return model

    @staticmethod
    def build_common_fields(
        mod: ModelOnDisk,
        override_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Builds the common fields for all model configs.

        Args:
            mod: The model on disk to extract fields from.
            overrides: A optional dictionary of fields to override. These fields will take precedence over the values
                extracted from the model on disk.

        - Casts string fields to their Enum types.
        - Does not validate the fields against the model config schema.
        """

        _overrides: dict[str, Any] = override_fields or {}
        fields: dict[str, Any] = {}

        if "type" in _overrides:
            fields["type"] = ModelType(_overrides["type"])

        if "format" in _overrides:
            fields["format"] = ModelFormat(_overrides["format"])

        if "base" in _overrides:
            fields["base"] = BaseModelType(_overrides["base"])

        if "source_type" in _overrides:
            fields["source_type"] = ModelSourceType(_overrides["source_type"])

        if "variant" in _overrides:
            fields["variant"] = variant_type_adapter.validate_strings(_overrides["variant"])

        fields["path"] = mod.path.as_posix()
        fields["source"] = _overrides.get("source") or fields["path"]
        fields["source_type"] = _overrides.get("source_type") or ModelSourceType.Path
        fields["name"] = _overrides.get("name") or mod.name
        fields["hash"] = _overrides.get("hash") or mod.hash()
        fields["key"] = _overrides.get("key") or uuid_string()
        fields["description"] = _overrides.get("description")
        fields["file_size"] = _overrides.get("file_size") or mod.size()

        return fields

    @staticmethod
    def _validate_path_looks_like_model(path: Path) -> None:
        """Perform basic sanity checks to ensure a path looks like a model.

        This prevents wasting time trying to identify obviously non-model paths like
        home directories or downloads folders. Raises RuntimeError if the path doesn't
        pass basic checks.

        Args:
            path: The path to validate

        Raises:
            ValueError: If the path doesn't look like a model
        """
        if path.is_file():
            # For files, just check the extension
            if path.suffix.lower() not in _MODEL_EXTENSIONS:
                raise ValueError(
                    f"File extension {path.suffix} is not a recognized model format. "
                    f"Expected one of: {', '.join(sorted(_MODEL_EXTENSIONS))}"
                )
        else:
            # Recognized Diffusers/Transformers configs are safe model markers. A generic config.json
            # is not sufficient because many large application directories contain one.
            recognized_root_config = False
            for config_name in _CONFIG_FILES:
                config_path = path / config_name
                if not config_path.exists():
                    continue
                try:
                    # Model config.json files are UTF-8; read explicitly so a non-ASCII value does not
                    # raise UnicodeDecodeError under a cp1252 (Windows) locale and get mis-treated as
                    # "unrecognized", which would wrongly reject a valid model directory.
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                recognized_root_config = _is_known_model_marker(config_name, config)
                if recognized_root_config:
                    break
            if recognized_root_config:
                return

            # For directories, do a quick file count check with early exit
            total_files = 0
            # Ignore hidden files and directories
            paths_to_check = (
                p
                for p in path.rglob("*")
                if not p.name.startswith(".") and not any(part.startswith(".") for part in p.parts)
            )
            for item in paths_to_check:
                if item.is_file():
                    total_files += 1
                    if total_files > _MAX_FILES_IN_MODEL_DIR:
                        raise ValueError(
                            f"Directory contains more than {_MAX_FILES_IN_MODEL_DIR} files. "
                            "This looks like a general-purpose directory rather than a model. "
                            "Please provide a path to a specific model file or model directory."
                        )

            # Otherwise, search for model files within depth limit
            def find_model_files(current_path: Path, depth: int) -> bool:
                if depth > _MAX_SEARCH_DEPTH:
                    return False
                try:
                    for item in current_path.iterdir():
                        if item.is_file() and item.suffix.lower() in _MODEL_EXTENSIONS:
                            return True
                        elif item.is_dir() and find_model_files(item, depth + 1):
                            return True
                except PermissionError:
                    pass
                return False

            if not find_model_files(path, 0):
                raise ValueError(
                    f"No model files or config files found in directory {path}. "
                    f"Expected to find model files with extensions: {', '.join(sorted(_MODEL_EXTENSIONS))} "
                    f"or config files: {', '.join(sorted(_CONFIG_FILES))}"
                )

    @staticmethod
    def matches_sort_key(m: AnyModelConfig) -> int:
        """Sort key function to prioritize model config matches in case of multiple matches."""

        # It is possible that we have multiple matches. We need to prioritize them.

        # Known cases where multiple matches can occur:
        # - SD main models can look like a LoRA when they have merged in LoRA weights. Prefer the main model.
        # - SD main models in diffusers format can look like a CLIP Embed; they have a text_encoder folder with
        #   a config.json file. Prefer the main model.

        # Given the above cases, we can prioritize the matches by type. If we find more cases, we may need a more
        # sophisticated approach.
        match m.type:
            case ModelType.Main:
                return 0
            case ModelType.LoRA:
                return 1
            case ModelType.CLIPEmbed:
                return 2
            case _:
                return 3

    @staticmethod
    def from_model_on_disk(
        mod: str | Path | ModelOnDisk,
        override_fields: dict[str, Any] | None = None,
        hash_algo: HASHING_ALGORITHMS = "blake3_single",
        allow_unknown: bool = True,
    ) -> ModelClassificationResult:
        """Classify a model on disk and return the best matching model config.

        Args:
            mod: The model on disk to classify. Can be a path (str or Path) or a ModelOnDisk instance.
            override_fields: Optional dictionary of fields to override. These fields will take precedence
                over the values extracted from the model on disk, but this cannot force a match if the
                model on disk doesn't actually match the config class.
            hash_algo: The hashing algorithm to use when computing the model hash if needed.

        Returns:
            A ModelClassificationResult containing the best matching model config (or None if no match)
            and a mapping of all attempted model config classes to either an instance of that class (if it matched)
            or an Exception (if it didn't match or an error occurred during matching).

        Raises:
            ValueError: If the provided path doesn't look like a model.
        """
        if isinstance(mod, Path | str):
            mod = ModelOnDisk(Path(mod), hash_algo)

        # Perform basic sanity checks before attempting any config matching
        # This rejects obviously non-model paths early, saving time
        ModelConfigFactory._validate_path_looks_like_model(mod.path)

        # We will always need these fields to build any model config.
        fields = ModelConfigFactory.build_common_fields(mod, override_fields)

        # Store results as a mapping of config class to either an instance of that class or an exception
        # that was raised when trying to build it.
        details: dict[str, AnyModelConfig | Exception] = {}

        # Try to build an instance of each model config class that uses the classify API.
        # Each class will either return an instance of itself or raise NotAMatch if it doesn't match.
        # Other exceptions may be raised if something unexpected happens during matching or building.
        for candidate_class in filter(lambda x: x is not Unknown_Config, Config_Base.CONFIG_CLASSES):
            candidate_name = candidate_class.__name__
            try:
                candidate_fields = fields
                # Preserve the explicit encoder choice for InvokeAI IP-Adapter probes; this field is not part of
                # the common model record changes, but re-identification can carry it from the stored config.
                if (
                    override_fields is not None
                    and "image_encoder_model_id" in override_fields
                    and "image_encoder_model_id" in candidate_class.model_fields
                ):
                    candidate_fields = {**fields, "image_encoder_model_id": override_fields["image_encoder_model_id"]}
                # Technically, from_model_on_disk returns a Config_Base, but in practice it will always be a member of
                # the AnyModelConfig union.
                candidate = candidate_class.from_model_on_disk(mod, candidate_fields)
                ModelConfigFactory._raise_for_unsupported_gguf_quantization(mod, candidate)
                details[candidate_name] = candidate  # type: ignore
            except NotAMatchError as e:
                # This means the model didn't match this config class. It's not an error, just no match.
                details[candidate_name] = ModelConfigFactory._detach_traceback(e)
            except InvalidMatchError as e:
                # This means the model *is* this config class' kind of model, but is unusable (e.g. a
                # truncated checkpoint). Recorded like any other result here; the fallback below is
                # what treats it differently from a plain no-match.
                details[candidate_name] = ModelConfigFactory._detach_traceback(e)
            except ValidationError as e:
                # This means the model matched, but we couldn't create the pydantic model instance for the config.
                # Maybe invalid overrides were provided?
                details[candidate_name] = ModelConfigFactory._detach_traceback(e)
            except Exception as e:
                # Some other unexpected error occurred. Store the exception for reporting later.
                details[candidate_name] = ModelConfigFactory._detach_traceback(e)

        # Extract just the successful matches
        matches = [r for r in details.values() if isinstance(r, Config_Base)]

        if not matches:
            if any(isinstance(r, InvalidMatchError) for r in details.values()):
                # A config class recognised the model and rejected it as unusable. That is not the same
                # as "unidentifiable": falling back to Unknown_Config here would register a file we know
                # to be broken as a normal model record, so the rejection wins over allow_unknown.
                return ModelClassificationResult(config=None, details=details)
            if not allow_unknown:
                # No matches and we are not allowed to fall back to Unknown_Config
                return ModelClassificationResult(config=None, details=details)
            else:
                # Fall back to Unknown_Config
                # This should always succeed as Unknown_Config.from_model_on_disk never raises NotAMatch
                config = Unknown_Config.from_model_on_disk(mod, fields)
                details[Unknown_Config.__name__] = config
                return ModelClassificationResult(config=config, details=details)

        matches.sort(key=ModelConfigFactory.matches_sort_key)
        config = matches[0]

        # Now do any post-processing needed for specific model types/bases/etc.
        match config.type:
            case ModelType.Main:
                # Variant, name and path all narrow the result: four architectures have
                # per-variant defaults, and ERNIE-Image-Turbo has no variant on the config at all,
                # so its name is the only signal. What each architecture recommends lives in
                # invokeai/backend/architectures/defs/.
                variant = getattr(config, "variant", None)
                config.default_settings = _identified_default_settings(
                    resolve_default_settings(config.base, variant, config.name, config.path),
                    MainModelDefaultSettings,
                    mod,
                    override_fields,
                    config=config,
                )
            case ModelType.ControlNet | ModelType.T2IAdapter | ModelType.ControlLoRa:
                config.default_settings = _identified_default_settings(
                    ControlAdapterDefaultSettings.from_model_name(config.name),
                    ControlAdapterDefaultSettings,
                    mod,
                    override_fields,
                    config=config,
                )
            case ModelType.LoRA:
                config.default_settings = _identified_default_settings(
                    LoraModelDefaultSettings(),
                    LoraModelDefaultSettings,
                    mod,
                    override_fields,
                    config=config,
                )
            case _:
                pass

        return ModelClassificationResult(config=config, details=details)


MODEL_NAME_TO_PREPROCESSOR = {
    "canny": "canny_image_processor",
    "mlsd": "mlsd_image_processor",
    "depth": "depth_anything_image_processor",
    "bae": "normalbae_image_processor",
    "normal": "normalbae_image_processor",
    "sketch": "pidi_image_processor",
    "scribble": "lineart_image_processor",
    "lineart anime": "lineart_anime_image_processor",
    "lineart_anime": "lineart_anime_image_processor",
    "lineart": "lineart_image_processor",
    "soft": "hed_image_processor",
    "softedge": "hed_image_processor",
    "hed": "hed_image_processor",
    "shuffle": "content_shuffle_image_processor",
    "pose": "dw_openpose_image_processor",
    "mediapipe": "mediapipe_face_processor",
    "pidi": "pidi_image_processor",
    "zoe": "zoe_depth_image_processor",
    "color": "color_map_image_processor",
}
