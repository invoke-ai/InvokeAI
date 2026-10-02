import re
from abc import ABC
from pathlib import Path
from typing import Any, Literal, Self

import torch
from pydantic import BaseModel, Field

from invokeai.backend.ltx2 import checkpoint_layout as ltx2_layout
from invokeai.backend.model_manager.checkpoint_prefix import COMFYUI_KEY_PREFIXES
from invokeai.backend.model_manager.configs.base import (
    Checkpoint_Config_Base,
    Config_Base,
    Diffusers_Config_Base,
    SubmodelDefinition,
)
from invokeai.backend.model_manager.configs.clip_embed import get_clip_variant_type_from_config

# Re-exported: `MainModelDefaultSettings` moved to its own module so the architecture registry
# can hold instances of it without this module — which now looks the values *up* — becoming part
# of an import cycle. Kept importable from here, where every caller already expects it.
from invokeai.backend.model_manager.configs.default_settings import (  # noqa: E402
    DEFAULTS_PRECISION as DEFAULTS_PRECISION,
)
from invokeai.backend.model_manager.configs.default_settings import (
    MainModelDefaultSettings as MainModelDefaultSettings,
)
from invokeai.backend.model_manager.configs.flux2_variant import (
    flux2_variant_from_context_dim,
    flux2_variant_from_hidden_size,
)
from invokeai.backend.model_manager.configs.identification_utils import (
    InvalidMatchError,
    NotAMatchError,
    common_config_paths,
    get_config_dict_or_raise,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
    state_dict_has_any_keys_exact,
)
from invokeai.backend.model_manager.configs.qwen3_encoder import _SDNQ_LOADABLE_QWEN_ARCHITECTURES
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    Flux2VariantType,
    FluxVariantType,
    Krea2VariantType,
    LTX2VariantType,
    MiniMaxH3VariantType,
    ModelFormat,
    ModelRepoVariant,
    ModelType,
    ModelVariantType,
    QwenImageVariantType,
    SchedulerPredictionType,
    SubModelType,
    WanVariantType,
    ZImageVariantType,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.int8_convrot import INT8_TENSORWISE_FORMAT, read_comfy_quant_markers
from invokeai.backend.quantization.sdnq.detection import is_sdnq_folder
from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor


class Main_Config_Base(ABC, BaseModel):
    type: Literal[ModelType.Main] = Field(default=ModelType.Main)
    trigger_phrases: set[str] | None = Field(
        default=None,
        description="Set of trigger phrases for this model",
    )
    default_settings: MainModelDefaultSettings | None = Field(
        default=None,
        description="Default settings for this model",
    )


def _has_bnb_nf4_keys(state_dict: dict[str | int, Any]) -> bool:
    bnb_nf4_keys = {
        "double_blocks.0.img_attn.proj.weight.quant_state.bitsandbytes__nf4",
        "model.diffusion_model.double_blocks.0.img_attn.proj.weight.quant_state.bitsandbytes__nf4",
    }
    return any(key in state_dict for key in bnb_nf4_keys)


def _has_ggml_tensors(state_dict: dict[str | int, Any]) -> bool:
    return any(isinstance(v, GGMLTensor) for v in state_dict.values())


def _has_sdnq_tensors(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains SDNQTensor instances."""
    return any(isinstance(v, SDNQTensor) for v in state_dict.values())


def _has_sdnq_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict has SDNQ-style keys (weight + scale pairs).

    SDNQ quantized models store weights with associated scale tensors.
    """
    keys = {k for k in state_dict.keys() if isinstance(k, str)}
    for key in keys:
        if key.endswith(".weight"):
            base = key[:-7]
            if f"{base}.scale" in keys:
                return True
    return False


def _has_main_keys(state_dict: dict[str | int, Any]) -> bool:
    for key in state_dict.keys():
        if isinstance(key, int):
            continue
        elif key.startswith(
            (
                "cond_stage_model.",
                "first_stage_model.",
                "model.diffusion_model.",
                # Some FLUX checkpoint files contain transformer keys prefixed with "model.diffusion_model".
                # This prefix is typically used to distinguish between multiple models bundled in a single file.
                "model.diffusion_model.double_blocks.",
            )
        ):
            return True
        elif key.startswith("double_blocks.") and "ip_adapter" not in key:
            # FLUX models in the official BFL format contain keys with the "double_blocks." prefix, but we must be
            # careful to avoid false positives on XLabs FLUX IP-Adapter models.
            return True
    return False


def _has_z_image_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains Z-Image S3-DiT transformer keys.

    This function returns True only for Z-Image main models, not LoRAs.
    LoRAs are excluded by checking for LoRA-specific weight suffixes.
    """
    # Z-Image specific keys that distinguish it from other models
    z_image_specific_keys = {
        "cap_embedder",  # Caption embedder - unique to Z-Image
        "context_refiner",  # Context refiner blocks
        "cap_pad_token",  # Caption padding token
    }

    # LoRA-specific suffixes - if present, this is a LoRA not a main model
    lora_suffixes = (
        ".lora_down.weight",
        ".lora_up.weight",
        ".lora_A.weight",
        ".lora_B.weight",
        ".dora_scale",
        ".alpha",
    )

    # First pass: check if any key has LoRA suffixes - if so, this is a LoRA not a main model
    for key in state_dict.keys():
        if isinstance(key, int):
            continue
        if key.endswith(lora_suffixes):
            return False

    # Second pass: check for Z-Image specific key parts
    for key in state_dict.keys():
        if isinstance(key, int):
            continue
        # Handle both direct keys (cap_embedder.0.weight) and
        # ComfyUI-style keys (model.diffusion_model.cap_embedder.0.weight)
        key_parts = key.split(".")
        for part in key_parts:
            if part in z_image_specific_keys:
                return True

    return False


def _get_krea2_variant_from_name(name: str) -> Krea2VariantType:
    """Guess the Krea-2 variant from a single-file/GGUF filename.

    Turbo and Raw (Base) share the identical transformer architecture, so a single-file checkpoint
    cannot be distinguished from its weights. Filenames with a "raw"/"base" token (e.g. "Krea-2-Raw",
    "krea2_base_q4") indicate the undistilled Base model; everything else defaults to the distilled
    Turbo. The user can override the variant in the model manager.
    """
    lowered = name.lower()
    # "turbo" is a strong positive signal for the distilled checkpoint and wins outright, so a Turbo file
    # whose name merely *contains* "base"/"raw" as a substring (e.g. "baseline", "database", "raw_export")
    # is not misread as Base.
    if "turbo" in lowered:
        return Krea2VariantType.Turbo
    # Otherwise match "raw"/"base" only as a whole token delimited by non-alphanumeric separators
    # ("-", "_", ".") - not as an arbitrary substring.
    tokens = re.split(r"[^a-z0-9]+", lowered)
    if "raw" in tokens or "base" in tokens:
        return Krea2VariantType.Base
    return Krea2VariantType.Turbo


def _has_krea2_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains Krea-2 (Krea2Transformer2DModel) transformer keys.

    Krea-2's single-stream MMDiT has a distinctive text-fusion stage; the ``text_fusion.``
    prefix (with ``layerwise_blocks`` / ``refiner_blocks`` / ``projector``) is unique to it.
    Returns True only for Krea-2 main models, not LoRAs.
    """
    # The text-fusion stage is unique to Krea-2. Diffusers naming uses `text_fusion`/`time_mod_proj`;
    # the native/ComfyUI GGUF conversion uses the compact `txtfusion`/`tproj` names instead.
    krea2_specific_keys = {
        "text_fusion",  # text-fusion stage (diffusers naming) - unique to Krea-2
        "txtfusion",  # text-fusion stage (native/ComfyUI GGUF naming)
        "time_mod_proj",  # timestep modulation projection (diffusers)
    }
    # Corroborating image-input signals: `img_in` (diffusers) / `first` (native), or the timestep
    # modulation projection (`tproj` native).
    krea2_corroborating_keys = {"img_in", "first", "tproj"}

    lora_suffixes = (
        ".lora_down.weight",
        ".lora_up.weight",
        ".lora_A.weight",
        ".lora_B.weight",
        ".dora_scale",
        ".alpha",
    )

    # If any key has a LoRA suffix, this is a LoRA, not a main model.
    for key in state_dict.keys():
        if isinstance(key, int):
            continue
        if key.endswith(lora_suffixes):
            return False

    has_text_fusion = False
    has_corroborator = False
    for key in state_dict.keys():
        if isinstance(key, int):
            continue
        # Handle both direct keys and ComfyUI-style (model.diffusion_model.*) keys.
        key_parts = key.split(".")
        if any(part in krea2_specific_keys for part in key_parts):
            has_text_fusion = True
        if any(part in krea2_corroborating_keys for part in key_parts):
            has_corroborator = True
    # Require the distinctive text-fusion stage; the image-input key is a corroborating signal.
    return has_text_fusion and has_corroborator


class Main_SD_Checkpoint_Config_Base(Checkpoint_Config_Base, Main_Config_Base):
    """Model config for main checkpoint models."""

    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    prediction_type: SchedulerPredictionType = Field()
    variant: ModelVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_base(mod)

        prediction_type = override_fields.pop("prediction_type", None) or cls._get_scheduler_prediction_type_or_raise(
            mod
        )

        variant = override_fields.pop("variant", None) or cls._get_variant_or_raise(mod)

        return cls(**override_fields, prediction_type=prediction_type, variant=variant)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        state_dict = mod.load_state_dict()

        key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in state_dict and state_dict[key_name].shape[-1] == 768:
            return BaseModelType.StableDiffusion1
        if key_name in state_dict and state_dict[key_name].shape[-1] == 1024:
            return BaseModelType.StableDiffusion2

        key_name = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in state_dict and state_dict[key_name].shape[-1] == 2048:
            return BaseModelType.StableDiffusionXL
        elif key_name in state_dict and state_dict[key_name].shape[-1] == 1280:
            return BaseModelType.StableDiffusionXLRefiner

        raise NotAMatchError("unable to determine base type from state dict")

    @classmethod
    def _get_scheduler_prediction_type_or_raise(cls, mod: ModelOnDisk) -> SchedulerPredictionType:
        base = cls.model_fields["base"].default

        if base is BaseModelType.StableDiffusion2:
            state_dict = mod.load_state_dict()
            key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
            if key_name in state_dict and state_dict[key_name].shape[-1] == 1024:
                if "global_step" in state_dict:
                    if state_dict["global_step"] == 220000:
                        return SchedulerPredictionType.Epsilon
                    elif state_dict["global_step"] == 110000:
                        return SchedulerPredictionType.VPrediction
            return SchedulerPredictionType.VPrediction
        else:
            return SchedulerPredictionType.Epsilon

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> ModelVariantType:
        base = cls.model_fields["base"].default

        state_dict = mod.load_state_dict()
        key_name = "model.diffusion_model.input_blocks.0.0.weight"

        if key_name not in state_dict:
            raise NotAMatchError("unable to determine model variant from state dict")

        in_channels = state_dict["model.diffusion_model.input_blocks.0.0.weight"].shape[1]

        match in_channels:
            case 4:
                return ModelVariantType.Normal
            case 5:
                # Only SD2 has a depth variant
                assert base is BaseModelType.StableDiffusion2, f"unexpected unet in_channels 5 for base '{base}'"
                return ModelVariantType.Depth
            case 9:
                return ModelVariantType.Inpaint
            case _:
                raise NotAMatchError(f"unrecognized unet in_channels {in_channels} for base '{base}'")

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatchError("state dict does not look like a main model")


class Main_Checkpoint_SD1_Config(Main_SD_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class Main_Checkpoint_SD2_Config(Main_SD_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class Main_Checkpoint_SDXL_Config(Main_SD_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class Main_Checkpoint_SDXLRefiner_Config(Main_SD_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXLRefiner] = Field(default=BaseModelType.StableDiffusionXLRefiner)


def _is_flux2_model(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict is a FLUX.2 model by examining context_embedder dimensions.

    FLUX.2 Klein uses Qwen3 encoder with larger context dimension:
    - FLUX.1: context_in_dim = 4096 (T5)
    - FLUX.2 Klein 4B: context_in_dim = 7680 (3×Qwen3-4B hidden size)
    - FLUX.2 Klein 8B: context_in_dim = 12288 (3×Qwen3-8B hidden size)

    Also checks for FLUX.2-specific 32-channel latent space (in_channels=128 after packing).
    """
    # Check context_embedder input dimension (most reliable)
    # Weight shape: [hidden_size, context_in_dim]
    for key in {"context_embedder.weight", "model.diffusion_model.context_embedder.weight"}:
        if key in state_dict:
            weight = state_dict[key]
            if hasattr(weight, "shape") and len(weight.shape) >= 2:
                context_in_dim = weight.shape[1]
                # FLUX.2 has context_in_dim > 4096 (Qwen3 vs T5)
                if context_in_dim > 4096:
                    return True

    # Also check in_channels - FLUX.2 uses 128 (32 latent channels × 4 packing)
    for key in {"img_in.weight", "model.diffusion_model.img_in.weight"}:
        if key in state_dict:
            in_channels = state_dict[key].shape[1]
            # FLUX.2 uses 128 in_channels (32 latent channels × 4)
            # FLUX.1 uses 64 in_channels (16 latent channels × 4)
            if in_channels == 128:
                return True

    return False


def _has_flux2_diffusers_transformer_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check for the bare diffusers-style FLUX.2 transformer layout.

    A FLUX.2 single-file SDNQ checkpoint can ship its keys in the bare diffusers layout
    (``transformer_blocks.*``, ``context_embedder.*``) rather than the BFL / ComfyUI layout that
    ``_has_main_keys`` recognizes. The FLUX.2 single-file loader looks for exactly these keys, so we
    must treat their presence as "looks like a main model" for identification purposes.
    """
    for key in state_dict.keys():
        if isinstance(key, str) and (key.startswith("transformer_blocks.") or key.startswith("context_embedder.")):
            return True
    return False


# Letters, not substrings: `_filename_suggests_base` used to test `"base" in name`, which fires on
# "database" and "basement". Digits stay adjacent on purpose -- "base9b" is a name, "database" is
# not a claim.
_SUGGESTS_BASE = re.compile(r"(?<![a-z])base(?![a-z])")
_SUGGESTS_DISTILLED = re.compile(r"(?<![a-z])distill")
# "undistilled" and "non-distilled" are the Base claim spelled out, so they have to be read before
# the word they contain. A lookbehind cannot do this: it has to span an optional separator.
_DENIES_DISTILLED = re.compile(r"(?<![a-z])(?:non|not|un)[-_ ]?distill")


def _filename_suggests_base(name: str) -> bool:
    """Whether a model's name says it is the Base (undistilled) variant.

    Klein 9B Base and Klein 9B have identical architectures and identical keys, so the name is the
    only thing left to read. Getting it wrong is not cosmetic: the two ship different default step
    counts (28 against 4), so a distilled model identified as Base generates at seven times the cost
    and a Base model identified as distilled generates at four steps it was never trained for.

    Read in order of how specific the claim is: a denial of distillation, then a claim of it, then
    the bare word "base". A name carrying both "distilled" and "base" is naming its ancestry rather
    than its variant, which is how the community repacks are named -- and which *source* the same
    weights arrive from decides the name: installed by path
    `Winnougan/Klein9b-Distilled-Base-INT8-Convrot` is `flux-2-klein-9b-int8-convrot` and identifies
    correctly, installed by repo id it is the repo's name and used not to.
    """
    lowered = name.lower()
    if _DENIES_DISTILLED.search(lowered):
        return True
    if _SUGGESTS_DISTILLED.search(lowered):
        return False
    return bool(_SUGGESTS_BASE.search(lowered))


def _get_flux2_variant(state_dict: dict[str | int, Any]) -> Flux2VariantType | None:
    """Determine FLUX.2 variant from state dict.

    Distinguishes between variants based on context embedding dimension:
    - Klein 4B: context_in_dim = 7680 (3 × Qwen3-4B hidden_size 2560)
    - Klein 9B: context_in_dim = 12288 (3 × Qwen3-8B hidden_size 4096)
    - Dev:      context_in_dim = 15360 (3 × Mistral Small 3.1 hidden_size 5120)

    Note: Klein 9B (distilled) and Klein 9B Base (undistilled) have identical architectures
    and cannot be distinguished from the state dict alone. This function defaults to Klein9B
    for all 9B models. Callers should use filename heuristics to detect Klein9BBase.

    Supports both BFL format (checkpoint) and diffusers format keys:
    - BFL format: txt_in.weight (context embedder)
    - Diffusers format: context_embedder.weight
    """
    # Check context_embedder to determine variant
    # Support both BFL format (txt_in.weight) and diffusers format (context_embedder.weight)
    context_keys = {
        # Diffusers format
        "context_embedder.weight",
        "model.diffusion_model.context_embedder.weight",
        # BFL format (used by checkpoint/GGUF models)
        "txt_in.weight",
        "model.diffusion_model.txt_in.weight",
    }
    for key in context_keys:
        if key in state_dict:
            weight = state_dict[key]
            # Handle GGUF quantized tensors which use tensor_shape instead of shape
            if hasattr(weight, "tensor_shape"):
                shape = weight.tensor_shape
            elif hasattr(weight, "shape"):
                shape = weight.shape
            else:
                continue
            if len(shape) >= 2:
                context_in_dim = shape[1]
                # Determine variant based on context dimension. Callers use filename
                # heuristics to upgrade Klein4B/Klein9B to their Base variants.
                variant = flux2_variant_from_context_dim(context_in_dim)
                if variant is not None:
                    return variant
                if context_in_dim > 4096:
                    # Unknown FLUX.2 variant, default to 4B
                    return Flux2VariantType.Klein4B

    # Check in_channels as backup - can only confirm it's FLUX.2, not which variant
    for key in {"img_in.weight", "model.diffusion_model.img_in.weight"}:
        if key in state_dict:
            weight = state_dict[key]
            # Handle GGUF quantized tensors
            if hasattr(weight, "tensor_shape"):
                in_channels = weight.tensor_shape[1]
            elif hasattr(weight, "shape"):
                in_channels = weight.shape[1]
            else:
                continue
            if in_channels == 128:
                # It's FLUX.2 but we can't determine which Klein variant, default to 4B
                return Flux2VariantType.Klein4B

    return None


def _get_flux_variant(state_dict: dict[str | int, Any]) -> FluxVariantType | None:
    # FLUX Model variant types are distinguished by input channels and the presence of certain keys.

    # Input channels are derived from the shape of either "img_in.weight" or "model.diffusion_model.img_in.weight".
    #
    # Known models that use the latter key:
    # - https://civitai.com/models/885098?modelVersionId=990775
    # - https://civitai.com/models/1018060?modelVersionId=1596255
    # - https://civitai.com/models/978314/ultrareal-fine-tune?modelVersionId=1413133
    #
    # Input channels for known FLUX models:
    # - Unquantized Dev and Schnell have in_channels=64
    # - BNB-NF4 Dev and Schnell have in_channels=1
    # - FLUX Fill has in_channels=384
    # - Unsure of quantized FLUX Fill models
    # - Unsure of GGUF-quantized models

    in_channels = None
    for key in {"img_in.weight", "model.diffusion_model.img_in.weight"}:
        if key in state_dict:
            in_channels = state_dict[key].shape[1]
            break

    if in_channels is None:
        # TODO(psyche): Should we have a graceful fallback here? Previously we fell back to the "normal" variant,
        # but this variant is no longer used for FLUX models. If we get here, but the model is definitely a FLUX
        # model, we should figure out a good fallback value.
        return None

    # Because FLUX Dev and Schnell models have the same in_channels, we need to check for the presence of
    # certain keys to distinguish between them.
    is_flux_dev = (
        "guidance_in.out_layer.weight" in state_dict
        or "model.diffusion_model.guidance_in.out_layer.weight" in state_dict
    )

    if is_flux_dev and in_channels == 384:
        return FluxVariantType.DevFill
    elif is_flux_dev:
        return FluxVariantType.Dev
    else:
        # Must be a Schnell model...?
        return FluxVariantType.Schnell


class Main_Checkpoint_FLUX_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for main checkpoint models."""

    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)

    variant: FluxVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_is_flux(mod)

        cls._validate_does_not_look_like_bnb_quantized(mod)

        cls._validate_does_not_look_like_gguf_quantized(mod)

        variant = override_fields.pop("variant", None) or cls._get_variant_or_raise(mod)

        return cls(**override_fields, variant=variant)

    @classmethod
    def _validate_is_flux(cls, mod: ModelOnDisk) -> None:
        state_dict = mod.load_state_dict()
        if not state_dict_has_any_keys_exact(
            state_dict,
            {
                "double_blocks.0.img_attn.norm.key_norm.scale",
                "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale",
            },
        ):
            raise NotAMatchError("state dict does not look like a FLUX checkpoint")

        # Exclude FLUX.2 models - they have their own config class
        if _is_flux2_model(state_dict):
            raise NotAMatchError("model is a FLUX.2 model, not FLUX.1")

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> FluxVariantType:
        # FLUX Model variant types are distinguished by input channels and the presence of certain keys.
        state_dict = mod.load_state_dict()
        variant = _get_flux_variant(state_dict)

        if variant is None:
            # TODO(psyche): Should we have a graceful fallback here? Previously we fell back to the "normal" variant,
            # but this variant is no longer used for FLUX models. If we get here, but the model is definitely a FLUX
            # model, we should figure out a good fallback value.
            raise NotAMatchError("unable to determine model variant from state dict")

        return variant

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatchError("state dict does not look like a main model")

    @classmethod
    def _validate_does_not_look_like_bnb_quantized(cls, mod: ModelOnDisk) -> None:
        has_bnb_nf4_keys = _has_bnb_nf4_keys(mod.load_state_dict())
        if has_bnb_nf4_keys:
            raise NotAMatchError("state dict looks like bnb quantized nf4")

    @classmethod
    def _validate_does_not_look_like_gguf_quantized(cls, mod: ModelOnDisk):
        has_ggml_tensors = _has_ggml_tensors(mod.load_state_dict())
        if has_ggml_tensors:
            raise NotAMatchError("state dict looks like GGUF quantized")


class Main_Checkpoint_Flux2_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for FLUX.2 checkpoint models (e.g. Klein)."""

    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)

    variant: Flux2VariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_is_flux2(mod)

        cls._validate_does_not_look_like_bnb_quantized(mod)

        cls._validate_does_not_look_like_gguf_quantized(mod)

        variant = override_fields.pop("variant", None) or cls._get_variant_or_raise(mod)

        return cls(**override_fields, variant=variant)

    @classmethod
    def _validate_is_flux2(cls, mod: ModelOnDisk) -> None:
        """Validate that this is a FLUX.2 model, not FLUX.1."""
        state_dict = mod.load_state_dict()
        if not _is_flux2_model(state_dict):
            raise NotAMatchError("state dict does not look like a FLUX.2 model")

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> Flux2VariantType:
        state_dict = mod.load_state_dict()
        variant = _get_flux2_variant(state_dict)

        if variant is None:
            raise NotAMatchError("unable to determine FLUX.2 model variant from state dict")

        # Base (undistilled) and distilled variants share identical architectures.
        # Use filename heuristic to detect the Base variant.
        if variant == Flux2VariantType.Klein9B and _filename_suggests_base(mod.name):
            return Flux2VariantType.Klein9BBase
        if variant == Flux2VariantType.Klein4B and _filename_suggests_base(mod.name):
            return Flux2VariantType.Klein4BBase

        return variant

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatchError("state dict does not look like a main model")

    @classmethod
    def _validate_does_not_look_like_bnb_quantized(cls, mod: ModelOnDisk) -> None:
        has_bnb_nf4_keys = _has_bnb_nf4_keys(mod.load_state_dict())
        if has_bnb_nf4_keys:
            raise NotAMatchError("state dict looks like bnb quantized nf4")

    @classmethod
    def _validate_does_not_look_like_gguf_quantized(cls, mod: ModelOnDisk):
        has_ggml_tensors = _has_ggml_tensors(mod.load_state_dict())
        if has_ggml_tensors:
            raise NotAMatchError("state dict looks like GGUF quantized")


class Main_BnBNF4_FLUX_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for main checkpoint models."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    format: Literal[ModelFormat.BnbQuantizednf4b] = Field(default=ModelFormat.BnbQuantizednf4b)

    variant: FluxVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_model_looks_like_bnb_quantized(mod)

        variant = override_fields.pop("variant", None) or cls._get_variant_or_raise(mod)

        return cls(**override_fields, variant=variant)

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> FluxVariantType:
        # FLUX Model variant types are distinguished by input channels and the presence of certain keys.
        state_dict = mod.load_state_dict()
        variant = _get_flux_variant(state_dict)

        if variant is None:
            # TODO(psyche): Should we have a graceful fallback here? Previously we fell back to the "normal" variant,
            # but this variant is no longer used for FLUX models. If we get here, but the model is definitely a FLUX
            # model, we should figure out a good fallback value.
            raise NotAMatchError("unable to determine model variant from state dict")

        return variant

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatchError("state dict does not look like a main model")

    @classmethod
    def _validate_model_looks_like_bnb_quantized(cls, mod: ModelOnDisk) -> None:
        has_bnb_nf4_keys = _has_bnb_nf4_keys(mod.load_state_dict())
        if not has_bnb_nf4_keys:
            raise NotAMatchError("state dict does not look like bnb quantized nf4")


class Main_GGUF_FLUX_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for main checkpoint models."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)

    variant: FluxVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_looks_like_gguf_quantized(mod)

        cls._validate_is_not_flux2(mod)

        variant = override_fields.pop("variant", None) or cls._get_variant_or_raise(mod)

        return cls(**override_fields, variant=variant)

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> FluxVariantType:
        # FLUX Model variant types are distinguished by input channels and the presence of certain keys.
        state_dict = mod.load_state_dict()
        variant = _get_flux_variant(state_dict)

        if variant is None:
            # TODO(psyche): Should we have a graceful fallback here? Previously we fell back to the "normal" variant,
            # but this variant is no longer used for FLUX models. If we get here, but the model is definitely a FLUX
            # model, we should figure out a good fallback value.
            raise NotAMatchError("unable to determine model variant from state dict")

        return variant

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatchError("state dict does not look like a main model")

    @classmethod
    def _validate_looks_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        has_ggml_tensors = _has_ggml_tensors(mod.load_state_dict())
        if not has_ggml_tensors:
            raise NotAMatchError("state dict does not look like GGUF quantized")

    @classmethod
    def _validate_is_not_flux2(cls, mod: ModelOnDisk) -> None:
        """Validate that this is NOT a FLUX.2 model."""
        state_dict = mod.load_state_dict()
        if _is_flux2_model(state_dict):
            raise NotAMatchError("model is a FLUX.2 model, not FLUX.1")


class Main_GGUF_Flux2_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for GGUF-quantized FLUX.2 checkpoint models (e.g. Klein)."""

    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)

    variant: Flux2VariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_looks_like_gguf_quantized(mod)

        cls._validate_is_flux2(mod)

        variant = override_fields.pop("variant", None) or cls._get_variant_or_raise(mod)

        return cls(**override_fields, variant=variant)

    @classmethod
    def _validate_is_flux2(cls, mod: ModelOnDisk) -> None:
        """Validate that this is a FLUX.2 model, not FLUX.1."""
        state_dict = mod.load_state_dict()
        if not _is_flux2_model(state_dict):
            raise NotAMatchError("state dict does not look like a FLUX.2 model")

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> Flux2VariantType:
        state_dict = mod.load_state_dict()
        variant = _get_flux2_variant(state_dict)

        if variant is None:
            raise NotAMatchError("unable to determine FLUX.2 model variant from state dict")

        # Base (undistilled) and distilled variants share identical architectures.
        # Use filename heuristic to detect the Base variant.
        if variant == Flux2VariantType.Klein9B and _filename_suggests_base(mod.name):
            return Flux2VariantType.Klein9BBase
        if variant == Flux2VariantType.Klein4B and _filename_suggests_base(mod.name):
            return Flux2VariantType.Klein4BBase

        return variant

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatchError("state dict does not look like a main model")

    @classmethod
    def _validate_looks_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        has_ggml_tensors = _has_ggml_tensors(mod.load_state_dict())
        if not has_ggml_tensors:
            raise NotAMatchError("state dict does not look like GGUF quantized")


class Main_Diffusers_FLUX_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    """Model config for FLUX.1 models in diffusers format."""

    base: Literal[BaseModelType.Flux] = Field(BaseModelType.Flux)
    variant: FluxVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # Check for FLUX-specific pipeline or transformer class names
        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "FluxPipeline",
                "FluxFillPipeline",
                "FluxTransformer2DModel",
            },
        )

        # Reject SDNQ-quantized pipelines so Main_SDNQ_Diffusers_FLUX_Config matches instead.
        if (mod.path / "transformer").is_dir() and _is_sdnq_folder(mod.path / "transformer"):
            raise NotAMatchError("transformer is SDNQ-quantized; use Main_SDNQ_Diffusers_FLUX_Config")

        variant = override_fields.pop("variant", None) or cls._get_variant_or_raise(mod)

        repo_variant = override_fields.pop("repo_variant", None) or cls._get_repo_variant_or_raise(mod)

        return cls(
            **override_fields,
            variant=variant,
            repo_variant=repo_variant,
        )

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> FluxVariantType:
        """Determine the FLUX variant from the transformer config.

        FLUX variants are distinguished by:
        - in_channels: 64 for Dev/Schnell, 384 for DevFill
        - guidance_embeds: True for Dev, False for Schnell
        """
        transformer_config = get_config_dict_or_raise(mod.path / "transformer" / "config.json")

        in_channels = transformer_config.get("in_channels", 64)
        guidance_embeds = transformer_config.get("guidance_embeds", False)

        # DevFill has 384 input channels
        if in_channels == 384:
            return FluxVariantType.DevFill

        # Dev has guidance_embeds=True, Schnell has guidance_embeds=False
        if guidance_embeds:
            return FluxVariantType.Dev
        else:
            return FluxVariantType.Schnell


class Main_Diffusers_Flux2_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    """Model config for FLUX.2 models in diffusers format (FLUX.2 Klein and FLUX.2 [dev])."""

    base: Literal[BaseModelType.Flux2] = Field(BaseModelType.Flux2)
    variant: Flux2VariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # A FLUX.2 *main* model is a full diffusers pipeline: a `model_index.json` at
        # the root, or at least the transformer packaged as a `transformer/` subfolder.
        # A loose transformer-only checkout — just the contents of `transformer/`, with
        # a root `config.json` whose `_class_name` is `Flux2Transformer2DModel` — is NOT
        # a usable main model: the loader unconditionally appends `vae/` / `text_encoder/`
        # subfolders that don't exist and fails with an OSError mid-queue. Reject that
        # layout here so it falls through to a non-main classification instead of
        # registering as a broken pipeline. (The standalone `transformer/` still matches
        # via the pipeline layout below when it ships inside a full folder.)
        if not (mod.path / "model_index.json").exists() and not (mod.path / "transformer").exists():
            raise NotAMatchError(
                "directory is not a full FLUX.2 pipeline (no model_index.json and no transformer/ subfolder); "
                "a loose transformer-only checkout cannot be used as a FLUX.2 main model"
            )

        # Check for FLUX.2-specific pipeline class names
        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "Flux2KleinPipeline",
                "Flux2Pipeline",
                "Flux2Transformer2DModel",
            },
        )

        # Reject SDNQ-quantized pipelines so the SDNQ-specific config matches them instead.
        # Without this both configs accept the same folder and identification can latch onto
        # the wrong one (the plain diffusers loader would then mis-read packed uint8 weights
        # as bf16 and crash with size-mismatch errors at first inference).
        if (mod.path / "transformer").is_dir() and _is_sdnq_folder(mod.path / "transformer"):
            raise NotAMatchError("transformer is SDNQ-quantized; use Main_SDNQ_Diffusers_Flux2_Config")

        variant = override_fields.pop("variant", None) or cls._get_variant_or_raise(mod)

        repo_variant = override_fields.pop("repo_variant", None) or cls._get_repo_variant_or_raise(mod)

        return cls(
            **override_fields,
            variant=variant,
            repo_variant=repo_variant,
        )

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> Flux2VariantType:
        """Determine the FLUX.2 variant from the transformer config.

        FLUX.2 variants are distinguished by joint_attention_dim (= 3 × text encoder hidden_size):
        - Klein 4B/4B Base: 7680 (3 × Qwen3-4B 2560)
        - Klein 9B/9B Base: 12288 (3 × Qwen3-8B 4096)
        - Dev:              15360 (3 × Mistral Small 3.1 5120)

        Klein distilled and Base variants share identical architectures; the Base variant
        is detected by a filename heuristic.
        """
        # Try transformer/config.json first (full pipeline), fall back to root config.json
        # (loose transformer-only checkouts).
        transformer_config_path = mod.path / "transformer" / "config.json"
        root_config_path = mod.path / "config.json"
        if transformer_config_path.exists():
            transformer_config = get_config_dict_or_raise(transformer_config_path)
        else:
            transformer_config = get_config_dict_or_raise(root_config_path)

        joint_attention_dim = transformer_config.get("joint_attention_dim", 4096)

        # Determine variant based on joint_attention_dim (= context_in_dim).
        variant = flux2_variant_from_context_dim(joint_attention_dim)
        if variant is None:
            # Unknown or FLUX.1-sized joint_attention_dim — default to Klein 4B.
            return Flux2VariantType.Klein4B
        # Klein 4B/9B share their architecture with the corresponding Base variant; use the
        # filename heuristic to distinguish. Dev has no Base variant.
        if variant is Flux2VariantType.Klein9B and _filename_suggests_base(mod.name):
            return Flux2VariantType.Klein9BBase
        if variant is Flux2VariantType.Klein4B and _filename_suggests_base(mod.name):
            return Flux2VariantType.Klein4BBase
        return variant


class Main_SD_Diffusers_Config_Base(Diffusers_Config_Base, Main_Config_Base):
    prediction_type: SchedulerPredictionType = Field()
    variant: ModelVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        raise_for_class_name(
            common_config_paths(mod.path),
            {
                # SD 1.x and 2.x
                "StableDiffusionPipeline",
                "StableDiffusionInpaintPipeline",
                # SDXL
                "StableDiffusionXLPipeline",
                "StableDiffusionXLInpaintPipeline",
                # SDXL Refiner
                "StableDiffusionXLImg2ImgPipeline",
                # TODO(psyche): Do we actually support LCM models? I don't see using this class anywhere in the codebase.
                "LatentConsistencyModelPipeline",
            },
        )

        cls._validate_base(mod)

        variant = override_fields.pop("variant", None) or cls._get_variant_or_raise(mod)

        prediction_type = override_fields.pop("prediction_type", None) or cls._get_scheduler_prediction_type_or_raise(
            mod
        )

        repo_variant = override_fields.pop("repo_variant", None) or cls._get_repo_variant_or_raise(mod)

        return cls(
            **override_fields,
            variant=variant,
            prediction_type=prediction_type,
            repo_variant=repo_variant,
        )

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        # Handle pipelines with a UNet (i.e SD 1.x, SD2.x, SDXL).
        unet_conf = get_config_dict_or_raise(mod.path / "unet" / "config.json")
        cross_attention_dim = unet_conf.get("cross_attention_dim")
        match cross_attention_dim:
            case 768:
                return BaseModelType.StableDiffusion1
            case 1024:
                return BaseModelType.StableDiffusion2
            case 1280:
                return BaseModelType.StableDiffusionXLRefiner
            case 2048:
                return BaseModelType.StableDiffusionXL
            case _:
                raise NotAMatchError(f"unrecognized cross_attention_dim {cross_attention_dim}")

    @classmethod
    def _get_scheduler_prediction_type_or_raise(cls, mod: ModelOnDisk) -> SchedulerPredictionType:
        scheduler_conf = get_config_dict_or_raise(mod.path / "scheduler" / "scheduler_config.json")

        # TODO(psyche): Is epsilon the right default or should we raise if it's not present?
        prediction_type = scheduler_conf.get("prediction_type", "epsilon")

        match prediction_type:
            case "v_prediction":
                return SchedulerPredictionType.VPrediction
            case "epsilon":
                return SchedulerPredictionType.Epsilon
            case _:
                raise NotAMatchError(f"unrecognized scheduler prediction_type {prediction_type}")

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> ModelVariantType:
        base = cls.model_fields["base"].default
        unet_config = get_config_dict_or_raise(mod.path / "unet" / "config.json")
        in_channels = unet_config.get("in_channels")

        match in_channels:
            case 4:
                return ModelVariantType.Normal
            case 5:
                # Only SD2 has a depth variant
                assert base is BaseModelType.StableDiffusion2, f"unexpected unet in_channels 5 for base '{base}'"
                return ModelVariantType.Depth
            case 9:
                return ModelVariantType.Inpaint
            case _:
                raise NotAMatchError(f"unrecognized unet in_channels {in_channels} for base '{base}'")


class Main_Diffusers_SD1_Config(Main_SD_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(BaseModelType.StableDiffusion1)


class Main_Diffusers_SD2_Config(Main_SD_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(BaseModelType.StableDiffusion2)


class Main_Diffusers_SDXL_Config(Main_SD_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(BaseModelType.StableDiffusionXL)


class Main_Diffusers_SDXLRefiner_Config(Main_SD_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXLRefiner] = Field(BaseModelType.StableDiffusionXLRefiner)


class Main_Diffusers_SD3_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion3] = Field(BaseModelType.StableDiffusion3)
    submodels: dict[SubModelType, SubmodelDefinition] | None = Field(
        description="Loadable submodels in this model",
        default=None,
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # This check implies the base type - no further validation needed.
        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "StableDiffusion3Pipeline",
                "SD3Transformer2DModel",
            },
        )

        submodels = override_fields.pop("submodels", None) or cls._get_submodels_or_raise(mod)

        repo_variant = override_fields.pop("repo_variant", None) or cls._get_repo_variant_or_raise(mod)

        return cls(
            **override_fields,
            submodels=submodels,
            repo_variant=repo_variant,
        )

    @classmethod
    def _get_submodels_or_raise(cls, mod: ModelOnDisk) -> dict[SubModelType, SubmodelDefinition]:
        # Example: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/model_index.json
        config = get_config_dict_or_raise(common_config_paths(mod.path))

        submodels: dict[SubModelType, SubmodelDefinition] = {}

        for key, value in config.items():
            # Anything that starts with an underscore is top-level metadata, not a submodel
            if key.startswith("_") or not (isinstance(value, list) and len(value) == 2):
                continue
            # The key is something like "transformer" and is a submodel - it will be in a dir of the same name.
            # The value value is something like ["diffusers", "SD3Transformer2DModel"]
            _library_name, class_name = value

            match class_name:
                case "CLIPTextModelWithProjection":
                    model_type = ModelType.CLIPEmbed
                    path_or_prefix = (mod.path / key).resolve().as_posix()

                    # We need to read the config to determine the variant of the CLIP model.
                    clip_embed_config = get_config_dict_or_raise(
                        {
                            mod.path / key / "config.json",
                            mod.path / key / "model_index.json",
                        }
                    )
                    variant = get_clip_variant_type_from_config(clip_embed_config)
                    submodels[SubModelType(key)] = SubmodelDefinition(
                        path_or_prefix=path_or_prefix,
                        model_type=model_type,
                        variant=variant,
                    )
                case "SD3Transformer2DModel":
                    model_type = ModelType.Main
                    path_or_prefix = (mod.path / key).resolve().as_posix()
                    variant = None
                    submodels[SubModelType(key)] = SubmodelDefinition(
                        path_or_prefix=path_or_prefix,
                        model_type=model_type,
                        variant=variant,
                    )
                case _:
                    pass

        return submodels


class Main_Diffusers_CogView4_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    base: Literal[BaseModelType.CogView4] = Field(BaseModelType.CogView4)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # This check implies the base type - no further validation needed.
        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "CogView4Pipeline",
            },
        )

        repo_variant = override_fields.pop("repo_variant", None) or cls._get_repo_variant_or_raise(mod)

        return cls(
            **override_fields,
            repo_variant=repo_variant,
        )


def _has_anima_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains Anima model keys.

    Anima models are identified by the presence of `llm_adapter` keys
    (unique to Anima - the LLM Adapter that bridges Qwen3 text encoder to the Cosmos DiT)
    alongside Cosmos Predict2 DiT keys (blocks, t_embedder, x_embedder, final_layer).

    The checkpoint keys may have a `net.` prefix (e.g. `net.llm_adapter.`, `net.blocks.`)
    or a `model.diffusion_model.` prefix (ComfyUI bundled checkpoint format).
    """
    has_llm_adapter = False
    has_cosmos_dit = False

    # LLM adapter key prefixes — support bare, `net.`, and `model.diffusion_model.` prefixes
    llm_adapter_prefixes = (
        "llm_adapter.",
        "net.llm_adapter.",
        "model.diffusion_model.llm_adapter.",
    )

    # Cosmos DiT key prefixes — support bare, `net.`, and `model.diffusion_model.` prefixes
    cosmos_prefixes = (
        "blocks.",
        "t_embedder.",
        "x_embedder.",
        "final_layer.",
        "net.blocks.",
        "net.t_embedder.",
        "net.x_embedder.",
        "net.final_layer.",
        "model.diffusion_model.blocks.",
        "model.diffusion_model.t_embedder.",
        "model.diffusion_model.x_embedder.",
        "model.diffusion_model.final_layer.",
    )

    for key in state_dict.keys():
        if isinstance(key, int):
            continue
        if any(key.startswith(p) for p in llm_adapter_prefixes):
            has_llm_adapter = True
        if any(key.startswith(p) for p in cosmos_prefixes):
            has_cosmos_dit = True
        if has_llm_adapter and has_cosmos_dit:
            return True

    return False


class Main_Diffusers_ZImage_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    """Model config for Z-Image diffusers models (Z-Image-Turbo, Z-Image-Base)."""

    base: Literal[BaseModelType.ZImage] = Field(BaseModelType.ZImage)
    variant: ZImageVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # This check implies the base type - no further validation needed.
        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "ZImagePipeline",
            },
        )

        # Reject SDNQ-quantized pipelines so Main_SDNQ_Diffusers_ZImage_Config matches them instead.
        # Without this both configs accept the same ZImagePipeline folder and identification can
        # latch onto the plain diffusers one (which would then mis-read packed uint8 weights as bf16
        # and crash at inference). It also breaks the self-contained SDNQ path, since a pipeline
        # mis-identified as plain diffusers would force the user to select separate VAE/Qwen3 sources.
        if (mod.path / "transformer").is_dir() and _is_sdnq_folder(mod.path / "transformer"):
            raise NotAMatchError("transformer is SDNQ-quantized; use Main_SDNQ_Diffusers_ZImage_Config")

        variant = override_fields.pop("variant", None) or cls._get_variant_or_raise(mod)

        repo_variant = override_fields.pop("repo_variant", None) or cls._get_repo_variant_or_raise(mod)

        return cls(
            **override_fields,
            variant=variant,
            repo_variant=repo_variant,
        )

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> ZImageVariantType:
        """Determine Z-Image variant from the scheduler config.

        Z-Image variants are distinguished by the scheduler shift value:
        - Turbo (distilled): shift = 3.0
        - Base (undistilled): shift = 6.0
        """
        scheduler_config = get_config_dict_or_raise(mod.path / "scheduler" / "scheduler_config.json")

        shift = scheduler_config.get("shift", 3.0)

        # ZBase (undistilled) uses shift = 6.0, Turbo uses shift = 3.0
        if shift >= 5.0:
            return ZImageVariantType.ZBase
        else:
            return ZImageVariantType.Turbo


class Main_Checkpoint_ZImage_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for Z-Image single-file checkpoint models (safetensors, etc)."""

    base: Literal[BaseModelType.ZImage] = Field(default=BaseModelType.ZImage)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    variant: ZImageVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_z_image_model(mod)

        cls._validate_does_not_look_like_gguf_quantized(mod)

        variant = override_fields.pop("variant", None) or ZImageVariantType.Turbo

        return cls(**override_fields, variant=variant)

    @classmethod
    def _validate_looks_like_z_image_model(cls, mod: ModelOnDisk) -> None:
        has_z_image_keys = _has_z_image_keys(mod.load_state_dict())
        if not has_z_image_keys:
            raise NotAMatchError("state dict does not look like a Z-Image model")

    @classmethod
    def _validate_does_not_look_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        has_ggml_tensors = _has_ggml_tensors(mod.load_state_dict())
        if has_ggml_tensors:
            raise NotAMatchError("state dict looks like GGUF quantized")


class Main_GGUF_ZImage_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for GGUF-quantized Z-Image transformer models."""

    base: Literal[BaseModelType.ZImage] = Field(default=BaseModelType.ZImage)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)
    variant: ZImageVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_z_image_model(mod)

        cls._validate_looks_like_gguf_quantized(mod)

        variant = override_fields.pop("variant", None) or ZImageVariantType.Turbo

        return cls(**override_fields, variant=variant)

    @classmethod
    def _validate_looks_like_z_image_model(cls, mod: ModelOnDisk) -> None:
        has_z_image_keys = _has_z_image_keys(mod.load_state_dict())
        if not has_z_image_keys:
            raise NotAMatchError("state dict does not look like a Z-Image model")

    @classmethod
    def _validate_looks_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        has_ggml_tensors = _has_ggml_tensors(mod.load_state_dict())
        if not has_ggml_tensors:
            raise NotAMatchError("state dict does not look like GGUF quantized")


class Main_Diffusers_Ideogram4_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    """Model config for Ideogram 4 diffusers models (nf4 / fp8 quantized).

    The on-disk layout is a diffusers pipeline folder bundling two transformers
    (transformer/ + unconditional_transformer/), a Qwen3-VL text_encoder/ + tokenizer/,
    and a FLUX.2-style vae/. Quantization (nf4 vs fp8) lives inside the component folders
    and is detected by the loader, not here.
    """

    base: Literal[BaseModelType.Ideogram4] = Field(BaseModelType.Ideogram4)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # The Ideogram4Pipeline class name in model_index.json uniquely identifies this base.
        raise_for_class_name(
            common_config_paths(mod.path),
            {"Ideogram4Pipeline"},
        )

        repo_variant = override_fields.pop("repo_variant", None) or cls._get_repo_variant_or_raise(mod)

        return cls(
            **override_fields,
            repo_variant=repo_variant,
        )


# Comfy-Org ships the two Ideogram 4 branches as separate single files whose tensors are
# key-for-key and shape-for-shape identical. Which branch a file holds is recorded only in its
# safetensors metadata, and the pair is not interchangeable: swapping them turns the guided
# branch into the unguided one and vice versa, which produces images with no visible error.
_IDEOGRAM4_METADATA_KEY = "model_type"
_IDEOGRAM4_BRANCH_BY_METADATA = {
    "ideogram4_cond": "conditional",
    "ideogram4_uncond": "unconditional",
}

Ideogram4Branch = Literal["conditional", "unconditional"]


def _has_ideogram4_keys(state_dict: dict[str | int, Any]) -> bool:
    """Fingerprint for Ideogram 4 single-file transformers.

    ``embed_image_indicator`` is the token-type embedding for Ideogram's packed ``[text][image]``
    sequence and no other architecture probed here carries it; the two projections pin the layer
    layout the loader builds (``input_proj`` is the 128-channel patch input, ``adaln_proj`` the
    512-wide AdaLN conditioning trunk).
    """
    return all(
        key in state_dict
        for key in (
            "embed_image_indicator.weight",
            "input_proj.weight",
            "adaln_proj.weight",
            "final_layer.linear.weight",
        )
    )


def _ideogram4_branch_from_filename(filename: str) -> Ideogram4Branch:
    """The branch for a file that declares none.

    Not only a fallback for a stripped re-upload: the released `int8_convrot` pair carries no
    `model_type` at all, so this is the sole path for that build.

    Only the unconditional file is named for its branch, so anything else is read as the conditional
    branch — the same default direction as the released naming. Both spellings are accepted: the
    filename says "unconditional" and the metadata this stands in for says "uncond", and a repack
    that drops the metadata is exactly the kind of tool that would name the file after it.
    """
    lowered = filename.lower()
    return "unconditional" if "unconditional" in lowered or "uncond" in lowered else "conditional"


class Main_Checkpoint_Ideogram4_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for Ideogram 4 single-file transformer checkpoints (safetensors).

    One file holds ONE of the two dual-branch transformers. The Qwen3-VL 8B text encoder and the
    FLUX.2 VAE are separate models, selected on the loader node; the diffusers pipeline folder
    (``Main_Diffusers_Ideogram4_Config``) is the variant that bundles everything.

    Quantization: plain bf16/fp16, ComfyUI "scaled fp8" and ComfyUI ``int8_tensorwise``(+convrot)
    all load. The ``nvfp4`` repack of the same files is recognised and rejected with
    `InvalidMatchError`, which keeps it out of the database entirely -- `NotAMatchError` would let
    it fall through to `Unknown_Config` and register a 5 GiB file that nothing can ever load.
    """

    base: Literal[BaseModelType.Ideogram4] = Field(default=BaseModelType.Ideogram4)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    branch: Ideogram4Branch = Field(
        description="Which of Ideogram 4's two transformer branches this file holds. Read from the "
        "file's `model_type` metadata where it has any, and from the filename otherwise — the "
        "released int8 build records no metadata at all, so keep those files under their published "
        "names. Rename and re-install to correct it."
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()
        if not _has_ideogram4_keys(state_dict):
            raise NotAMatchError("state dict does not look like an Ideogram 4 transformer")

        if _has_ggml_tensors(state_dict):
            raise NotAMatchError("GGUF-quantized Ideogram 4 checkpoints are not supported yet")

        cls._raise_for_unsupported_quantization(mod, state_dict)

        branch = override_fields.pop("branch", None) or cls._branch_or_raise(mod)

        return cls(**override_fields, branch=branch)

    @classmethod
    def _branch_or_raise(cls, mod: ModelOnDisk) -> Ideogram4Branch:
        """The branch, from the file's own declaration where it has one.

        A declaration this does not recognise is refused rather than ignored. Falling back to the
        filename there would quietly classify a future release (an `ideogram4_5_cond`, an edit
        build) as one of *these* two branches, and the loader node trusts the recorded branch
        precisely because it came from the file — which is how a wrong model would end up guiding
        against a right one with nothing in the log.
        """
        declared = mod.metadata().get(_IDEOGRAM4_METADATA_KEY)
        if not declared:
            return _ideogram4_branch_from_filename(mod.path.name)
        branch = _IDEOGRAM4_BRANCH_BY_METADATA.get(declared)
        if branch is None:
            raise InvalidMatchError(
                f"this file declares model_type '{declared}', which is not one of Ideogram 4's two "
                f"transformer branches ({', '.join(sorted(_IDEOGRAM4_BRANCH_BY_METADATA))}). It is most "
                "likely a newer or different Ideogram model that this version cannot run."
            )
        return branch

    @classmethod
    def _raise_for_unsupported_quantization(cls, mod: ModelOnDisk, state_dict: dict[str | int, Any]) -> None:
        """Refuse the quantization schemes this loader cannot build.

        `InvalidMatchError`, not `NotAMatchError`: the file *is* an Ideogram 4 transformer, so the
        right outcome is a refusal the installer shows, not a fall-through to `Unknown_Config` that
        registers it as a model nothing can load.

        Two of them. nvfp4 packs two codes per byte, so its uint8 weights are indistinguishable from
        the `comfy_quant` markers every repack carries -- including the two supported ones; the
        per-tensor `weight_scale_2` is what only nvfp4 writes.

        And int8 weights *without* a readable `int8_tensorwise` marker: the loader refuses those
        (`reject_unmarked_int8_weights`), because a rotated weight loaded as if it were not one
        generates noise. Refusing them here too is what keeps that refusal at install time -- a
        torchao or `int8_dynamic` repack of this architecture would otherwise register as a 9 GiB
        model, pull in its three starter dependencies, and fail at the first render.

        The markers come from the file's header rather than from `state_dict`: identification loads
        tensors on the meta device, so it has every dtype and shape but no bytes to parse. That read
        is a header parse plus one seek per marker, and it only happens for a file that has int8
        weights to explain in the first place.
        """
        if any(isinstance(key, str) and key.endswith(".weight_scale_2") for key in state_dict):
            raise InvalidMatchError(
                "this is an nvfp4-quantized Ideogram 4 transformer, which is not supported yet. "
                "Install the fp8_scaled or int8_convrot build instead."
            )

        int8_weights = sorted(
            key
            for key, value in state_dict.items()
            if isinstance(key, str) and key.endswith(".weight") and getattr(value, "dtype", None) is torch.int8
        )
        if not int8_weights:
            return

        try:
            markers = read_comfy_quant_markers(mod.path)
        except Exception:
            # Not readable safetensors, so there are no markers to find and nothing explains the
            # int8 weights. The refusal below is the right answer for that file too.
            markers = {}
        unmarked = [
            key
            for key in int8_weights
            if markers.get(key[: -len(".weight")], {}).get("format") != INT8_TENSORWISE_FORMAT
        ]
        if unmarked:
            raise InvalidMatchError(
                f"{len(unmarked)} int8 weight(s) in this Ideogram 4 transformer carry no readable "
                f"'{INT8_TENSORWISE_FORMAT}' marker (e.g. '{unmarked[0]}'), so the quantization scheme "
                "cannot be identified. Only Comfy-Org's int8_convrot build is supported."
            )


class Main_Diffusers_Krea2_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    """Model config for Krea-2 diffusers models (Krea-2-Turbo)."""

    base: Literal[BaseModelType.Krea2] = Field(BaseModelType.Krea2)
    variant: Krea2VariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # This check implies the base type - no further validation needed.
        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "Krea2Pipeline",
            },
        )

        variant = override_fields.pop("variant", None) or cls._get_variant(mod)

        repo_variant = override_fields.pop("repo_variant", None) or cls._get_repo_variant_or_raise(mod)

        return cls(
            **override_fields,
            variant=variant,
            repo_variant=repo_variant,
        )

    @classmethod
    def _get_variant(cls, mod: ModelOnDisk) -> Krea2VariantType:
        """Determine the Krea-2 variant from the pipeline-level ``is_distilled`` flag.

        Krea-2-Turbo sets ``is_distilled=true`` in model_index.json (distilled, 8 steps, CFG off);
        Krea-2-Raw sets ``is_distilled=false`` (undistilled Base, more steps, CFG on). The transformer
        architectures are identical, so this flag is the only reliable discriminator.
        """
        # model_index.json was already validated by the class-name check in from_model_on_disk, so a
        # read/parse failure here is a genuine identification error and is allowed to propagate rather
        # than being silently registered as Turbo (which would give a Raw model the wrong defaults).
        config = get_config_dict_or_raise(mod.path / "model_index.json")
        if config.get("is_distilled", False) is False:
            return Krea2VariantType.Base
        return Krea2VariantType.Turbo


class Main_Diffusers_MiniMaxH3_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    """Model config for MiniMax H3 (Hailuo 3.0) diffusers-format models."""

    base: Literal[BaseModelType.MiniMaxH3] = Field(BaseModelType.MiniMaxH3)
    variant: MiniMaxH3VariantType = Field()
    components_only: bool = Field(
        default=False,
        description="Whether the folder holds only the shared components (tokenizer, processor, VAEs) "
        "without transformer weights - a slim install whose transformer and text encoder must be "
        "supplied as single-file overrides at generation time.",
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # H3 ships as a Modular Diffusers pipeline: the root config is modular_model_index.json, not
        # model_index.json, and the class name implies the base type. The HF repo's FL2VA/ and
        # Ref2VA/ subtrees are the original remote-code checkpoints and declare "MiniMaxH3Pipeline"
        # instead - deliberately not matched, since their custom Python cannot be run here.
        raise_for_class_name(
            mod.path / "modular_model_index.json",
            {"MiniMaxH3ModularPipeline"},
        )

        # The jointly-denoised audio track is what distinguishes H3 from every other video family.
        # Require the audio VAE so a partial download fails identification rather than failing
        # mid-generation.
        raise_for_class_name(
            mod.path / "audio_vae" / "config.json",
            {"AutoencoderKLMiniMaxH3Audio"},
        )

        # An override may arrive as the raw string; normalize so the folder lookup below compares enums.
        variant = MiniMaxH3VariantType(override_fields.pop("variant", None) or cls._get_variant(mod))

        repo_variant = override_fields.pop("repo_variant", None) or cls._get_repo_variant_or_raise(mod)

        # A slim ("components-only") install carries the transformer's config.json for variant
        # identification but no weight shards - the transformer and text encoder come from
        # single-file installs selected in the model loader instead. Record that here so the UI
        # can require those selections up front rather than failing mid-generation.
        components_only = override_fields.pop("components_only", None)
        if components_only is None:
            if variant is MiniMaxH3VariantType.REF2VA:
                # Ref2VA folder weights are not folder-loadable in this version (`SubModelType`
                # has no `transformer_ref` member), so a Ref2VA install always needs the
                # single-file overrides - weight shards present or not. Marking it
                # components-only makes the UI require them up front instead of failing
                # minutes into a run.
                components_only = True
            else:
                transformer_dir = mod.path / "transformer"
                components_only = not any(
                    any(transformer_dir.glob(pattern))
                    for pattern in ("*.safetensors", "*.bin", "*.pth", "*.pt", "*.ckpt")
                )

        return cls(
            **override_fields,
            variant=variant,
            repo_variant=repo_variant,
            components_only=components_only,
        )

    @classmethod
    def _get_variant(cls, mod: ModelOnDisk) -> MiniMaxH3VariantType:
        """Determine the H3 variant from which task transformer is present.

        H3's task checkpoints share every component except the transformer folder: ``transformer``
        (FL2VA: text / first/last-frame to audio-video) vs ``transformer_ref`` (Ref2VA: multi-
        reference). A folder holding both (the full official repo) identifies as FL2VA - that is
        the folder-loadable variant. A ``transformer_ref``-only folder identifies as REF2VA, but
        note its weights are NOT folder-loadable in this version (``SubModelType`` has no
        ``transformer_ref`` member); the supported Ref2VA generation path is a components install
        plus a single-file transformer override selected in the model loader.
        """
        transformer_config = mod.path / "transformer" / "config.json"
        if transformer_config.exists():
            raise_for_class_name(transformer_config, {"MiniMaxH3Transformer3DModel"})
            return MiniMaxH3VariantType.FL2VA
        ref_transformer_config = mod.path / "transformer_ref" / "config.json"
        if ref_transformer_config.exists():
            raise_for_class_name(ref_transformer_config, {"MiniMaxH3Transformer3DModel"})
            return MiniMaxH3VariantType.REF2VA
        raise NotAMatchError("no transformer folder (`transformer/` or `transformer_ref/`)")


def _has_minimax_h3_keys(state_dict: dict[str | int, Any]) -> bool:
    """Fingerprint for MiniMax H3 single-file transformer checkpoints (remote-code key layout,
    as shipped by MiniMax's release and the Comfy-Org repacks, bf16 or int8 alike).

    The joint audio+video patch projections are unique to H3 across every family probed here
    (Wan keys on `patch_embedding`, FLUX on `double_blocks`, Z-Image on `cap_embedder`, ...);
    the fused-qkv block key pins the remote-code layout the loader's converter expects.
    """
    return all(
        k in state_dict for k in ("audio_patch_proj.weight", "video_patch_proj.weight", "blocks.0.attn.qkv_proj.weight")
    )


class Main_Checkpoint_MiniMaxH3_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for MiniMax H3 single-file transformer checkpoints (safetensors).

    Covers MiniMax's single-file FL2VA transformer repacks (Comfy-Org and mirrors): bf16 or
    Comfy ``int8_tensorwise``(+convrot) quantized, full or AdaLN-pruned ("adaln curves"). The
    file holds ONLY the transformer - the text encoder, VAEs, tokenizer and processor must come
    from an installed H3 diffusers-layout folder.

    The FL2VA and Ref2VA task transformers are key-for-key (and, except the non-pruned
    int8_convrot repacks, byte-size) indistinguishable, so the FILENAME is the variant
    classifier and ``variant`` is the user-correctable override for renamed files (e.g.
    re-uploads). A misclassified variant loads and runs but produces degraded output - FL2VA
    expects no reference conditioning rows, Ref2VA expects them - which is why the override
    exists rather than any attempt at content sniffing.
    """

    base: Literal[BaseModelType.MiniMaxH3] = Field(default=BaseModelType.MiniMaxH3)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    variant: MiniMaxH3VariantType = Field()
    pruned: bool = Field(description="Whether this is an AdaLN-pruned ('adaln curves') checkpoint.")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()
        if not _has_minimax_h3_keys(state_dict):
            raise NotAMatchError("state dict does not look like a MiniMax H3 transformer")

        if _has_ggml_tensors(state_dict):
            raise NotAMatchError("GGUF-quantized MiniMax H3 checkpoints are not supported yet")

        variant = override_fields.pop("variant", None) or (
            MiniMaxH3VariantType.REF2VA if "ref2va" in mod.path.name.lower() else MiniMaxH3VariantType.FL2VA
        )
        pruned = "adaln_t_table" in state_dict

        return cls(**override_fields, variant=variant, pruned=pruned)


# ---------------------------------------------------------------------------------------------
# LTX-2
# ---------------------------------------------------------------------------------------------

LTX2_SUPPORTED_GENERATIONS = frozenset({"2.5"})
"""The LTX-2 generations this version can run. 2.0/2.3 use a Gemma-3 encoder, bundled single files
and different transformer switches; their files are recognised as LTX-2 and refused (an invalid
match, so they are not registered as unknown models) rather than half-loaded."""


def _ltx2_generation_from_keys(stripped_keys: set[str]) -> str | None:
    """The generation a file's structure implies, for files whose header carries no version.

    LTX-2.5 dropped the video feed-forward biases (2.0/2.3 keep them on both streams), and its video
    VAE decoder gained a fourth upsampling stage (``up_blocks.8``; 2.0/2.3 stop at ``up_blocks.6``).
    Either component identifies the generation on its own.
    """
    if "transformer_blocks.0.ff.net.0.proj.weight" in stripped_keys:
        video_ff_bias = "transformer_blocks.0.ff.net.2.bias" in stripped_keys
        audio_ff_bias = "transformer_blocks.0.audio_ff.net.2.bias" in stripped_keys
        return "2.5" if (not video_ff_bias and audio_ff_bias) else None
    for prefix in ("", "vae."):
        if f"{prefix}decoder.up_blocks.0.res_blocks.0.conv1.conv.weight" in stripped_keys:
            return "2.5" if f"{prefix}decoder.up_blocks.8.res_blocks.0.conv1.conv.weight" in stripped_keys else None
    return None


def _ltx2_generation_or_raise(mod: ModelOnDisk, path: Path, stripped_keys: set[str] | None) -> str:
    """The generation a dating file belongs to: the header version when the file carries one, its
    structure otherwise. An LTX-2 file of a generation this version cannot run is an *invalid* match:
    recognised, and refused rather than registered as an unknown model."""
    generation = ltx2_layout.generation_from_version(ltx2_layout.header_model_version(mod.metadata(path)))
    if generation is None and stripped_keys is not None:
        generation = _ltx2_generation_from_keys(stripped_keys)
    if generation is None:
        raise NotAMatchError(f"cannot tell which LTX-2 generation {path.name} belongs to")
    if generation not in LTX2_SUPPORTED_GENERATIONS:
        raise InvalidMatchError(
            f"{path.name} is an LTX-{generation} file; only LTX-{'/'.join(sorted(LTX2_SUPPORTED_GENERATIONS))} "
            "is supported in this version"
        )
    return generation


class Main_Diffusers_LTX2_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    """An LTX-2 folder of per-component single files in the official key layout.

    This is how the ``DeepBeepMeep/LTX-2`` mirror distributes LTX-2.5: one safetensors per component
    rather than a diffusers ``model_index.json`` tree. (The official ``Lightricks/LTX-2.5`` files use
    the same layout for the VAEs, upsamplers and transformer, but keep the connectors inside the
    transformer file and the text projection inside the text-encoder file; a folder of official files
    therefore lacks a text-projection component in this version.) The folder is
    the *component source* of a generation -- video VAE, audio VAE, vocoder, text projection, the two
    text connectors and the latent upsamplers -- while the 22B transformer normally comes from a
    single-file record selected in the model loader (bf16, int8-convrot or nvfp4). A folder that
    also holds a transformer file is a full install.

    ``components`` maps each role (see ``invokeai.backend.ltx2.checkpoint_layout``) to the file in the
    folder that carries it, so the loader never re-classifies at generation time.
    """

    base: Literal[BaseModelType.LTX2] = Field(BaseModelType.LTX2)
    variant: LTX2VariantType = Field()
    generation: str = Field(description="The LTX-2 generation the folder's files belong to, e.g. '2.5'.")
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Component role -> weight file name inside the folder.",
    )
    components_only: bool = Field(
        default=False,
        description="Whether the folder holds only the shared components (VAEs, vocoder, connectors, "
        "upsamplers) without a transformer - the transformer must then be supplied as a single-file "
        "selection at generation time.",
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)
        raise_for_override_fields(cls, override_fields)

        if any((mod.path / name).exists() for name in ("model_index.json", "modular_model_index.json")):
            # The diffusers-layout repos (`Lightricks/LTX-2.5-Diffusers`) are a different install shape
            # this version does not load; leave them unclaimed rather than half-matched.
            raise NotAMatchError("diffusers-layout LTX-2 folders are not supported in this version")

        components: dict[str, str] = {}
        generations: dict[str, str] = {}
        for weight_file in sorted(p for p in mod.path.glob("*.safetensors") if p.is_file()):
            state_dict = mod.load_state_dict(weight_file)
            keys = {k for k in state_dict if isinstance(k, str)}
            roles = ltx2_layout.classify_roles(state_dict, mod.metadata(weight_file))
            if not roles:
                continue
            for role in roles:
                # First file wins for a duplicated role; sorted order keeps the choice deterministic.
                components.setdefault(role, weight_file.name)
            if roles & {ltx2_layout.ROLE_TRANSFORMER, ltx2_layout.ROLE_VIDEO_VAE}:
                # The transformer and the video VAE are the two components whose structure dates
                # the release; every such file is dated so a mixed folder is caught here, not at
                # the strict load of whichever file sorted first.
                stripped = {ltx2_layout.strip_transformer_prefix(k) for k in keys}
                generations[weight_file.name] = _ltx2_generation_or_raise(mod, weight_file, stripped)

        # The audio track is what distinguishes LTX-2 from every other video family: a folder without
        # both VAEs is not an LTX-2 component source, whatever else it holds.
        if ltx2_layout.ROLE_VIDEO_VAE not in components or ltx2_layout.ROLE_AUDIO_VAE not in components:
            if ltx2_layout.ROLE_DIFFUSION_VIDEO_VAE in components:
                raise NotAMatchError(
                    "folder holds no LTX-2 video VAE + audio VAE pair (its video VAE is the diffusion-decoder "
                    "variant, which this version does not use - install the conv video VAE)"
                )
            raise NotAMatchError("folder holds no LTX-2 video VAE + audio VAE pair")
        if not generations:
            raise NotAMatchError("cannot tell which LTX-2 generation the folder's files belong to")
        if len(set(generations.values())) > 1:
            raise InvalidMatchError(
                "folder mixes LTX-2 generations: "
                + ", ".join(f"{name} is {gen}" for name, gen in sorted(generations.items()))
            )
        generation = override_fields.pop("generation", None) or next(iter(generations.values()))
        components = override_fields.pop("components", None) or components

        transformer_file = components.get(ltx2_layout.ROLE_TRANSFORMER)
        components_only = override_fields.pop("components_only", None)
        if components_only is None:
            components_only = transformer_file is None

        variant = override_fields.pop("variant", None)
        if variant is None:
            variant = (
                LTX2VariantType.Distilled
                if transformer_file is not None and ltx2_layout.is_distilled_filename(transformer_file)
                else LTX2VariantType.Dev
            )

        repo_variant = override_fields.pop("repo_variant", None) or cls._get_repo_variant_or_raise(mod)
        return cls(
            **override_fields,
            variant=LTX2VariantType(variant),
            generation=generation,
            components=components,
            components_only=components_only,
            repo_variant=repo_variant,
        )


class Main_Checkpoint_LTX2_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """An LTX-2 single-file transformer (safetensors) in the official key layout.

    Covers the bf16, Comfy ``int8_tensorwise``(+convrot) and ``nvfp4`` releases; the quantization
    scheme is read from the file's own markers by the loader, not recorded here. The file may hold
    only the transformer (the 2.5 releases) or bundle other components beside it (earlier
    generations) -- either way everything else comes from an installed LTX-2 component folder.

    Dev and Distilled transformers are key-for-key identical, so the FILENAME is the variant
    classifier and ``variant`` is the user-correctable override for renamed files. A misclassified
    variant runs but samples with the wrong recipe (guided schedule vs fixed distilled sigmas).
    """

    base: Literal[BaseModelType.LTX2] = Field(default=BaseModelType.LTX2)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    variant: LTX2VariantType = Field()
    generation: str = Field(description="The LTX-2 generation the transformer belongs to, e.g. '2.5'.")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        raise_for_override_fields(cls, override_fields)

        if mod.path.suffix.lower() != ".safetensors":
            raise NotAMatchError(f"expected a .safetensors file, got {mod.path.suffix or '(no suffix)'}")

        state_dict = mod.load_state_dict()
        keys = {k for k in state_dict if isinstance(k, str)}
        if ltx2_layout.ROLE_TRANSFORMER not in ltx2_layout.classify_roles(state_dict, mod.metadata()):
            raise NotAMatchError("state dict does not look like an LTX-2 transformer")
        if _has_ggml_tensors(state_dict):
            raise NotAMatchError("GGUF-quantized LTX-2 checkpoints are not supported yet")

        generation = override_fields.pop("generation", None) or _ltx2_generation_or_raise(
            mod, mod.path, {ltx2_layout.strip_transformer_prefix(k) for k in keys}
        )
        variant = override_fields.pop("variant", None) or (
            LTX2VariantType.Distilled if ltx2_layout.is_distilled_filename(mod.path.name) else LTX2VariantType.Dev
        )
        return cls(**override_fields, variant=LTX2VariantType(variant), generation=generation)


class Main_Checkpoint_Krea2_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for Krea-2 single-file checkpoint models (safetensors, etc)."""

    base: Literal[BaseModelType.Krea2] = Field(default=BaseModelType.Krea2)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    variant: Krea2VariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if mod.path.suffix.lower() != ".safetensors":
            raise NotAMatchError(f"expected a .safetensors file, got {mod.path.suffix or '(no suffix)'}")

        cls._validate_looks_like_krea2_model(mod)

        cls._validate_does_not_look_like_gguf_quantized(mod)

        variant = override_fields.pop("variant", None) or _get_krea2_variant_from_name(mod.path.name)

        return cls(**override_fields, variant=variant)

    @classmethod
    def _validate_looks_like_krea2_model(cls, mod: ModelOnDisk) -> None:
        if not _has_krea2_keys(mod.load_state_dict()):
            raise NotAMatchError("state dict does not look like a Krea-2 model")

    @classmethod
    def _validate_does_not_look_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        if _has_ggml_tensors(mod.load_state_dict()):
            raise NotAMatchError("state dict looks like GGUF quantized")


class Main_GGUF_Krea2_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for GGUF-quantized Krea-2 transformer models (single-file)."""

    base: Literal[BaseModelType.Krea2] = Field(default=BaseModelType.Krea2)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)
    variant: Krea2VariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_krea2_model(mod)

        cls._validate_looks_like_gguf_quantized(mod)

        variant = override_fields.pop("variant", None) or _get_krea2_variant_from_name(mod.path.name)

        return cls(**override_fields, variant=variant)

    @classmethod
    def _validate_looks_like_krea2_model(cls, mod: ModelOnDisk) -> None:
        if not _has_krea2_keys(mod.load_state_dict()):
            raise NotAMatchError("state dict does not look like a Krea-2 model")

    @classmethod
    def _validate_looks_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        if not _has_ggml_tensors(mod.load_state_dict()):
            raise NotAMatchError("state dict does not look like GGUF quantized")


class Main_Diffusers_QwenImage_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    """Model config for Qwen Image diffusers models (both txt2img and edit)."""

    base: Literal[BaseModelType.QwenImage] = Field(BaseModelType.QwenImage)
    variant: QwenImageVariantType | None = Field(default=None)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # This check implies the base type - no further validation needed.
        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "QwenImagePlusPipeline",
                "QwenImageEditPlusPipeline",
                "QwenImagePipeline",
            },
        )

        repo_variant = override_fields.pop("repo_variant", None) or cls._get_repo_variant_or_raise(mod)
        variant = override_fields.pop("variant", None) or cls._get_qwen_image_variant(mod)

        return cls(
            **override_fields,
            repo_variant=repo_variant,
            variant=variant,
        )

    @classmethod
    def _get_qwen_image_variant(cls, mod: ModelOnDisk) -> QwenImageVariantType:
        """Detect whether this is an edit or txt2img model from the pipeline class name."""
        import json

        model_index = mod.path / "model_index.json"
        if model_index.exists():
            with open(model_index) as f:
                config = json.load(f)
            class_name = config.get("_class_name", "")
            if "Edit" in class_name:
                return QwenImageVariantType.Edit
        return QwenImageVariantType.Generate


# ComfyUI single-file checkpoints prefix every transformer key with one of these. The loaders strip
# them before instantiating the model (`CheckpointPrefix`), and detection has to strip them too or
# the two disagree: a file identification accepts, the loader then refuses with every key unexpected.
# Shared rather than restated, which is what made them drift; the *operation* still differs, because
# detection looks for evidence in individual names instead of normalising a whole file.


def _strip_comfyui_key_prefix(key: str) -> str:
    """Strip a leading ComfyUI `model.diffusion_model.` / `diffusion_model.` prefix from a key."""
    for prefix in COMFYUI_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _has_qwen_image_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains Qwen Image Edit transformer keys.

    Qwen Image Edit uses 'txt_in' and 'txt_norm' instead of 'context_embedder' (FLUX).
    This distinguishes it from FLUX and other architectures. ComfyUI-style prefixes are
    stripped first so prefixed checkpoints are detected and reach the loader.
    """
    keys = [_strip_comfyui_key_prefix(k) for k in state_dict.keys() if isinstance(k, str)]
    has_txt_in = any(k.startswith("txt_in.") for k in keys)
    has_txt_norm = any(k.startswith("txt_norm.") for k in keys)
    has_img_in = any(k.startswith("img_in.") for k in keys)
    # Must NOT have context_embedder (which would indicate FLUX)
    has_context_embedder = any("context_embedder" in k for k in keys)
    return has_txt_in and has_txt_norm and has_img_in and not has_context_embedder


# Matches "edit" as a standalone token (delimited by start/end or any non-alphanumeric
# separator), so `qwen_image_edit_2509` matches but `credited` / `edited` / `unedited` do not.
_EDIT_TOKEN_RE = re.compile(r"(?:^|[^a-z0-9])edit(?:[^a-z0-9]|$)")


def _infer_qwen_image_variant(sd: dict[str | int, Any], path: Path) -> QwenImageVariantType:
    """Infer Qwen Image variant from state dict marker or filename heuristic.

    Edit-variant models include an `__index_timestep_zero__` tensor used by the
    `zero_cond_t` dual-modulation path. Falls back to a filename "edit" token check
    for converters that don't emit the marker.
    """
    marker = "__index_timestep_zero__"
    if marker in sd or any(isinstance(k, str) and _strip_comfyui_key_prefix(k) == marker for k in sd):
        return QwenImageVariantType.Edit
    if _EDIT_TOKEN_RE.search(path.stem.lower()):
        return QwenImageVariantType.Edit
    return QwenImageVariantType.Generate


class Main_Checkpoint_QwenImage_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for Qwen Image single-file checkpoint models (safetensors, etc).

    Covers raw bf16/fp16 checkpoints and ComfyUI-style fp8_scaled and nvfp4 checkpoints.
    The loader keeps scaled fp8 weights when fp8 compute is available or the
    `default_settings.fp8_storage` toggle is on (which also re-casts the rest to fp8),
    and dequantizes them to bf16 at load time otherwise.
    """

    base: Literal[BaseModelType.QwenImage] = Field(default=BaseModelType.QwenImage)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    variant: QwenImageVariantType | None = Field(default=None)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        sd = mod.load_state_dict()

        if not _has_qwen_image_keys(sd):
            raise NotAMatchError("state dict does not look like a Qwen Image model")

        if _has_ggml_tensors(sd):
            raise NotAMatchError("state dict looks like GGUF quantized")

        explicit_variant = override_fields.pop("variant", None) or _infer_qwen_image_variant(sd, mod.path)

        return cls(**override_fields, variant=explicit_variant)


class Main_GGUF_QwenImage_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for GGUF-quantized Qwen Image transformer models."""

    base: Literal[BaseModelType.QwenImage] = Field(default=BaseModelType.QwenImage)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)
    variant: QwenImageVariantType | None = Field(default=None)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        sd = mod.load_state_dict()

        if not _has_qwen_image_keys(sd):
            raise NotAMatchError("state dict does not look like a Qwen Image Edit model")

        if not _has_ggml_tensors(sd):
            raise NotAMatchError("state dict does not look like GGUF quantized")

        explicit_variant = override_fields.pop("variant", None) or _infer_qwen_image_variant(sd, mod.path)

        return cls(**override_fields, variant=explicit_variant)


def _has_wan_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains Wan 2.2 transformer keys.

    Two layouts are accepted:

    * **Diffusers** (city96-style GGUF, Wan-AI/*-Diffusers safetensors): the text
      projection is named ``condition_embedder.text_embedder.linear_1``.
    * **Native upstream** (QuantStack-style GGUF, ComfyUI, Wan-AI's non-Diffusers
      releases): the text projection is named ``text_embedding.0``.

    Both layouts share ``patch_embedding.weight`` as the input conv. Combined with
    the text-projection fingerprint, this won't collide with FLUX
    (``double_blocks/single_blocks``), Qwen Image (``txt_in/img_in``), Z-Image
    (``cap_embedder``), or Anima (``llm_adapter``).

    Tolerates both bare keys and the ComfyUI ``model.diffusion_model.`` /
    ``diffusion_model.`` prefixes.
    """
    text_proj_options = (
        "condition_embedder.text_embedder.linear_1.weight",
        "text_embedding.0.weight",
    )
    prefixes = ("", "model.diffusion_model.", "diffusion_model.")
    keys = state_dict.keys()
    if not any((p + "patch_embedding.weight") in keys for p in prefixes):
        return False
    return any((p + needle) in keys for p in prefixes for needle in text_proj_options)


def _is_native_wan_layout(state_dict: dict[str | int, Any]) -> bool:
    """True if the state dict uses the native upstream Wan key layout.

    Native layout uses ``text_embedding.0/2``, ``self_attn``/``cross_attn``,
    ``ffn.0/2``, ``head.head``, ``head.modulation``, etc. — what ComfyUI and
    QuantStack ship. Diffusers layout uses ``condition_embedder.*``, ``attn1``/
    ``attn2``, ``ffn.net.*``, ``proj_out``, ``scale_shift_table``.
    """
    prefixes = ("", "model.diffusion_model.", "diffusion_model.")
    keys = state_dict.keys()
    return any((p + "text_embedding.0.weight") in keys for p in prefixes)


_WAN_KEY_PREFIXES = ("", "model.diffusion_model.", "diffusion_model.")


def _wan_patch_embedding_shape(state_dict: dict[str | int, Any]) -> tuple[int, ...] | None:
    """Return the shape of ``patch_embedding.weight``, tolerating ComfyUI prefixes.

    Works for both plain tensors and GGMLTensors (which carry the logical shape on
    ``tensor_shape`` because their storage is the packed quantized blob).
    """
    for prefix in _WAN_KEY_PREFIXES:
        tensor = state_dict.get(prefix + "patch_embedding.weight")
        if tensor is None:
            continue
        shape = getattr(tensor, "tensor_shape", None)
        if shape is None:
            shape = getattr(tensor, "shape", None)
        if shape is None:
            return None
        return tuple(int(dim) for dim in shape)
    return None


def _detect_wan_variant_from_state_dict(state_dict: dict[str | int, Any]) -> WanVariantType | None:
    """Determine A14B (T2V vs I2V) vs TI2V-5B from the transformer state dict.

    ``patch_embedding.weight`` has shape ``[inner_dim, in_channels, T, H, W]``;
    ``in_channels`` uniquely identifies the Wan 2.2 variant:

    - 16 → T2V-A14B (noise latents only).
    - 36 → I2V-A14B (16 noise + 16 ref-image latents + 4 first-frame mask,
      concatenated along the channel dim — see diffusers
      ``WanImageToVideoPipeline.prepare_latents``).
    - 48 → TI2V-5B (Wan2.2-VAE z_dim=48).

    Returns None if the tensor is missing or the channel count is unrecognised.
    """
    shape = _wan_patch_embedding_shape(state_dict)
    if shape is None or len(shape) < 2:
        return None
    inner_dim, in_channels = shape[0], shape[1]

    # in_channels alone is ambiguous outside the three supported releases: the wider
    # Wan family reuses these channel counts at other widths (Fun-Control-14B is
    # 48-channel but 5120-wide, i.e. A14B-sized, not TI2V-5B). Require the width to
    # agree, so a derivative we don't support falls through to None rather than being
    # mislabelled — a wrong variant pins `expert`, picks the wrong default settings,
    # and hides the low-noise partner picker.
    #
    # A14B is uniquely 5120-wide and TI2V-5B uniquely 3072-wide across Wan 2.2.
    if in_channels == 16 and inner_dim == 5120:
        return WanVariantType.T2V_A14B
    if in_channels == 36 and inner_dim == 5120:
        return WanVariantType.I2V_A14B
    if in_channels == 48 and inner_dim == 3072:
        return WanVariantType.TI2V_5B
    return None


def _has_wan_transformer_block_weights(state_dict: dict[str | int, Any]) -> bool:
    """True if the state dict carries a transformer block's *own* attention weight.

    ``_has_wan_keys`` only looks at the input conv and the text projection, and a
    Wan LoRA can legitimately ship both: I2V adapters bundle a full replacement
    ``patch_embedding`` because they change ``in_channels`` from 16 to 36. Such a
    file matches both the LoRA and the main-model probes, and ``matches_sort_key``
    ranks Main above LoRA — so it would be pulled out of the LoRA pickers and into
    the main-model dropdown, where it can only fail to load.

    Requiring a bare ``blocks.0.<attn>.<q>.weight`` separates them positively: a
    LoRA stores ``...q.lora_A.weight`` / ``...q.lora_down.weight`` and never the
    undecorated weight. Keys are matched exactly, so a LoRA's decorated key cannot
    satisfy this.

    Deliberately a *positive* structural test rather than a "reject anything with
    lora_A keys" exclusion: main models with merged-in LoRA weights sometimes retain
    those keys (see ``LoRA_LyCORIS_*_Config._validate_looks_like_lora``), and
    rejecting them would be the same over-restrictiveness this probe exists to fix.
    """
    attention_weights = (
        "blocks.0.self_attn.q.weight",  # native upstream / ComfyUI layout
        "blocks.0.attn1.to_q.weight",  # diffusers layout
    )
    keys = state_dict.keys()
    return any((prefix + weight) in keys for prefix in _WAN_KEY_PREFIXES for weight in attention_weights)


def _find_wan_2_1_marker(state_dict: dict[str | int, Any]) -> str | None:
    """Return a human-readable reason if the state dict is architecturally Wan 2.1.

    Wan 2.1 and Wan 2.2 share a key layout, so the two families can only be told
    apart by architecture. Three markers are decisive, and all three describe
    things Wan 2.2 never ships:

    * **CLIP image embedder** (``img_emb.proj.*`` / ``condition_embedder.image_embedder.*``).
      Wan 2.1 I2V conditioned on CLIP-vision features via ``image_dim``. Wan 2.2
      I2V-A14B dropped that entirely and concatenates VAE latents instead, so any
      36-channel model carrying an image embedder is Wan 2.1.
    * **1536-dim inner width** — the Wan 2.1 T2V-1.3B model. The Wan 2.2 family is
      5120 (A14B) or 3072 (TI2V-5B).
    * **VACE blocks** (``vace_blocks.*``) — the Wan 2.1 VACE editing variant, which
      needs a control branch InvokeAI's Wan pipeline doesn't drive.

    Wan 2.1 T2V-14B is *not* detectable this way: it is shape-identical to a single
    Wan 2.2 A14B expert. Callers that care fall back to the filename/metadata gate.
    """
    keys = state_dict.keys()
    image_embedder_markers = ("img_emb.proj.0.weight", "condition_embedder.image_embedder.norm1.weight")
    if any((prefix + marker) in keys for prefix in _WAN_KEY_PREFIXES for marker in image_embedder_markers):
        return (
            "state dict has a CLIP image embedder (img_emb), which is a Wan 2.1 I2V feature; "
            "Wan 2.2 I2V conditions on VAE latents instead"
        )

    shape = _wan_patch_embedding_shape(state_dict)
    if shape is not None and len(shape) >= 1 and shape[0] == 1536:
        return "state dict has a 1536-dim transformer, which is the Wan 2.1 T2V-1.3B architecture"

    return None


def _find_unsupported_wan_variant_marker(state_dict: dict[str | int, Any]) -> str | None:
    """Return a reason if this is a Wan variant the plain transformer can't represent.

    These are Wan 2.2-era models built on extra conditioning branches that
    ``WanTransformer3DModel`` simply doesn't have. Loading them would not error:
    ``load_state_dict(strict=False)`` drops the extra modules as unexpected keys and
    produces a model that silently ignores the conditioning it was built around
    (real ``wan2.2_animate_14B_bf16.safetensors``: 127 of its 1441 keys are
    ``face_adapter``/``motion_encoder``). Refusing is the honest outcome.

    Checked before the Wan 2.1 markers because Animate also carries ``img_emb``, so
    it would otherwise be reported as a Wan 2.1 I2V model.
    """
    keys = [key for key in state_dict.keys() if isinstance(key, str)]

    def has(*markers: str) -> bool:
        return any(marker in key for key in keys for marker in markers)

    if has("face_adapter.", "motion_encoder."):
        return (
            "state dict has face-adapter / motion-encoder branches, which belong to Wan Animate; "
            "character animation and replacement are not supported yet"
        )

    if has("audio_injector.", "casual_audio_encoder.", "cond_encoder.", "frame_packer."):
        return (
            "state dict has audio-conditioning branches, which belong to Wan S2V; "
            "audio-driven video is not supported yet"
        )

    if has("control_adapter."):
        # Fun-Control-Camera is the dangerous one: 36 in-channels and otherwise
        # key-identical to plain I2V-A14B, and it ships as a properly tagged
        # high/low pair, so the expert-pairing check passes too. Without this it
        # would load and render as an ordinary I2V, ignoring every camera input.
        return (
            "state dict has a control-adapter branch, which belongs to the Wan Fun-Control family; "
            "camera and control conditioning are not supported yet"
        )

    if has("vace_blocks."):
        return "state dict has VACE control blocks, and VACE models are not supported yet"

    return None


# Tokens that turn an adjacent bare ``high``/``low`` into an adjective about
# something other than the noise level, so it must not be read as an MoE expert.
#
# Deliberately short. Real releases sit expert markers next to plenty of unrelated
# words — ``..._LOW_lightning_edition``, ``..._high_lighting_fp16`` — so anything
# broader here starts costing true positives, which is the more damaging error for
# a checkpoint whose expert can only be recovered from its name.
_WAN_EXPERT_DISQUALIFIERS = frozenset(
    {
        "angle",
        "cfg",
        "guidance",
        "vram",
        "ram",
        "mem",
        "memory",
        "step",
        "steps",
        "res",
        "resolution",
        "quality",
        "speed",
        "fps",
        "bit",
        "bits",
    }
)


def _detect_wan_expert(filename: str) -> Literal["high", "low", "none"]:
    """Filename heuristic for the A14B dual-expert MoE.

    Two conventions dominate and both have to work, because the expert cannot be
    read off the weights:

    * ``high_noise`` / ``low_noise`` and its spellings — hyphenated, underscored,
      spaced, fused (``highnoise``, ``LOWNOISEFP8``), camel-cased (``HighNoise``),
      reversed (``noise_high``). This is what Comfy-Org's repackaged repos use.
    * A **bare** ``HIGH`` / ``LOW`` token, e.g.
      ``Wan2_2-T2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors``. This is what the
      widely-mirrored Kijai fp8 catalogue and many CivitAI fine-tunes use, so
      refusing to read it would leave most single-file A14B models unpairable.

    Some releases also declare the expert in file metadata; ``_resolve_wan_expert``
    consults that when the name yields nothing.

    Precedence, in order:

    1. Explicit ``..._noise`` markers outrank bare tokens found elsewhere in the
       name, so ``..._4step_LOW_lightning_high_noise`` reads as the high-noise file.
    2. Within a tier, one distinct marker wins. If **both** appear, the file serves
       both experts (``... I2V HIGH+LOW ...``, ``..._low_high_noise_...`` — both are
       real release patterns) or the name is simply ambiguous, so return 'none'
       rather than guess. For a LoRA 'none' means "apply to both", which is the
       right answer for those; for a main it leaves the role to the explicit
       transformer-slot wiring instead of guessing from an ambiguous name.

    Note the tiers are reconciled the same way. An earlier revision returned on the
    first ``noise`` marker it saw, which meant the two spellings of the same
    both-experts name disagreed: ``HIGH-LOW`` gave 'none' but ``low_high_noise``
    gave 'high'.

    A bare token is ignored when a neighbour marks it as an adjective about
    something else (``lowVRAM``, ``low-cfg``, ``Low Angle``) — see
    ``_WAN_EXPERT_DISQUALIFIERS``. TI2V-5B is handled structurally by the callers,
    which force 'none' because the model is single-transformer.

    Matching is per token, so a bare marker can't fire on a substring
    (``highway``), and the fused form is anchored at a token boundary so
    ``slow_noise`` and ``flownoise`` are left alone.

    Returns 'none' for an untagged filename.
    """
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", filename).lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", name) if token]

    markers = ("high", "low")
    explicit: list[str] = []
    bare: list[str] = []

    index = 0
    while index < len(tokens):
        token = tokens[index]

        # Fused with 'noise'. Anchored: startswith catches `lownoisefp8`, and only the
        # reversed form may match at the end — `endswith("lownoise")` would wrongly
        # claim `slownoise`.
        fused = next((m for m in markers if token.startswith(f"{m}noise") or token.endswith(f"noise{m}")), None)
        if fused is not None:
            explicit.append(fused)
            index += 1
            continue

        if token not in markers:
            index += 1
            continue

        # Consume the whole *run* of adjacent marker tokens, so a single 'noise'
        # qualifies all of them. `low_high_noise` names one file holding both experts
        # (real: moriqqe/Mabrle_wan2.2_low_high_noise); scoring only the marker
        # touching 'noise' would read it as the high-noise expert alone.
        start = index
        while index < len(tokens) and tokens[index] in markers:
            index += 1
        run = tokens[start:index]
        previous = tokens[start - 1] if start > 0 else None
        following = tokens[index] if index < len(tokens) else None

        # Only the token *following* the run can disqualify: these are adjective-noun
        # pairs, so the noun comes second. "low angle" is a camera angle, but
        # "Angle HIGH" is the high-noise expert of a camera-angle LoRA.
        #
        # A disqualifier only consumes the marker it is actually attached to — the last
        # one in the run — not the whole run. `HIGH_lowVRAM` is the high-noise expert of
        # a low-VRAM build: dropping both markers there loses a correct tag, which for a
        # main model disables the pair checks and for a LoRA silently applies a
        # single-expert distill to both experts.
        #
        # Checked ahead of the 'noise' test below, and it clears `following` with it: a
        # disqualifier sits after the run, so an adjacent 'noise' would have to precede
        # it. ('low noise' can't trip this — 'noise' is not itself a disqualifier.)
        if following in _WAN_EXPERT_DISQUALIFIERS:
            run = run[:-1]
            following = None
        if not run:
            continue
        if previous == "noise" or following == "noise":
            explicit.extend(run)
        else:
            bare.extend(run)

    # An explicit `..._noise` marker outranks a bare token found elsewhere in the
    # name. Within each tier, both experts named means the file serves both — real
    # releases do ship that way — so return 'none' rather than pick one. For a LoRA
    # 'none' means "apply to both", which is the right answer for those.
    for candidates in (explicit, bare):
        distinct = set(candidates)
        if len(distinct) == 1:
            return candidates[0]  # type: ignore[return-value]
        if distinct:
            return "none"
    return "none"


def _resolve_wan_expert(
    mod: ModelOnDisk, override_fields: dict[str, Any], variant: WanVariantType
) -> Literal["high", "low", "none"]:
    """Settle the MoE expert field: explicit override, then filename, then metadata.

    The override is consumed here so it can't reach the constructor twice. Note it is
    not reachable through the install API today — ``ModelRecordChanges`` has no
    ``expert`` field and ``ModelConfigFactory.build_common_fields`` forwards a fixed
    whitelist that excludes it — so in practice a mis-detected expert can only be
    corrected by renaming the file and re-importing.

    TI2V-5B is a single-transformer model, so the expert is meaningless there and is
    pinned to 'none'. That keeps the record neutral for every expert-aware consumer;
    the frontend's low-noise partner picker also excludes TI2V-5B structurally, even
    if a legacy record carries a misleading ``low`` tag.

    Metadata is consulted only as a fallback, not as the primary signal, even though
    it is the more trustworthy of the two. Renaming a file is the one lever a user
    has to correct a mis-detected expert — there is no UI for the field — and the
    Wan model loader's error message tells them to use it. Letting an embedded
    ``model_type`` override the name would take that lever away.

    Coverage is uneven, which is why both signals are needed. Sampled 2026-08-13:
    every Wan 2.2 safetensors in ``Kijai/WanVideo_comfy_fp8_scaled`` carries
    ``__metadata__["model_type"]`` naming the expert (``Wan2_2-I2V-A14B-high``),
    while none of the ``Comfy-Org/Wan_2.2_ComfyUI_Repackaged`` files carry any
    ``__metadata__`` at all. GGUF releases use ``general.name`` instead.
    """
    explicit_expert = override_fields.pop("expert", None)
    if explicit_expert is not None:
        return explicit_expert  # type: ignore[no-any-return]

    if variant == WanVariantType.TI2V_5B:
        return "none"

    expert = _detect_wan_expert(mod.path.stem)
    if expert != "none":
        return expert

    metadata = mod.metadata()
    declared = metadata.get("model_type") or metadata.get("general.name") or ""
    if declared:
        return _detect_wan_expert(declared)
    return "none"


class Main_GGUF_Wan_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for GGUF-quantized Wan 2.2 transformer models.

    A14B's MoE ships as two GGUF files (one per expert); ``expert`` records
    which one this is so the model loader invocation can pair them. TI2V-5B
    is a single-transformer model and stores ``expert='none'``.
    """

    base: Literal[BaseModelType.Wan] = Field(default=BaseModelType.Wan)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)
    variant: WanVariantType = Field()
    expert: Literal["high", "low", "none"] = Field(
        default="none",
        description="For Wan 2.2 A14B's dual-expert MoE: 'high' for the high-noise expert, "
        "'low' for the low-noise expert. 'none' for single-transformer models (TI2V-5B).",
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        raise_for_override_fields(cls, override_fields)

        sd = mod.load_state_dict()

        if not _has_ggml_tensors(sd):
            raise NotAMatchError("state dict does not look like GGUF quantized")
        if not _has_wan_keys(sd):
            raise NotAMatchError("state dict does not look like a Wan transformer")
        if not _has_wan_transformer_block_weights(sd):
            raise NotAMatchError(
                "state dict has no undecorated transformer block weights — it looks like a Wan LoRA "
                "or adapter rather than a full transformer"
            )
        unsupported_reason = _find_unsupported_wan_variant_marker(sd)
        if unsupported_reason is not None:
            raise NotAMatchError(unsupported_reason)
        gguf_name = mod.metadata().get("general.name", "")
        normalized_identity = "".join(
            character for character in f"{mod.path.stem} {gguf_name}".lower() if character.isalnum()
        )
        if "wan21" in normalized_identity:
            raise NotAMatchError("Wan 2.1 GGUF models are not supported by the Wan 2.2 loader")
        # A misnamed Wan 2.1 GGUF slips past the name check above; the architectural
        # markers don't care what the file is called.
        wan_2_1_reason = _find_wan_2_1_marker(sd)
        if wan_2_1_reason is not None:
            raise NotAMatchError(f"Wan 2.1 GGUF models are not supported by the Wan 2.2 loader: {wan_2_1_reason}")

        explicit_variant = override_fields.pop("variant", None)
        variant = explicit_variant or _detect_wan_variant_from_state_dict(sd)
        if variant is None:
            raise NotAMatchError("could not determine Wan variant from state dict")
        if variant in (WanVariantType.T2V_A14B, WanVariantType.I2V_A14B) and "wan22" not in normalized_identity:
            raise NotAMatchError("Wan A14B GGUF filename or metadata must identify the model as Wan 2.2")

        expert = _resolve_wan_expert(mod, override_fields, variant)

        return cls(**override_fields, variant=variant, expert=expert)


class Main_Checkpoint_Wan_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for single-file Wan 2.2 transformer checkpoints (safetensors).

    This is the format the community ships on CivitAI and in ComfyUI-oriented
    Hugging Face repos: one ``.safetensors`` per transformer, in either the native
    upstream key layout or the diffusers one, optionally under a
    ``model.diffusion_model.`` prefix, and optionally ComfyUI ``fp8_scaled``
    quantized. The loader normalises all of those.

    As with GGUF, A14B's MoE arrives as two files (one per expert); ``expert``
    records which one this is so the Wan model loader invocation can pair them.
    TI2V-5B is single-transformer and stores ``expert='none'``.
    """

    base: Literal[BaseModelType.Wan] = Field(default=BaseModelType.Wan)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    variant: WanVariantType = Field()
    expert: Literal["high", "low", "none"] = Field(
        default="none",
        description="For Wan 2.2 A14B's dual-expert MoE: 'high' for the high-noise expert, "
        "'low' for the low-noise expert. 'none' for single-transformer models (TI2V-5B).",
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        raise_for_override_fields(cls, override_fields)

        # The loader reads this format with safetensors.torch.load_file, so claiming a
        # pickle here would let the model install and then fail with an opaque
        # "header too large" error at first generation. Wan isn't distributed as
        # .ckpt/.pt/.bin, so refusing them costs nothing.
        if mod.path.suffix.lower() != ".safetensors":
            raise NotAMatchError(f"single-file Wan checkpoints must be .safetensors, not {mod.path.suffix or 'None'}")

        sd = mod.load_state_dict()

        if not _has_wan_keys(sd):
            raise NotAMatchError("state dict does not look like a Wan transformer")
        if _has_ggml_tensors(sd):
            raise NotAMatchError("state dict looks like GGUF quantized")
        if not _has_wan_transformer_block_weights(sd):
            raise NotAMatchError(
                "state dict has no undecorated transformer block weights — it looks like a Wan LoRA "
                "or adapter rather than a full transformer"
            )

        # Checked before the Wan 2.1 markers: Animate carries img_emb too, so the
        # order is what makes the rejection reason accurate.
        unsupported_reason = _find_unsupported_wan_variant_marker(sd)
        if unsupported_reason is not None:
            raise NotAMatchError(unsupported_reason)

        # Wan 2.1 shares Wan 2.2's key layout, so reject it on architecture rather
        # than on the filename. Unlike the GGUF probe we deliberately do *not*
        # require the name to say "wan2.2": community fine-tunes routinely drop the
        # version from the filename, and rejecting them was the whole complaint in
        # #9463. The residual ambiguity is Wan 2.1 T2V-14B, which is shape-identical
        # to a Wan 2.2 A14B expert — that one is caught by the explicit "wan2.1" name
        # check below, and otherwise imports as A14B.
        wan_2_1_reason = _find_wan_2_1_marker(sd)
        if wan_2_1_reason is not None:
            raise NotAMatchError(f"Wan 2.1 models are not supported by the Wan 2.2 loader: {wan_2_1_reason}")

        normalized_identity = "".join(character for character in mod.path.stem.lower() if character.isalnum())
        if "wan21" in normalized_identity:
            raise NotAMatchError("Wan 2.1 models are not supported by the Wan 2.2 loader")

        explicit_variant = override_fields.pop("variant", None)
        variant = explicit_variant or _detect_wan_variant_from_state_dict(sd)
        if variant is None:
            raise NotAMatchError("could not determine Wan variant from state dict")

        expert = _resolve_wan_expert(mod, override_fields, variant)

        return cls(**override_fields, variant=variant, expert=expert)


class Main_Diffusers_Wan_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    """Model config for Wan 2.2 diffusers models.

    Covers both the dual-expert T2V-A14B family and the single-transformer TI2V-5B
    family. Variant is detected from the on-disk transformer config (latent channel
    count) plus the presence of a sibling ``transformer_2/`` directory.
    """

    base: Literal[BaseModelType.Wan] = Field(default=BaseModelType.Wan)
    variant: WanVariantType = Field()
    has_dual_expert: bool = Field(
        default=False,
        description="Whether this model ships two transformer experts (Wan 2.2 A14B MoE). False for TI2V-5B.",
    )
    boundary_ratio: float | None = Field(
        default=None,
        description="MoE expert switch point as a fraction of num_train_timesteps (typically 1000). "
        "None for single-transformer models. Read from model_index.json by Diffusers' WanPipeline.",
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # Wan repos ship with WanPipeline (T2V) or WanImageToVideoPipeline (I2V/TI2V).
        # Either class name is sufficient to identify a Wan diffusers model.
        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "WanPipeline",
                "WanImageToVideoPipeline",
            },
        )

        repo_variant = override_fields.pop("repo_variant", None) or cls._get_repo_variant_or_raise(mod)

        explicit_variant = override_fields.pop("variant", None)
        has_dual_expert = (mod.path / "transformer_2" / "config.json").exists()
        variant = explicit_variant or cls._detect_wan_variant(mod, has_dual_expert)
        boundary_ratio = override_fields.pop("boundary_ratio", None)
        if boundary_ratio is None:
            boundary_ratio = cls._read_boundary_ratio(mod)

        return cls(
            **override_fields,
            repo_variant=repo_variant,
            variant=variant,
            has_dual_expert=has_dual_expert,
            boundary_ratio=boundary_ratio,
        )

    @classmethod
    def _read_boundary_ratio(cls, mod: ModelOnDisk) -> float | None:
        """Pull ``boundary_ratio`` from ``model_index.json`` if present.

        Diffusers' ``WanPipeline.__init__`` registers it via ``register_to_config``,
        which persists it as a top-level key in the saved pipeline config.
        """
        try:
            model_index = get_config_dict_or_raise(mod.path / "model_index.json")
        except NotAMatchError:
            return None
        value = model_index.get("boundary_ratio")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _detect_wan_variant(cls, mod: ModelOnDisk, has_dual_expert: bool) -> WanVariantType:
        """Detect Wan variant from transformer + VAE config.

        - T2V-A14B: dual transformer experts, standard Wan VAE (z_dim=16),
          transformer ``in_channels=16`` (text-only conditioning).
        - I2V-A14B: dual transformer experts, standard Wan VAE,
          transformer ``in_channels=36`` (text + VAE-encoded reference image
          + first-frame mask concatenated along the channel dim).
        - TI2V-5B: single transformer, Wan2.2-VAE (z_dim=48).
        """
        if has_dual_expert:
            # Disambiguate T2V vs I2V via the transformer's input channel count.
            # Wan 2.2 I2V uses VAE-latent concatenation: 16 noise + 16 ref-image
            # latents + 4 first-frame mask = 36. (Wan 2.1 I2V used CLIP-vision
            # via ``image_dim``; that mechanism is absent in Wan 2.2.)
            in_channels = cls._transformer_in_channels(mod)
            if in_channels == 36:
                return WanVariantType.I2V_A14B
            return WanVariantType.T2V_A14B

        # Single-transformer model: distinguish TI2V-5B from any future single-expert
        # A14B-derived release by inspecting the VAE latent dimension.
        try:
            vae_config = get_config_dict_or_raise(mod.path / "vae" / "config.json")
            z_dim = vae_config.get("z_dim")
            if z_dim is not None and int(z_dim) >= 32:
                return WanVariantType.TI2V_5B
        except NotAMatchError:
            # No VAE config to inspect — fall through to the heuristic path below.
            pass

        # Filename / repo-name heuristic as a last resort.
        name = mod.path.name.lower()
        if "5b" in name or "ti2v" in name:
            return WanVariantType.TI2V_5B
        return WanVariantType.T2V_A14B

    @staticmethod
    def _transformer_in_channels(mod: ModelOnDisk) -> int | None:
        """Read ``in_channels`` from ``transformer/config.json``.

        For Wan 2.2 A14B, this is the canonical discriminator between T2V
        (``in_channels=16``) and I2V (``in_channels=36``). Returns None if the
        config can't be read.
        """
        try:
            transformer_config = get_config_dict_or_raise(mod.path / "transformer" / "config.json")
        except NotAMatchError:
            return None
        value = transformer_config.get("in_channels")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None


class Main_Checkpoint_Anima_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for Anima single-file checkpoint models (safetensors).

    Anima is built on NVIDIA Cosmos Predict2 DiT with a custom LLM Adapter
    that bridges Qwen3 0.6B text encoder outputs to the DiT.
    """

    base: Literal[BaseModelType.Anima] = Field(default=BaseModelType.Anima)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_anima_model(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_looks_like_anima_model(cls, mod: ModelOnDisk) -> None:
        has_anima_keys = _has_anima_keys(mod.load_state_dict())
        if not has_anima_keys:
            raise NotAMatchError("state dict does not look like an Anima model")


class Main_SDNQ_FLUX_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for SDNQ-quantized FLUX transformer models."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    format: Literal[ModelFormat.SDNQQuantized] = Field(default=ModelFormat.SDNQQuantized)

    variant: FluxVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_is_not_flux2(mod)

        cls._validate_looks_like_sdnq_quantized(mod)

        variant = override_fields.get("variant") or cls._get_variant_or_raise(mod)

        return cls(**override_fields, variant=variant)

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> FluxVariantType:
        state_dict = mod.load_state_dict()
        variant = _get_flux_variant(state_dict)

        if variant is None:
            raise NotAMatchError("unable to determine model variant from state dict")

        return variant

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatchError("state dict does not look like a main model")

    @classmethod
    def _validate_is_not_flux2(cls, mod: ModelOnDisk) -> None:
        """Reject FLUX.2 SDNQ checkpoints so they route to Main_SDNQ_Flux2_Config instead.

        This config only checks for generic main-model keys plus SDNQ keys, so without this guard a
        prefixed FLUX.2 SDNQ state dict could be accepted as base=flux and later loaded by the
        FLUX.1 loader.
        """
        state_dict = mod.load_state_dict()
        if _is_flux2_model(state_dict):
            raise NotAMatchError("model is a FLUX.2 model, not FLUX.1; use Main_SDNQ_Flux2_Config")

    @classmethod
    def _validate_looks_like_sdnq_quantized(cls, mod: ModelOnDisk) -> None:
        state_dict = mod.load_state_dict()
        if not _has_sdnq_keys(state_dict) and not _has_sdnq_tensors(state_dict):
            raise NotAMatchError("state dict does not look like SDNQ quantized")


class Main_SDNQ_Flux2_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for SDNQ-quantized FLUX.2 transformer models (e.g. Klein 4B / 9B)."""

    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    format: Literal[ModelFormat.SDNQQuantized] = Field(default=ModelFormat.SDNQQuantized)

    variant: Flux2VariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_is_flux2(mod)

        cls._validate_looks_like_sdnq_quantized(mod)

        variant = override_fields.pop("variant", None) or cls._get_variant_or_raise(mod)

        return cls(**override_fields, variant=variant)

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        state_dict = mod.load_state_dict()
        # Only the bare diffusers layout (transformer_blocks. / context_embedder.) is accepted: the
        # single-file SDNQ FLUX.2 loader consumes those keys directly and neither strips the
        # model.diffusion_model. prefix nor converts BFL double_blocks. keys. Accepting a BFL /
        # ComfyUI checkpoint here (via _has_main_keys) would classify a checkpoint the loader then
        # fails to load with missing/unexpected keys, so we reject it during identification instead.
        if _has_flux2_diffusers_transformer_keys(state_dict):
            return
        raise NotAMatchError(
            "state dict is not in the bare diffusers layout the single-file SDNQ FLUX.2 loader supports"
        )

    @classmethod
    def _validate_is_flux2(cls, mod: ModelOnDisk) -> None:
        state_dict = mod.load_state_dict()
        if not _is_flux2_model(state_dict):
            raise NotAMatchError("state dict does not look like a FLUX.2 model")

    @classmethod
    def _validate_looks_like_sdnq_quantized(cls, mod: ModelOnDisk) -> None:
        state_dict = mod.load_state_dict()
        if not _has_sdnq_keys(state_dict) and not _has_sdnq_tensors(state_dict):
            raise NotAMatchError("state dict does not look like SDNQ quantized")

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> Flux2VariantType:
        state_dict = mod.load_state_dict()
        variant = _get_flux2_variant(state_dict)

        if variant is None:
            raise NotAMatchError("unable to determine FLUX.2 model variant from state dict")

        if variant == Flux2VariantType.Klein9B and _filename_suggests_base(mod.name):
            return Flux2VariantType.Klein9BBase
        if variant == Flux2VariantType.Klein4B and _filename_suggests_base(mod.name):
            return Flux2VariantType.Klein4BBase

        return variant


class Main_SDNQ_ZImage_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for SDNQ-quantized Z-Image transformer models."""

    base: Literal[BaseModelType.ZImage] = Field(default=BaseModelType.ZImage)
    format: Literal[ModelFormat.SDNQQuantized] = Field(default=ModelFormat.SDNQQuantized)
    variant: ZImageVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_z_image_model(mod)

        cls._validate_looks_like_sdnq_quantized(mod)

        variant = override_fields.pop("variant", None) or ZImageVariantType.Turbo

        return cls(**override_fields, variant=variant)

    @classmethod
    def _validate_looks_like_z_image_model(cls, mod: ModelOnDisk) -> None:
        has_z_image_keys = _has_z_image_keys(mod.load_state_dict())
        if not has_z_image_keys:
            raise NotAMatchError("state dict does not look like a Z-Image model")

    @classmethod
    def _validate_looks_like_sdnq_quantized(cls, mod: ModelOnDisk) -> None:
        state_dict = mod.load_state_dict()
        if not _has_sdnq_keys(state_dict) and not _has_sdnq_tensors(state_dict):
            raise NotAMatchError("state dict does not look like SDNQ quantized")


# Tokenizer class names a Qwen3-encoder pipeline can advertise in model_index.json. Qwen models
# (incl. Qwen3) use the Qwen2 tokenizer classes, in slow and fast variants.
_QWEN_TOKENIZER_CLASS_NAMES = {"Qwen2Tokenizer", "Qwen2TokenizerFast"}

# Text-encoder class names the SDNQ pipeline loaders can actually instantiate. They build a text-only
# Qwen3ForCausalLM for the discovered text_encoder/ folder, so they can only load a Qwen3 model — not
# a Qwen2 causal LM (missing Qwen3 q/k-norm params) and not the multimodal Qwen2VLForConditionalGeneration
# (visual tower). Recording any of those as a self-contained TextEncoder would mark the pipeline
# complete even though the loader would fail, so discovery narrows to the loadable Qwen3 set.
_SDNQ_PIPELINE_TEXT_ENCODER_CLASS_NAMES = _SDNQ_LOADABLE_QWEN_ARCHITECTURES


# Files a pipeline component folder must actually ship to be loadable. Weight-bearing components
# (transformer / text_encoder / vae) are loaded with a `from_pretrained`-style call that needs both a
# config and at least one weight file; the tokenizer folder carries no weights, only its vocab/config.
_COMPONENT_CONFIG_FILENAMES = ("config.json", "model_index.json")
_COMPONENT_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")
# ...but a *quantized* component is read by `sdnq_sd_loader`, which globs `*.safetensors` and raises
# when it finds none. Anything else in an SDNQ folder is not a weight that component can be loaded
# from, so accepting it would record a submodel the loader is guaranteed to choke on.
_SDNQ_COMPONENT_WEIGHT_SUFFIXES = (".safetensors",)
# Submodel slots that ship vocab/config only. They need no weight file, and nothing can be
# mis-instantiated against them, so they are also exempt from the declared-class requirement.
_WEIGHTLESS_SUBMODEL_TYPES = frozenset({SubModelType.Tokenizer, SubModelType.Tokenizer2})

# FLUX.1 SDNQ pipeline components, keyed by the class name `model_index.json` advertises. The value
# is the slot it fills, the model type recorded for it, and the class names the component's own config
# may declare — the tokenizers list both spellings transformers writes.
# Slots whose SDNQ loader calls `sdnq_sd_loader` unconditionally, so the component must ship
# safetensors whatever its folder declares. FLUX.2 and Z-Image build their transformer and Qwen3
# text encoder that way; FLUX.1 does so only for the transformer, because its CLIP / T5 / VAE
# branch on `_is_sdnq_folder` and fall back to `from_pretrained`.
# The transformer is read with `sdnq_sd_loader` in every SDNQ pipeline, which is why it is also the
# slot identification validates before accepting the folder as one.
_SDNQ_TRANSFORMER_ONLY = frozenset({SubModelType.Transformer})
_SDNQ_PIPELINE_ALWAYS_SDNQ_LOADED = _SDNQ_TRANSFORMER_ONLY | {SubModelType.TextEncoder}
_SDNQ_FLUX1_ALWAYS_SDNQ_LOADED = _SDNQ_TRANSFORMER_ONLY

_SDNQ_FLUX1_COMPONENT_BY_CLASS_NAME: dict[str, tuple[SubModelType, ModelType, set[str]]] = {
    "FluxTransformer2DModel": (SubModelType.Transformer, ModelType.Main, {"FluxTransformer2DModel"}),
    "CLIPTextModel": (SubModelType.TextEncoder, ModelType.CLIPEmbed, {"CLIPTextModel"}),
    "T5EncoderModel": (SubModelType.TextEncoder2, ModelType.T5Encoder, {"T5EncoderModel"}),
    "AutoencoderKL": (SubModelType.VAE, ModelType.VAE, {"AutoencoderKL"}),
    "CLIPTokenizer": (SubModelType.Tokenizer, ModelType.CLIPEmbed, {"CLIPTokenizer", "CLIPTokenizerFast"}),
    "T5TokenizerFast": (SubModelType.Tokenizer2, ModelType.T5Encoder, {"T5Tokenizer", "T5TokenizerFast"}),
    "T5Tokenizer": (SubModelType.Tokenizer2, ModelType.T5Encoder, {"T5Tokenizer", "T5TokenizerFast"}),
}
_TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "spiece.model",
)


# Class names that legitimately identify each component of an SDNQ pipeline, used both to map a
# `model_index.json` entry onto a submodel type and to check the component's own config agrees.
# `AutoencoderKL` and `AutoencoderKLFlux2` are interchangeable spellings of the VAE slot.
_SDNQ_FLUX2_TRANSFORMER_CLASS_NAMES = {"Flux2Transformer2DModel"}
_SDNQ_ZIMAGE_TRANSFORMER_CLASS_NAMES = {"ZImageTransformer2DModel"}
_SDNQ_VAE_CLASS_NAMES = {"AutoencoderKLFlux2", "AutoencoderKL"}

# Keys a component config uses to name its own class. Transformers models use `architectures`,
# diffusers models `_class_name`, and tokenizers `tokenizer_class`.
_COMPONENT_CLASS_CONFIG_FILENAMES = ("config.json", "model_index.json", "tokenizer_config.json")


def _sdnq_component_declared_classes(component_path: Path) -> set[str]:
    """Class names the component's *own* config files declare, if any."""
    import json

    declared: set[str] = set()
    for filename in _COMPONENT_CLASS_CONFIG_FILENAMES:
        config_path = component_path / filename
        if not config_path.is_file():
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(config, dict):
            continue
        architectures = config.get("architectures")
        if isinstance(architectures, list):
            declared.update(name for name in architectures if isinstance(name, str))
        for key in ("_class_name", "tokenizer_class"):
            name = config.get(key)
            if isinstance(name, str):
                declared.add(name)
    return declared


def _sdnq_component_matches_advertised_class(
    component_path: Path, accepted_class_names: set[str], *, require_declaration: bool
) -> bool:
    """True if the component's own config agrees with the class `model_index.json` advertises for it.

    `model_index.json` records what a pipeline *claims* each component is, and that claim is what
    selects the loader — but nothing forces it to be true. A folder can advertise `Qwen3ForCausalLM`
    for `text_encoder/` and ship a Qwen2 or multimodal Qwen2-VL model there; the folder is populated,
    so a file-presence check passes and the pipeline is recorded as self-contained. The mismatch then
    only surfaces at generation time, when the loader builds a Qwen3ForCausalLM against a state dict
    that cannot satisfy it.

    With `require_declaration`, a component that declares *nothing* is rejected too, not just one that
    contradicts. Silence is not evidence: "some config.json plus some weight file" is exactly the
    shape an unrelated model has, so accepting it re-opens the hole for every component whose config
    happens to omit the key. `save_pretrained` writes `_class_name` (diffusers) or `architectures`
    (transformers) for anything it saves, so a weight-bearing component that names no class is not a
    normally-produced one. Set it for those; the tokenizer stays lenient because it carries no weights
    (nothing can be mis-instantiated against it) and `tokenizer_class` is less consistently written.

    Rejecting here is not fatal to the install: the component is left unrecorded, the pipeline is
    simply not self-contained, and the user wires that component up explicitly — which is the correct
    outcome for a folder we cannot confirm.
    """
    declared = _sdnq_component_declared_classes(component_path)
    if not declared:
        return not require_declaration
    return bool(declared & accepted_class_names)


def _sdnq_component_dir_is_populated(
    component_path: Path,
    submodel_type: SubModelType,
    always_sdnq_loaded: frozenset[SubModelType] = frozenset(),
) -> bool:
    """True if `component_path` holds the files the component's loader needs.

    An existing-but-empty folder is not enough: an interrupted download leaves the component
    directories created but empty (or holding only a partial file), and recording such a component as
    a submodel makes is_self_contained_sdnq_pipeline() report the pipeline as complete. Readiness then
    permits generation and the invocations select the main model as the component source, and the
    failure only surfaces when the loader tries to read the empty vae/ text_encoder/ tokenizer/ folder.

    "Files its loader needs" is per component, not one list for all of them. `sdnq_sd_loader` globs
    `*.safetensors` and raises if it finds none, so a `.gguf` or `.bin` in a component it reads is
    not a usable weight however well-formed the folder otherwise looks. A component loaded with
    `from_pretrained` instead — typically the VAE, which SDNQ exports leave in bfloat16 — may
    legitimately ship any of the formats diffusers reads.

    Which of the two applies is a property of the *loader*, not of the folder. `always_sdnq_loaded`
    names the slots whose loader calls `sdnq_sd_loader` unconditionally: FLUX.2 and Z-Image do that
    for the transformer and text encoder, FLUX.1 only for the transformer (its CLIP / T5 / VAE
    branch on the folder marker). Relying on `quantization_config.json` alone was not enough — a
    component carrying SDNQ weight/scale keys but no marker file reads as unquantized, so a `.bin`
    passed discovery and then failed in the loader that was always going to use `sdnq_sd_loader`.
    """
    if not component_path.is_dir():
        return False

    # Both tokenizer slots are weightless — FLUX.1 pipelines carry a CLIP tokenizer plus a T5
    # `tokenizer_2`, and requiring weights of either would reject every real one.
    if submodel_type in _WEIGHTLESS_SUBMODEL_TYPES:
        return any((component_path / name).is_file() for name in _TOKENIZER_FILENAMES)

    read_by_sdnq_loader = submodel_type in always_sdnq_loaded or _is_sdnq_folder(component_path)
    accepted_suffixes = _SDNQ_COMPONENT_WEIGHT_SUFFIXES if read_by_sdnq_loader else _COMPONENT_WEIGHT_SUFFIXES
    has_config = any((component_path / name).is_file() for name in _COMPONENT_CONFIG_FILENAMES)
    has_weights = any(entry.is_file() and entry.suffix in accepted_suffixes for entry in component_path.iterdir())
    return has_config and has_weights


class Main_SDNQ_Diffusers_Flux2_Config(Main_Config_Base, Config_Base):
    """Model config for SDNQ-quantized FLUX.2 models in diffusers format
    (Flux2KleinPipeline / Flux2Pipeline folder with transformer/, text_encoder/, vae/, ...)."""

    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    format: Literal[ModelFormat.SDNQQuantized] = Field(default=ModelFormat.SDNQQuantized)

    variant: Flux2VariantType = Field()
    repo_variant: ModelRepoVariant = Field(default=ModelRepoVariant.Default)
    submodels: dict[SubModelType, SubmodelDefinition] | None = Field(
        description="Loadable submodels in this model",
        default=None,
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_flux2_diffusers(mod)

        cls._validate_has_sdnq_transformer(mod)

        variant = override_fields.get("variant") or cls._get_variant_or_raise(mod)
        repo_variant = override_fields.get("repo_variant") or cls._get_repo_variant(mod)
        # Submodels are always rediscovered from disk, never replayed from a serialized config. A
        # persisted map records what existed at install time, so a component deleted since would
        # still be reported present and the pipeline still "self-contained" — and the failure would
        # surface only when a loader opens a folder that is gone. The filesystem is the authority.
        override_fields = {k: v for k, v in override_fields.items() if k != "submodels"}
        submodels = cls._get_submodels(mod)

        return cls(**override_fields, variant=variant, repo_variant=repo_variant, submodels=submodels)

    @classmethod
    def _validate_looks_like_flux2_diffusers(cls, mod: ModelOnDisk) -> None:
        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "Flux2Pipeline",
                "Flux2KleinPipeline",
                "Flux2Transformer2DModel",
            },
        )

    @classmethod
    def _validate_has_sdnq_transformer(cls, mod: ModelOnDisk) -> None:
        transformer_path = mod.path / "transformer"
        if not transformer_path.is_dir():
            raise NotAMatchError("no transformer subfolder found")

        if not _is_sdnq_folder(transformer_path):
            raise NotAMatchError("transformer is not SDNQ quantized")

        # ...and it must actually be loadable. A marker (or SDNQ-shaped config) over a folder with no
        # weight file the loader can read is not a pipeline: discovery would leave `Transformer` out
        # of `submodels` while this check still said "yes, SDNQ pipeline", so the model installed and
        # only failed when a loader opened the transformer path at generation time. Same predicate
        # discovery uses, so identification and discovery cannot disagree about the transformer.
        if not _sdnq_component_dir_is_populated(transformer_path, SubModelType.Transformer, _SDNQ_TRANSFORMER_ONLY):
            raise NotAMatchError("transformer folder holds no weights its loader can read")

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> Flux2VariantType:
        """Determine the Flux2 variant from the transformer config + filename heuristic."""
        transformer_config = get_config_dict_or_raise(mod.path / "transformer" / "config.json")

        # Read the geometry through `flux2_variant.py`, the single source of truth the checkpoint,
        # diffusers and LoRA paths already use. Hand-rolling the comparison here is what made every
        # SDNQ pipeline a Klein: [dev]'s 15360 / 6144 matched no branch and fell into the `else`,
        # so a [dev] pipeline was silently identified as Klein4B rather than as itself.
        joint_attention_dim = transformer_config.get("joint_attention_dim")
        hidden_size = transformer_config.get("attention_head_dim", 128) * transformer_config.get(
            "num_attention_heads", 24
        )
        variant = (
            (flux2_variant_from_context_dim(joint_attention_dim) if joint_attention_dim is not None else None)
            or flux2_variant_from_hidden_size(hidden_size)
            or Flux2VariantType.Klein4B
        )

        # Base variants are architecturally identical to their distilled counterpart, so only the
        # name separates them. [dev] has no base counterpart and is left alone.
        if _filename_suggests_base(mod.name):
            if variant == Flux2VariantType.Klein9B:
                return Flux2VariantType.Klein9BBase
            if variant == Flux2VariantType.Klein4B:
                return Flux2VariantType.Klein4BBase
        return variant

    @classmethod
    def _get_repo_variant(cls, mod: ModelOnDisk) -> ModelRepoVariant:
        weight_files = list(mod.path.glob("**/*.safetensors"))
        weight_files.extend(list(mod.path.glob("**/*.bin")))
        for x in weight_files:
            if ".fp16" in x.suffixes:
                return ModelRepoVariant.FP16
            if "openvino_model" in x.name:
                return ModelRepoVariant.OpenVINO
            if "flax_model" in x.name:
                return ModelRepoVariant.Flax
            if x.suffix == ".onnx":
                return ModelRepoVariant.ONNX
        return ModelRepoVariant.Default

    @classmethod
    def _get_submodels(cls, mod: ModelOnDisk) -> dict[SubModelType, SubmodelDefinition]:
        config = get_config_dict_or_raise(common_config_paths(mod.path))

        submodels: dict[SubModelType, SubmodelDefinition] = {}

        for key, value in config.items():
            if key.startswith("_") or not (isinstance(value, list) and len(value) == 2):
                continue

            _library_name, class_name = value

            if class_name is None:
                continue

            match class_name:
                case "Flux2Transformer2DModel":
                    submodel_type, model_type = SubModelType.Transformer, ModelType.Main
                    accepted_class_names = _SDNQ_FLUX2_TRANSFORMER_CLASS_NAMES
                case name if name in _SDNQ_PIPELINE_TEXT_ENCODER_CLASS_NAMES:
                    submodel_type, model_type = SubModelType.TextEncoder, ModelType.Qwen3Encoder
                    accepted_class_names = _SDNQ_PIPELINE_TEXT_ENCODER_CLASS_NAMES
                case name if name in _QWEN_TOKENIZER_CLASS_NAMES:
                    submodel_type, model_type = SubModelType.Tokenizer, ModelType.Qwen3Encoder
                    accepted_class_names = _QWEN_TOKENIZER_CLASS_NAMES
                case "AutoencoderKLFlux2" | "AutoencoderKL":
                    submodel_type, model_type = SubModelType.VAE, ModelType.VAE
                    accepted_class_names = _SDNQ_VAE_CLASS_NAMES
                case _:
                    continue

            # model_index.json only advertises which components a pipeline *should* have; a partial or
            # interrupted download can retain the original index while its component folders are
            # missing or still empty. Record a submodel only when the folder actually holds the files
            # its loader needs, otherwise a partial pipeline is treated as self-contained
            # (is_self_contained_sdnq_pipeline) and the loaders later request fixed vae/ text_encoder/
            # tokenizer/ subfolders that have nothing to load.
            component_path = mod.path / key
            if not _sdnq_component_dir_is_populated(component_path, submodel_type, _SDNQ_PIPELINE_ALWAYS_SDNQ_LOADED):
                continue

            # Populated is not the same as correct: the index's class name is a claim about the
            # folder, not a fact. Require the component's own config to agree before recording it.
            if not _sdnq_component_matches_advertised_class(
                component_path,
                accepted_class_names,
                require_declaration=submodel_type not in _WEIGHTLESS_SUBMODEL_TYPES,
            ):
                continue

            submodels[submodel_type] = SubmodelDefinition(
                path_or_prefix=component_path.resolve().as_posix(),
                model_type=model_type,
                variant=None,
            )

        return submodels


class Main_SDNQ_Diffusers_ZImage_Config(Main_Config_Base, Config_Base):
    """Model config for SDNQ-quantized Z-Image models in diffusers format (full ZImagePipeline folder)."""

    base: Literal[BaseModelType.ZImage] = Field(default=BaseModelType.ZImage)
    format: Literal[ModelFormat.SDNQQuantized] = Field(default=ModelFormat.SDNQQuantized)
    variant: ZImageVariantType = Field()

    repo_variant: ModelRepoVariant = Field(default=ModelRepoVariant.Default)
    submodels: dict[SubModelType, SubmodelDefinition] | None = Field(
        description="Loadable submodels in this model",
        default=None,
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_z_image_diffusers(mod)

        cls._validate_has_sdnq_transformer(mod)

        variant = override_fields.get("variant") or cls._get_variant_or_default(mod)

        repo_variant = override_fields.get("repo_variant") or cls._get_repo_variant(mod)

        # Submodels are always rediscovered from disk, never replayed from a serialized config. A
        # persisted map records what existed at install time, so a component deleted since would
        # still be reported present and the pipeline still "self-contained" — and the failure would
        # surface only when a loader opens a folder that is gone. The filesystem is the authority.
        override_fields = {k: v for k, v in override_fields.items() if k != "submodels"}
        submodels = cls._get_submodels(mod)

        return cls(**override_fields, variant=variant, repo_variant=repo_variant, submodels=submodels)

    @classmethod
    def _get_variant_or_default(cls, mod: ModelOnDisk) -> ZImageVariantType:
        """Determine Z-Image variant from the scheduler config (same heuristic as the unquantized diffusers config).

        Turbo (distilled) uses shift = 3.0, ZBase (undistilled) uses shift = 6.0.
        """
        try:
            scheduler_config = get_config_dict_or_raise(mod.path / "scheduler" / "scheduler_config.json")
            shift = scheduler_config.get("shift", 3.0)
        except NotAMatchError:
            return ZImageVariantType.Turbo
        return ZImageVariantType.ZBase if shift >= 5.0 else ZImageVariantType.Turbo

    @classmethod
    def _validate_looks_like_z_image_diffusers(cls, mod: ModelOnDisk) -> None:
        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "ZImagePipeline",
                "ZImageTransformer2DModel",
            },
        )

    @classmethod
    def _validate_has_sdnq_transformer(cls, mod: ModelOnDisk) -> None:
        transformer_path = mod.path / "transformer"
        if not transformer_path.is_dir():
            raise NotAMatchError("no transformer subfolder found")

        if not _is_sdnq_folder(transformer_path):
            raise NotAMatchError("transformer is not SDNQ quantized")

        # ...and it must actually be loadable. A marker (or SDNQ-shaped config) over a folder with no
        # weight file the loader can read is not a pipeline: discovery would leave `Transformer` out
        # of `submodels` while this check still said "yes, SDNQ pipeline", so the model installed and
        # only failed when a loader opened the transformer path at generation time. Same predicate
        # discovery uses, so identification and discovery cannot disagree about the transformer.
        if not _sdnq_component_dir_is_populated(transformer_path, SubModelType.Transformer, _SDNQ_TRANSFORMER_ONLY):
            raise NotAMatchError("transformer folder holds no weights its loader can read")

    @classmethod
    def _get_repo_variant(cls, mod: ModelOnDisk) -> ModelRepoVariant:
        weight_files = list(mod.path.glob("**/*.safetensors"))
        weight_files.extend(list(mod.path.glob("**/*.bin")))
        for x in weight_files:
            if ".fp16" in x.suffixes:
                return ModelRepoVariant.FP16
            if "openvino_model" in x.name:
                return ModelRepoVariant.OpenVINO
            if "flax_model" in x.name:
                return ModelRepoVariant.Flax
            if x.suffix == ".onnx":
                return ModelRepoVariant.ONNX
        return ModelRepoVariant.Default

    @classmethod
    def _get_submodels(cls, mod: ModelOnDisk) -> dict[SubModelType, SubmodelDefinition]:
        config = get_config_dict_or_raise(common_config_paths(mod.path))

        submodels: dict[SubModelType, SubmodelDefinition] = {}

        for key, value in config.items():
            if key.startswith("_") or not (isinstance(value, list) and len(value) == 2):
                continue

            _library_name, class_name = value

            if class_name is None:
                continue

            match class_name:
                case "ZImageTransformer2DModel":
                    submodel_type, model_type = SubModelType.Transformer, ModelType.Main
                    accepted_class_names = _SDNQ_ZIMAGE_TRANSFORMER_CLASS_NAMES
                case name if name in _SDNQ_PIPELINE_TEXT_ENCODER_CLASS_NAMES:
                    submodel_type, model_type = SubModelType.TextEncoder, ModelType.Qwen3Encoder
                    accepted_class_names = _SDNQ_PIPELINE_TEXT_ENCODER_CLASS_NAMES
                case name if name in _QWEN_TOKENIZER_CLASS_NAMES:
                    submodel_type, model_type = SubModelType.Tokenizer, ModelType.Qwen3Encoder
                    accepted_class_names = _QWEN_TOKENIZER_CLASS_NAMES
                case "AutoencoderKL":
                    submodel_type, model_type = SubModelType.VAE, ModelType.VAE
                    accepted_class_names = _SDNQ_VAE_CLASS_NAMES
                case _:
                    continue

            # See the FLUX.2 _get_submodels note: only record a component whose folder actually holds
            # the files its loader needs, so a partial download with a complete model_index.json isn't
            # mis-classified as a self-contained SDNQ pipeline.
            component_path = mod.path / key
            if not _sdnq_component_dir_is_populated(component_path, submodel_type, _SDNQ_PIPELINE_ALWAYS_SDNQ_LOADED):
                continue

            # ...and whose own config agrees with the class the index advertises for it.
            if not _sdnq_component_matches_advertised_class(
                component_path,
                accepted_class_names,
                require_declaration=submodel_type not in _WEIGHTLESS_SUBMODEL_TYPES,
            ):
                continue

            submodels[submodel_type] = SubmodelDefinition(
                path_or_prefix=component_path.resolve().as_posix(),
                model_type=model_type,
                variant=None,
            )

        return submodels


def _is_sdnq_folder(folder_path: Path) -> bool:
    """Check if a folder contains SDNQ-quantized model weights.

    Delegates to the shared detector so identification and the loaders cannot disagree about a
    markerless export — see `invokeai.backend.quantization.sdnq.detection`.
    """
    return is_sdnq_folder(folder_path)


class Main_SDNQ_Diffusers_FLUX_Config(Main_Config_Base, Config_Base):
    """Model config for SDNQ-quantized FLUX models in diffusers format (folder with transformer, text_encoder, etc.)."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    format: Literal[ModelFormat.SDNQQuantized] = Field(default=ModelFormat.SDNQQuantized)

    variant: FluxVariantType = Field()
    repo_variant: ModelRepoVariant = Field(default=ModelRepoVariant.Default)
    submodels: dict[SubModelType, SubmodelDefinition] | None = Field(
        description="Loadable submodels in this model",
        default=None,
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_flux_diffusers(mod)

        cls._validate_has_sdnq_transformer(mod)

        variant = override_fields.get("variant") or cls._get_variant_or_raise(mod)

        repo_variant = override_fields.get("repo_variant") or cls._get_repo_variant(mod)

        # Submodels are always rediscovered from disk, never replayed from a serialized config. A
        # persisted map records what existed at install time, so a component deleted since would
        # still be reported present and the pipeline still "self-contained" — and the failure would
        # surface only when a loader opens a folder that is gone. The filesystem is the authority.
        override_fields = {k: v for k, v in override_fields.items() if k != "submodels"}
        submodels = cls._get_submodels(mod)

        return cls(**override_fields, variant=variant, repo_variant=repo_variant, submodels=submodels)

    @classmethod
    def _validate_looks_like_flux_diffusers(cls, mod: ModelOnDisk) -> None:
        """Check if this looks like a Flux diffusers model by checking for FluxPipeline or FluxTransformer2DModel."""
        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "FluxPipeline",
                "FluxTransformer2DModel",
            },
        )

    @classmethod
    def _validate_has_sdnq_transformer(cls, mod: ModelOnDisk) -> None:
        """Check if the transformer subfolder contains SDNQ quantization."""
        transformer_path = mod.path / "transformer"
        if not transformer_path.is_dir():
            raise NotAMatchError("no transformer subfolder found")

        if not _is_sdnq_folder(transformer_path):
            raise NotAMatchError("transformer is not SDNQ quantized")

        # ...and it must actually be loadable. A marker (or SDNQ-shaped config) over a folder with no
        # weight file the loader can read is not a pipeline: discovery would leave `Transformer` out
        # of `submodels` while this check still said "yes, SDNQ pipeline", so the model installed and
        # only failed when a loader opened the transformer path at generation time. Same predicate
        # discovery uses, so identification and discovery cannot disagree about the transformer.
        if not _sdnq_component_dir_is_populated(transformer_path, SubModelType.Transformer, _SDNQ_TRANSFORMER_ONLY):
            raise NotAMatchError("transformer folder holds no weights its loader can read")

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> FluxVariantType:
        """Determine the Flux variant from the transformer config."""
        transformer_config = get_config_dict_or_raise(mod.path / "transformer" / "config.json")

        # Check for guidance_embeds to determine if it's Dev or Schnell
        guidance_embeds = transformer_config.get("guidance_embeds", False)
        in_channels = transformer_config.get("in_channels", 64)

        if guidance_embeds and in_channels == 384:
            return FluxVariantType.DevFill
        elif guidance_embeds:
            return FluxVariantType.Dev
        else:
            return FluxVariantType.Schnell

    @classmethod
    def _get_repo_variant(cls, mod: ModelOnDisk) -> ModelRepoVariant:
        """Determine the repo variant from the model files."""
        weight_files = list(mod.path.glob("**/*.safetensors"))
        weight_files.extend(list(mod.path.glob("**/*.bin")))
        for x in weight_files:
            if ".fp16" in x.suffixes:
                return ModelRepoVariant.FP16
            if "openvino_model" in x.name:
                return ModelRepoVariant.OpenVINO
            if "flax_model" in x.name:
                return ModelRepoVariant.Flax
            if x.suffix == ".onnx":
                return ModelRepoVariant.ONNX
        return ModelRepoVariant.Default

    @classmethod
    def _get_submodels(cls, mod: ModelOnDisk) -> dict[SubModelType, SubmodelDefinition]:
        """Extract submodels from model_index.json for Flux SDNQ diffusers format."""
        config = get_config_dict_or_raise(common_config_paths(mod.path))

        submodels: dict[SubModelType, SubmodelDefinition] = {}

        for key, value in config.items():
            # Skip metadata fields and invalid entries
            if key.startswith("_") or not (isinstance(value, list) and len(value) == 2):
                continue

            _library_name, class_name = value

            # Skip null entries
            if class_name is None:
                continue

            resolved = _SDNQ_FLUX1_COMPONENT_BY_CLASS_NAME.get(class_name)
            if resolved is None:
                continue
            submodel_type, model_type, accepted_class_names = resolved

            # Same two guards as the FLUX.2 / Z-Image pipelines: the index advertises what a pipeline
            # *should* have, so a partial download keeps a complete index over missing or empty
            # folders, and the advertised class is a claim about the folder rather than a fact.
            # Without these, a half-downloaded pipeline looks complete until generation time.
            component_path = mod.path / key
            if not _sdnq_component_dir_is_populated(component_path, submodel_type, _SDNQ_FLUX1_ALWAYS_SDNQ_LOADED):
                continue
            if not _sdnq_component_matches_advertised_class(
                component_path,
                accepted_class_names,
                require_declaration=submodel_type not in _WEIGHTLESS_SUBMODEL_TYPES,
            ):
                continue

            submodels[submodel_type] = SubmodelDefinition(
                path_or_prefix=component_path.resolve().as_posix(),
                model_type=model_type,
                variant=None,
            )

        return submodels


class Main_Diffusers_ErnieImage_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    """Model config for ERNIE-Image diffusers models (ERNIE-Image, ERNIE-Image-Turbo)."""

    base: Literal[BaseModelType.ErnieImage] = Field(BaseModelType.ErnieImage)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        raise_for_class_name(
            common_config_paths(mod.path),
            {"ErnieImagePipeline"},
        )

        repo_variant = override_fields.get("repo_variant") or cls._get_repo_variant_or_raise(mod)

        return cls(
            **override_fields,
            repo_variant=repo_variant,
        )


def _has_ernie_image_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains ERNIE-Image transformer keys.

    The single-file release carries the same keys as the diffusers checkpoint. Four of them
    together are the fingerprint: `x_embedder.proj` (the patch projection, a conv), `text_proj` (the
    Mistral3 conditioning projection), the model-level `adaLN_modulation.1` and `final_norm.linear`,
    the output modulation no sibling architecture spells that way. Anima also has an
    `x_embedder`, but it is identified by its `llm_adapter`, which ERNIE-Image does not have; the
    text projection keeps this clear of Wan (`text_embedding.0` / `condition_embedder`), Qwen Image
    (`txt_in`/`img_in`) and Z-Image (`cap_embedder`).
    """
    keys = state_dict.keys()
    return all(
        key in keys
        for key in (
            "x_embedder.proj.weight",
            "text_proj.weight",
            "adaLN_modulation.1.weight",
            "final_norm.linear.weight",
        )
    )


class Main_Checkpoint_ErnieImage_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for ERNIE-Image single-file checkpoint models (safetensors).

    ERNIE-Image and ERNIE-Image-Turbo share an architecture and a key layout, so a single file
    cannot be told apart from its weights. The architecture's default settings pick the Turbo
    numbers from the model name, exactly as they do for the diffusers pipelines.
    """

    base: Literal[BaseModelType.ErnieImage] = Field(default=BaseModelType.ErnieImage)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()

        if not _has_ernie_image_keys(state_dict):
            raise NotAMatchError("state dict does not look like an ERNIE-Image model")

        if _has_ggml_tensors(state_dict):
            raise NotAMatchError("state dict looks like GGUF quantized")

        return cls(**override_fields)
