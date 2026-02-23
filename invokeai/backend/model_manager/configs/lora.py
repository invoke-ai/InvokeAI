from abc import ABC
from pathlib import Path
from typing import (
    Any,
    Literal,
    Self,
)

from pydantic import BaseModel, ConfigDict, Field

from invokeai.backend.model_manager.configs.base import (
    Config_Base,
)
from invokeai.backend.model_manager.configs.controlnet import ControlAdapterDefaultSettings
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
    state_dict_has_any_keys_ending_with,
    state_dict_has_any_keys_starting_with,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.omi import flux_dev_1_lora, stable_diffusion_xl_1_lora
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    Flux2VariantType,
    FluxLoRAFormat,
    ModelFormat,
    ModelType,
    ZImageVariantType,
)
from invokeai.backend.model_manager.util.model_util import lora_token_vector_length
from invokeai.backend.patches.lora_conversions.flux_control_lora_utils import is_state_dict_likely_flux_control


class LoraModelDefaultSettings(BaseModel):
    weight: float | None = Field(default=None, ge=-1, le=2, description="Default weight for this model")
    model_config = ConfigDict(extra="forbid")


class LoRA_Config_Base(ABC, BaseModel):
    """Base class for LoRA models."""

    type: Literal[ModelType.LoRA] = Field(default=ModelType.LoRA)
    trigger_phrases: set[str] | None = Field(
        default=None,
        description="Set of trigger phrases for this model",
    )
    default_settings: LoraModelDefaultSettings | None = Field(
        default=None,
        description="Default settings for this model",
    )


def _get_flux_lora_format(mod: ModelOnDisk) -> FluxLoRAFormat | None:
    # TODO(psyche): Moving this import to the function to avoid circular imports. Refactor later.
    from invokeai.backend.patches.lora_conversions.formats import flux_format_from_state_dict

    state_dict = mod.load_state_dict()
    value = flux_format_from_state_dict(state_dict, mod.metadata())
    return value


# FLUX.2 Klein context_in_dim values: 3 * Qwen3 hidden_size
# Klein 4B: 3 * 2560 = 7680, Klein 9B: 3 * 4096 = 12288
_FLUX2_CONTEXT_IN_DIMS = {7680, 12288}

# FLUX.2 Klein vec_in_dim values: Qwen3 hidden_size
# Klein 4B: 2560 (Qwen3-4B), Klein 9B: 4096 (Qwen3-8B)
_FLUX2_VEC_IN_DIMS = {2560, 4096}

# FLUX.1 hidden_size is 3072. Klein 9B uses hidden_size=4096.
# Klein 4B also uses 3072, so hidden_size alone can't distinguish Klein 4B from FLUX.1.
_FLUX1_HIDDEN_SIZE = 3072

# FLUX.1 uses mlp_ratio=4 (ffn_dim=12288 for hidden_size=3072).
# Klein 4B uses mlp_ratio=6 (ffn_dim=18432 for hidden_size=3072).
_FLUX1_MLP_RATIO = 4


def _is_flux2_lora(mod: ModelOnDisk) -> bool:
    """Check if a FLUX-format LoRA is specifically for FLUX.2 (Klein) rather than FLUX.1.

    Detection is based on:
    1. Tensor shapes of embedding layers (context_embedder, vector_in) that differ between FLUX.1 and FLUX.2
    2. Hidden size of attention layers (3072 for FLUX.1/Klein 4B, 4096 for Klein 9B)

    Returns False for ambiguous LoRAs (e.g. Klein 4B transformer-only LoRAs with no distinguishing layers).
    """
    state_dict = mod.load_state_dict()
    return _is_flux2_lora_state_dict(state_dict)


def _is_flux2_lora_state_dict(state_dict: dict[str | int, Any]) -> bool:
    """Check state dict tensor shapes for FLUX.2 Klein-specific dimensions."""
    # Check diffusers/PEFT format keys (with various prefixes).
    # This covers both Flux.1 diffusers naming AND Flux2 Klein diffusers naming.
    for prefix in ["transformer.", "base_model.model.", ""]:
        # Check context_embedder (txt_in) dimensions
        # FLUX.1: context_in_dim=4096, FLUX.2 Klein 4B: 7680, Klein 9B: 12288
        ctx_key_a = f"{prefix}context_embedder.lora_A.weight"
        if ctx_key_a in state_dict:
            return state_dict[ctx_key_a].shape[1] in _FLUX2_CONTEXT_IN_DIMS

        # Check vector_in (time_text_embed.text_embedder) dimensions
        # FLUX.1: vec_in_dim=768, FLUX.2 Klein 4B: 2560, Klein 9B: 4096
        vec_key_a = f"{prefix}time_text_embed.text_embedder.linear_1.lora_A.weight"
        if vec_key_a in state_dict:
            return state_dict[vec_key_a].shape[1] in _FLUX2_VEC_IN_DIMS

        # Check Flux2 Klein diffusers naming: fused QKV+MLP in single blocks.
        # This key only exists in Flux2 models (Flux.1 uses separate to_q/to_k/to_v + proj_mlp).
        fused_key_a = f"{prefix}single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight"
        if fused_key_a in state_dict:
            return True

        # Check Flux2 Klein diffusers naming: ff.linear_in (Flux.1 uses ff.net.0.proj).
        ff_key_a = f"{prefix}transformer_blocks.0.ff.linear_in.lora_A.weight"
        if ff_key_a in state_dict:
            return True

    # Check BFL PEFT format (diffusion_model.* or base_model.model.* prefix with BFL layer names).
    # Klein 9B has hidden_size=4096 (vs 3072 for FLUX.1 and Klein 4B).
    # Klein 4B has same hidden_size as FLUX.1 (3072) but different mlp_ratio (6 vs 4),
    # and different txt_in/vector_in dimensions.
    _bfl_prefixes = ("diffusion_model.", "base_model.model.")
    bfl_hidden_size: int | None = None
    for key in state_dict:
        if not isinstance(key, str):
            continue
        if not key.startswith(_bfl_prefixes):
            continue

        # BFL PEFT: attention projection → check hidden_size
        if key.endswith(".img_attn.proj.lora_A.weight"):
            bfl_hidden_size = state_dict[key].shape[1]
            if bfl_hidden_size != _FLUX1_HIDDEN_SIZE:
                return True
            # hidden_size=3072 is ambiguous (could be Klein 4B or FLUX.1), keep checking

        # BFL PEFT: context_embedder/txt_in
        elif "txt_in" in key and key.endswith("lora_A.weight"):
            return state_dict[key].shape[1] in _FLUX2_CONTEXT_IN_DIMS

        # BFL PEFT: vector_in
        elif "vector_in" in key and key.endswith("lora_A.weight"):
            return state_dict[key].shape[1] in _FLUX2_VEC_IN_DIMS

    # BFL PEFT: hidden_size matches FLUX.1. Check MLP ratio to distinguish Klein 4B.
    # Klein 4B uses mlp_ratio=6 (ffn_dim=18432), FLUX.1 uses mlp_ratio=4 (ffn_dim=12288).
    if bfl_hidden_size == _FLUX1_HIDDEN_SIZE:
        for key in state_dict:
            if not isinstance(key, str):
                continue
            if key.startswith(_bfl_prefixes) and key.endswith(".img_mlp.0.lora_B.weight"):
                ffn_dim = state_dict[key].shape[0]
                if ffn_dim != bfl_hidden_size * _FLUX1_MLP_RATIO:
                    return True
                break

    # Check kohya format: look for context_embedder or vector_in keys
    # Kohya format uses lora_unet_ prefix with underscores instead of dots
    for key in state_dict:
        if not isinstance(key, str):
            continue
        if key.startswith("lora_unet_txt_in.") or key.startswith("lora_unet_context_embedder."):
            if key.endswith("lora_down.weight"):
                return state_dict[key].shape[1] in _FLUX2_CONTEXT_IN_DIMS
        if key.startswith("lora_unet_vector_in.") or key.startswith("lora_unet_time_text_embed_text_embedder_"):
            if key.endswith("lora_down.weight"):
                return state_dict[key].shape[1] in _FLUX2_VEC_IN_DIMS

    return False


def _get_flux2_lora_variant(state_dict: dict[str | int, Any]) -> Flux2VariantType | None:
    """Determine FLUX.2 Klein variant (4B vs 9B) from a LoRA state dict.

    Detection is based on tensor dimensions that differ between Klein 4B and Klein 9B:
    - hidden_size from attention projection: 3072 = Klein 4B, 4096 = Klein 9B
    - context_in_dim from context embedder: 7680 = Klein 4B, 12288 = Klein 9B
    - vec_in_dim from vector embedder: 2560 = Klein 4B, 4096 = Klein 9B

    Returns None if the variant cannot be determined (e.g. LoRA only targets layers
    with identical dimensions across variants).
    """
    KLEIN_4B_CONTEXT_DIM = 7680  # 3 * 2560
    KLEIN_9B_CONTEXT_DIM = 12288  # 3 * 4096
    KLEIN_4B_VEC_DIM = 2560
    KLEIN_9B_VEC_DIM = 4096
    KLEIN_4B_HIDDEN_SIZE = 3072
    KLEIN_9B_HIDDEN_SIZE = 4096

    # Check diffusers/PEFT format keys
    for prefix in ["transformer.", "base_model.model.", ""]:
        # Context embedder (txt_in) dimensions
        ctx_key_a = f"{prefix}context_embedder.lora_A.weight"
        if ctx_key_a in state_dict:
            dim = state_dict[ctx_key_a].shape[1]
            if dim == KLEIN_4B_CONTEXT_DIM:
                return Flux2VariantType.Klein4B
            if dim == KLEIN_9B_CONTEXT_DIM:
                return Flux2VariantType.Klein9B
            return None

        # Vector embedder dimensions
        vec_key_a = f"{prefix}time_text_embed.text_embedder.linear_1.lora_A.weight"
        if vec_key_a in state_dict:
            dim = state_dict[vec_key_a].shape[1]
            if dim == KLEIN_4B_VEC_DIM:
                return Flux2VariantType.Klein4B
            if dim == KLEIN_9B_VEC_DIM:
                return Flux2VariantType.Klein9B
            return None

        # Attention projection hidden_size (Flux.1 diffusers naming)
        attn_key_a = f"{prefix}transformer_blocks.0.attn.to_out.0.lora_A.weight"
        if attn_key_a in state_dict:
            dim = state_dict[attn_key_a].shape[1]
            if dim == KLEIN_4B_HIDDEN_SIZE:
                return Flux2VariantType.Klein4B
            if dim == KLEIN_9B_HIDDEN_SIZE:
                return Flux2VariantType.Klein9B
            return None

        # Attention projection hidden_size (Flux2 Klein diffusers naming)
        attn_key_a2 = f"{prefix}transformer_blocks.0.attn.to_add_out.lora_A.weight"
        if attn_key_a2 in state_dict:
            dim = state_dict[attn_key_a2].shape[1]
            if dim == KLEIN_4B_HIDDEN_SIZE:
                return Flux2VariantType.Klein4B
            if dim == KLEIN_9B_HIDDEN_SIZE:
                return Flux2VariantType.Klein9B
            return None

        # Fused QKV+MLP hidden_size (Flux2 Klein diffusers naming)
        fused_key_a = f"{prefix}single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight"
        if fused_key_a in state_dict:
            dim = state_dict[fused_key_a].shape[1]
            if dim == KLEIN_4B_HIDDEN_SIZE:
                return Flux2VariantType.Klein4B
            if dim == KLEIN_9B_HIDDEN_SIZE:
                return Flux2VariantType.Klein9B
            return None

    # Check BFL PEFT format (diffusion_model.* or base_model.model.* prefix with BFL names)
    _bfl_prefixes = ("diffusion_model.", "base_model.model.")
    for key in state_dict:
        if not isinstance(key, str):
            continue
        if not key.startswith(_bfl_prefixes):
            continue

        # BFL PEFT: context embedder (txt_in)
        if "txt_in" in key and key.endswith("lora_A.weight"):
            dim = state_dict[key].shape[1]
            if dim == KLEIN_4B_CONTEXT_DIM:
                return Flux2VariantType.Klein4B
            if dim == KLEIN_9B_CONTEXT_DIM:
                return Flux2VariantType.Klein9B
            return None

        # BFL PEFT: vector embedder (vector_in)
        if "vector_in" in key and key.endswith("lora_A.weight"):
            dim = state_dict[key].shape[1]
            if dim == KLEIN_4B_VEC_DIM:
                return Flux2VariantType.Klein4B
            if dim == KLEIN_9B_VEC_DIM:
                return Flux2VariantType.Klein9B
            return None

        # BFL PEFT: attention projection
        if key.endswith(".img_attn.proj.lora_A.weight"):
            dim = state_dict[key].shape[1]
            if dim == KLEIN_4B_HIDDEN_SIZE:
                return Flux2VariantType.Klein4B
            if dim == KLEIN_9B_HIDDEN_SIZE:
                return Flux2VariantType.Klein9B
            return None

    # Check kohya format
    for key in state_dict:
        if not isinstance(key, str):
            continue
        if key.startswith("lora_unet_txt_in.") or key.startswith("lora_unet_context_embedder."):
            if key.endswith("lora_down.weight"):
                dim = state_dict[key].shape[1]
                if dim == KLEIN_4B_CONTEXT_DIM:
                    return Flux2VariantType.Klein4B
                if dim == KLEIN_9B_CONTEXT_DIM:
                    return Flux2VariantType.Klein9B
                return None
        if key.startswith("lora_unet_vector_in.") or key.startswith("lora_unet_time_text_embed_text_embedder_"):
            if key.endswith("lora_down.weight"):
                dim = state_dict[key].shape[1]
                if dim == KLEIN_4B_VEC_DIM:
                    return Flux2VariantType.Klein4B
                if dim == KLEIN_9B_VEC_DIM:
                    return Flux2VariantType.Klein9B
                return None

    return None


class LoRA_OMI_Config_Base(LoRA_Config_Base):
    format: Literal[ModelFormat.OMI] = Field(default=ModelFormat.OMI)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_omi_lora(mod)

        cls._validate_base(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _validate_looks_like_omi_lora(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model metadata does not look like an OMI LoRA."""
        flux_format = _get_flux_lora_format(mod)
        if flux_format in [FluxLoRAFormat.Control, FluxLoRAFormat.Diffusers]:
            raise NotAMatchError("model looks like ControlLoRA or Diffusers LoRA")

        metadata = mod.metadata()

        metadata_looks_like_omi_lora = (
            bool(metadata.get("modelspec.sai_model_spec"))
            and metadata.get("ot_branch") == "omi_format"
            and metadata.get("modelspec.architecture", "").split("/")[1].lower() == "lora"
        )

        if not metadata_looks_like_omi_lora:
            raise NotAMatchError("metadata does not look like OMI LoRA")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> Literal[BaseModelType.Flux, BaseModelType.StableDiffusionXL]:
        metadata = mod.metadata()
        architecture = metadata["modelspec.architecture"]

        if architecture == stable_diffusion_xl_1_lora:
            return BaseModelType.StableDiffusionXL
        elif architecture == flux_dev_1_lora:
            return BaseModelType.Flux
        else:
            raise NotAMatchError(f"unrecognised/unsupported architecture for OMI LoRA: {architecture}")


class LoRA_OMI_SDXL_Config(LoRA_OMI_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class LoRA_OMI_FLUX_Config(LoRA_OMI_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class LoRA_LyCORIS_Config_Base(LoRA_Config_Base):
    """Model config for LoRA/Lycoris models."""

    type: Literal[ModelType.LoRA] = Field(default=ModelType.LoRA)
    format: Literal[ModelFormat.LyCORIS] = Field(default=ModelFormat.LyCORIS)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_lora(mod)

        cls._validate_base(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _validate_looks_like_lora(cls, mod: ModelOnDisk) -> None:
        # First rule out ControlLoRA
        flux_format = _get_flux_lora_format(mod)
        if flux_format in [FluxLoRAFormat.Control]:
            raise NotAMatchError("model looks like Control LoRA")

        # If it's a recognized Flux LoRA format (Kohya, Diffusers, OneTrainer, AIToolkit, XLabs, etc.),
        # it's valid and we skip the heuristic check
        if flux_format is not None:
            return

        # Note: Existence of these key prefixes/suffixes does not guarantee that this is a LoRA.
        # Some main models have these keys, likely due to the creator merging in a LoRA.
        has_key_with_lora_prefix = state_dict_has_any_keys_starting_with(
            mod.load_state_dict(),
            {
                "lora_te_",
                "lora_unet_",
                "lora_te1_",
                "lora_te2_",
                "lora_transformer_",
            },
        )

        has_key_with_lora_suffix = state_dict_has_any_keys_ending_with(
            mod.load_state_dict(),
            {
                "to_k_lora.up.weight",
                "to_q_lora.down.weight",
                "lora_A.weight",
                "lora_B.weight",
            },
        )

        if not has_key_with_lora_prefix and not has_key_with_lora_suffix:
            raise NotAMatchError("model does not match LyCORIS LoRA heuristics")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        if _get_flux_lora_format(mod):
            if _is_flux2_lora(mod):
                return BaseModelType.Flux2
            return BaseModelType.Flux

        state_dict = mod.load_state_dict()
        # If we've gotten here, we assume that the model is a Stable Diffusion model
        token_vector_length = lora_token_vector_length(state_dict)
        if token_vector_length == 768:
            return BaseModelType.StableDiffusion1
        elif token_vector_length == 1024:
            return BaseModelType.StableDiffusion2
        elif token_vector_length == 1280:
            return BaseModelType.StableDiffusionXL  # recognizes format at https://civitai.com/models/224641
        elif token_vector_length == 2048:
            return BaseModelType.StableDiffusionXL
        else:
            raise NotAMatchError(f"unrecognized token vector length {token_vector_length}")


class LoRA_LyCORIS_SD1_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class LoRA_LyCORIS_SD2_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class LoRA_LyCORIS_SDXL_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class LoRA_LyCORIS_FLUX_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class LoRA_LyCORIS_Flux2_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    """Model config for FLUX.2 (Klein) LoRA models in LyCORIS format."""

    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    variant: Flux2VariantType | None = Field(default=None)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        raise_for_override_fields(cls, override_fields)
        cls._validate_looks_like_lora(mod)
        cls._validate_base(mod)
        override_fields.setdefault("variant", _get_flux2_lora_variant(mod.load_state_dict()))
        return cls(**override_fields)

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        if _get_flux_lora_format(mod) and _is_flux2_lora(mod):
            return BaseModelType.Flux2
        raise NotAMatchError("model is not a FLUX.2 LoRA")


class LoRA_LyCORIS_ZImage_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    """Model config for Z-Image LoRA models in LyCORIS format."""

    base: Literal[BaseModelType.ZImage] = Field(default=BaseModelType.ZImage)
    variant: ZImageVariantType | None = Field(default=None)

    @classmethod
    def _validate_looks_like_lora(cls, mod: ModelOnDisk) -> None:
        """Z-Image LoRAs have different key patterns than SD/SDXL LoRAs.

        Z-Image LoRAs use keys like:
        - diffusion_model.layers.X.attention.to_k.lora_down.weight (DoRA format)
        - diffusion_model.layers.X.attention.to_k.lora_A.weight (PEFT format)
        - diffusion_model.layers.X.attention.to_k.dora_scale (DoRA scale)
        """
        state_dict = mod.load_state_dict()

        # Check for Z-Image specific LoRA patterns
        has_z_image_lora_keys = state_dict_has_any_keys_starting_with(
            state_dict,
            {
                "diffusion_model.layers.",  # Z-Image S3-DiT layer pattern
            },
        )

        # Also check for LoRA weight suffixes (various formats)
        has_lora_suffix = state_dict_has_any_keys_ending_with(
            state_dict,
            {
                "lora_A.weight",
                "lora_B.weight",
                "lora_down.weight",
                "lora_up.weight",
                "dora_scale",
            },
        )

        if has_z_image_lora_keys and has_lora_suffix:
            return

        raise NotAMatchError("model does not match Z-Image LoRA heuristics")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        """Z-Image LoRAs are identified by their diffusion_model.layers structure.

        Z-Image uses S3-DiT architecture with layer names like:
        - diffusion_model.layers.0.attention.to_k.lora_A.weight
        - diffusion_model.layers.0.feed_forward.w1.lora_A.weight
        """
        state_dict = mod.load_state_dict()

        # Check for Z-Image transformer layer patterns
        # Z-Image uses diffusion_model.layers.X structure (unlike Flux which uses double_blocks/single_blocks)
        has_z_image_keys = state_dict_has_any_keys_starting_with(
            state_dict,
            {
                "diffusion_model.layers.",  # Z-Image S3-DiT layer pattern
            },
        )

        # If it looks like a Z-Image LoRA, return ZImage base
        if has_z_image_keys:
            return BaseModelType.ZImage

        raise NotAMatchError("model does not look like a Z-Image LoRA")


class ControlAdapter_Config_Base(ABC, BaseModel):
    default_settings: ControlAdapterDefaultSettings | None = Field(None)


class ControlLoRA_LyCORIS_FLUX_Config(ControlAdapter_Config_Base, Config_Base):
    """Model config for Control LoRA models."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    type: Literal[ModelType.ControlLoRa] = Field(default=ModelType.ControlLoRa)
    format: Literal[ModelFormat.LyCORIS] = Field(default=ModelFormat.LyCORIS)

    trigger_phrases: set[str] | None = Field(None)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_control_lora(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_looks_like_control_lora(cls, mod: ModelOnDisk) -> None:
        state_dict = mod.load_state_dict()

        if not is_state_dict_likely_flux_control(state_dict):
            raise NotAMatchError("model state dict does not look like a Flux Control LoRA")


class LoRA_Diffusers_Config_Base(LoRA_Config_Base):
    """Model config for LoRA/Diffusers models."""

    # TODO(psyche): Needs base handling. For FLUX, the Diffusers format does not indicate a folder model; it indicates
    # the weights format. FLUX Diffusers LoRAs are single files.

    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_base(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        if _get_flux_lora_format(mod):
            if _is_flux2_lora(mod):
                return BaseModelType.Flux2
            return BaseModelType.Flux

        # If we've gotten here, we assume that the LoRA is a Stable Diffusion LoRA
        path_to_weight_file = cls._get_weight_file_or_raise(mod)
        state_dict = mod.load_state_dict(path_to_weight_file)
        token_vector_length = lora_token_vector_length(state_dict)

        match token_vector_length:
            case 768:
                return BaseModelType.StableDiffusion1
            case 1024:
                return BaseModelType.StableDiffusion2
            case 1280:
                return BaseModelType.StableDiffusionXL  # recognizes format at https://civitai.com/models/224641
            case 2048:
                return BaseModelType.StableDiffusionXL
            case _:
                raise NotAMatchError(f"unrecognized token vector length {token_vector_length}")

    @classmethod
    def _get_weight_file_or_raise(cls, mod: ModelOnDisk) -> Path:
        suffixes = ["bin", "safetensors"]
        weight_files = [mod.path / f"pytorch_lora_weights.{sfx}" for sfx in suffixes]
        for wf in weight_files:
            if wf.exists():
                return wf
        raise NotAMatchError("missing pytorch_lora_weights.bin or pytorch_lora_weights.safetensors")


class LoRA_Diffusers_SD1_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class LoRA_Diffusers_SD2_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class LoRA_Diffusers_SDXL_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class LoRA_Diffusers_FLUX_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class LoRA_Diffusers_Flux2_Config(LoRA_Diffusers_Config_Base, Config_Base):
    """Model config for FLUX.2 (Klein) LoRA models in Diffusers format."""

    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    variant: Flux2VariantType | None = Field(default=None)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)
        raise_for_override_fields(cls, override_fields)
        cls._validate_base(mod)
        path_to_weight_file = cls._get_weight_file_or_raise(mod)
        state_dict = mod.load_state_dict(path_to_weight_file)
        override_fields.setdefault("variant", _get_flux2_lora_variant(state_dict))
        return cls(**override_fields)

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        path_to_weight_file = cls._get_weight_file_or_raise(mod)
        state_dict = mod.load_state_dict(path_to_weight_file)
        if _is_flux2_lora_state_dict(state_dict):
            return BaseModelType.Flux2
        raise NotAMatchError("model is not a FLUX.2 Diffusers LoRA")


class LoRA_Diffusers_ZImage_Config(LoRA_Diffusers_Config_Base, Config_Base):
    """Model config for Z-Image LoRA models in Diffusers format."""

    base: Literal[BaseModelType.ZImage] = Field(default=BaseModelType.ZImage)
    variant: ZImageVariantType | None = Field(default=None)
