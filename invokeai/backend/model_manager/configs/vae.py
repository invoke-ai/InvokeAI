import re
from typing import (
    Literal,
    Self,
)

from pydantic import Field
from typing_extensions import Any

from invokeai.backend.flux.util import is_flux_family_vae_config
from invokeai.backend.model_manager.configs.backbone_names import backbone_from_components, name_components
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    common_config_paths,
    get_config_dict_or_raise,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
    state_dict_has_any_keys_starting_with,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)
from invokeai.backend.sd3.vae import is_sd3_vae_config

REGEX_TO_BASE: dict[str, BaseModelType] = {
    r"xl": BaseModelType.StableDiffusionXL,
    r"sd2": BaseModelType.StableDiffusion2,
    r"vae": BaseModelType.StableDiffusion1,
    r"FLUX.1-schnell_ae": BaseModelType.Flux,
}

# Standalone `AutoencoderKL` VAEs by latent width. Within one of these families the weights cannot say
# which base a VAE belongs to: SD1, SD2 and SDXL share one network, and FLUX.1 and SD3 share another
# (same 244 tensors, same shapes). So an explicit `base` override, a folder's `config.json` or the name
# has to decide within a family. Across families the weights are authoritative: nothing files a
# 16-channel VAE under SD1.
_VAE_FAMILIES: dict[int, frozenset[BaseModelType]] = {
    4: frozenset({BaseModelType.StableDiffusion1, BaseModelType.StableDiffusion2, BaseModelType.StableDiffusionXL}),
    16: frozenset({BaseModelType.Flux, BaseModelType.StableDiffusion3}),
}


def _override_fits_latent_width(
    override_fields: dict[str, Any], base: BaseModelType, latent_channels: int | None
) -> bool:
    """Whether an explicit `base` override decides the match for a VAE of this latent width.

    `raise_for_override_fields` has already held the override to the candidate class's `base`, so this
    only asks whether the weights allow it: the override chooses within the family they establish, and
    never files a VAE whose latent width no family has under a base it cannot load as.
    """
    if override_fields.get("base") is None or latent_channels is None:
        return False
    family = _VAE_FAMILIES.get(latent_channels)
    return family is not None and base in family


def _is_qwen_image_vae(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict is a Qwen Image VAE (AutoencoderKLQwenImage).

    Qwen Image VAE can be identified by:
    1. Diffusers-format encoder/decoder keys (`encoder.conv_in`, `decoder.conv_in`)
    2. 5-dimensional convolution weights (3D causal convolutions vs. standard 2D conv in SD/SDXL/FLUX VAEs)
    3. 16-dimensional latent space (z_dim=16)

    Note: Wan 2.2 A14B reuses the same architecture (AutoencoderKLWan with z_dim=16),
    so this function returns True for both. Disambiguation between the two for
    standalone files relies on the filename heuristic in :func:`_is_wan_vae` and
    config registration order.
    """
    decoder_conv_in_key = "decoder.conv_in.weight"
    if decoder_conv_in_key not in state_dict:
        return False
    weight = state_dict[decoder_conv_in_key]
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) != 5:
        return False
    # z_dim is the input channel dim of decoder.conv_in
    return shape[1] == 16


def _wan_vae_z_dim(state_dict: dict[str | int, Any]) -> int | None:
    """Return ``z_dim`` for a Wan-family VAE state dict, or ``None`` if it isn't one.

    Wan-family VAEs (AutoencoderKLWan) have 5D convolution weights and a
    decoder.conv_in input channel count of 16 (Wan 2.1 / A14B / Qwen Image) or
    48 (Wan 2.2 TI2V-5B's Wan2.2-VAE).
    """
    decoder_conv_in_key = "decoder.conv_in.weight"
    if decoder_conv_in_key not in state_dict:
        return None
    weight = state_dict[decoder_conv_in_key]
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) != 5:
        return None
    z = int(shape[1])
    return z if z in (16, 48) else None


def _filename_suggests_wan(mod: ModelOnDisk) -> bool:
    """Filename heuristic to distinguish standalone Wan VAE files from Qwen Image VAEs.

    Both use the same ``AutoencoderKLWan`` architecture for 16-channel files, so the
    state dict alone can't tell them apart. Filenames in the wild (community ports,
    ComfyUI repacks) typically include ``wan`` for Wan releases.
    """
    return "wan" in mod.path.name.lower()


def _latent_channels(state_dict: dict[str | int, Any]) -> int | None:
    """A 2-D autoencoder's latent width: the input channels of its first decoder convolution."""
    weight = state_dict.get("decoder.conv_in.weight")
    return None if weight is None else int(weight.shape[1])


def _is_qwen_image21_vae(state_dict: dict[str | int, Any]) -> bool:
    """Whether the state dict is Qwen-Image-2.1's RGBA VAE (AutoencoderKLQwenImage21), in either layout.

    It decodes 64 latent channels into 4 (RGBA). Its convolutions are 2-D: the diffusers export stores
    them as 4-D weights, ComfyUI's Wan-style export as 5-D with a temporal extent of 1. That temporal
    extent is what separates the ComfyUI file from a Wan or Anima VAE, whose causal convolutions are 3 deep.
    """

    def shape(key: str) -> tuple[int, ...] | None:
        weight = state_dict.get(key)
        return None if weight is None else tuple(weight.shape)

    diffusers_in, diffusers_out = shape("decoder.conv_in.weight"), shape("encoder.conv_in.weight")
    if diffusers_in is not None and diffusers_out is not None:
        return len(diffusers_in) == 4 and diffusers_in[1] == 64 and diffusers_out[1] == 4
    comfy_in, comfy_out = shape("decoder.conv1.weight"), shape("encoder.conv1.weight")
    if comfy_in is not None and comfy_out is not None:
        return len(comfy_in) == 5 and comfy_in[1] == 64 and comfy_in[2] == 1 and comfy_out[1] == 4
    return False


def _is_flux2_vae(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict is a FLUX.2 VAE (AutoencoderKLFlux2).

    FLUX.2 VAE can be identified by:
    1. Batch Normalization layers (bn.running_mean, bn.running_var) - unique to FLUX.2
    2. 32-dimensional latent space (decoder.conv_in has 32 input channels)

    FLUX.1 VAE has 16-dimensional latent space and no BatchNorm layers.
    """
    # Check for BN layer which is unique to FLUX.2 VAE
    has_bn = "bn.running_mean" in state_dict or "bn.running_var" in state_dict

    # Check for 32-channel latent space (FLUX.2 has 32, FLUX.1 has 16)
    decoder_conv_in_key = "decoder.conv_in.weight"
    has_32_latent_channels = decoder_conv_in_key in state_dict and state_dict[decoder_conv_in_key].shape[1] == 32

    return has_bn or has_32_latent_channels


class VAE_Checkpoint_Config_Base(Checkpoint_Config_Base):
    """Model config for standalone VAE models."""

    type: Literal[ModelType.VAE] = Field(default=ModelType.VAE)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_vae(mod)

        cls._validate_base(mod, override_fields)

        return cls(**override_fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> None:
        """Raise `NotAMatch` if the model base does not match this config class.

        A single file carries weights and nothing else, so within a latent family (see
        `_VAE_FAMILIES`) an explicit `base` override outranks anything the name suggests. It used to be
        validated and then overruled here, which filed an SD2 VAE installed as `sd-2` as Unknown.
        """
        expected_base = cls.model_fields["base"].default
        if _override_fits_latent_width(override_fields, expected_base, _latent_channels(mod.load_state_dict())):
            return
        recognized_base = cls._get_base_or_raise(mod, override_fields)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _validate_looks_like_vae(cls, mod: ModelOnDisk) -> None:
        state_dict = mod.load_state_dict()
        if not state_dict_has_any_keys_starting_with(
            state_dict,
            {
                "encoder.conv_in",
                "decoder.conv_in",
            },
        ):
            raise NotAMatchError("model does not match Checkpoint VAE heuristics")

        # Exclude FLUX.2 VAEs - they have their own config class
        if _is_flux2_vae(state_dict):
            raise NotAMatchError("model is a FLUX.2 VAE, not a standard VAE")

        # Exclude Qwen Image / Wan VAEs - they share the AutoencoderKLWan
        # architecture and each has its own config class.
        if _is_qwen_image_vae(state_dict) or _wan_vae_z_dim(state_dict) is not None:
            raise NotAMatchError("model is a Wan-family VAE, not a standard VAE")

        # A latent width no AutoencoderKL family has is not one of these VAEs, whatever its name says.
        # Without this, `_get_base_or_raise` falls through to the name and files a 64-channel
        # `qwen_image_2.1_vae.safetensors` as SD1 because the name contains "vae".
        latent_channels = _latent_channels(state_dict)
        if latent_channels is not None and latent_channels not in _VAE_FAMILIES:
            raise NotAMatchError(f"{latent_channels} latent channels is not a standard VAE")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> BaseModelType:
        # First, try to identify by latent space dimensions (most reliable)
        latent_channels = _latent_channels(mod.load_state_dict())
        if latent_channels == 16:
            # FLUX.1 or SD3, which only the name can tell apart. One naming neither -- the BFL
            # `ae.safetensors` among them -- stays FLUX.1, as every 16-channel VAE was filed before SD3
            # had a class. A name outside the family (a 16-channel file called "sdxl_vae") contradicts
            # the weights and is ignored rather than obeyed.
            named = backbone_from_components(name_components(mod, override_fields))
            return named if named in _VAE_FAMILIES[16] else BaseModelType.Flux
        elif latent_channels == 4:
            # SD/SDXL VAE has 4-dimensional latent space
            # Try to distinguish SD1/SD2/SDXL by name, fallback to SD1
            for regexp, base in REGEX_TO_BASE.items():
                if re.search(regexp, mod.path.name, re.IGNORECASE):
                    return base
            # Default to SD1 if we can't determine from name
            return BaseModelType.StableDiffusion1

        # Fallback: guess based on name
        for regexp, base in REGEX_TO_BASE.items():
            if re.search(regexp, mod.path.name, re.IGNORECASE):
                return base

        raise NotAMatchError("cannot determine base type")


class VAE_Checkpoint_SD1_Config(VAE_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class VAE_Checkpoint_SD2_Config(VAE_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class VAE_Checkpoint_SDXL_Config(VAE_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class VAE_Checkpoint_FLUX_Config(VAE_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class VAE_Checkpoint_SD3_Config(VAE_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion3] = Field(default=BaseModelType.StableDiffusion3)


class VAE_Checkpoint_Flux2_Config(Checkpoint_Config_Base, Config_Base):
    """Model config for FLUX.2 VAE checkpoint models (AutoencoderKLFlux2)."""

    type: Literal[ModelType.VAE] = Field(default=ModelType.VAE)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_vae(mod)

        cls._validate_is_flux2_vae(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_looks_like_vae(cls, mod: ModelOnDisk) -> None:
        if not state_dict_has_any_keys_starting_with(
            mod.load_state_dict(),
            {
                "encoder.conv_in",
                "decoder.conv_in",
            },
        ):
            raise NotAMatchError("model does not match Checkpoint VAE heuristics")

    @classmethod
    def _validate_is_flux2_vae(cls, mod: ModelOnDisk) -> None:
        """Validate that this is a FLUX.2 VAE, not FLUX.1."""
        state_dict = mod.load_state_dict()
        if not _is_flux2_vae(state_dict):
            raise NotAMatchError("state dict does not look like a FLUX.2 VAE")


class VAE_Checkpoint_QwenImage_Config(Checkpoint_Config_Base, Config_Base):
    """Model config for Qwen Image VAE checkpoint models (AutoencoderKLQwenImage)."""

    type: Literal[ModelType.VAE] = Field(default=ModelType.VAE)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    base: Literal[BaseModelType.QwenImage] = Field(default=BaseModelType.QwenImage)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()
        if not _is_qwen_image_vae(state_dict):
            raise NotAMatchError("state dict does not look like a Qwen Image VAE")

        # Defer to VAE_Checkpoint_Wan_Config for files whose names indicate Wan
        # (both architectures are 16-channel AutoencoderKLWan and otherwise
        # indistinguishable from the state dict alone).
        if _filename_suggests_wan(mod):
            raise NotAMatchError("filename suggests a Wan VAE, not Qwen Image")

        return cls(**override_fields)


class VAE_Checkpoint_Wan_Config(Checkpoint_Config_Base, Config_Base):
    """Model config for Wan 2.2 VAE checkpoint models (AutoencoderKLWan).

    Distinguishes A14B (z_dim=16, standard Wan VAE) from TI2V-5B (z_dim=48,
    Wan2.2-VAE) via the input channel count of ``decoder.conv_in.weight``.
    """

    type: Literal[ModelType.VAE] = Field(default=ModelType.VAE)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    base: Literal[BaseModelType.Wan] = Field(default=BaseModelType.Wan)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")
    latent_channels: Literal[16, 48] = Field(
        description="VAE latent channel count: 16 for A14B (standard Wan VAE) or 48 for TI2V-5B (Wan2.2-VAE)."
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()
        z_dim = _wan_vae_z_dim(state_dict)
        if z_dim is None:
            raise NotAMatchError("state dict does not look like a Wan VAE")

        # 48-channel files are unambiguously Wan2.2-VAE (TI2V-5B). 16-channel
        # files are architecturally identical to Qwen Image's VAE; require the
        # filename to suggest Wan to claim them, otherwise let the QwenImage
        # config win.
        latent_channels: int = z_dim
        if latent_channels == 16 and not _filename_suggests_wan(mod):
            raise NotAMatchError(
                "16-channel AutoencoderKLWan VAE without 'wan' in filename — deferring to Qwen Image VAE config."
            )

        explicit = override_fields.pop("latent_channels", None)
        if explicit is not None:
            latent_channels = int(explicit)

        return cls(**override_fields, latent_channels=latent_channels)


class VAE_Diffusers_Wan_Config(Diffusers_Config_Base, Config_Base):
    """Model config for Wan 2.2 VAE in diffusers folder layout (AutoencoderKLWan)."""

    type: Literal[ModelType.VAE] = Field(default=ModelType.VAE)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)
    base: Literal[BaseModelType.Wan] = Field(default=BaseModelType.Wan)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")
    latent_channels: Literal[16, 48] = Field(
        default=16,
        description="VAE latent channel count: 16 for A14B or 48 for TI2V-5B's Wan2.2-VAE.",
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        raise_for_class_name(
            common_config_paths(mod.path),
            {"AutoencoderKLWan"},
        )

        # Read z_dim from the diffusers config to set latent_channels.
        latent_channels: int = 16
        try:
            config = get_config_dict_or_raise(common_config_paths(mod.path))
            z = config.get("z_dim")
            if z is not None and int(z) in (16, 48):
                latent_channels = int(z)
        except NotAMatchError:
            pass

        explicit = override_fields.pop("latent_channels", None)
        if explicit is not None:
            latent_channels = int(explicit)

        return cls(**override_fields, latent_channels=latent_channels)


def _has_anima_vae_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict looks like an Anima QwenImage VAE (AutoencoderKLQwenImage).

    The Anima VAE has a distinctive structure with:
    - encoder.downsamples.* (instead of encoder.down_blocks)
    - decoder.upsamples.* (instead of decoder.up_blocks)
    - decoder.head.* / decoder.middle.*
    - Top-level conv1/conv2 weights
    """
    required_prefixes = {
        "encoder.downsamples.",
        "decoder.upsamples.",
        "decoder.middle.",
    }
    # Qwen-Image-2.1's ComfyUI export shares these block names; its 64-channel RGBA latent does not.
    return all(any(str(k).startswith(prefix) for k in state_dict) for prefix in required_prefixes) and not (
        _is_qwen_image21_vae(state_dict)
    )


class VAE_Checkpoint_Anima_Config(Checkpoint_Config_Base, Config_Base):
    """Model config for Anima QwenImage VAE checkpoint models (AutoencoderKLQwenImage)."""

    type: Literal[ModelType.VAE] = Field(default=ModelType.VAE)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    base: Literal[BaseModelType.Anima] = Field(default=BaseModelType.Anima)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()
        if not _has_anima_vae_keys(state_dict):
            raise NotAMatchError("state dict does not look like an Anima QwenImage VAE")

        return cls(**override_fields)


class VAE_Checkpoint_QwenImage21_Config(Checkpoint_Config_Base, Config_Base):
    """Model config for Qwen-Image-2.1 VAE single files (AutoencoderKLQwenImage21, diffusers or ComfyUI layout)."""

    type: Literal[ModelType.VAE] = Field(default=ModelType.VAE)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    base: Literal[BaseModelType.QwenImage21] = Field(default=BaseModelType.QwenImage21)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if not _is_qwen_image21_vae(mod.load_state_dict()):
            raise NotAMatchError("state dict does not look like a Qwen-Image-2.1 VAE")

        return cls(**override_fields)


class VAE_Diffusers_QwenImage21_Config(Diffusers_Config_Base, Config_Base):
    """Model config for a Qwen-Image-2.1 VAE folder in diffusers format (AutoencoderKLQwenImage21)."""

    type: Literal[ModelType.VAE] = Field(default=ModelType.VAE)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)
    base: Literal[BaseModelType.QwenImage21] = Field(default=BaseModelType.QwenImage21)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        raise_for_class_name(common_config_paths(mod.path), {"AutoencoderKLQwenImage21"})

        return cls(**override_fields)


class VAE_Diffusers_Config_Base(Diffusers_Config_Base):
    """Model config for standalone VAE models (diffusers version)."""

    type: Literal[ModelType.VAE] = Field(default=ModelType.VAE)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "AutoencoderKL",
                "AutoencoderTiny",
            },
        )

        cls._validate_base(mod, override_fields)

        return cls(**override_fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> None:
        """Raise `NotAMatch` if the model base does not match this config class.

        A folder's `config.json` names the constants that normalise its latents, and for a 16-channel
        VAE those are the whole difference between FLUX.1 and SD3. So there the config decides, and a
        config that names neither is filed under neither, override or not: FLUX.1 and SD3 latents
        would both be decoded with the wrong constants. Within the 4-channel family the config is a
        weak hint, and an explicit `base` override outranks it.
        """
        expected_base = cls.model_fields["base"].default
        config_dict = get_config_dict_or_raise(common_config_paths(mod.path))
        # diffusers' own default, for the configs that leave it out.
        latent_channels = config_dict.get("latent_channels", 4)

        if latent_channels == 16:
            if is_flux_family_vae_config(config_dict):
                recognized_base = BaseModelType.Flux
            elif is_sd3_vae_config(config_dict):
                recognized_base = BaseModelType.StableDiffusion3
            else:
                raise NotAMatchError(
                    "16-channel autoencoder normalised as neither FLUX.1 nor SD3 "
                    f"(scaling_factor={config_dict.get('scaling_factor')}, shift_factor={config_dict.get('shift_factor')})"
                )
        elif latent_channels == 4:
            if _override_fits_latent_width(override_fields, expected_base, latent_channels):
                return
            recognized_base = cls._get_4_channel_base(mod, config_dict, override_fields.get("name"))
        else:
            raise NotAMatchError(f"{latent_channels}-channel autoencoder is not an SD, FLUX.1 or SD3 VAE")

        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _config_looks_like_sdxl(cls, config: dict[str, Any]) -> bool:
        # Heuristic: These config values that distinguish Stability's SD 1.x VAE from their SDXL VAE.
        return config.get("scaling_factor", 0) == 0.13025 and config.get("sample_size") in [512, 1024]

    @classmethod
    def _name_looks_like_sdxl(cls, mod: ModelOnDisk, override_name: str | None = None) -> bool:
        # Heuristic: SD and SDXL VAE are the same shape (3-channel RGB to 4-channel float scaled down
        # by a factor of 8), so we can't necessarily tell them apart by config hyperparameters. Best
        # we can do is guess based on name.
        return bool(re.search(r"xl\b", override_name or mod.path.name, re.IGNORECASE))

    @classmethod
    def _get_4_channel_base(
        cls, mod: ModelOnDisk, config_dict: dict[str, Any], override_name: str | None = None
    ) -> BaseModelType:
        # Unfortunately it is difficult to distinguish SD1 and SDXL VAEs by config alone, so we may need to
        # guess based on name if the config is inconclusive.
        if cls._config_looks_like_sdxl(config_dict):
            return BaseModelType.StableDiffusionXL
        elif cls._name_looks_like_sdxl(mod, override_name):
            return BaseModelType.StableDiffusionXL
        else:
            # TODO(psyche): Figure out how to positively identify SD1 here, and raise if we can't. Until then, YOLO.
            return BaseModelType.StableDiffusion1


class VAE_Diffusers_SD1_Config(VAE_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class VAE_Diffusers_SDXL_Config(VAE_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class VAE_Diffusers_FLUX_Config(VAE_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class VAE_Diffusers_SD3_Config(VAE_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion3] = Field(default=BaseModelType.StableDiffusion3)


class VAE_Diffusers_Flux2_Config(Diffusers_Config_Base, Config_Base):
    """Model config for FLUX.2 VAE models in diffusers format (AutoencoderKLFlux2)."""

    type: Literal[ModelType.VAE] = Field(default=ModelType.VAE)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)
    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "AutoencoderKLFlux2",
            },
        )

        return cls(**override_fields)
