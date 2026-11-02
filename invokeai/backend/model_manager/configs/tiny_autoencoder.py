"""Tiny AutoEncoders (madebyollin's TAESD family), used only to render denoising previews.

A TAE decodes the latents an architecture denoises in, so a record is filed under the base whose
`LatentSpace` its weights fit, and serves every base sharing that space (`latent_space_with_channels`).
Only single files are TAEs here: a diffusers `AutoencoderTiny` folder stays a VAE, as it always was.

The weights fix the latent width and the decoder architecture. Two widths are shared by two bases with
the same default decoder -- 4 channels (SD 1, SDXL) and 16 (SD3, FLUX.1) -- and there an explicit
`base` override decides, else the TAE's published name. A file that names none of them is refused
rather than guessed: a wrong guess would install cleanly and render every preview in wrong colours.
"""

import re
from typing import ClassVar, Literal, Self

from pydantic import Field
from typing_extensions import Any

from invokeai.backend.architectures import LatentSpace, latent_space_with_channels
from invokeai.backend.model_manager.configs.backbone_names import name_components
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    InvalidMatchError,
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.tiny_autoencoder.checkpoint import (
    TAESD_CHECKPOINT_SUFFIXES,
    TaesdArch,
    TaesdLayout,
    detect_taesd_layout,
    is_tinier_taesd,
)

# madebyollin's names, as they appear in file names, repo names and install URLs. Boundary-aware, so
# `taesd` matches neither `taesdxl` nor `taesd3`.
_TAE_NAME_PATTERNS: tuple[tuple[BaseModelType, re.Pattern[str]], ...] = tuple(
    (base, re.compile(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])"))
    for base, name in (
        (BaseModelType.StableDiffusionXL, "taesdxl"),
        (BaseModelType.StableDiffusion3, "taesd3"),
        (BaseModelType.StableDiffusion1, "taesd"),
        (BaseModelType.Flux, "taef1"),
        (BaseModelType.Flux2, "taef2"),
        (BaseModelType.QwenImage21, "taeqi2_1"),
    )
)


def _tae_named_in(components: tuple[str, ...]) -> BaseModelType | None:
    """The base named by the most specific name component that names exactly one TAE."""
    for component in components:
        named = {base for base, pattern in _TAE_NAME_PATTERNS if pattern.search(component.lower())}
        if len(named) == 1:
            return named.pop()
    return None


class TinyAutoencoder_Checkpoint_Config_Base(Checkpoint_Config_Base):
    """Model config for a Tiny AutoEncoder preview decoder."""

    type: Literal[ModelType.TinyAutoencoder] = Field(default=ModelType.TinyAutoencoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    latent_channels: int = Field(description="Latent channels the decoder consumes.")

    ARCH: ClassVar[TaesdArch]
    """The decoder architecture this base's TAE uses."""

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if mod.path.suffix.lower() not in TAESD_CHECKPOINT_SUFFIXES:
            raise NotAMatchError(f"the Tiny AutoEncoder loader does not read {mod.path.suffix} files")

        state_dict = mod.load_state_dict()
        layout = detect_taesd_layout(state_dict)
        if layout is None:
            if is_tinier_taesd(state_dict):
                raise InvalidMatchError(
                    "This is the 'tinier' TAEQI2.1 preview decoder, which InvokeAI does not support. "
                    "Install taeqi2_1.safetensors from the same repository instead."
                )
            raise NotAMatchError("state dict is not a TAESD decoder")

        base: BaseModelType = cls.model_fields["base"].default
        candidates = _bases_fitting(layout)
        if base not in candidates:
            if override_fields.get("base") is not None:
                # The override has already been held to this class's base, so the user asked for a base
                # whose latents this decoder cannot read. Falling through would file it as Unknown.
                raise InvalidMatchError(
                    f"This preview decoder ({layout.latent_channels} latent channels, {layout.arch} architecture) "
                    f"cannot decode {base.value} latents."
                )
            raise NotAMatchError(f"a {layout.latent_channels}-channel {layout.arch} TAE is not for {base.value}")

        if len(candidates) > 1 and override_fields.get("base") is None:
            named = _tae_named_in(name_components(mod, override_fields))
            if named not in candidates:
                names = " or ".join(sorted(b.value for b in candidates))
                raise InvalidMatchError(
                    f"This preview decoder reads {layout.latent_channels}-channel latents, which {names} share, "
                    "and its name does not say which it is for. Install it from the starter models, or name the "
                    "file after the TAE it is (taesd, taesdxl, taesd3 or taef1)."
                )
            if named is not base:
                raise NotAMatchError(f"named for {named.value}, not {base.value}")

        return cls(**override_fields, latent_channels=layout.latent_channels)


class TinyAutoencoder_Checkpoint_SD1_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)
    ARCH: ClassVar[TaesdArch] = "default"


class TinyAutoencoder_Checkpoint_SDXL_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)
    ARCH: ClassVar[TaesdArch] = "default"


class TinyAutoencoder_Checkpoint_SD3_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion3] = Field(default=BaseModelType.StableDiffusion3)
    ARCH: ClassVar[TaesdArch] = "default"


class TinyAutoencoder_Checkpoint_FLUX_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    ARCH: ClassVar[TaesdArch] = "default"


class TinyAutoencoder_Checkpoint_Flux2_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    ARCH: ClassVar[TaesdArch] = "flux_2"


class TinyAutoencoder_Checkpoint_QwenImage21_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.QwenImage21] = Field(default=BaseModelType.QwenImage21)
    ARCH: ClassVar[TaesdArch] = "f16"


_CONFIG_CLASSES: tuple[type[TinyAutoencoder_Checkpoint_Config_Base], ...] = (
    TinyAutoencoder_Checkpoint_SD1_Config,
    TinyAutoencoder_Checkpoint_SDXL_Config,
    TinyAutoencoder_Checkpoint_SD3_Config,
    TinyAutoencoder_Checkpoint_FLUX_Config,
    TinyAutoencoder_Checkpoint_Flux2_Config,
    TinyAutoencoder_Checkpoint_QwenImage21_Config,
)


def _base_of(config: type[TinyAutoencoder_Checkpoint_Config_Base]) -> BaseModelType:
    return config.model_fields["base"].default


def _bases_fitting(layout: TaesdLayout) -> frozenset[BaseModelType]:
    """Every base with a TAE config whose architecture and latent space this decoder fits."""
    return frozenset(
        _base_of(config)
        for config in _CONFIG_CLASSES
        if config.ARCH == layout.arch
        and latent_space_with_channels(_base_of(config), layout.latent_channels) is not None
    )


def is_tiny_autoencoder_supported(space: LatentSpace) -> bool:
    """Whether InvokeAI identifies a TAE that decodes `space`, i.e. whether there is one worth installing.

    Every TAE config class has a starter model, so this is also whether a starter serves the space.
    """
    return any(latent_space_with_channels(_base_of(config), space.channels) is space for config in _CONFIG_CLASSES)
