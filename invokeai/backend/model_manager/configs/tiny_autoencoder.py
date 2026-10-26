"""Tiny AutoEncoders (madebyollin's TAESD and TAEHV families), used only to render denoising previews.

A TAE decodes the latents an architecture denoises in, so a record is filed under the base whose
`LatentSpace` its weights fit, and serves every base sharing that space (`latent_space_with_channels`).
Only single files are TAEs here: a diffusers `AutoencoderTiny` folder stays a VAE, as it always was.

The weights fix the latent width, the compression and the decoder architecture. Some networks are not
specific to one base or one TAE: 4 channels serve SD 1 and SDXL, 16 serve SD3 and FLUX.1, taew2_1's network
is also published as taehv (HunyuanVideo) and taecvx (CogVideoX), and taeltx2_3's as taeltx_2 (LTX-2).
There an explicit `base` override decides, else the TAE's published name. A file that names none of them
is refused rather than guessed: a wrong guess would install cleanly and render every preview in wrong
colours. A file named for one of the unsupported twins is refused whatever the override says.

Names are read from the most specific path segment first (`name_components`, then each component's
segments from the last), so the TAEW2.1 starter's `.../madebyollin/taehv/raw/<sha>/safetensors/
taew2_1.safetensors` is named by its file, not by the `taehv` repository it is published in.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
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
from invokeai.backend.model_manager.starter_models import STARTER_MODELS, StarterModel
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.tiny_autoencoder.checkpoint import (
    TAELTX2_3,
    TAEW2_1,
    TINY_AUTOENCODER_CHECKPOINT_SUFFIXES,
    TinyAutoencoderArch,
    TinyAutoencoderLayout,
    detect_tiny_autoencoder_layout,
    is_tinier_taesd,
)

# madebyollin's names, as they appear in file names, repo names and install URLs, with the base and latent
# width each is published for. Boundary-aware, so `taesd` matches neither `taesdxl` nor `taesd3`.
_TAE_NAMES: tuple[tuple[str, BaseModelType, int], ...] = (
    ("taesdxl", BaseModelType.StableDiffusionXL, 4),
    ("taesd3", BaseModelType.StableDiffusion3, 16),
    ("taesd", BaseModelType.StableDiffusion1, 4),
    ("taef1", BaseModelType.Flux, 16),
    ("taef2", BaseModelType.Flux2, 32),
    ("taeqi2_1", BaseModelType.QwenImage21, 64),
    ("taew2_1", BaseModelType.Wan, 16),
    ("taew2_2", BaseModelType.Wan, 48),
    ("taeltx2_3", BaseModelType.LTX2, 128),
)


def _name_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])")


_TAE_NAME_PATTERNS = tuple(_name_pattern(name) for name, _, _ in _TAE_NAMES)


@dataclass(frozen=True)
class _Twin:
    """A published TAE with a supported layout's exact network, for latents InvokeAI does not preview with it."""

    label: str
    pattern: re.Pattern[str]
    refusal: str


_TWINS: dict[TinyAutoencoderLayout, tuple[_Twin, ...]] = {
    TAEW2_1: (
        _Twin(
            "HunyuanVideo (taehv)",
            _name_pattern("taehv"),
            "taehv is the HunyuanVideo preview decoder, which InvokeAI does not support. Wan, Qwen-Image, "
            "Krea-2 and Anima previews use taew2_1: install the TAEW2.1 starter model.",
        ),
        _Twin(
            "CogVideoX (taecvx)",
            _name_pattern("taecvx"),
            "taecvx is the CogVideoX preview decoder, which InvokeAI does not support. Wan, Qwen-Image, "
            "Krea-2 and Anima previews use taew2_1: install the TAEW2.1 starter model.",
        ),
    ),
    # On LTX-2.5 latents taeltx_2 scored 11.0 dB against the VAE, below the linear preview's 26.6
    # (taeltx2_3: 37.8). `taeltx2` without the `_3` counts as the same name.
    TAELTX2_3: (
        _Twin(
            "LTX-2 (taeltx_2)",
            re.compile(r"(?<![a-z0-9])taeltx_?2(?![a-z0-9])(?!_3(?![a-z0-9]))"),
            "taeltx_2 targets the LTX-2 VAE; LTX-2.5 needs taeltx2_3. Install TAELTX2.3 from the starter models.",
        ),
    ),
}


def _name_segments(components: tuple[str, ...]) -> Iterator[str]:
    """The name evidence, most specific first: each component's path segments from the last."""
    for component in components:
        for segment in reversed(re.split(r"[/\\:]+", component.lower())):
            if segment:
                yield segment


def named_tiny_autoencoder(components: tuple[str, ...]) -> tuple[str, BaseModelType, int] | None:
    """The published TAE `(name, base, latent channels)` the most specific segment naming exactly one names."""
    for segment in _name_segments(components):
        named = [
            entry for entry, pattern in zip(_TAE_NAMES, _TAE_NAME_PATTERNS, strict=True) if pattern.search(segment)
        ]
        if len(named) == 1:
            return named[0]
    return None


def _name_verdict(layout: TinyAutoencoderLayout, components: tuple[str, ...]) -> BaseModelType | _Twin | None:
    """What the most specific segment naming a supported TAE or one of `layout`'s twins says the file is."""
    twins = _TWINS.get(layout, ())
    for segment in _name_segments(components):
        twin = next((t for t in twins if t.pattern.search(segment)), None)
        if twin is not None:
            return twin
        named = {
            entry[1] for entry, pattern in zip(_TAE_NAMES, _TAE_NAME_PATTERNS, strict=True) if pattern.search(segment)
        }
        if len(named) == 1:
            return named.pop()
    return None


def _unnamed_message(layout: TinyAutoencoderLayout, candidates: frozenset[BaseModelType]) -> str:
    names = " or ".join(n for n, b, c in _TAE_NAMES if b in candidates and c == layout.latent_channels)
    if len(candidates) > 1:
        bases = " or ".join(sorted(b.value for b in candidates))
        reason = f"reads {layout.latent_channels}-channel latents, which {bases} share,"
    else:
        twins = " and ".join(twin.label for twin in _TWINS[layout])
        reason = f"has the network of {names} and of the {twins} decoders,"
    return (
        f"This preview decoder {reason} and nothing in its name says which it is. Install it from the starter "
        f"models, or, if it is {names}, name the file after it."
    )


class TinyAutoencoder_Checkpoint_Config_Base(Checkpoint_Config_Base):
    """Model config for a Tiny AutoEncoder preview decoder."""

    type: Literal[ModelType.TinyAutoencoder] = Field(default=ModelType.TinyAutoencoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    latent_channels: int = Field(description="Latent channels the decoder consumes.")

    ARCH: ClassVar[TinyAutoencoderArch]
    """The decoder architecture this base's TAEs use."""

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if mod.path.suffix.lower() not in TINY_AUTOENCODER_CHECKPOINT_SUFFIXES:
            raise NotAMatchError(f"the Tiny AutoEncoder loader does not read {mod.path.suffix} files")

        state_dict = mod.load_state_dict()
        layout = detect_tiny_autoencoder_layout(state_dict)
        if layout is None:
            if is_tinier_taesd(state_dict):
                raise InvalidMatchError(
                    "This is the 'tinier' TAEQI2.1 preview decoder, which InvokeAI does not support. "
                    "Install taeqi2_1.safetensors from the same repository instead."
                )
            raise NotAMatchError("state dict is not a supported Tiny AutoEncoder decoder")

        base: BaseModelType = cls.model_fields["base"].default
        candidates = _bases_fitting(layout)
        if base not in candidates:
            if override_fields.get("base") is not None:
                # The override has already been held to this class's base, so the user asked for a base
                # whose latents this decoder cannot read. Falling through would file it as Unknown.
                raise InvalidMatchError(
                    f"This preview decoder ({layout.latent_channels} latent channels, {layout.arch} architecture, "
                    f"{layout.spatial_compression}x) cannot decode {base.value} latents."
                )
            raise NotAMatchError(f"a {layout.latent_channels}-channel {layout.arch} TAE is not for {base.value}")

        verdict = _name_verdict(layout, name_components(mod, override_fields))
        if isinstance(verdict, _Twin):
            raise InvalidMatchError(verdict.refusal)

        if (len(candidates) > 1 or layout in _TWINS) and override_fields.get("base") is None:
            if verdict not in candidates:
                raise InvalidMatchError(_unnamed_message(layout, candidates))
            if verdict is not base:
                raise NotAMatchError(f"named for {verdict.value}, not {base.value}")

        return cls(**override_fields, latent_channels=layout.latent_channels)


class TinyAutoencoder_Checkpoint_SD1_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)
    ARCH: ClassVar[TinyAutoencoderArch] = "default"


class TinyAutoencoder_Checkpoint_SDXL_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)
    ARCH: ClassVar[TinyAutoencoderArch] = "default"


class TinyAutoencoder_Checkpoint_SD3_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion3] = Field(default=BaseModelType.StableDiffusion3)
    ARCH: ClassVar[TinyAutoencoderArch] = "default"


class TinyAutoencoder_Checkpoint_FLUX_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    ARCH: ClassVar[TinyAutoencoderArch] = "default"


class TinyAutoencoder_Checkpoint_Flux2_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    ARCH: ClassVar[TinyAutoencoderArch] = "flux_2"


class TinyAutoencoder_Checkpoint_QwenImage21_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.QwenImage21] = Field(default=BaseModelType.QwenImage21)
    ARCH: ClassVar[TinyAutoencoderArch] = "f16"


class TinyAutoencoder_Checkpoint_Wan_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    """taew2_1 (16 channels, also Qwen-Image, Krea-2, Anima) or taew2_2 (48, TI2V-5B): `latent_channels` decides."""

    base: Literal[BaseModelType.Wan] = Field(default=BaseModelType.Wan)
    ARCH: ClassVar[TinyAutoencoderArch] = "taehv"


class TinyAutoencoder_Checkpoint_LTX2_Config(TinyAutoencoder_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.LTX2] = Field(default=BaseModelType.LTX2)
    ARCH: ClassVar[TinyAutoencoderArch] = "taehv"


_CONFIG_CLASSES: tuple[type[TinyAutoencoder_Checkpoint_Config_Base], ...] = (
    TinyAutoencoder_Checkpoint_SD1_Config,
    TinyAutoencoder_Checkpoint_SDXL_Config,
    TinyAutoencoder_Checkpoint_SD3_Config,
    TinyAutoencoder_Checkpoint_FLUX_Config,
    TinyAutoencoder_Checkpoint_Flux2_Config,
    TinyAutoencoder_Checkpoint_QwenImage21_Config,
    TinyAutoencoder_Checkpoint_Wan_Config,
    TinyAutoencoder_Checkpoint_LTX2_Config,
)


def _base_of(config: type[TinyAutoencoder_Checkpoint_Config_Base]) -> BaseModelType:
    return config.model_fields["base"].default


def _bases_fitting(layout: TinyAutoencoderLayout) -> frozenset[BaseModelType]:
    """Every base with a TAE config whose architecture, latent width and compression this decoder fits."""
    fitting = set()
    for config in _CONFIG_CLASSES:
        space = latent_space_with_channels(_base_of(config), layout.latent_channels)
        if config.ARCH == layout.arch and space is not None and space.spatial_compression == layout.spatial_compression:
            fitting.add(_base_of(config))
    return frozenset(fitting)


def tiny_autoencoder_starters(space: LatentSpace) -> list[StarterModel]:
    """The starter models that install a TAE for `space`, read from the catalog by the TAE each one installs."""
    starters = []
    for starter in STARTER_MODELS:
        if starter.type is not ModelType.TinyAutoencoder:
            continue
        named = named_tiny_autoencoder((starter.source,))
        if named is not None and latent_space_with_channels(starter.base, named[2]) is space:
            starters.append(starter)
    return starters
