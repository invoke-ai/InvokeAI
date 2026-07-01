"""Model configs for PiD (Pixel Diffusion Decoder) checkpoints.

PiD decoders are released by NVIDIA at https://huggingface.co/nvidia/PiD and
ship per supported backbone (FLUX.1, FLUX.2, SD3) in two resolution presets
(`res2k_sr4x_*` and `res2kto4k_sr4x_*`). See `LICENSE-PiD.txt` at the repo
root — code is Apache-2.0, weights are NSCLv1 (non-commercial / research).
"""

import re
from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    PiDDecoderVariantType,
)

# Marker substring produced by `PidNet.lq_proj` (see
# invokeai/backend/pid/_src/networks/pid_net.py). The pretrained PixDiT_T2I
# weights do not contain `lq_proj`, so its presence in any key is diagnostic
# of a PiD-style checkpoint. We match by substring (not prefix) because the
# official `.pth` files keep PidDistillModel's `net.` prefix, so keys look
# like `net.lq_proj.layers.0.weight`.
_PID_MARKER_SUBSTRING = "lq_proj"


def _looks_like_pid_decoder(state_dict: dict[str | int, Any]) -> bool:
    return any(isinstance(k, str) and _PID_MARKER_SUBSTRING in k for k in state_dict)


# The latent input projection (`lq_proj.latent_proj.0`) is a Conv2d whose
# in-channel count equals the backbone's latent channel count — the released
# sr4x checkpoints apply no spatial fold here, so the Conv's dim-1 is exactly
# `lq_latent_channels` (see `_PER_BACKBONE` in invokeai/backend/pid/decode.py):
# FLUX.1 / SD3 = 16, FLUX.2 = 128. This is the only architectural dimension
# that varies between backbones and is therefore a filename-independent
# discriminator between FLUX.2 and the 16-channel family. (FLUX.1 and SD3 are
# architecturally identical and cannot be told apart from the weights alone.)
# We match the key by suffix because the official `.pth` keep the `net.` prefix.
_LATENT_PROJ_KEY_SUFFIX = "lq_proj.latent_proj.0.weight"

_LATENT_CHANNELS_TO_BASES: dict[int, set[BaseModelType]] = {
    16: {BaseModelType.Flux, BaseModelType.StableDiffusion3},
    128: {BaseModelType.Flux2},
}


def _latent_channels_from_state_dict(state_dict: dict[str | int, Any]) -> int | None:
    """Read the backbone's latent channel count from the `lq_proj` input Conv.

    Returns None if the diagnostic weight is absent or not a 4D conv tensor.
    """
    for k, v in state_dict.items():
        if isinstance(k, str) and k.endswith(_LATENT_PROJ_KEY_SUFFIX):
            shape = getattr(v, "shape", None)
            if shape is not None and len(shape) == 4:
                return int(shape[1])
    return None


def _name_for_matching(mod: ModelOnDisk) -> str:
    """Searchable name for backbone/variant heuristics.

    NVIDIA distributes PiD checkpoints as
    ``PiD_res2k_sr4x_official_<backbone>_distill_4step/model_ema_bf16.pth`` — the
    backbone + variant live in the *directory* name, not the weights filename.
    We therefore match against both the filename and its parent directory.
    """
    return f"{mod.path.parent.name} {mod.path.name}"


def _backbone_from_filename(name: str) -> BaseModelType | None:
    """Heuristic backbone match against NVIDIA's checkpoint filename conventions.

    Returns None if no backbone can be inferred.
    """
    n = name.lower()
    # Order matters: 'flux2' must match before 'flux'.
    if re.search(r"\bflux[_-]?2\b|flux2", n):
        return BaseModelType.Flux2
    if "flux" in n:
        return BaseModelType.Flux
    if re.search(r"\bsd[_-]?3\b|sd3", n):
        return BaseModelType.StableDiffusion3
    return None


def _variant_from_filename(name: str) -> PiDDecoderVariantType:
    """Map NVIDIA's `res2k_sr4x` / `res2kto4k_sr4x` filename slice to a variant.

    Defaults to ``Res2k_Sr4x`` when no clear marker is present.
    """
    n = name.lower()
    if "res2kto4k" in n or "res2k_to_4k" in n or "res2k_to4k" in n:
        return PiDDecoderVariantType.Res2kTo4k_Sr4x
    return PiDDecoderVariantType.Res2k_Sr4x


class PiDDecoder_Checkpoint_Config_Base(Checkpoint_Config_Base):
    """Shared logic for PiD decoder checkpoint configs.

    Concrete subclasses pin `base` to a specific backbone. Backbone matching is
    driven primarily by the latent channel count read from the weights, with the
    filename / directory name as a tie-breaker for the architecturally identical
    FLUX.1 / SD3 pair. `variant` is carried as data without participating in the
    discriminator tag (one config class per backbone).
    """

    type: Literal[ModelType.PiDDecoder] = Field(default=ModelType.PiDDecoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()
        if not _looks_like_pid_decoder(state_dict):
            raise NotAMatchError("state dict does not look like a PiD decoder (no 'lq_proj.*' keys)")

        # Whether the caller explicitly pinned a base (e.g. a starter-model install passes base=sd-3).
        # In the ambiguous 16-channel FLUX.1/SD3 case this override is trusted when the filename is silent.
        had_base_override = override_fields.get("base") is not None
        cls._validate_base(mod, state_dict, had_base_override=had_base_override)

        variant = override_fields.pop("variant", None) or _variant_from_filename(_name_for_matching(mod))
        return cls(**override_fields, variant=variant)

    @classmethod
    def _validate_base(
        cls, mod: ModelOnDisk, state_dict: dict[str | int, Any], *, had_base_override: bool = False
    ) -> None:
        """Confirm this checkpoint belongs to the config's pinned backbone.

        The latent channel count (read from the weights) is authoritative and
        separates FLUX.2 (128ch) from the 16ch family. FLUX.1 and SD3 share an
        identical architecture, so within the 16ch family we fall back to the
        filename / directory name, defaulting to FLUX.1 when it is silent.

        ``had_base_override`` is True when the caller explicitly pinned ``base``
        (e.g. a starter-model install). In the ambiguous 16ch case, a trusted
        override wins over the FLUX.1 default — necessary because the HF
        single-file download renames the parent directory, dropping the
        ``…official_sd3_distill…`` hint that would otherwise identify SD3.
        """
        expected_base = cls.model_fields["base"].default
        channels = _latent_channels_from_state_dict(state_dict)

        if channels is not None:
            candidate_bases = _LATENT_CHANNELS_TO_BASES.get(channels)
            if candidate_bases is None:
                raise NotAMatchError(
                    f"PiD checkpoint has {channels} latent channels; no supported backbone uses this "
                    "(supported: 16 for FLUX.1/SD3, 128 for FLUX.2)"
                )
            if expected_base not in candidate_bases:
                raise NotAMatchError(f"latent channels={channels} do not match backbone {expected_base}")
            if len(candidate_bases) > 1:
                # Ambiguous 16ch family — disambiguate FLUX.1 vs SD3 by name.
                named_base = _backbone_from_filename(_name_for_matching(mod))
                if named_base in candidate_bases:
                    if named_base is not expected_base:
                        raise NotAMatchError(f"name indicates {named_base}, not {expected_base}")
                elif had_base_override:
                    # Name is silent, but the caller explicitly pinned this base → trust it.
                    return
                elif expected_base is not BaseModelType.Flux:
                    # Name gives no usable hint and no override → default the family to FLUX.1.
                    raise NotAMatchError("ambiguous 16-channel PiD checkpoint; defaulting to FLUX.1")
            return

        # No diagnostic weight (unexpected) → fall back to filename-only matching.
        inferred_base = _backbone_from_filename(_name_for_matching(mod))
        if inferred_base is None:
            raise NotAMatchError(
                "cannot determine PiD decoder backbone from weights or filename (expected one of: flux, flux2, sd3)"
            )
        if inferred_base is not expected_base:
            raise NotAMatchError(f"backbone is {inferred_base}, not {expected_base}")


class PiDDecoder_Checkpoint_FLUX_Config(PiDDecoder_Checkpoint_Config_Base, Config_Base):
    """PiD decoder for the FLUX.1 backbone (16-channel latent)."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    variant: PiDDecoderVariantType = Field(description="Resolution preset of the PiD decoder checkpoint.")


class PiDDecoder_Checkpoint_Flux2_Config(PiDDecoder_Checkpoint_Config_Base, Config_Base):
    """PiD decoder for the FLUX.2 backbone (128-channel latent)."""

    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    variant: PiDDecoderVariantType = Field(description="Resolution preset of the PiD decoder checkpoint.")


class PiDDecoder_Checkpoint_SD3_Config(PiDDecoder_Checkpoint_Config_Base, Config_Base):
    """PiD decoder for the Stable Diffusion 3 backbone (16-channel latent)."""

    base: Literal[BaseModelType.StableDiffusion3] = Field(default=BaseModelType.StableDiffusion3)
    variant: PiDDecoderVariantType = Field(description="Resolution preset of the PiD decoder checkpoint.")
