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

    Concrete subclasses pin `base` to a specific backbone; backbone matching is
    performed against the filename (NVIDIA's distribution names backbone +
    variant unambiguously). `variant` is carried as data without participating
    in the discriminator tag (one config class per backbone).
    """

    type: Literal[ModelType.PiDDecoder] = Field(default=ModelType.PiDDecoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        raise_for_override_fields(cls, override_fields)

        if not _looks_like_pid_decoder(mod.load_state_dict()):
            raise NotAMatchError("state dict does not look like a PiD decoder (no 'lq_proj.*' keys)")

        cls._validate_base(mod)

        variant = override_fields.pop("variant", None) or _variant_from_filename(mod.path.name)
        return cls(**override_fields, variant=variant)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        expected_base = cls.model_fields["base"].default
        inferred_base = _backbone_from_filename(mod.path.name)
        if inferred_base is None:
            raise NotAMatchError(
                "cannot determine PiD decoder backbone from filename (expected one of: flux, flux2, sd3)"
            )
        if inferred_base is not expected_base:
            raise NotAMatchError(f"backbone is {inferred_base}, not {expected_base}")


class PiDDecoder_Checkpoint_FLUX_Config(PiDDecoder_Checkpoint_Config_Base, Config_Base):
    """PiD decoder for the FLUX.1 backbone (16-channel latent)."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    variant: PiDDecoderVariantType = Field(description="Resolution preset of the PiD decoder checkpoint.")


class PiDDecoder_Checkpoint_Flux2_Config(PiDDecoder_Checkpoint_Config_Base, Config_Base):
    """PiD decoder for the FLUX.2 backbone (32-channel latent)."""

    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    variant: PiDDecoderVariantType = Field(description="Resolution preset of the PiD decoder checkpoint.")


class PiDDecoder_Checkpoint_SD3_Config(PiDDecoder_Checkpoint_Config_Base, Config_Base):
    """PiD decoder for the Stable Diffusion 3 backbone (16-channel latent)."""

    base: Literal[BaseModelType.StableDiffusion3] = Field(default=BaseModelType.StableDiffusion3)
    variant: PiDDecoderVariantType = Field(description="Resolution preset of the PiD decoder checkpoint.")
