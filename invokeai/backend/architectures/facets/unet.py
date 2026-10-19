"""How far a UNet-based architecture downscales the latent image internally.

Kept apart from `LatentSpaceFacet` on purpose: this is a property of the *UNet* (SD1 downscales 8x
internally, SDXL 4x), unrelated to the VAE latent geometry next door. The two happen to both be
small integers about downscaling, which is exactly why they should not share a field.
"""

from dataclasses import dataclass

from invokeai.backend.architectures.facet import Facet
from invokeai.backend.architectures.registry import get
from invokeai.backend.model_manager.taxonomy import BaseModelType


@dataclass(frozen=True)
class UNetDownscaleFacet(Facet):
    """Optional: only SD1 and SDXL have a UNet in the sense T2I-Adapter conditioning needs.

    Every other architecture legitimately does not declare it, so this facet is not `REQUIRED` and
    the accessor -- not `require()` -- carries the error, preserving the message the two duplicated
    dispatches it replaces raised.
    """

    max_unet_downscale: int


def get_max_unet_downscale(base: BaseModelType) -> int:
    """The maximum amount the UNet downscales the latent image internally.

    Raises for architectures without a UNet, which is what the T2I-Adapter call sites did before
    this facet existed.
    """
    facet = get(base, UNetDownscaleFacet)
    if facet is None:
        # The message is reproduced verbatim, including how the enum renders: BaseModelType is a
        # `str, Enum` mixin rather than a StrEnum, so this interpolates as "BaseModelType.Flux".
        raise ValueError(f"Unexpected T2I-Adapter base model type: '{base}'.")
    return facet.max_unet_downscale
