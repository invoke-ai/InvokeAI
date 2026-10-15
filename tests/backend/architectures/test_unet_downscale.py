"""`get_max_unet_downscale` replaces two verbatim-duplicated dispatches.

They lived in `denoise_latents.run_t2i_adapters` and `T2IAdapterExt.__init__`, identical down to
the comment and the error string. The message is reproduced exactly, because it is user-facing.
"""

import pytest

from invokeai.backend.architectures import generative_bases, get_max_unet_downscale
from invokeai.backend.model_manager.taxonomy import BaseModelType

HAS_UNET = {BaseModelType.StableDiffusion1: 8, BaseModelType.StableDiffusionXL: 4}


@pytest.mark.parametrize(("base", "expected"), sorted(HAS_UNET.items(), key=lambda item: item[0].value))
def test_returns_the_declared_downscale(base: BaseModelType, expected: int) -> None:
    assert get_max_unet_downscale(base) == expected


@pytest.mark.parametrize("base", sorted(set(generative_bases()) - set(HAS_UNET), key=lambda b: b.value))
def test_raises_for_architectures_without_a_unet(base: BaseModelType) -> None:
    # Verbatim, including the quoting and how the enum renders. BaseModelType is a `str, Enum`
    # mixin rather than a StrEnum, so it interpolates as "BaseModelType.Flux", not "flux".
    with pytest.raises(ValueError) as exc_info:
        get_max_unet_downscale(base)

    assert str(exc_info.value) == f"Unexpected T2I-Adapter base model type: '{base}'."


def test_the_sd1_and_sdxl_values_are_the_pre_registry_ones() -> None:
    # SD1's UNet downscales 8x internally, SDXL's 4x. Pinned separately from the parametrized test
    # so that an edit to HAS_UNET cannot silently redefine what is being asserted.
    assert get_max_unet_downscale(BaseModelType.StableDiffusion1) == 8
    assert get_max_unet_downscale(BaseModelType.StableDiffusionXL) == 4
