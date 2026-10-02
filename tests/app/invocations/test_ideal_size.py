"""Ideal Size computes from what the architecture declares, not from a list of six bases."""

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.ideal_size import IdealSizeInvocation
from invokeai.app.invocations.model import ModelIdentifierField, UNetField
from invokeai.backend.architectures import ArchitectureError, generative_bases
from invokeai.backend.model_manager.taxonomy import AnyVariant, BaseModelType, ModelType, WanVariantType


def _unet(base: BaseModelType) -> UNetField:
    """A UNetField the node can hold. Only `unet.key` is read, and only to look up the config."""
    identifier = ModelIdentifierField(key="test", hash="test", name="test", base=base, type=ModelType.Main)
    return UNetField(unet=identifier, scheduler=identifier, loras=[])


def _invoke(
    base: BaseModelType,
    width: int = 1024,
    height: int = 576,
    multiplier: float = 1.0,
    variant: AnyVariant | None = None,
) -> Any:
    node = IdealSizeInvocation(width=width, height=height, multiplier=multiplier, unet=_unet(base))
    context = MagicMock()
    # No `variant` attribute at all unless one is asked for: plenty of model configs have none, and
    # the node has to cope with that rather than only with `variant=None`.
    config = SimpleNamespace(base=base) if variant is None else SimpleNamespace(base=base, variant=variant)
    context.models.get_config.return_value = config
    return node.invoke(context)


# What the node returns for each architecture at its own field defaults (1024x576). Derived by
# hand, not from the implementation: the node squares the architecture's recommended width to get a
# target area, fits that area to the requested 1.7778 aspect, floors, and trims to the dimension
# grid. SD 1.x: 512^2 = 262144 px; height = sqrt(262144 / 1.7778) = 384; width = 682.67 -> 682 ->
# trimmed to 680 on the 8-pixel grid.
#
# Both halves are pinned deliberately. The dimension follows `DefaultSettingsFacet.width`, so a
# model-card width edited for a UX reason would otherwise silently move every legacy workflow's
# output resolution; the grid follows `FeaturesFacet.dimension_grid`, which the old code hardcoded
# to 8 for everything -- a FLUX size came back off-grid and the denoise node then rejected it.
IDEAL_SIZE_AT_NODE_DEFAULTS: dict[BaseModelType, tuple[int, int]] = {
    BaseModelType.Anima: (1360, 768),
    BaseModelType.CogView4: (1344, 768),
    BaseModelType.ErnieImage: (1360, 768),
    BaseModelType.Flux: (1360, 768),
    BaseModelType.Flux2: (1360, 768),
    BaseModelType.Ideogram4: (1360, 768),
    BaseModelType.Krea2: (1360, 768),
    # Declared 1248x704 (non-square, like H3): 878592 px at 1.7778 -> height sqrt(494208) = 703.0
    # -> 703 -> 672 on the 32 grid; width 1249.9 -> 1249 -> 1248.
    BaseModelType.LTX2: (1248, 672),
    # The one architecture whose declared default is not square (1344x768). Squaring its width
    # asked for 1.75x the area it was trained on and returned (1792, 992); taking both
    # dimensions as an area keeps it at its own scale. Every other row is unchanged by that,
    # because their declared defaults are square.
    BaseModelType.MiniMaxH3: (1344, 736),
    BaseModelType.QwenImage: (1360, 768),
    BaseModelType.StableDiffusion1: (680, 384),
    BaseModelType.StableDiffusion2: (1024, 576),
    BaseModelType.StableDiffusion3: (1360, 768),
    BaseModelType.StableDiffusionXL: (1360, 768),
    BaseModelType.StableDiffusionXLRefiner: (1360, 768),
    BaseModelType.Wan: (1360, 768),
    BaseModelType.ZImage: (1360, 768),
}


@pytest.mark.parametrize(("base", "expected"), sorted(IDEAL_SIZE_AT_NODE_DEFAULTS.items(), key=lambda i: i[0].value))
def test_the_ideal_size_is_pinned_for_every_architecture(base: BaseModelType, expected: tuple[int, int]) -> None:
    """The old dispatch was an if/elif over six bases ending in `raise ValueError(Unsupported model
    type)`, which fired here -- at generation time, after the model had loaded -- for the other
    nine. Every architecture now has an answer, and these are the answers."""
    output = _invoke(base)
    assert (output.width, output.height) == expected


def test_every_architecture_is_pinned() -> None:
    """A new architecture has to land in the table above, rather than going unchecked."""
    assert set(IDEAL_SIZE_AT_NODE_DEFAULTS) == set(generative_bases())


def test_wan_uses_the_grid_of_the_variant_it_was_given() -> None:
    """TI2V-5B takes multiples of 32 where A14B takes 16, and `wan_denoise` only rejects the
    mismatch inside `invoke()`. 1360 is the A14B answer and `1360 % 32 == 16`, so a TI2V-5B model
    routed through this node used to produce a width its own denoise node refuses."""
    a14b = _invoke(BaseModelType.Wan, variant=WanVariantType.T2V_A14B)
    assert (a14b.width, a14b.height) == IDEAL_SIZE_AT_NODE_DEFAULTS[BaseModelType.Wan]
    assert a14b.width % 32 == 16, "otherwise the case below proves nothing"

    ti2v = _invoke(BaseModelType.Wan, variant=WanVariantType.TI2V_5B)
    assert (ti2v.width, ti2v.height) == (1344, 768)
    assert ti2v.width % 32 == 0 and ti2v.height % 32 == 0


def test_an_architecture_without_dimensions_says_so() -> None:
    """The SDXL refiner declares a canvas; nothing declares none today, so this pins the message
    rather than a current state."""
    with pytest.raises(ArchitectureError, match="no default dimensions"):
        node = IdealSizeInvocation(width=1024, height=576, unet=_unet(BaseModelType.Any))
        context = MagicMock()
        context.models.get_config.return_value = SimpleNamespace(base=BaseModelType.Any)
        node.invoke(context)


def test_the_multiplier_still_scales_the_area() -> None:
    single = _invoke(BaseModelType.StableDiffusionXL)
    doubled = _invoke(BaseModelType.StableDiffusionXL, multiplier=2.0)
    assert doubled.width > single.width
    assert math.isclose(doubled.width / single.width, 2.0, rel_tol=0.02)
