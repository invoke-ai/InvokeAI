import math
from typing import Tuple

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import FieldDescriptions, InputField, OutputField
from invokeai.app.invocations.model import UNetField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.architectures import (
    ArchitectureError,
    FeaturesFacet,
    require,
    resolve_default_settings,
)


@invocation_output("ideal_size_output")
class IdealSizeOutput(BaseInvocationOutput):
    """Base class for invocations that output an image"""

    width: int = OutputField(description="The ideal width of the image (in pixels)")
    height: int = OutputField(description="The ideal height of the image (in pixels)")


@invocation(
    "ideal_size",
    # No architecture suffix: the node reads what the architecture declares, so it now answers for
    # every registered one rather than for the six the old if/elif named.
    title="Ideal Size",
    tags=["latents", "math", "ideal_size"],
    category="latents",
    # Minor, not patch: the fields are unchanged, so existing workflows still load, but the answer
    # moved. FLUX, FLUX.2 and SD3 are trimmed to their declared 16 rather than to a hardcoded 8,
    # and every architecture outside the six the old if/elif named returns a size where it raised.
    version="1.1.0",
)
class IdealSizeInvocation(BaseInvocation):
    """Calculates the ideal size for generation to avoid duplication"""

    width: int = InputField(default=1024, description="Final image width")
    height: int = InputField(default=576, description="Final image height")
    unet: UNetField = InputField(description=FieldDescriptions.unet)
    multiplier: float = InputField(
        default=1.0,
        description="Amount to multiply the model's dimensions by when calculating the ideal size (may result in "
        "initial generation artifacts if too large)",
    )

    def trim_to_multiple_of(self, *args: int, multiple_of: int) -> Tuple[int, ...]:
        return tuple((x - x % multiple_of) for x in args)

    def invoke(self, context: InvocationContext) -> IdealSizeOutput:
        unet_config = context.models.get_config(self.unet.unet.key)
        aspect = self.width / self.height

        # Both numbers come from what the architecture declares. This was an if/elif over six bases
        # ending in `raise ValueError(f"Unsupported model type: ...")`, which fired here — at
        # generation time — for the nine architectures nobody had added to it. The grid was
        # hardcoded to 8 besides, so a FLUX or CogView 4 size could come back off-grid.
        settings = resolve_default_settings(unet_config.base)
        if settings is None or settings.width is None or settings.height is None:
            raise ArchitectureError(
                f"Architecture '{unet_config.base.value}' declares no default dimensions, so there is no "
                "ideal size to compute from."
            )
        model_width = settings.width * self.multiplier
        model_height = settings.height * self.multiplier
        # The variant matters where an architecture's grid depends on it -- Wan TI2V-5B wants
        # multiples of 32 where A14B wants 16. `wan_denoise` reads it off the same config the same
        # way; a config that has no `variant` field, or whose variant is not named, gets the base
        # grid.
        variant = getattr(unet_config, "variant", None)
        grid = require(unet_config.base, FeaturesFacet).resolve_dimension_grid(variant)
        min_dimension = math.floor(min(model_width, model_height) * 0.5)
        # Area, not a squared width. The declared default is a product decision rather than the
        # native training resolution, and for fifteen of sixteen architectures it is square, so
        # this is identical to the old `width * width` there. MiniMax H3 is the exception at
        # 1344x768: squaring 1344 asked for 75% more area than it was trained on. The node is
        # reachable for any `ModelType.Main` -- the loaders emit a `UNetField` only for SD, but
        # `MetadataToModelInvocation` emits one for anything -- so that path is real.
        model_area = model_width * model_height

        if aspect > 1.0:
            init_height = max(min_dimension, math.sqrt(model_area / aspect))
            init_width = init_height * aspect
        else:
            init_width = max(min_dimension, math.sqrt(model_area * aspect))
            init_height = init_width / aspect

        scaled_width, scaled_height = self.trim_to_multiple_of(
            math.floor(init_width),
            math.floor(init_height),
            multiple_of=grid,
        )

        return IdealSizeOutput(width=scaled_width, height=scaled_height)
