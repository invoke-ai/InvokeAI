from abc import abstractmethod
from typing import Literal, get_args

from PIL import Image

from invokeai.app.invocations.fields import ColorField, ImageField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.misc import SEED_MAX
from invokeai.backend.image_util.infill_methods.cv2_inpaint import cv2_inpaint
from invokeai.backend.image_util.infill_methods.lama import LaMA
from invokeai.backend.image_util.infill_methods.mosaic import infill_mosaic
from invokeai.backend.image_util.infill_methods.patchmatch import PatchMatch, infill_patchmatch
from invokeai.backend.image_util.infill_methods.tile import infill_tile
from invokeai.backend.util.logging import InvokeAILogger

from .baseinvocation import BaseInvocation, invocation
from .fields import InputField, WithBoard, WithMetadata
from .image import PIL_RESAMPLING_MAP, PIL_RESAMPLING_MODES

logger = InvokeAILogger.get_logger()


def get_infill_methods():
    methods = Literal["tile", "color", "lama", "cv2"]  # TODO: add mosaic back
    if PatchMatch.patchmatch_available():
        methods = Literal["patchmatch", "tile", "color", "lama", "cv2"]  # TODO: add mosaic back
    return methods


INFILL_METHODS = get_infill_methods()
DEFAULT_INFILL_METHOD = "patchmatch" if "patchmatch" in get_args(INFILL_METHODS) else "tile"


class InfillImageProcessorInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Base class for invocations that preprocess images for Infilling"""

    image: ImageField = InputField(description="The image to process")

    @abstractmethod
    def infill(self, image: Image.Image, context: InvocationContext) -> Image.Image:
        """Infill the image with the specified method"""
        pass

    def load_image(self, context: InvocationContext) -> tuple[Image.Image, bool]:
        """Process the image to have an alpha channel before being infilled"""
        image = context.images.get_pil(self.image.image_name)
        has_alpha = True if image.mode == "RGBA" else False
        return image, has_alpha

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Retrieve and process image to be infilled
        input_image, has_alpha = self.load_image(context)

        # If the input image has no alpha channel, return it
        if has_alpha is False:
            return ImageOutput.build(context.images.get_dto(self.image.image_name))

        # Perform Infill action
        infilled_image = self.infill(input_image, context)

        # Create ImageDTO for Infilled Image
        infilled_image_dto = context.images.save(image=infilled_image)

        # Return Infilled Image
        return ImageOutput.build(infilled_image_dto)


@invocation("infill_rgba", title="Solid Color Infill", tags=["image", "inpaint"], category="inpaint", version="1.2.2")
class InfillColorInvocation(InfillImageProcessorInvocation):
    """Infills transparent areas of an image with a solid color"""

    color: ColorField = InputField(
        default=ColorField(r=127, g=127, b=127, a=255),
        description="The color to use to infill",
    )

    def infill(self, image: Image.Image, context: InvocationContext):
        solid_bg = Image.new("RGBA", image.size, self.color.tuple())
        infilled = Image.alpha_composite(solid_bg, image.convert("RGBA"))
        infilled.paste(image, (0, 0), image.split()[-1])
        return infilled


@invocation("infill_tile", title="Tile Infill", tags=["image", "inpaint"], category="inpaint", version="1.2.3")
class InfillTileInvocation(InfillImageProcessorInvocation):
    """Infills transparent areas of an image with tiles of the image"""

    tile_size: int = InputField(default=32, ge=1, description="The tile size (px)")
    seed: int = InputField(
        default=0,
        ge=0,
        le=SEED_MAX,
        description="The seed to use for tile generation (omit for random)",
    )

    def infill(self, image: Image.Image, context: InvocationContext):
        output = infill_tile(image, seed=self.seed, tile_size=self.tile_size)
        return output.infilled


@invocation(
    "infill_patchmatch", title="PatchMatch Infill", tags=["image", "inpaint"], category="inpaint", version="1.2.2"
)
class InfillPatchMatchInvocation(InfillImageProcessorInvocation):
    """Infills transparent areas of an image using the PatchMatch algorithm"""

    downscale: float = InputField(default=2.0, gt=0, description="Run patchmatch on downscaled image to speedup infill")
    resample_mode: PIL_RESAMPLING_MODES = InputField(default="bicubic", description="The resampling mode")

    def infill(self, image: Image.Image, context: InvocationContext):
        resample_mode = PIL_RESAMPLING_MAP[self.resample_mode]

        width = int(image.width / self.downscale)
        height = int(image.height / self.downscale)

        infilled = image.resize(
            (width, height),
            resample=resample_mode,
        )
        infilled = infill_patchmatch(image)
        infilled = infilled.resize(
            (image.width, image.height),
            resample=resample_mode,
        )
        infilled.paste(image, (0, 0), mask=image.split()[-1])

        return infilled


@invocation("infill_lama", title="LaMa Infill", tags=["image", "inpaint"], category="inpaint", version="1.2.2")
class LaMaInfillInvocation(InfillImageProcessorInvocation):
    """Infills transparent areas of an image using the LaMa model"""

    def infill(self, image: Image.Image, context: InvocationContext):
        lama = LaMA(context)
        return lama(image)


@invocation("infill_cv2", title="CV2 Infill", tags=["image", "inpaint"], category="inpaint", version="1.2.2")
class CV2InfillInvocation(InfillImageProcessorInvocation):
    """Infills transparent areas of an image using OpenCV Inpainting"""

    def infill(self, image: Image.Image, context: InvocationContext):
        return cv2_inpaint(image)


# @invocation(
#     "infill_mosaic", title="Mosaic Infill", tags=["image", "inpaint", "outpaint"], category="inpaint", version="1.0.0"
# )
class MosaicInfillInvocation(InfillImageProcessorInvocation):
    """Infills transparent areas of an image with a mosaic pattern drawing colors from the rest of the image"""

    image: ImageField = InputField(description="The image to infill")
    tile_width: int = InputField(default=64, description="Width of the tile")
    tile_height: int = InputField(default=64, description="Height of the tile")
    min_color: ColorField = InputField(
        default=ColorField(r=0, g=0, b=0, a=255),
        description="The min threshold for color",
    )
    max_color: ColorField = InputField(
        default=ColorField(r=255, g=255, b=255, a=255),
        description="The max threshold for color",
    )

    def infill(self, image: Image.Image, context: InvocationContext):
        return infill_mosaic(image, (self.tile_width, self.tile_height), self.min_color.tuple(), self.max_color.tuple())
