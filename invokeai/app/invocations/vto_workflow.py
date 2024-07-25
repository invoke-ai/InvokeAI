import cv2
import numpy as np
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.vto_workflow.extract_channel import ImageChannel, extract_channel
from invokeai.backend.vto_workflow.overlay_pattern import multiply_images
from invokeai.backend.vto_workflow.seamless_mapping import map_seamless_tiles


@invocation("vto", title="Virtual Try-On", tags=["vto"], category="vto", version="1.1.0")
class VTOInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Virtual try-on."""

    original_image: ImageField = InputField(description="The input image")
    clothing_mask: ImageField = InputField(description="Clothing mask.")
    pattern_image: ImageField = InputField(description="Pattern image.")
    pattern_vertical_repeats: float = InputField(
        description="Number of vertical repeats for the pattern.", gt=0.01, default=1.0
    )

    shading_max: float = InputField(
        description="The lightness of the light spots on the clothing. Default is 1.0. Typically in the range [0.7, 1.2]. Must be > shading_min",
        default=1.0,
        ge=0.0,
    )
    shading_min: float = InputField(
        description="The lightness of the dark spots on the clothing. Default id 0.5. Typically in the range [0.2, 0.7]",
        default=0.5,
        ge=0.0,
    )

    mask_dilation: int = InputField(
        description="The number of pixels to dilate the mask by. Default is 1.",
        default=1,
        ge=0,
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # TODO(ryand): Avoid all the unnecessary flip-flopping between PIL and numpy.
        original_image = context.images.get_pil(self.original_image.image_name)
        clothing_mask = context.images.get_pil(self.clothing_mask.image_name)
        pattern_image = context.images.get_pil(self.pattern_image.image_name)

        shadows = extract_channel(np.array(original_image), ImageChannel.LAB_L)

        # Clip the shadows to the 0.05 and 0.95 percentiles to eliminate outliers.
        shadows = np.clip(shadows, np.percentile(shadows, 5), np.percentile(shadows, 95))

        # Normalize the shadows to the range [shading_min, shading_max].
        assert self.shading_min < self.shading_max
        shadows = shadows.astype(np.float32)
        shadows = (shadows - shadows.min()) / (shadows.max() - shadows.min())
        shadows = self.shading_min + (self.shading_max - self.shading_min) * shadows
        shadows = np.clip(shadows, 0.0, 1.0)
        shadows = (shadows * 255).astype(np.uint8)

        expanded_pattern = map_seamless_tiles(
            seamless_tile=pattern_image,
            target_hw=(original_image.height, original_image.width),
            num_repeats_h=self.pattern_vertical_repeats,
        )

        pattern_with_shadows = multiply_images(expanded_pattern, Image.fromarray(shadows))

        # Dilate the mask.
        clothing_mask_np = np.array(clothing_mask)
        if self.mask_dilation > 0:
            clothing_mask_np = cv2.dilate(clothing_mask_np, np.ones((3, 3), np.uint8), iterations=self.mask_dilation)

        # Merge the pattern with the model image.
        pattern_with_shadows_np = np.array(pattern_with_shadows)
        original_image_np = np.array(original_image)
        merged_image = np.where(clothing_mask_np[:, :, None], pattern_with_shadows_np, original_image_np)
        merged_image = Image.fromarray(merged_image)

        image_dto = context.images.save(image=merged_image)
        return ImageOutput.build(image_dto)
