import numpy as np
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.vto_workflow.extract_channel import ImageChannel, extract_channel
from invokeai.backend.vto_workflow.overlay_pattern import multiply_images
from invokeai.backend.vto_workflow.seamless_mapping import map_seamless_tiles


@invocation("vto", title="Virtual Try-On", tags=["vto"], category="vto", version="1.0.0")
class VTOInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Virtual try-on."""

    original_image: ImageField = InputField(description="The input image")
    clothing_mask: ImageField = InputField(description="Clothing mask.")
    pattern_image: ImageField = InputField(description="Pattern image.")
    pattern_vertical_repeats: int = InputField(description="Number of vertical repeats for the pattern.", default=1)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # TODO(ryand): Avoid all the unnecessary flip-flopping between PIL and numpy.
        original_image = context.images.get_pil(self.original_image.image_name)
        clothing_mask = context.images.get_pil(self.clothing_mask.image_name)
        pattern_image = context.images.get_pil(self.pattern_image.image_name)

        shadows = extract_channel(np.array(original_image), ImageChannel.LAB_L)

        expanded_pattern = map_seamless_tiles(
            seamless_tile=pattern_image,
            target_hw=(original_image.height, original_image.width),
            num_repeats_h=self.pattern_vertical_repeats,
        )

        pattern_with_shadows = multiply_images(expanded_pattern, Image.fromarray(shadows))

        # Merge the pattern with the model image.
        pattern_with_shadows_np = np.array(pattern_with_shadows)
        clothing_mask_np = np.array(clothing_mask)
        original_image_np = np.array(original_image)
        merged_image = np.where(clothing_mask_np[:, :, None], pattern_with_shadows_np, original_image_np)
        merged_image = Image.fromarray(merged_image)

        image_dto = context.images.save(image=merged_image)
        return ImageOutput.build(image_dto)
