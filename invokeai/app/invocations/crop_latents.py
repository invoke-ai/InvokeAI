from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, LatentsField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


# The Crop Latents node was copied from @skunkworxdark's implementation here:
# https://github.com/skunkworxdark/XYGrid_nodes/blob/74647fa9c1fa57d317a94bd43ca689af7f0aae5e/images_to_grids.py#L1117C1-L1167C80
@invocation(
    "crop_latents",
    title="Crop Latents",
    tags=["latents", "crop"],
    category="latents",
    version="1.0.2",
)
# TODO(ryand): Named `CropLatentsCoreInvocation` to prevent a conflict with custom node `CropLatentsInvocation`.
# Currently, if the class names conflict then 'GET /openapi.json' fails.
class CropLatentsCoreInvocation(BaseInvocation):
    """Crops a latent-space tensor to a box specified in image-space. The box dimensions and coordinates must be
    divisible by the latent scale factor of 8.
    """

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    x: int = InputField(
        ge=0,
        multiple_of=LATENT_SCALE_FACTOR,
        description="The left x coordinate (in px) of the crop rectangle in image space. This value will be converted to a dimension in latent space.",
    )
    y: int = InputField(
        ge=0,
        multiple_of=LATENT_SCALE_FACTOR,
        description="The top y coordinate (in px) of the crop rectangle in image space. This value will be converted to a dimension in latent space.",
    )
    width: int = InputField(
        ge=1,
        multiple_of=LATENT_SCALE_FACTOR,
        description="The width (in px) of the crop rectangle in image space. This value will be converted to a dimension in latent space.",
    )
    height: int = InputField(
        ge=1,
        multiple_of=LATENT_SCALE_FACTOR,
        description="The height (in px) of the crop rectangle in image space. This value will be converted to a dimension in latent space.",
    )

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.tensors.load(self.latents.latents_name)

        x1 = self.x // LATENT_SCALE_FACTOR
        y1 = self.y // LATENT_SCALE_FACTOR
        x2 = x1 + (self.width // LATENT_SCALE_FACTOR)
        y2 = y1 + (self.height // LATENT_SCALE_FACTOR)

        cropped_latents = latents[..., y1:y2, x1:x2]

        name = context.tensors.save(tensor=cropped_latents)

        return LatentsOutput.build(latents_name=name, latents=cropped_latents)
