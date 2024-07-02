import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    InputField,
    UIType,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.spandrel_image_to_image_model import SpandrelImageToImageModel


@invocation("upscale_spandrel", title="Upscale (spandrel)", tags=["upscale"], category="upscale", version="1.0.0")
class UpscaleSpandrelInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Upscales an image using any upscaler supported by spandrel (https://github.com/chaiNNer-org/spandrel)."""

    image: ImageField = InputField(description="The input image")
    spandrel_image_to_image_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.spandrel_image_to_image_model, ui_type=UIType.SpandrelImageToImageModel
    )

    @torch.inference_mode()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        # Load the model.
        spandrel_model_info = context.models.load(self.spandrel_image_to_image_model)

        with spandrel_model_info as spandrel_model:
            assert isinstance(spandrel_model, SpandrelImageToImageModel)

            # Prepare input image for inference.
            image_tensor = SpandrelImageToImageModel.pil_to_tensor(image)
            image_tensor = image_tensor.to(device=spandrel_model.device, dtype=spandrel_model.dtype)

            # Run inference.
            image_tensor = spandrel_model.run(image_tensor)

        # Convert the output tensor to a PIL image.
        pil_image = SpandrelImageToImageModel.tensor_to_pil(image_tensor)
        image_dto = context.images.save(image=pil_image)
        return ImageOutput.build(image_dto)
