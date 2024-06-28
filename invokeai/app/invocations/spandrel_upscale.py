import numpy as np
import torch
from PIL import Image

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


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to torch.Tensor.

    Args:
        image (Image.Image): A PIL Image with shape (H, W, C) and values in the range [0, 255].

    Returns:
        torch.Tensor: A torch.Tensor with shape (N, C, H, W) and values in the range [0, 1].
    """
    image_np = np.array(image)
    # (H, W, C) -> (C, H, W)
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = image_np / 255
    image_tensor = torch.from_numpy(image_np).float()
    # (C, H, W) -> (N, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert torch.Tensor to PIL Image.

    Args:
        tensor (torch.Tensor): A torch.Tensor with shape (N, C, H, W) and values in the range [0, 1].

    Returns:
        Image.Image: A PIL Image with shape (H, W, C) and values in the range [0, 255].
    """
    # (N, C, H, W) -> (C, H, W)
    tensor = tensor.squeeze(0)
    # (C, H, W) -> (H, W, C)
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.clamp(0, 1)
    tensor = (tensor * 255).cpu().detach().numpy().astype(np.uint8)
    image = Image.fromarray(tensor)
    return image


@invocation("upscale_spandrel", title="Upscale (spandrel)", tags=["upscale"], category="upscale", version="1.0.0")
class UpscaleSpandrelInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Upscales an image using any upscaler supported by spandrel (https://github.com/chaiNNer-org/spandrel)."""

    image: ImageField = InputField(description="The input image")
    spandrel_image_to_image_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.spandrel_image_to_image_model, ui_type=UIType.LoRAModel
    )

    @torch.inference_mode()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        # Load the model.
        spandrel_model_info = context.models.load(self.spandrel_image_to_image_model)

        with spandrel_model_info as spandrel_model:
            assert isinstance(spandrel_model, SpandrelImageToImageModel)

            # Prepare input image for inference.
            image_tensor = pil_to_tensor(image)
            image_tensor = image_tensor.to(device=spandrel_model.device, dtype=spandrel_model.dtype)

            # Run inference.
            image_tensor = spandrel_model.run(image_tensor)

        # Convert the output tensor to a PIL image.
        pil_image = tensor_to_pil(image_tensor)
        image_dto = context.images.save(image=pil_image)
        return ImageOutput.build(image_dto)
