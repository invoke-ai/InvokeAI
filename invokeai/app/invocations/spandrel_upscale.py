import numpy as np
import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice


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
    # TODO(ryand): Figure out how to handle all the spandrel models so that you don't have to enter a string.
    model_path: str = InputField(description="The path to the upscaling model to use.")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        # Load the model.
        # TODO(ryand): Integrate with the model manager.
        model = ModelLoader().load_from_file(self.model_path)
        if not isinstance(model, ImageModelDescriptor):
            raise ValueError(
                f"Loaded a spandrel model of type '{type(model)}'. Only image-to-image models are supported "
                "('ImageModelDescriptor')."
            )

        # Select model device and dtype.
        torch_dtype = TorchDevice.choose_torch_dtype()
        torch_device = TorchDevice.choose_torch_device()
        if (torch_dtype == torch.float16 and not model.supports_half) or (
            torch_dtype == torch.bfloat16 and not model.supports_bfloat16
        ):
            context.logger.warning(
                f"The configured dtype ('{torch_dtype}') is not supported by the {type(model.model)} model. Falling "
                "back to 'float32'."
            )
            torch_dtype = torch.float32
        model.to(device=torch_device, dtype=torch_dtype)

        # Prepare input image for inference.
        image_tensor = pil_to_tensor(image)
        image_tensor = image_tensor.to(device=torch_device, dtype=torch_dtype)

        # Run inference.
        image_tensor = model(image_tensor)

        # Convert the output tensor to a PIL image.
        pil_image = tensor_to_pil(image_tensor)
        image_dto = context.images.save(image=pil_image)
        return ImageOutput.build(image_dto)
