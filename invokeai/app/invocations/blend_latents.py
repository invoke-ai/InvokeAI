from typing import Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, Input, InputField, LatentsField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice


def slerp(
    t: Union[float, np.ndarray],
    v0: Union[torch.Tensor, np.ndarray],
    v1: Union[torch.Tensor, np.ndarray],
    device: torch.device,
    DOT_THRESHOLD: float = 0.9995,
):
    """
    Spherical linear interpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                            colineal. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    """
    inputs_are_torch = False
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1, np.ndarray):
        inputs_are_torch = True
        v1 = v1.detach().cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(device)

    return v2


@invocation(
    "lblend",
    title="Blend Latents",
    tags=["latents", "blend", "mask"],
    category="latents",
    version="1.1.0",
)
class BlendLatentsInvocation(BaseInvocation):
    """Blend two latents using a given alpha. If a mask is provided, the second latents will be masked before blending.
    Latents must have same size. Masking functionality added by @dwringer."""

    latents_a: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    latents_b: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    mask: Optional[ImageField] = InputField(default=None, description="Mask for blending in latents B")
    alpha: float = InputField(ge=0, default=0.5, description=FieldDescriptions.blend_alpha)

    def prep_mask_tensor(self, mask_image: Image.Image) -> torch.Tensor:
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        mask_tensor = image_resized_to_grid_as_tensor(mask_image, normalize=False)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_tensor

    def replace_tensor_from_masked_tensor(
        self, tensor: torch.Tensor, other_tensor: torch.Tensor, mask_tensor: torch.Tensor
    ):
        output = tensor.clone()
        mask_tensor = mask_tensor.expand(output.shape)
        if output.dtype != torch.float16:
            output = torch.add(output, mask_tensor * torch.sub(other_tensor, tensor))
        else:
            output = torch.add(output, mask_tensor.half() * torch.sub(other_tensor, tensor))
        return output

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents_a = context.tensors.load(self.latents_a.latents_name)
        latents_b = context.tensors.load(self.latents_b.latents_name)
        if self.mask is None:
            mask_tensor = torch.zeros(latents_a.shape[-2:])
        else:
            mask_tensor = self.prep_mask_tensor(context.images.get_pil(self.mask.image_name))
            mask_tensor = tv_resize(mask_tensor, latents_a.shape[-2:], T.InterpolationMode.BILINEAR, antialias=False)

        latents_b = self.replace_tensor_from_masked_tensor(latents_b, latents_a, mask_tensor)

        if latents_a.shape != latents_b.shape:
            raise ValueError("Latents to blend must be the same size.")

        device = TorchDevice.choose_torch_device()

        # blend
        blended_latents = slerp(self.alpha, latents_a, latents_b, device)

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        blended_latents = blended_latents.to("cpu")
        torch.cuda.empty_cache()

        name = context.tensors.save(tensor=blended_latents)
        return LatentsOutput.build(latents_name=name, latents=blended_latents)
