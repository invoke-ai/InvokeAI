"""MiniMax H3 latents-to-image invocation: the still-image (frame extraction) path.

Paired with Denoise - MiniMax H3 at ``num_frames=5`` (the single-block minimum), this is the
text-to-image mode: decode the clip and save one frame as a regular gallery image.
"""

import torch
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.minimax_h3_latents_to_video import decode_video_latents
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation(
    "minimax_h3_latents_to_image",
    title="Latents to Image - MiniMax H3",
    tags=["latents", "image", "vae", "l2i", "minimax"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class MiniMaxH3LatentsToImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decode MiniMax H3 video latents and save a single frame as an image."""

    video_latents: LatentsField = InputField(description=FieldDescriptions.latents, input=Input.Connection)
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection, title="Video VAE")
    frame_index: int = InputField(default=0, ge=0, description="Which decoded frame to save.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.video_latents.latents_name)
        decoded = decode_video_latents(context, self.vae, latents)  # [C, T, H, W] in [0, 1]

        num_frames = decoded.shape[1]
        if self.frame_index >= num_frames:
            raise ValueError(f"frame_index {self.frame_index} is out of range for a {num_frames}-frame clip.")

        frame = decoded[:, self.frame_index].permute(1, 2, 0)  # [H, W, C]
        img_pil = Image.fromarray((255.0 * frame).round().clamp(0, 255).byte().cpu().numpy())

        image_dto = context.images.save(image=img_pil)
        return ImageOutput.build(image_dto)
