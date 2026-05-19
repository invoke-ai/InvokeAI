"""Reference-image (VAE-latent) encoder for Wan 2.2 I2V-A14B.

Wan 2.2 I2V conditions on a reference image by VAE-encoding it and
concatenating the resulting latents to the noise latents along the channel
dim. This invocation produces the 20-channel condition tensor (4-ch first-
frame mask + 16-ch image latents) the denoise loop will consume.
"""

import torch
from diffusers.models.autoencoders import AutoencoderKLWan

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import WanRefImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.wan.extensions.wan_ref_image_extension import (
    encode_reference_image_to_condition,
)


@invocation(
    "wan_ref_image_encoder",
    title="Reference Image - Wan 2.2",
    tags=["image", "conditioning", "wan", "i2v"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class WanRefImageEncoderInvocation(BaseInvocation):
    """VAE-encode a reference image into Wan 2.2 I2V conditioning.

    Output is a ``[1, 20, 1, height // 8, width // 8]`` condition tensor that
    the denoise loop concatenates to the 16-channel noise latents each step,
    producing the 36-channel input the I2V-A14B transformer expects.

    Only works with I2V-A14B (the denoise loop's variant gate enforces this).
    For T2V or TI2V-5B, omit this node entirely.
    """

    image: ImageField = InputField(description="Reference image to condition on.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection, title="VAE")
    # Must match wan_denoise's width/height. multiple_of=16 (not 8) because
    # Wan's transformer patch_size=(1, 2, 2) needs latent H/W to be even.
    width: int = InputField(
        default=1024,
        multiple_of=16,
        description="Width to resize the reference image to (must match denoise width).",
    )
    height: int = InputField(
        default=1024,
        multiple_of=16,
        description="Height to resize the reference image to (must match denoise height).",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> WanRefImageOutput:
        pil_image = context.images.get_pil(self.image.image_name, "RGB")

        vae_info = context.models.load(self.vae.vae)
        device = TorchDevice.choose_torch_device()
        target_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        with vae_info.model_on_device() as (_, vae):
            if not isinstance(vae, AutoencoderKLWan):
                raise TypeError(f"Reference-image encoder requires AutoencoderKLWan, got {type(vae).__name__}.")
            context.util.signal_progress("VAE-encoding reference image")
            condition = encode_reference_image_to_condition(
                image=pil_image,
                vae=vae,
                width=self.width,
                height=self.height,
                device=device,
                dtype=target_dtype,
            )

        condition = condition.detach().to("cpu")
        name = context.tensors.save(tensor=condition)
        return WanRefImageOutput.build(condition_tensor_name=name, width=self.width, height=self.height)
