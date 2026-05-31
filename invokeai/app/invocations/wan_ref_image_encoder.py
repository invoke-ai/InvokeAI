"""Reference-image (VAE-latent) encoder for Wan 2.2 I2V-A14B.

Wan 2.2 I2V conditions on a reference image by VAE-encoding it and
concatenating the resulting latents to the noise latents along the channel
dim. This invocation produces the 20-channel condition tensor (4-ch first-
frame mask + 16-ch image latents) the denoise loop will consume.

Supports both single-frame (image I2V, ``num_frames=1``) and multi-frame
(video I2V, e.g. ``num_frames=81``) condition tensors.
"""

from typing import Optional

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
    encode_reference_image_to_ti2v_condition,
    encode_reference_image_to_video_condition,
)


@invocation(
    "wan_ref_image_encoder",
    title="Reference Image - Wan 2.2",
    tags=["image", "conditioning", "wan", "i2v"],
    category="conditioning",
    version="1.2.0",
    classification=Classification.Prototype,
)
class WanRefImageEncoderInvocation(BaseInvocation):
    """VAE-encode a reference image into Wan 2.2 I2V conditioning.

    Output is a ``[1, 20, T_lat, height // 8, width // 8]`` condition tensor
    that the denoise loop concatenates to the 16-channel noise latents each
    step, producing the 36-channel input the I2V-A14B transformer expects.

    For image (single-frame) I2V leave ``num_frames=1`` (T_lat=1). For video
    I2V set ``num_frames`` to match the value on the video-denoise node
    (e.g. 81 for the Wan 2.2 reference defaults).

    Supply an optional ``end_image`` for **first-last-frame interpolation
    (FLF2V)** — the model then interpolates the motion from ``image`` (first
    frame) to ``end_image`` (final frame). FLF2V is I2V-A14B video only
    (``num_frames > 1``); it is not supported for TI2V-5B or single-frame I2V.

    Only works with I2V-A14B (the denoise loop's variant gate enforces this).
    For T2V or TI2V-5B, omit this node entirely.
    """

    image: ImageField = InputField(description="Reference image to condition on (the first frame).")
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
    num_frames: int = InputField(
        default=1,
        ge=1,
        description="Pixel-frame count to build the condition for. Use 1 for single-frame image "
        "I2V. For video I2V, set this to match the video-denoise node's num_frames (and ensure "
        "(num_frames - 1) %% 4 == 0, e.g. 81).",
        title="Number of Frames",
    )
    end_image: Optional[ImageField] = InputField(
        default=None,
        description="Optional end frame for first-last-frame interpolation (FLF2V). When set, the "
        "video interpolates from the reference image (first frame) to this image (final frame). "
        "I2V-A14B video only (num_frames > 1); not supported for TI2V-5B or single-frame I2V.",
        title="End Image (FLF2V)",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> WanRefImageOutput:
        if self.num_frames > 1 and (self.num_frames - 1) % 4 != 0:
            raise ValueError(
                f"num_frames must satisfy (num_frames - 1) %% 4 == 0 for the Wan VAE's temporal "
                f"compression (got {self.num_frames}). Try 5, 9, 13, ..., 81, 85, ..."
            )

        pil_image = context.images.get_pil(self.image.image_name, "RGB")
        end_pil_image = context.images.get_pil(self.end_image.image_name, "RGB") if self.end_image is not None else None

        vae_info = context.models.load(self.vae.vae)
        device = TorchDevice.choose_torch_device()
        target_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        with vae_info.model_on_device() as (_, vae):
            if not isinstance(vae, AutoencoderKLWan):
                raise TypeError(f"Reference-image encoder requires AutoencoderKLWan, got {type(vae).__name__}.")
            context.util.signal_progress(
                ("VAE-encoding FLF2V start+end images" if end_pil_image is not None else "VAE-encoding reference image")
                + (f" ({self.num_frames} frames)" if self.num_frames > 1 else "")
            )
            # Free cached allocator blocks left over from earlier nodes (denoise expert
            # swaps in particular can leave the cache fragmented in ways that look like
            # free VRAM but fail a single large contiguous request). Mirrors the
            # pattern used in wan_latents_to_image.py / wan_latents_to_video.py.
            TorchDevice.empty_cache()
            # Pick the encoder path by VAE z_dim: 48 means the Wan 2.2-VAE (TI2V-5B),
            # which uses a single-frame 48-channel condition that the denoise loop
            # blends with the noisy latents at every step (expand_timesteps path).
            # 16 means the standard Wan VAE (A14B), which uses the 20-channel
            # mask + latent condition concatenated to noise along the channel dim.
            is_ti2v_5b = getattr(vae.config, "z_dim", 16) == 48
            if end_pil_image is not None and (is_ti2v_5b or self.num_frames <= 1):
                raise ValueError(
                    "End-image (FLF2V) interpolation is only supported for I2V-A14B video "
                    f"(num_frames > 1). Got {'TI2V-5B' if is_ti2v_5b else 'single-frame I2V'}. "
                    "Remove the End Image input, or use an A14B VAE with num_frames > 1."
                )
            if is_ti2v_5b:
                # TI2V-5B I2V needs latent H/W to be even for the transformer
                # patch_size=(1,2,2), so pixel dims must be multiples of 32
                # (16x VAE * 2 transformer patch). A14B's 8x VAE only needed
                # multiples of 16.
                if self.width % 32 != 0 or self.height % 32 != 0:
                    raise ValueError(
                        f"TI2V-5B I2V requires width and height to be multiples of 32 "
                        f"(got {self.width}x{self.height}). The Wan 2.2-VAE uses 16x "
                        f"spatial compression and the transformer adds a 2x patch on "
                        f"top, so pixel dims must divide by 32 for the patchify step."
                    )
                condition = encode_reference_image_to_ti2v_condition(
                    image=pil_image,
                    vae=vae,
                    width=self.width,
                    height=self.height,
                    device=device,
                    dtype=target_dtype,
                )
            elif self.num_frames <= 1:
                condition = encode_reference_image_to_condition(
                    image=pil_image,
                    vae=vae,
                    width=self.width,
                    height=self.height,
                    device=device,
                    dtype=target_dtype,
                )
            else:
                condition = encode_reference_image_to_video_condition(
                    image=pil_image,
                    vae=vae,
                    width=self.width,
                    height=self.height,
                    num_frames=self.num_frames,
                    device=device,
                    dtype=target_dtype,
                    last_image=end_pil_image,
                )

        condition = condition.detach().to("cpu")
        # Release this node's VAE-encode intermediates before the next node tries to
        # partial-load the denoise transformer — the OOM we saw in PR #9163 review
        # was the I2V expert load racing against still-cached encode activations.
        TorchDevice.empty_cache()
        name = context.tensors.save(tensor=condition)
        return WanRefImageOutput.build(
            condition_tensor_name=name,
            width=self.width,
            height=self.height,
            num_frames=self.num_frames,
        )
