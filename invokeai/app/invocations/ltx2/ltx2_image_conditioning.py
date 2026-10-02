"""LTX-2 first-frame conditioning: H.264 re-compress, fit to the canvas, VAE-encode."""

import torch

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    LTX2VideoConditioningField,
    OutputField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ltx2.constants import LTX2_CANVAS_MULTIPLE, LTX2_IMAGE_CRF
from invokeai.backend.ltx2.image_conditioning import encode_image_latents, fit_to_canvas, recompress_h264
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_ltx2


@invocation_output("ltx2_image_conditioning_output")
class LTX2ImageConditioningOutput(BaseInvocationOutput):
    """A VAE-encoded first frame for LTX-2 image-to-video."""

    video_conditioning: LTX2VideoConditioningField = OutputField(description=FieldDescriptions.ltx2_video_conditioning)


@invocation(
    "ltx2_image_conditioning",
    title="Image Conditioning - LTX-2",
    tags=["conditioning", "image", "ltx", "ltx2", "video"],
    category="conditioning",
    version="1.1.0",
    classification=Classification.Prototype,
)
class LTX2ImageConditioningInvocation(BaseInvocation):
    """Encodes an image as a held frame of an LTX-2 generation.

    The image is re-compressed as a single H.264 frame before it is encoded. That is not an
    optimization: LTX-2 was trained on frames that had been through a video codec, and a pristine
    source is far enough out of that distribution that the clip visibly drifts away from it over
    the first second. 18 is the CRF LTX-2.5 was trained against.
    """

    image: ImageField = InputField(description="The image the video starts from.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection, title="Video VAE")
    width: int = InputField(
        default=1248, gt=0, multiple_of=LTX2_CANVAS_MULTIPLE, description="Canvas width (must match Denoise)."
    )
    height: int = InputField(
        default=704, gt=0, multiple_of=LTX2_CANVAS_MULTIPLE, description="Canvas height (must match Denoise)."
    )
    strength: float = InputField(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="How strongly the generation is held to this frame. 1 keeps it exactly; lower values "
        "let the model redraw it, which can hide a source that does not match the prompt.",
    )
    frame_index: int = InputField(
        default=0,
        description="Latent frame to hold this image at. 0 is the first frame; negative counts from the end, "
        "so -1 makes it the last. A frame held anywhere but 0 is appended to the model's sequence rather "
        "than overwriting the grid, which is what lets it coexist with a first frame.",
    )
    crf: int = InputField(
        default=LTX2_IMAGE_CRF,
        ge=0,
        le=51,
        description="H.264 quality factor the image is re-compressed at before encoding. Higher is more "
        "compressed; 0 skips re-compression, which is off-distribution for this model.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LTX2ImageConditioningOutput:
        image = context.images.get_pil(self.image.image_name)

        # Re-compression comes first, at the source's own resolution, the way the released pipeline
        # applies it: the codec's artefacts are then resampled by the fit, not stamped on after it.
        context.util.signal_progress("Preparing the LTX-2 image conditioning")
        prepared = fit_to_canvas(recompress_h264(image, self.crf), self.height, self.width)

        vae_info = context.models.load(self.vae.vae)
        estimated_working_memory = estimate_vae_working_memory_ltx2(
            operation="encode",
            vae=vae_info.model,
            pixel_height=self.height,
            pixel_width=self.width,
            pixel_frames=1,
        )
        with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (_, vae):
            context.util.signal_progress("Running the LTX-2 video VAE encode")
            latents = encode_image_latents(vae, prepared, device=get_effective_device(vae))

        return LTX2ImageConditioningOutput(
            video_conditioning=LTX2VideoConditioningField(
                latents_name=context.tensors.save(tensor=latents),
                width=self.width,
                height=self.height,
                strength=self.strength,
                frame_index=self.frame_index,
            )
        )
