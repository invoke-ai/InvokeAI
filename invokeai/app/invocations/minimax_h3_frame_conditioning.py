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
    MiniMaxH3FrameConditioningField,
    OutputField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from invokeai.backend.minimax_h3.keyframe_conditioning import encode_keyframes, prepare_keyframes
from invokeai.backend.minimax_h3.packing import MINIMAX_H3_CANVAS_MULTIPLE
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device


@invocation_output("minimax_h3_frame_conditioning_output")
class MiniMaxH3FrameConditioningOutput(BaseInvocationOutput):
    """Output of the MiniMax H3 keyframe VAE-encoder."""

    frame_conditioning: MiniMaxH3FrameConditioningField = OutputField(
        description=FieldDescriptions.minimax_h3_frame_conditioning
    )


@invocation(
    "minimax_h3_frame_conditioning",
    title="Frame Conditioning - MiniMax H3",
    tags=["conditioning", "minimax", "video", "i2v"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class MiniMaxH3FrameConditioningInvocation(BaseInvocation):
    """VAE-encodes first/last keyframes into MiniMax H3 conditioning rows.

    The rows are clean (the denoise node noise-augments them with the request seed). The same
    images and width/height must also be wired to the Prompt - MiniMax H3 node: the keyframes
    are part of both the packed sequence and the text conditioning.
    """

    first_image: ImageField | None = InputField(
        default=None, description="Keyframe the video starts from (stretched onto the canvas)."
    )
    last_image: ImageField | None = InputField(
        default=None, description="Keyframe the video ends on (cover-cropped onto the canvas)."
    )
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection, title="Video VAE")
    width: int = InputField(
        default=1344, gt=0, multiple_of=MINIMAX_H3_CANVAS_MULTIPLE, description="Target canvas width."
    )
    height: int = InputField(
        default=768, gt=0, multiple_of=MINIMAX_H3_CANVAS_MULTIPLE, description="Target canvas height."
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> MiniMaxH3FrameConditioningOutput:
        if self.first_image is None and self.last_image is None:
            raise ValueError("Frame Conditioning needs a first image, a last image, or both.")

        first = context.images.get_pil(self.first_image.image_name) if self.first_image else None
        last = context.images.get_pil(self.last_image.image_name) if self.last_image else None
        keyframes, anchors = prepare_keyframes(first, last, self.height, self.width)

        vae_info = context.models.load(self.vae.vae)
        if not isinstance(vae_info.model, AutoencoderKLMiniMaxH3):
            raise TypeError(
                f"Expected AutoencoderKLMiniMaxH3 for the MiniMax H3 video VAE, got {type(vae_info.model).__name__}."
            )
        with vae_info.model_on_device() as (_, vae):
            assert isinstance(vae, AutoencoderKLMiniMaxH3)
            context.util.signal_progress("Encoding MiniMax H3 keyframes")
            rows = encode_keyframes(vae, keyframes, device=get_effective_device(vae))

        name = context.tensors.save(tensor=rows.detach().to("cpu"))
        return MiniMaxH3FrameConditioningOutput(
            frame_conditioning=MiniMaxH3FrameConditioningField(
                condition_rows_name=name,
                keyframe_anchors=list(anchors),
                width=self.width,
                height=self.height,
            )
        )
