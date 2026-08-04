import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    UIComponent,
)
from invokeai.app.invocations.model import MiniMaxH3TextEncoderField
from invokeai.app.invocations.primitives import MiniMaxH3ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.minimax_h3.keyframe_conditioning import prepare_keyframes
from invokeai.backend.minimax_h3.packing import MINIMAX_H3_CANVAS_MULTIPLE
from invokeai.backend.minimax_h3.text_conditioning import encode_prompt
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    MiniMaxH3ConditioningInfo,
)


@invocation(
    "minimax_h3_text_encoder",
    title="Prompt - MiniMax H3",
    tags=["prompt", "conditioning", "minimax", "video"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
    idle_gpu_offloadable=True,
)
class MiniMaxH3TextEncoderInvocation(BaseInvocation):
    """Encodes a prompt (and optional first/last keyframes) for MiniMax H3.

    The conditioning is Qwen3-VL-32B's *unnormalized* layer-50 hidden state. H3 is
    guidance-distilled: there is no negative prompt. For first/last-frame video, the keyframes
    are ALSO part of the text conditioning (a "<Picture i>: " label plus a vision block per
    keyframe), so the same images must be wired here and to the Frame Conditioning node, with
    the same width/height as the denoise node.
    """

    prompt: str = InputField(description="Text prompt for MiniMax H3.", ui_component=UIComponent.Textarea)
    text_encoder: MiniMaxH3TextEncoderField = InputField(
        title="Qwen3-VL Encoder",
        description=FieldDescriptions.minimax_h3_text_encoder,
        input=Input.Connection,
    )
    first_image: ImageField | None = InputField(
        default=None, description="Optional keyframe the video starts from (must match Frame Conditioning)."
    )
    last_image: ImageField | None = InputField(
        default=None, description="Optional keyframe the video ends on (must match Frame Conditioning)."
    )
    width: int = InputField(
        default=1344, gt=0, multiple_of=MINIMAX_H3_CANVAS_MULTIPLE, description="Target canvas width."
    )
    height: int = InputField(
        default=768, gt=0, multiple_of=MINIMAX_H3_CANVAS_MULTIPLE, description="Target canvas height."
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> MiniMaxH3ConditioningOutput:
        keyframes = []
        anchors: tuple[str, ...] = ()
        if self.first_image is not None or self.last_image is not None:
            first = context.images.get_pil(self.first_image.image_name) if self.first_image else None
            last = context.images.get_pil(self.last_image.image_name) if self.last_image else None
            keyframes, anchors = prepare_keyframes(first, last, self.height, self.width)

        # Load the largest model first: loading the ~22 GB text encoder evicts unlocked small
        # entries from the RAM cache, so tokenizer/processor loaded before it would be dropped
        # and re-read ("model loading order is non-optimal", issue #7513).
        text_encoder_info = context.models.load(self.text_encoder.text_encoder)
        tokenizer_info = context.models.load(self.text_encoder.tokenizer)
        processor_info = context.models.load(self.text_encoder.processor)
        with (
            tokenizer_info.model_on_device() as (_, tokenizer),
            processor_info.model_on_device() as (_, processor),
            text_encoder_info.model_on_device() as (_, text_encoder),
        ):
            device = get_effective_device(text_encoder)
            context.util.signal_progress("Running Qwen3-VL text encoder")
            prompt_embeds, text_token_tags = encode_prompt(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                processor=processor,
                prompt=self.prompt,
                keyframe_images=keyframes,
                device=device,
            )

        # Persist on CPU; required by the idle-GPU-offload contract of this node.
        conditioning_data = ConditioningFieldData(
            conditionings=[
                MiniMaxH3ConditioningInfo(
                    prompt_embeds=prompt_embeds.detach().to("cpu"),
                    text_token_tags=text_token_tags.detach().to("cpu"),
                    keyframe_anchors=anchors,
                    width=self.width if anchors else None,
                    height=self.height if anchors else None,
                )
            ]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return MiniMaxH3ConditioningOutput.build(conditioning_name)
