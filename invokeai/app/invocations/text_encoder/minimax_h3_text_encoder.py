import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    MiniMaxH3ReferenceMediaField,
    UIComponent,
)
from invokeai.app.invocations.minimax_h3.minimax_h3_reference import (
    _ResolvedVideoRange,
    load_reference_video_frames,
    normalize_reference_list,
    reference_has_audio,
    reference_kind,
    reference_signature_entry,
)
from invokeai.app.invocations.model import MiniMaxH3TextEncoderField
from invokeai.app.invocations.primitives import MiniMaxH3ConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.minimax_h3.keyframe_conditioning import prepare_keyframes
from invokeai.backend.minimax_h3.packing import MINIMAX_H3_CANVAS_MULTIPLE
from invokeai.backend.minimax_h3.reference_conditioning import (
    normalize_reference_image,
    resolve_reference_image_short_edge,
)
from invokeai.backend.minimax_h3.text_conditioning import (
    MiniMaxH3TextReference,
    encode_prompt,
    encode_prompt_ref2va,
)
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
    version="1.1.0",
    classification=Classification.Prototype,
    idle_gpu_offloadable=True,
)
class MiniMaxH3TextEncoderInvocation(BaseInvocation):
    """Encodes a prompt (and optional first/last keyframes, or Ref2VA references) for MiniMax H3.

    The conditioning is Qwen3-VL-32B's *unnormalized* layer-50 hidden state. H3 is
    guidance-distilled: there is no negative prompt. For first/last-frame video, the keyframes
    are ALSO part of the text conditioning (a "<Picture i>: " label plus a vision block per
    keyframe), so the same images must be wired here and to the Frame Conditioning node, with
    the same width/height as the denoise node. For reference-to-video (Ref2VA), the same
    ordered references must be wired here and to the Reference Conditioning node, with the
    same Number of Frames — the references appear in the conditioning as per-modality labels
    and vision blocks (a reference video is read at 2 fps).
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
    references: MiniMaxH3ReferenceMediaField | list[MiniMaxH3ReferenceMediaField] | None = InputField(
        default=None,
        description="Ref2VA references, in conditioning order (must match Reference Conditioning - MiniMax H3). "
        "Mutually exclusive with keyframes.",
        input=Input.Connection,
    )
    num_frames: int | None = InputField(
        default=None,
        gt=0,
        description="The generation's frame count. Required with references (they are truncated to it); "
        "must match the Denoise and Reference Conditioning nodes.",
    )
    width: int = InputField(
        default=1344, gt=0, multiple_of=MINIMAX_H3_CANVAS_MULTIPLE, description="Target canvas width."
    )
    height: int = InputField(
        default=768, gt=0, multiple_of=MINIMAX_H3_CANVAS_MULTIPLE, description="Target canvas height."
    )

    def _build_text_references(
        self, context: InvocationContext, references: list[MiniMaxH3ReferenceMediaField], num_frames: int
    ) -> tuple[list[MiniMaxH3TextReference], list[str]]:
        """Normalize the reference media for the vision context, in packed order.

        The normalization is intentionally identical to Reference Conditioning - MiniMax H3's
        (same helpers, same trim, same canvas rules); the signature entries computed here are
        what the denoise node compares against that node's, so the two sides cannot silently
        disagree. All video decoding happens here, before the text encoder loads.
        """
        text_references: list[MiniMaxH3TextReference] = []
        signatures: list[str] = []
        for reference in references:
            if reference.image is not None:
                image = context.images.get_pil(reference.image.image_name)
                short_edge = resolve_reference_image_short_edge(
                    image.width, image.height, reference.image_detail, self.width * self.height
                )
                normalized = normalize_reference_image(image, short_edge)
                text_references.append(MiniMaxH3TextReference(kind="image", image=normalized))
                signatures.append(reference_signature_entry(reference, normalized.size, num_frames))
                continue
            kind = reference_kind(reference)
            has_audio = reference_has_audio(reference)
            if kind == "video":
                span = _ResolvedVideoRange(context, reference)
                frames = load_reference_video_frames(context, reference, span, num_frames)
                text_references.append(MiniMaxH3TextReference(kind="video", has_audio=has_audio, frames=frames))
            else:
                # An audio-only reference contributes its label alone; whether the file
                # actually carries sound is enforced by the Reference Conditioning node.
                text_references.append(MiniMaxH3TextReference(kind="audio", has_audio=True))
            signatures.append(reference_signature_entry(reference, None, num_frames))
        return text_references, signatures

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> MiniMaxH3ConditioningOutput:
        has_keyframes = self.first_image is not None or self.last_image is not None
        if self.references is not None and has_keyframes:
            raise ValueError(
                "Keyframes (first/last image) and Ref2VA references are mutually exclusive - a request is "
                "either first/last-frame-to-video or reference-to-video."
            )

        keyframes = []
        anchors: tuple[str, ...] = ()
        text_references: list[MiniMaxH3TextReference] = []
        reference_signature: tuple[str, ...] = ()
        if has_keyframes:
            first = context.images.get_pil(self.first_image.image_name) if self.first_image else None
            last = context.images.get_pil(self.last_image.image_name) if self.last_image else None
            keyframes, anchors = prepare_keyframes(first, last, self.height, self.width)
        elif self.references is not None:
            if self.num_frames is None:
                raise ValueError(
                    "num_frames is required with references: they are truncated to the generated duration."
                )
            reference_list = normalize_reference_list(self.references)
            text_references, signatures = self._build_text_references(context, reference_list, self.num_frames)
            reference_signature = tuple(signatures)

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
            if text_references:
                prompt_embeds, text_token_tags = encode_prompt_ref2va(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    processor=processor,
                    prompt=self.prompt,
                    references=text_references,
                    device=device,
                )
            else:
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
                    reference_signature=reference_signature,
                    reference_num_frames=self.num_frames if reference_signature else None,
                )
            ]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return MiniMaxH3ConditioningOutput.build(conditioning_name)
