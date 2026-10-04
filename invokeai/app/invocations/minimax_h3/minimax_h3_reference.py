"""MiniMax H3 Ref2VA reference invocations.

Three nodes cover the reference side of a Ref2VA request:

- ``minimax_h3_image_reference`` / ``minimax_h3_video_reference`` are cheap descriptor
  nodes — they carry the media name plus its conditioning options, so an ordered,
  heterogeneous reference list is expressible through a Collect node (order IS part of the
  request: a different order is a different generation).
- ``minimax_h3_reference_conditioning`` does all the VAE work: it normalizes and encodes
  every reference, in order, into clean conditioning rows.

The same reference list must also be wired to Prompt - MiniMax H3, which normalizes the
media independently for the Qwen3-VL vision context (the FL2VA wire-it-to-both doctrine);
the denoise node cross-checks the two sides via each reference's signature string.
"""

from typing import Literal

import numpy as np
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
    MiniMaxH3EncodedReferenceField,
    MiniMaxH3ReferenceConditioningField,
    MiniMaxH3ReferenceMediaField,
    OutputField,
    UIComponent,
    VideoField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_audio import extract_audio_pcm
from invokeai.app.util.video_thumbnails import decoder_frame_count, iter_video_frames, probe_video
from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from invokeai.backend.minimax_h3.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
from invokeai.backend.minimax_h3.packing import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    validate_reference_kinds,
)
from invokeai.backend.minimax_h3.presets import MINIMAX_H3_MIN_VIDEO_FRAMES
from invokeai.backend.minimax_h3.reference_conditioning import (
    encode_reference_audio,
    encode_reference_image,
    encode_reference_video,
    normalize_reference_audio,
    normalize_reference_image,
    normalize_reference_video_frames_streaming,
    resolve_reference_image_short_edge,
)
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_minimax_h3

# UI labels for the per-video conditioning selector.
VIDEO_CONDITIONING_LABELS: dict[str, str] = {
    "video_audio": "Video + audio",
    "video": "Video only",
    "audio": "Audio only",
}

IMAGE_DETAIL_LABELS: dict[str, str] = {
    "max": "Max (2048px, slower)",
    "match": "Match generation size",
}


@invocation_output("minimax_h3_reference_media_output")
class MiniMaxH3ReferenceMediaOutput(BaseInvocationOutput):
    """One ordered Ref2VA reference descriptor."""

    reference: MiniMaxH3ReferenceMediaField = OutputField(description=FieldDescriptions.minimax_h3_reference_media)


@invocation(
    "minimax_h3_image_reference",
    title="Image Reference - MiniMax H3",
    tags=["conditioning", "reference", "minimax", "video"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class MiniMaxH3ImageReferenceInvocation(BaseInvocation):
    """Describes one image reference for MiniMax H3 Ref2VA.

    Collect references in the order they should condition in — order is part of the request —
    and wire the collection to both Reference Conditioning - MiniMax H3 and Prompt - MiniMax H3.
    """

    image: ImageField = InputField(description="The reference image.")
    detail: Literal["max", "match"] = InputField(
        default="max",
        description="'max' conditions at a 2048px short edge (highest fidelity; a 2048px square reference adds "
        "~4096 rows to every denoising step). 'match' scales the reference to the generation's pixel area — "
        "several times faster.",
        ui_choice_labels=IMAGE_DETAIL_LABELS,
    )

    def invoke(self, context: InvocationContext) -> MiniMaxH3ReferenceMediaOutput:
        return MiniMaxH3ReferenceMediaOutput(
            reference=MiniMaxH3ReferenceMediaField(image=self.image, image_detail=self.detail)
        )


@invocation(
    "minimax_h3_video_reference",
    title="Video Reference - MiniMax H3",
    tags=["conditioning", "reference", "minimax", "video", "audio"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class MiniMaxH3VideoReferenceInvocation(BaseInvocation):
    """Describes one video reference for MiniMax H3 Ref2VA.

    'Audio only' conditions on the video's soundtrack alone (the standalone audio-reference
    kind). At least one reference of the request must contribute visuals, and a conditioning
    choice that includes audio requires the video to actually carry a soundtrack.
    """

    video: VideoField = InputField(description="The reference video.")
    conditioning: Literal["video_audio", "video", "audio"] = InputField(
        default="video_audio",
        description="Which streams this reference conditions: video + soundtrack, video only, or soundtrack only.",
        ui_choice_labels=VIDEO_CONDITIONING_LABELS,
    )
    start_frame: int = InputField(
        default=0,
        description="First source frame of the reference range (inclusive, 0-based; negative counts from the end).",
        ui_component=UIComponent.VideoFrameIndex,
    )
    end_frame: int = InputField(
        default=-1,
        description="Last source frame of the reference range (inclusive; negative counts from the end).",
        ui_component=UIComponent.VideoFrameIndex,
    )

    def invoke(self, context: InvocationContext) -> MiniMaxH3ReferenceMediaOutput:
        return MiniMaxH3ReferenceMediaOutput(
            reference=MiniMaxH3ReferenceMediaField(
                video=self.video,
                video_conditioning=self.conditioning,
                start_frame=self.start_frame,
                end_frame=self.end_frame,
            )
        )


def normalize_reference_list(
    references: MiniMaxH3ReferenceMediaField | list[MiniMaxH3ReferenceMediaField],
) -> list[MiniMaxH3ReferenceMediaField]:
    """A single reference or a Collect-fed list onto a validated, ordered list."""
    entries = references if isinstance(references, list) else [references]
    for index, entry in enumerate(entries):
        if (entry.image is None) == (entry.video is None):
            raise ValueError(f"Reference {index + 1} must carry exactly one of an image or a video.")
    validate_reference_kinds([reference_kind(entry) for entry in entries])
    return entries


def reference_kind(reference: MiniMaxH3ReferenceMediaField) -> str:
    """The packed-block kind of a reference: 'image', 'video', or 'audio' (audio-only video)."""
    if reference.image is not None:
        return "image"
    return "audio" if reference.video_conditioning == "audio" else "video"


def reference_has_audio(reference: MiniMaxH3ReferenceMediaField) -> bool:
    """Whether the reference contributes audio rows (and an '<Audio j>: ' label)."""
    return reference.video is not None and reference.video_conditioning in ("video_audio", "audio")


class _ResolvedVideoRange:
    """The trimmed source range of a video reference, resolved against the real frame count."""

    def __init__(self, context: InvocationContext, reference: MiniMaxH3ReferenceMediaField) -> None:
        assert reference.video is not None
        self.path = context.videos.get_path(reference.video.video_name)
        _width, _height, duration, fps = probe_video(self.path)
        n_frames = decoder_frame_count(self.path)
        if n_frames is None and duration and fps:
            n_frames = max(1, round(duration * fps))
        if n_frames is None or n_frames <= 0:
            raise ValueError(f"Could not determine the frame count of reference video {reference.video.video_name}.")
        if fps is not None and fps > 0:
            self.fps = float(fps)
        elif duration and duration > 0:
            self.fps = n_frames / float(duration)
        else:
            raise ValueError(f"Could not determine the frame rate of reference video {reference.video.video_name}.")

        start = reference.start_frame + n_frames if reference.start_frame < 0 else reference.start_frame
        if start < 0 or start >= n_frames:
            raise ValueError(
                f"start_frame={reference.start_frame} is out of range for a {n_frames}-frame reference video."
            )
        self.start = start
        # The sample LENGTH is pinned to what the clip can actually supply rather than
        # rejected: "start here and take N frames" past the last frame reads as "take the
        # rest of the clip", and the panel's frame count is an estimate the real count can
        # undershoot. A start frame that is itself off the end has no such reading, and an
        # end before the start is an inverted range, so both still raise.
        end = reference.end_frame + n_frames if reference.end_frame < 0 else reference.end_frame
        if end < self.start:
            raise ValueError(
                f"end_frame ({reference.end_frame} -> {end}) must not be before start_frame "
                f"({reference.start_frame} -> {self.start})."
            )
        self.end = min(end, n_frames - 1)


def load_reference_video_frames(
    context: InvocationContext, reference: MiniMaxH3ReferenceMediaField, span: _ResolvedVideoRange, num_frames: int
) -> np.ndarray:
    """Decode, trim and normalize a video reference's frames (24 fps, canvas, truncated).

    The decoder stream is fully drained (or early-stopped) here, BEFORE any GPU work — the
    app holds a single global video-stream slot with an inactivity timeout, so decode must
    never interleave with model execution.
    """

    def trimmed():
        for index, frame in enumerate(iter_video_frames(span.path, is_canceled=context.util.is_canceled)):
            if index < span.start:
                continue
            if index > span.end:
                return
            yield frame

    return normalize_reference_video_frames_streaming(trimmed(), span.fps, num_frames)


def load_reference_audio(
    reference: MiniMaxH3ReferenceMediaField, span: _ResolvedVideoRange, num_frames: int
) -> torch.Tensor:
    """Extract, trim and normalize a video reference's soundtrack to 32 kHz stereo.

    Raises with an actionable message when the container has no audio stream — the user
    asked for audio conditioning, so silence must not be conditioned on unannounced.
    """
    assert reference.video is not None
    extracted = extract_audio_pcm(span.path, float_pcm=True)
    if extracted is None:
        raise ValueError(
            f"Reference video {reference.video.video_name} has no audio track, but its conditioning is "
            f"'{VIDEO_CONDITIONING_LABELS[reference.video_conditioning]}'. Pick 'Video only' or use a video "
            "with sound."
        )
    pcm, rate = extracted
    window_start = round(span.start * rate / span.fps)
    window_end = round((span.end + 1) * rate / span.fps)
    # Only the generated duration ever survives normalize_reference_audio, so bound the
    # window here rather than carrying samples that are about to be dropped. This is what
    # keeps the silence-pad below cheap: np.pad allocates the WHOLE window, and an
    # audio-only reference defaults to its entire clip, so padding an unbounded window
    # would copy a full-length track to preserve an alignment truncation then discards.
    window_end = min(window_end, window_start + int(num_frames / MINIMAX_H3_FPS * rate))
    window = pcm[:, window_start:window_end]
    if window.shape[1] == 0:
        raise ValueError(
            f"The trimmed range of reference video {reference.video.video_name} lies past the end of its audio track."
        )
    if window.shape[1] < window_end - window_start:
        # Track shorter than the video: keep temporal alignment by padding with silence.
        window = np.pad(window, ((0, 0), (0, (window_end - window_start) - window.shape[1])))
    return normalize_reference_audio(window, rate, num_frames)


def reference_signature_entry(
    reference: MiniMaxH3ReferenceMediaField, normalized_image_size: tuple[int, int] | None, num_frames: int
) -> str:
    """The deterministic fingerprint of one reference, derivable identically on both the
    prompt and the encoder side, compared by the denoise node."""
    if reference.image is not None:
        assert normalized_image_size is not None
        width, height = normalized_image_size
        return f"image:{reference.image.image_name}:{reference.image_detail}:{width}x{height}"
    assert reference.video is not None
    kind = reference_kind(reference)
    streams = {"video_audio": "va", "video": "v", "audio": "a"}[reference.video_conditioning]
    return f"{kind}:{reference.video.video_name}:{streams}:{reference.start_frame}-{reference.end_frame}:{num_frames}"


@invocation_output("minimax_h3_reference_conditioning_output")
class MiniMaxH3ReferenceConditioningOutput(BaseInvocationOutput):
    """Output of the MiniMax H3 reference VAE-encoder."""

    reference_conditioning: MiniMaxH3ReferenceConditioningField = OutputField(
        description=FieldDescriptions.minimax_h3_reference_conditioning
    )


@invocation(
    "minimax_h3_reference_conditioning",
    title="Reference Conditioning - MiniMax H3",
    tags=["conditioning", "reference", "minimax", "video", "audio"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class MiniMaxH3ReferenceConditioningInvocation(BaseInvocation):
    """VAE-encodes an ordered list of Ref2VA references into conditioning rows.

    The rows are clean (the denoise node noise-augments the visual rows with the request
    seed; soundtrack rows stay clean). The same references, in the same order, must also be
    wired to Prompt - MiniMax H3: they are part of both the packed sequence and the text
    conditioning, and the denoise node rejects a mismatch.
    """

    references: MiniMaxH3ReferenceMediaField | list[MiniMaxH3ReferenceMediaField] = InputField(
        description="The references, in conditioning order (order is part of the request).",
        input=Input.Connection,
    )
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection, title="Video VAE")
    audio_vae: VAEField = InputField(
        description=FieldDescriptions.minimax_h3_audio_vae, input=Input.Connection, title="Audio VAE"
    )
    num_frames: int = InputField(
        default=124,
        gt=0,
        description="The frame count the generation will run at (17n+5). References are truncated to this "
        "duration; it must match the Denoise node's Number of Frames.",
    )
    width: int = InputField(
        default=1344,
        gt=0,
        multiple_of=MINIMAX_H3_CANVAS_MULTIPLE,
        description="Target canvas width (used only to size 'match'-detail image references).",
    )
    height: int = InputField(
        default=768,
        gt=0,
        multiple_of=MINIMAX_H3_CANVAS_MULTIPLE,
        description="Target canvas height (used only to size 'match'-detail image references).",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> MiniMaxH3ReferenceConditioningOutput:
        references = normalize_reference_list(self.references)
        if self.num_frames < MINIMAX_H3_MIN_VIDEO_FRAMES:
            raise ValueError(
                f"Reference-to-video needs at least {MINIMAX_H3_MIN_VIDEO_FRAMES} frames (the still-image "
                "clip is not supported)."
            )

        # Phase 1 — ingest and normalize every reference on the CPU. All video decoding
        # happens here, before any model load: the app's decoder has one global stream slot
        # and a per-frame inactivity timeout, so it must never wait on GPU work.
        normalized_images: list = [None] * len(references)
        normalized_frames: list = [None] * len(references)
        normalized_audio: list = [None] * len(references)
        signatures: list[str] = []
        for index, reference in enumerate(references):
            if context.util.is_canceled():
                raise CanceledException
            if reference.image is not None:
                image = context.images.get_pil(reference.image.image_name)
                short_edge = resolve_reference_image_short_edge(
                    image.width, image.height, reference.image_detail, self.width * self.height
                )
                normalized = normalize_reference_image(image, short_edge)
                normalized_images[index] = normalized
                signatures.append(reference_signature_entry(reference, normalized.size, self.num_frames))
                continue
            span = _ResolvedVideoRange(context, reference)
            if reference_kind(reference) == "video":
                normalized_frames[index] = load_reference_video_frames(context, reference, span, self.num_frames)
            if reference_has_audio(reference):
                normalized_audio[index] = load_reference_audio(reference, span, self.num_frames)
            signatures.append(reference_signature_entry(reference, None, self.num_frames))

        # Phase 2 — encode the visual references through the video VAE, in packed order.
        encoded: list[MiniMaxH3EncodedReferenceField] = [
            MiniMaxH3EncodedReferenceField(kind=reference_kind(reference)) for reference in references
        ]
        if any(image is not None for image in normalized_images) or any(
            frames is not None for frames in normalized_frames
        ):
            vae_info = context.models.load(self.vae.vae)
            if not isinstance(vae_info.model, AutoencoderKLMiniMaxH3):
                raise TypeError(
                    f"Expected AutoencoderKLMiniMaxH3 for the MiniMax H3 video VAE, got "
                    f"{type(vae_info.model).__name__}."
                )
            working_memory = 0
            for index in range(len(references)):
                if normalized_images[index] is not None:
                    width, height = normalized_images[index].size
                    working_memory = max(
                        working_memory,
                        estimate_vae_working_memory_minimax_h3("encode", vae_info.model, height, width, 1),
                    )
                elif normalized_frames[index] is not None:
                    frames = normalized_frames[index]
                    # encode_reference_video streams 17-frame chunks to the device (the
                    # moments are latent-sized), so the clip term of the estimate covers
                    # one chunk, not the whole reference - a 361-frame clip would otherwise
                    # reserve ~6 GB it never uses.
                    working_memory = max(
                        working_memory,
                        estimate_vae_working_memory_minimax_h3(
                            "encode",
                            vae_info.model,
                            frames.shape[1],
                            frames.shape[2],
                            min(frames.shape[0], int(vae_info.model.config.clip_length)),
                        ),
                    )
            with vae_info.model_on_device(working_mem_bytes=working_memory) as (_, vae):
                assert isinstance(vae, AutoencoderKLMiniMaxH3)
                device = get_effective_device(vae)
                for index in range(len(references)):
                    if context.util.is_canceled():
                        raise CanceledException
                    if normalized_images[index] is not None:
                        rows, latent_shape = encode_reference_image(vae, normalized_images[index], device)
                    elif normalized_frames[index] is not None:
                        rows, latent_shape = encode_reference_video(vae, normalized_frames[index], device)
                    else:
                        continue
                    encoded[index].video_rows_name = context.tensors.save(tensor=rows.detach().cpu())
                    encoded[index].latent_frames = latent_shape[0]
                    encoded[index].latent_height = latent_shape[1]
                    encoded[index].latent_width = latent_shape[2]
                    normalized_images[index] = None
                    normalized_frames[index] = None

        # Phase 3 — encode the soundtracks through the audio VAE, in packed order.
        if any(waveform is not None for waveform in normalized_audio):
            audio_vae_info = context.models.load(self.audio_vae.vae)
            if not isinstance(audio_vae_info.model, AutoencoderKLMiniMaxH3Audio):
                raise TypeError(
                    f"Expected AutoencoderKLMiniMaxH3Audio for the MiniMax H3 audio VAE, got "
                    f"{type(audio_vae_info.model).__name__}."
                )
            with audio_vae_info.model_on_device() as (_, audio_vae):
                assert isinstance(audio_vae, AutoencoderKLMiniMaxH3Audio)
                audio_device = get_effective_device(audio_vae)
                for index, waveform in enumerate(normalized_audio):
                    if waveform is None:
                        continue
                    rows = encode_reference_audio(audio_vae, waveform, audio_device)
                    encoded[index].audio_rows_name = context.tensors.save(tensor=rows.detach().cpu())

        return MiniMaxH3ReferenceConditioningOutput(
            reference_conditioning=MiniMaxH3ReferenceConditioningField(
                references=encoded,
                num_frames=self.num_frames,
                signature=signatures,
                width=self.width,
                height=self.height,
            )
        )
