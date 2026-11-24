"""Generate a video through MiniMax's hosted H3 API and save it in the Gallery."""

from contextlib import ExitStack
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_args

from invokeai.app.invocations.baseinvocation import BaseInvocation, Bottleneck, invocation
from invokeai.app.invocations.fields import (
    ImageField,
    Input,
    InputField,
    MetadataField,
    UIComponent,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.primitives import VideoOutput
from invokeai.app.services.external_generation.errors import (
    ExternalProviderNotConfiguredError,
    ExternalProviderRequestError,
)
from invokeai.app.services.external_generation.minimax_video import (
    GLOBAL_ONLY_MODELS,
    MAX_VIDEO_BYTES,
    MiniMaxModel,
    MiniMaxRatio,
    MiniMaxRegion,
    MiniMaxResolution,
    MiniMaxVideoRequest,
    generate_minimax_video,
    minimax_video_region,
    validate_minimax_image_size,
)
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.mp4_metadata import is_mp4_file
from invokeai.app.util.video_ingest import ingest_media_to_mp4, probe_media_streams
from invokeai.app.util.video_thumbnails import decoder_frame_count, probe_decodable_video


@invocation(
    "minimax_h3_video_generation",
    title="MiniMax H3 Hosted Video",
    tags=["minimax", "h3", "external", "video"],
    category="video",
    version="1.0.0",
    bottleneck=Bottleneck.Network,
    use_cache=False,
)
class MiniMaxH3VideoGenerationInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generate one video with the configured MiniMax account. Each submitted request can incur a charge."""

    model: MiniMaxModel = InputField(
        default="MiniMax-H3",
        description="MiniMax hosted video model",
        ui_choice_labels={
            model: f"{model} (international only)" if model in GLOBAL_ONLY_MODELS else model
            for model in get_args(MiniMaxModel)
        },
    )
    expected_region: MiniMaxRegion | None = InputField(
        default=None, input=Input.Direct, description="Require the region selected when the job was queued."
    )
    prompt: str = InputField(
        min_length=1, max_length=7000, description="Video prompt", ui_component=UIComponent.Textarea
    )
    resolution: MiniMaxResolution = InputField(default="768P", description="Output resolution")
    duration: int = InputField(default=5, ge=4, le=15, description="Video duration in seconds")
    ratio: MiniMaxRatio = InputField(
        default="16:9", description="Aspect ratio. First and last frames set it automatically."
    )
    generate_audio: bool = InputField(default=True, description="Generate synchronized audio")
    first_frame: ImageField | None = InputField(default=None, description="Starting frame")
    last_frame: ImageField | None = InputField(default=None, description="Ending frame")
    reference_images: list[ImageField] = InputField(
        default=[], max_length=9, description="Ordered reference images, used in place of first and last frames"
    )

    def invoke(self, context: InvocationContext) -> VideoOutput:
        api_key, base_url = context.config.get().minimax_credentials()
        if not api_key:
            raise ExternalProviderNotConfiguredError(
                "Configure a MiniMax API key in Model Manager to generate hosted video."
            )
        if self.expected_region is not None and minimax_video_region(base_url) != self.expected_region:
            raise ExternalProviderNotConfiguredError(
                "The MiniMax region changed after this job was queued. Review the region and queue the job again."
            )

        def check_canceled() -> None:
            if context.util.is_canceled():
                raise CanceledException

        check_canceled()
        context.videos.get_authorized_output_board()
        # Authorize and bound every input before decoding the pixels of even the first reference.
        for field in (self.first_frame, self.last_frame, *self.reference_images):
            if field is not None:
                dto = context.images.get_dto(field.image_name)
                validate_minimax_image_size(dto.width, dto.height)
        request = MiniMaxVideoRequest(
            model=self.model,
            prompt=self.prompt,
            resolution=self.resolution,
            duration=self.duration,
            ratio=self.ratio,
            generate_audio=self.generate_audio,
            first_frame=partial(context.images.get_pil, self.first_frame.image_name) if self.first_frame else None,
            last_frame=partial(context.images.get_pil, self.last_frame.image_name) if self.last_frame else None,
            reference_images=[partial(context.images.get_pil, image.image_name) for image in self.reference_images],
        )
        context.util.signal_progress("Submitting video to MiniMax")
        with (
            generate_minimax_video(
                api_key=api_key,
                base_url=base_url,
                request=request,
                is_canceled=context.util.is_canceled,
                on_progress=lambda status: context.util.signal_progress(f"MiniMax video {status}"),
            ) as path,
            ExitStack() as stack,
        ):
            if not is_mp4_file(path):
                raise ExternalProviderRequestError("MiniMax returned an invalid MP4 file.")
            streams = probe_media_streams(path)
            check_canceled()
            if not streams.has_video:
                raise ExternalProviderRequestError("MiniMax returned a file without a video stream.")
            if streams.video_codec != "h264" or streams.audio_codec not in (None, "aac"):
                context.util.signal_progress("Converting MiniMax video for Gallery playback")
                converted = Path(stack.enter_context(TemporaryDirectory())) / "gallery.mp4"
                ingest_media_to_mp4(
                    path,
                    converted,
                    max_output_bytes=MAX_VIDEO_BYTES,
                    probe=streams,
                    is_canceled=context.util.is_canceled,
                )
                path = converted
                check_canceled()
            (width, height, duration, fps), thumbnail = probe_decodable_video(
                path, is_canceled=context.util.is_canceled
            )
            check_canceled()
            # Container duration can include an audio tail beyond the last video frame.
            frames = decoder_frame_count(path) or round(duration * (fps or 0))
            check_canceled()
            metadata = {
                **(self.metadata.root if self.metadata else {}),
                "external_provider": "minimax",
                "external_model_id": self.model,
                "positive_prompt": self.prompt,
                "width": width,
                "height": height,
                "num_frames": frames,
            }
            if fps is not None:
                metadata["fps"] = fps
            else:
                metadata.pop("fps", None)
            video_dto = context.videos.save(
                source_path=path,
                width=width,
                height=height,
                duration=duration,
                fps=fps,
                first_frame=thumbnail,
                metadata=MetadataField(root=metadata),
            )
        return VideoOutput.build(video_dto, num_frames=frames)
