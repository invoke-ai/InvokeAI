"""Extract a single frame from a video as an image.

Enables I2V "shot extension": take the last frame of one clip and feed it back
in as the reference image for the next clip, then concatenate the MP4s
externally to get a video longer than the model's single-shot frame budget.
Also useful as a general-purpose video-to-image step.
"""

import imageio.v3 as iio

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    InputField,
    VideoField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_thumbnails import extract_video_frame, probe_video


@invocation(
    "video_frame_extract",
    title="Frame from Video",
    tags=["video", "image", "frame"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class VideoFrameExtractInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Extract a single frame from a video and save it as an image.

    ``frame_index`` is 0-based. Negative indices count from the end, so the
    default of -1 returns the final frame — the typical setup for chaining
    I2V clips into a longer sequence.
    """

    video: VideoField = InputField(description="The video to extract a frame from.")
    frame_index: int = InputField(
        default=-1,
        description="Index of the frame to extract. 0 = first frame, -1 = last frame, -2 = second-to-last, etc.",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        video_path = context.videos.get_path(self.video.video_name)

        # Resolve negative indices against the actual frame count rather than
        # trusting imageio plugins to accept index=-1 uniformly. Use the decoder's
        # frame count (iio.improps) when available — duration*fps can be off-by-one
        # for VFR uploads or containers with approximate metadata, causing
        # frame_index=-1 to point past the final frame.
        index = self.frame_index
        if index < 0:
            n_frames = _decoder_frame_count(video_path)
            if n_frames is None:
                _, _, duration, fps = probe_video(video_path)
                if not fps or duration <= 0:
                    raise ValueError(
                        f"Cannot resolve negative frame index for video {self.video.video_name}: "
                        f"probe returned duration={duration}, fps={fps}."
                    )
                n_frames = int(round(duration * fps))
            if n_frames <= 0:
                raise ValueError(f"Video {self.video.video_name} has no decodable frames (probed {n_frames}).")
            index = n_frames + index
            if index < 0:
                raise ValueError(f"frame_index {self.frame_index} is out of range for a {n_frames}-frame video.")

        frame = extract_video_frame(video_path, frame_index=index)
        if frame is None:
            raise ValueError(f"Failed to extract frame {index} from {self.video.video_name}.")

        image_dto = context.images.save(image=frame)
        return ImageOutput.build(image_dto=image_dto)


def _decoder_frame_count(video_path) -> int | None:
    """Return the exact decoded frame count, or None if neither backend can determine it.

    Tries imageio's improps first (works for a handful of codecs that expose nframes in
    container metadata). For libx264 streams imageio reports ``inf``, so we fall through
    to cv2's ``CAP_PROP_FRAME_COUNT`` which reads the actual packet count. Both sources
    are preferred over the ``duration * fps`` estimate used by the legacy code path,
    which can overshoot by one on VFR uploads or containers with imprecise metadata.
    """
    import math

    try:
        props = iio.improps(video_path, plugin="FFMPEG")
    except Exception:
        props = None
    shape = getattr(props, "shape", None) if props is not None else None
    if shape:
        n = shape[0]
        if not (isinstance(n, float) and not math.isfinite(n)):
            try:
                return int(n)
            except (TypeError, ValueError, OverflowError):
                pass

    # Fallback: cv2 reads libx264 frame counts exactly. We only import cv2 here because
    # it's a heavy module and the improps path covers other codecs without paying that cost.
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return None
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        return count if count > 0 else None
    except Exception:
        return None
