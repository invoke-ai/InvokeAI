"""The tail of a clip, encoded as the opening frames an LTX-2 extension continues from."""

from pathlib import Path

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
    Input,
    InputField,
    LTX2AudioConditioningField,
    LTX2VideoConditioningField,
    OutputField,
    VideoField,
)
from invokeai.app.invocations.model import LTX2VocoderField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.video_audio import extract_audio_pcm
from invokeai.backend.ltx2.audio_conditioning import encode_audio_latents
from invokeai.backend.ltx2.clip_frames import encode_canvas_clip, read_canvas_frames
from invokeai.backend.ltx2.constants import (
    LTX2_AUDIO_LATENTS_PER_SECOND,
    LTX2_CANVAS_MULTIPLE,
    LTX2_DEFAULT_FPS,
    LTX2_DEFAULT_TEMPORAL_TILE,
    LTX2_DEFAULT_TILE_SIZE,
    LTX2_FRAME_MODULUS,
    LTX2_LATENT_CHANNELS,
    LTX2_NUM_FRAMES_MAX,
)
from invokeai.backend.ltx2.packing import audio_latent_count, snap_num_frames_down, video_latent_shape
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.util.vae_working_memory import (
    estimate_audio_working_memory_ltx2,
    estimate_vae_working_memory_ltx2,
)

# Three latent frames. One would be a still, which says where the clip starts but nothing about how
# it was moving, and an extension seeded from a still stalls or lurches at the join. Seventeen pixel
# frames is a little under a second at 24 fps -- enough for the model to read the motion, short
# enough that it is not spending most of the generation replaying footage the user already has.
LTX2_DEFAULT_EXTEND_CONTEXT_FRAMES = 17

# A ceiling on the context, and not a formality: a tail read cannot stop early, so it buffers this
# many frames at the SOURCE's resolution before any of them are fitted to the canvas. Unbounded, a
# workflow asking for 1001 frames of a 1080p clip would reserve ~6 GiB of host memory -- and then
# encode all of it -- before the denoise refused the anchor for exceeding the generation's length.
# Nothing longer than the model's own maximum could be used anyway.
LTX2_MAX_EXTEND_CONTEXT_FRAMES = snap_num_frames_down(LTX2_NUM_FRAMES_MAX)


@invocation_output("ltx2_extend_conditioning_output")
class LTX2ExtendConditioningOutput(BaseInvocationOutput):
    """The source's tail, held as the opening of the continuation."""

    video_conditioning: LTX2VideoConditioningField = OutputField(
        description=FieldDescriptions.ltx2_video_conditioning, title="Image Conditioning"
    )
    context_frames: int = OutputField(
        description="Pixel frames of the source that are held. The generated clip reproduces these, so a "
        "join has to consume exactly this many to avoid repeating them."
    )
    audio_conditioning: LTX2AudioConditioningField | None = OutputField(
        default=None,
        description="The held opening of the source's soundtrack, or nothing when the clip is silent.",
        title="Audio Conditioning",
    )


@invocation(
    "ltx2_extend_conditioning",
    title="Extend Conditioning - LTX-2",
    tags=["ltx", "ltx2", "video", "extend", "conditioning"],
    category="conditioning",
    version="1.1.0",
    classification=Classification.Prototype,
)
class LTX2ExtendConditioningInvocation(BaseInvocation):
    """Encodes the end of a clip so LTX-2 can continue it.

    A continuation is image-to-video with a longer anchor: instead of one held frame the generation
    opens with several latent frames of the source, held clean the same way. That is what carries
    the source's *motion* across the join -- a single still frame fixes the picture but not its
    direction, and clips extended from one tend to stall or lurch where they meet.

    The held frames are regenerated as part of the output, so they appear in both clips. The join is
    what removes the duplication: a crossfade of ``context_frames`` consumes exactly those frames
    from each side, blending two renderings of the same moment into one.
    """

    video: VideoField = InputField(description="The clip to continue.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection, title="Video VAE")
    audio_vae: VAEField | None = InputField(
        default=None,
        description=FieldDescriptions.ltx2_audio_vae,
        input=Input.Connection,
        title="Audio VAE",
    )
    vocoder: LTX2VocoderField | None = InputField(
        default=None,
        description="The vocoder, whose checkpoint carries the log-mel filters the audio VAE encodes from.",
        input=Input.Connection,
        title="Vocoder",
    )
    fps: float = InputField(
        default=LTX2_DEFAULT_FPS,
        ge=1,
        le=120,
        description="The clip's frame rate, used to turn the held frame count into a span of its soundtrack.",
    )
    width: int = InputField(
        default=1248, gt=0, multiple_of=LTX2_CANVAS_MULTIPLE, description="Canvas width (must match Denoise)."
    )
    height: int = InputField(
        default=704, gt=0, multiple_of=LTX2_CANVAS_MULTIPLE, description="Canvas height (must match Denoise)."
    )
    context_frames: int = InputField(
        default=LTX2_DEFAULT_EXTEND_CONTEXT_FRAMES,
        ge=1,
        le=LTX2_MAX_EXTEND_CONTEXT_FRAMES,
        description="Frames of the source to carry into the continuation. The VAE encodes 8k + 1 "
        "frames, so this is snapped DOWN to the nearest 9, 17, 25, 33, 41, 49... -- asking for 24 "
        "holds 17. A short source is capped at what it has. The `context_frames` output reports "
        "what was actually held, which is why the join reads it from there rather than from here. "
        "More context means smoother motion and more rhythmic context across the join, and less of "
        "the generation left for new material.",
    )
    tile_size: int = InputField(
        default=LTX2_DEFAULT_TILE_SIZE,
        ge=64,
        description="Spatial encode tile, in source pixels. Smaller tiles need less VRAM and take longer.",
    )
    temporal_tile: int = InputField(
        default=LTX2_DEFAULT_TEMPORAL_TILE,
        ge=8,
        description="Temporal encode tile, in source frames. Smaller tiles need less VRAM and take longer.",
    )

    def _hold_soundtrack_opening(
        self, context: InvocationContext, path: Path, context_frames: int, source_frames: int | None
    ) -> LTX2AudioConditioningField | None:
        """The same span of the source's sound as the picture it holds, encoded for the denoise.

        The generated clip replays the source's last moments so the model can read its motion, and
        the join crossfades exactly those out of both halves. The picture survives that because both
        clips render the same instant -- but the soundtrack is sampled freely, so without this the
        blend fades newly invented audio in against the source's real audio and the new soundtrack
        audibly starts one overlap early.

        A silent clip holds nothing, which is not an error: the continuation simply invents its whole
        soundtrack, as it did before this existed.
        """
        if self.audio_vae is None or self.vocoder is None:
            return None

        decoded = extract_audio_pcm(path, float_pcm=True)
        if decoded is None:
            return None

        samples, sample_rate = decoded
        # Cut the decode back to the span the PICTURE occupies before taking its end. A decoded AAC
        # track runs past the last frame by the codec's end padding -- measured at 688 samples
        # (14.3 ms) on the muxes `extract_video_range` emits, which is what feeds this node -- and
        # the content is front-aligned, so slicing the tail off the untrimmed decode would hold
        # audio that starts 14 ms after the picture does and ends in padding. The join then blends
        # that against the source's own correctly-aligned tail: comb filtering across the whole
        # overlap, which is the artifact this method exists to remove. `video_concat` trims for the
        # same reason before its own crossfade.
        if source_frames is None:
            # Without the clip's length there is no way to say where its picture ends, and a hold
            # that is not anchored to the picture is worse than no hold at all.
            return None

        native = int(round(source_frames / self.fps * sample_rate))
        if samples.shape[1] > native:
            samples = samples[:, :native]
        elif samples.shape[1] < native:
            # The track ends before the picture does, which uploaded footage often does. Taking
            # the tail of the short track would hold an EARLIER instant than the frames held
            # beside it -- a clip whose sound stops two seconds early would hold audio from two
            # seconds before the frames it is meant to accompany, and the join would blend that
            # against the source's real soundtrack. Padding to the picture's span keeps the tail
            # slice anchored to the same instant; the silence is what the clip actually has there.
            samples = np.pad(samples, ((0, 0), (0, native - samples.shape[1])))

        # `context_frames` is what the picture read actually kept, so it never exceeds the clip's
        # own length and this slice is always fully inside the span trimmed above.
        wanted = int(round(context_frames / self.fps * sample_rate))

        audio_vae_info = context.models.load(self.audio_vae.vae)
        vocoder_info = context.models.load(self.vocoder.vocoder)
        context.util.signal_progress("Encoding the clip's closing sound for LTX-2 extension")

        with (
            audio_vae_info.model_on_device(
                working_mem_bytes=estimate_audio_working_memory_ltx2(
                    int(wanted / sample_rate * LTX2_AUDIO_LATENTS_PER_SECOND)
                )
            ) as (_, audio_vae),
            vocoder_info.model_on_device() as (_, vocoder),
        ):
            # The clip's TAIL, which is the part the continuation opens on.
            latents = encode_audio_latents(
                audio_vae, vocoder, torch.from_numpy(samples[:, -wanted:]), sample_rate=sample_rate
            )

        # Trimmed to the span the denoise sizes from the same frame count, so the two agree by
        # construction rather than by the encoder happening to land on the right number of rows.
        held = min(int(latents.shape[1]), audio_latent_count(context_frames, self.fps))

        return LTX2AudioConditioningField(
            latents_name=context.tensors.save(tensor=latents[:, :held]),
            num_audio_latents=held,
            num_frames=context_frames,
            fps=self.fps,
            source_video_name=self.video.video_name,
        )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LTX2ExtendConditioningOutput:
        path = context.videos.get_path(self.video.video_name)
        vae_info = context.models.load(self.vae.vae)
        if vae_info.config.base is not BaseModelType.LTX2:
            raise ValueError(f"Expected an LTX-2 video VAE; got a {vae_info.config.base.value} one.")

        # Snapped down first, so the read asks for a whole number of frame groups: the VAE encodes
        # 8k + 1 frames, and a ragged request would simply have its tail dropped afterwards.
        wanted = snap_num_frames_down(self.context_frames)
        context.util.signal_progress("Reading the clip's tail for LTX-2 extension")
        frames, source_frames = read_canvas_frames(
            path,
            width=self.width,
            height=self.height,
            cap=wanted,
            tail=True,
            is_canceled=context.util.is_canceled,
            # A tail read cannot stop early -- the last frames are only known once the clip ends --
            # so this is the longest decode of the two conditioning nodes, and the one that most
            # needs to say it is still working.
            on_progress=lambda read: context.util.signal_progress(
                f"Reading the clip's tail for LTX-2 extension ({read} frames)"
            ),
        )

        context_frames = snap_num_frames_down(len(frames))
        if context_frames < 1 + LTX2_FRAME_MODULUS:
            raise ValueError(
                f"'{self.video.video_name}' decoded to {len(frames)} frame(s), which is under one frame group. "
                f"An extension needs at least {1 + LTX2_FRAME_MODULUS} frames of the source to continue from."
            )
        # The tail is what matters, so a ragged read drops its HEAD rather than its end.
        del frames[: len(frames) - context_frames]

        working_memory = estimate_vae_working_memory_ltx2(
            "encode",
            vae_info.model,
            pixel_height=self.height,
            pixel_width=self.width,
            pixel_frames=context_frames,
            tile_size=self.tile_size,
            temporal_tile=self.temporal_tile,
            tiled=True,
        )
        context.util.signal_progress("Encoding the clip's tail for LTX-2 extension")

        with vae_info.model_on_device(working_mem_bytes=working_memory) as (_, vae):
            latents = encode_canvas_clip(
                vae,
                frames,
                tile_size=self.tile_size,
                temporal_tile=self.temporal_tile,
                is_canceled=context.util.is_canceled,
            )
        del frames

        expected = (1, LTX2_LATENT_CHANNELS, *video_latent_shape(context_frames, self.height, self.width))
        if tuple(latents.shape) != expected:
            raise ValueError(
                f"The clip's tail encoded to {tuple(latents.shape)} but {self.width}x{self.height} at "
                f"{context_frames} frames needs {expected}."
            )

        return LTX2ExtendConditioningOutput(
            audio_conditioning=self._hold_soundtrack_opening(context, path, context_frames, source_frames),
            video_conditioning=LTX2VideoConditioningField(
                latents_name=context.tensors.save(tensor=latents.cpu()),
                width=self.width,
                height=self.height,
                frame_index=0,
            ),
            context_frames=context_frames,
        )
