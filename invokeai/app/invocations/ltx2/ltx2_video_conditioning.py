"""A whole clip as LTX-2 conditioning: generate a soundtrack for existing picture."""

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
    LTX2FullVideoConditioningField,
    OutputField,
    VideoField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ltx2.clip_frames import encode_canvas_clip, read_canvas_frames
from invokeai.backend.ltx2.constants import (
    LTX2_CANVAS_MULTIPLE,
    LTX2_DEFAULT_TEMPORAL_TILE,
    LTX2_DEFAULT_TILE_SIZE,
    LTX2_FRAME_MODULUS,
    LTX2_LATENT_CHANNELS,
    LTX2_NUM_FRAMES_MAX,
)
from invokeai.backend.ltx2.packing import snap_num_frames_down, video_latent_shape
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_ltx2


@invocation_output("ltx2_video_conditioning_output")
class LTX2VideoConditioningOutput(BaseInvocationOutput):
    """A clip to generate a soundtrack for, and the geometry it pins."""

    video_conditioning: LTX2FullVideoConditioningField = OutputField(
        description=FieldDescriptions.ltx2_full_video_conditioning, title="Video Conditioning"
    )
    width: int = OutputField(description="Pixel width the clip was encoded at.")
    height: int = OutputField(description="Pixel height the clip was encoded at.")
    num_frames: int = OutputField(description="Frames encoded, snapped down to 8n + 1.")


@invocation(
    "ltx2_video_conditioning",
    title="Video Conditioning - LTX-2",
    tags=["ltx", "ltx2", "video", "audio", "conditioning"],
    category="conditioning",
    version="1.1.0",
    classification=Classification.Prototype,
)
class LTX2VideoConditioningInvocation(BaseInvocation):
    """Encodes a whole clip so LTX-2 can generate a soundtrack for it.

    The mirror of audio conditioning: every video token is held clean at each step and only the
    audio is sampled. The clip decides the generation's geometry -- its canvas, its length and its
    frame rate -- because the picture is a given rather than something being made.

    Frames are fitted to the canvas by cover-crop, as first-frame conditioning does, and the count
    is snapped *down* to the VAE's 8n + 1 grid; trailing frames beyond the last group are dropped
    rather than padded, since padding would invent picture for the model to score sound against.

    What comes back is the model's reconstruction of the clip, not the clip: the picture is
    cover-cropped to the canvas, VAE round-tripped and re-encoded, because the decode node renders
    the held latents like any others. That is the opposite of the choice audio conditioning makes
    for the soundtrack, which is muxed back in as supplied -- the symmetric treatment here would be
    to remux the source video stream, and it is not done yet.

    The encode is tiled, and not by preference: an untiled encode's activation grows with the whole
    clip rather than with one tile, so a 1248x704 clip of 121 frames would need about 65 GiB where
    tiled it needs 3.2, flat in the clip's length. Tiling costs blend seams -- measured at 16%
    relative rms against an untiled encode of the same smooth footage -- which is why the tile
    sizes are inputs rather than constants: a clip that fits a larger tile can be given one.
    """

    video: VideoField = InputField(description="The clip to generate a soundtrack for.")
    vae: VAEField = InputField(description=FieldDescriptions.vae, input=Input.Connection, title="Video VAE")
    width: int = InputField(
        default=768, gt=0, multiple_of=LTX2_CANVAS_MULTIPLE, description="Canvas width to encode at."
    )
    height: int = InputField(
        default=512, gt=0, multiple_of=LTX2_CANVAS_MULTIPLE, description="Canvas height to encode at."
    )
    fps: float = InputField(default=24.0, ge=1, le=120, description="The clip's frame rate, which the audio adopts.")
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

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LTX2VideoConditioningOutput:
        path = context.videos.get_path(self.video.video_name)
        vae_info = context.models.load(self.vae.vae)
        if vae_info.config.base is not BaseModelType.LTX2:
            raise ValueError(f"Expected an LTX-2 video VAE; got a {vae_info.config.base.value} one.")

        # A ceiling, and it stops the decode rather than checking after the fact: nothing upstream
        # bounds a clip's length -- `validate_num_frames` only checks the 8n + 1 grid -- so a
        # workflow handing this node a ten-minute recording would otherwise materialize every frame
        # before anything refused it. The cap is the grid value at or below the model's own maximum.
        context.util.signal_progress("Reading the clip for LTX-2 conditioning")
        frames, _ = read_canvas_frames(
            path,
            width=self.width,
            height=self.height,
            cap=snap_num_frames_down(LTX2_NUM_FRAMES_MAX),
            is_canceled=context.util.is_canceled,
            on_progress=lambda read: context.util.signal_progress(
                f"Reading the clip for LTX-2 conditioning ({read} frames)"
            ),
        )

        num_frames = snap_num_frames_down(len(frames))
        if num_frames < 1 + LTX2_FRAME_MODULUS:
            raise ValueError(
                f"'{self.video.video_name}' decoded to {len(frames)} frame(s), which is under one frame group. "
                f"Use a longer clip."
            )
        del frames[num_frames:]

        working_memory = estimate_vae_working_memory_ltx2(
            "encode",
            vae_info.model,
            pixel_height=self.height,
            pixel_width=self.width,
            pixel_frames=num_frames,
            tile_size=self.tile_size,
            temporal_tile=self.temporal_tile,
            tiled=True,
        )
        context.util.signal_progress("Encoding the clip for LTX-2 conditioning")

        with vae_info.model_on_device(working_mem_bytes=working_memory) as (_, vae):
            latents = encode_canvas_clip(
                vae,
                frames,
                tile_size=self.tile_size,
                temporal_tile=self.temporal_tile,
                is_canceled=context.util.is_canceled,
            )
        del frames

        expected = (1, LTX2_LATENT_CHANNELS, *video_latent_shape(num_frames, self.height, self.width))
        if tuple(latents.shape) != expected:
            raise ValueError(
                f"The clip encoded to {tuple(latents.shape)} but {self.width}x{self.height} at {num_frames} "
                f"frames needs {expected}."
            )

        return LTX2VideoConditioningOutput(
            video_conditioning=LTX2FullVideoConditioningField(
                latents_name=context.tensors.save(tensor=latents.cpu()),
                width=self.width,
                height=self.height,
                num_frames=num_frames,
                fps=self.fps,
                source_video_name=self.video.video_name,
            ),
            width=self.width,
            height=self.height,
            num_frames=num_frames,
        )
