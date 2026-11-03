"""The LTX-2 denoising loop.

One transformer forward produces both modalities' velocity predictions, so a guidance pass is a
pass for video *and* audio; the two are combined separately afterwards (see
:mod:`invokeai.backend.ltx2.guidance`) and stepped down the same sigma schedule.

Conditioning -- a first frame or a whole held modality today, keyframes later -- is carried by one
mechanism: a per-token mask over the packed video sequence. A conditioned token's timestep is
forced to 0 for every forward, and its value is restored from the clean latents after every step,
so the anchor neither drifts nor gets renoised by the ancestral sampler.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from diffusers.utils.torch_utils import randn_tensor

from invokeai.backend.ltx2.constants import (
    LTX2_ANCESTRAL_ETA,
    LTX2_ANCESTRAL_NOISE_SEED_OFFSET,
    LTX2_ANCESTRAL_S_NOISE,
    LTX2_AUDIO_LATENT_CHANNELS,
    LTX2_AUDIO_LATENT_MEL_BINS,
    LTX2_LATENT_CHANNELS,
    LTX2_REFINE_NOISE_SEED_OFFSET,
    LTX2_TEMPORAL_COMPRESSION,
)
from invokeai.backend.ltx2.guidance import (
    PASS_MODALITY,
    PASS_STG,
    PASS_UNCOND,
    LTX2Guidance,
    LTX2GuidancePass,
)
from invokeai.backend.ltx2.packing import (
    audio_latent_count,
    pack_audio_latents,
    pack_video_latents,
    prepare_keyframe_coords,
    validate_canvas,
    validate_num_frames,
    video_latent_shape,
    video_sequence_length,
)
from invokeai.backend.ltx2.sampling import build_refine_sigmas, build_sigmas, flow_step
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import LTX2ConditioningInfo
from invokeai.backend.util.cancel_hooks import cancel_before_forward


@dataclass
class LTX2DenoiseState:
    """Everything a denoise run needs that does not come from the models, built on the CPU."""

    video_latents: torch.Tensor
    """Packed, normalized, noised video latents. Shape: (1, video tokens, 128)."""

    audio_latents: torch.Tensor
    """Packed, normalized, noised audio latents. Shape: (1, audio latents, 128)."""

    sigmas: torch.Tensor
    """The noise levels to sample, terminal 0 included. Shape: (steps + 1,)."""

    latent_frames: int
    latent_height: int
    latent_width: int
    audio_latents_count: int

    conditioning_mask: torch.Tensor | None
    """How strongly each video token is held to its conditioning, 0 (free) to 1 (clean).

    Shape: (1, video tokens). A token's timestep is scaled by ``1 - mask`` and its x0 prediction is
    blended toward the clean value by ``mask``, so a fractional mask is a partial anchor rather
    than a switch.
    """

    clean_video_latents: torch.Tensor | None
    """The conditioned tokens' clean values, at full packed shape (0 elsewhere)."""

    eta: float
    noise_seed: int

    audio_conditioning_mask: torch.Tensor | None = None
    """How strongly each audio row is held to its conditioning, 0 (free) to 1 (clean).

    Shape: (1, audio latents). The mirror of ``conditioning_mask`` on the other modality: LTX-2
    conditions both streams through one mechanism, so audio-to-video is an all-ones mask here and
    video-to-audio is an all-ones mask on the video side.
    """

    clean_audio_latents: torch.Tensor | None = None
    """Packed, normalized audio latents the mask holds rows to. Shape: (1, audio latents, 128)."""

    keyframe_coords: torch.Tensor | None = None
    """RoPE coordinates for keyframe tokens appended to the sequence. Shape: (1, 3, patches, 2).

    A frame held at latent index 0 overwrites tokens already in the grid; one held anywhere else has
    nowhere in the grid to go, so it rides on the end of the sequence carrying its own position.
    """

    keyframe_tokens: int = 0
    """How many rows on the end of ``video_latents`` are appended keyframes rather than generation.

    They are conditioning the model reads, not picture it makes, so they are trimmed off before the
    latents are unpacked -- both for the decode and for the progress preview.
    """

    @property
    def generated_tokens(self) -> int:
        """Rows of ``video_latents`` that are the generation itself."""
        return self.video_latents.shape[1] - self.keyframe_tokens

    @property
    def num_steps(self) -> int:
        return self.sigmas.numel() - 1


def _hold_audio_prefix(
    audio_latents: torch.Tensor,
    prefix: torch.Tensor | None,
    audio_count: int,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Hold the opening of the soundtrack, leaving the rest to be generated.

    A continuation replays the source's last moments so the model can see its motion, and the join
    then crossfades exactly those frames out of both halves. The picture survives that because it is
    held -- both clips render the same instant, so the blend is of like for like. Without this the
    soundtrack does not: it is sampled freely from the first step, so the blend fades *newly
    invented* audio in against the source's real audio, and the new soundtrack audibly begins one
    overlap early.

    The mask is per row and already drives everything downstream, so holding a prefix needs no
    change to the loop -- only ones at the front of it instead of everywhere.
    """
    if prefix is None:
        return audio_latents, None, None

    held = prefix.shape[1] if prefix.ndim == 3 else 0
    expected = (1, held, LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_LATENT_MEL_BINS)
    if prefix.ndim != 3 or tuple(prefix.shape) != expected or not 0 < held <= audio_count:
        raise ValueError(
            f"The held opening of the soundtrack is {tuple(prefix.shape)} but this clip has "
            f"{audio_count} audio latents at {fps:g} fps, so it needs (1, 1..{audio_count}, "
            f"{LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_LATENT_MEL_BINS})."
        )

    clean = torch.zeros_like(audio_latents)
    clean[:, :held] = prefix.to(device="cpu", dtype=torch.float32)
    mask = torch.zeros((1, audio_count), dtype=torch.float32)
    mask[:, :held] = 1.0

    return torch.lerp(audio_latents, clean, mask.unsqueeze(-1)), mask, clean


def _validated_anchor(
    image_latents: torch.Tensor,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    width: int,
    height: int,
    hint: str,
) -> torch.Tensor:
    """A leading anchor: one latent frame for image-to-video, several for a video extension.

    A still frame tells the model where a clip starts but nothing about how it was moving, so an
    extension seeded from one tends to stall or lurch at the join. Several latent frames of the
    source carry its motion into the new clip, and they are held the same way -- the mask simply
    covers more of the front of the grid.
    """
    anchor = image_latents.to(device="cpu", dtype=torch.float32)
    valid = (
        anchor.ndim == 5
        and anchor.shape[:2] == (1, LTX2_LATENT_CHANNELS)
        and 0 < anchor.shape[2] <= latent_frames
        and anchor.shape[3:] == (latent_height, latent_width)
    )
    if not valid:
        raise ValueError(
            f"The image conditioning was encoded at {tuple(anchor.shape)} but this generation needs "
            f"(1, {LTX2_LATENT_CHANNELS}, 1..{latent_frames}, {latent_height}, {latent_width}). {hint}"
        )

    return anchor


def _append_keyframe(
    video_latents: torch.Tensor,
    conditioning_mask: torch.Tensor | None,
    clean_video_latents: torch.Tensor | None,
    *,
    keyframe_latents: torch.Tensor | None,
    keyframe_latent_index: int | None,
    keyframe_strength: float,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_frames: int,
    width: int,
    height: int,
    fps: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
    """Append a held frame to the end of a packed sequence, with the mask and coordinates it needs.

    Shared by the base and refine passes so a last frame survives a two-stage run the way a first
    frame does: the refine pass re-noises every token, so a keyframe that was only appended in
    stage one would be regenerated from the prompt in stage two.

    Returns the sequence, mask, clean values, the keyframe's RoPE coordinates and how many rows were
    appended; with no keyframe the first three are returned unchanged.
    """
    if keyframe_latents is None:
        return video_latents, conditioning_mask, clean_video_latents, None, 0

    if keyframe_latent_index is None or keyframe_latent_index <= 0:
        raise ValueError(
            f"A keyframe needs a latent index above 0; got {keyframe_latent_index}. Frame 0 is the "
            f"first frame, which is held by overwriting the grid rather than appending."
        )
    if keyframe_latent_index >= latent_frames:
        raise ValueError(
            f"A keyframe at latent index {keyframe_latent_index} is outside a {num_frames}-frame clip, "
            f"which has {latent_frames} latent frames."
        )
    if not 0.0 < keyframe_strength <= 1.0:
        raise ValueError(f"Keyframe strength must be in (0, 1]; got {keyframe_strength}.")

    keyframe = keyframe_latents.to(device="cpu", dtype=torch.float32)
    if (
        keyframe.ndim != 5
        or keyframe.shape[:2] != (1, LTX2_LATENT_CHANNELS)
        or keyframe.shape[3:]
        != (
            latent_height,
            latent_width,
        )
    ):
        raise ValueError(
            f"The keyframe was encoded at {tuple(keyframe.shape)} but this generation needs "
            f"(1, {LTX2_LATENT_CHANNELS}, frames, {latent_height}, {latent_width}). Re-run the "
            f"conditioning node at {width}x{height}."
        )

    # Drawn after everything else, so adding a last frame does not reshuffle the noise the rest of
    # the clip would have had -- the same seed keeps the same generation.
    keyframe_noise = randn_tensor(keyframe.shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32)
    keyframe_clean = pack_video_latents(keyframe)
    keyframe_tokens = keyframe_clean.shape[1]

    if conditioning_mask is None:
        conditioning_mask = torch.zeros((1, video_latents.shape[1]), dtype=torch.float32)
        clean_video_latents = torch.zeros_like(video_latents)
    assert clean_video_latents is not None

    video_latents = torch.cat(
        [video_latents, torch.lerp(pack_video_latents(keyframe_noise), keyframe_clean, keyframe_strength)], dim=1
    )
    conditioning_mask = torch.cat(
        [conditioning_mask, torch.full((1, keyframe_tokens), keyframe_strength, dtype=torch.float32)], dim=1
    )
    clean_video_latents = torch.cat([clean_video_latents, keyframe_clean], dim=1)
    # `(index - 1) * 8 + 1` is the pixel frame a latent frame starts at: latent 0 covers pixel frame 0
    # alone and every later latent covers the eight after it (`VideoConditionByKeyframeIndex`).
    keyframe_coords = prepare_keyframe_coords(
        keyframe.shape[2],
        latent_height,
        latent_width,
        pixel_frame_index=(keyframe_latent_index - 1) * LTX2_TEMPORAL_COMPRESSION + 1,
        num_pixel_frames=(keyframe.shape[2] - 1) * LTX2_TEMPORAL_COMPRESSION + 1,
        fps=fps,
    )

    return video_latents, conditioning_mask, clean_video_latents, keyframe_coords, keyframe_tokens


def build_denoise_state(
    *,
    num_frames: int,
    height: int,
    width: int,
    fps: float,
    seed: int,
    distilled: bool,
    num_steps: int,
    image_latents: torch.Tensor | None = None,
    conditioning_strength: float = 1.0,
    frozen_audio_latents: torch.Tensor | None = None,
    frozen_video_latents: torch.Tensor | None = None,
    audio_prefix_latents: torch.Tensor | None = None,
    keyframe_latents: torch.Tensor | None = None,
    keyframe_latent_index: int | None = None,
    keyframe_strength: float = 1.0,
) -> LTX2DenoiseState:
    """Noise, schedule and conditioning mask for one run.

    ``image_latents`` is a clean, normalized ``(1, 128, 1, h, w)`` encode of the first frame; when
    given it becomes latent frame 0, held to ``conditioning_strength``. Noise is drawn on the CPU
    (video first, then audio) so a request is reproducible across devices.

    ``frozen_audio_latents`` and ``frozen_video_latents`` hold a whole modality clean instead: the
    first is audio-to-video, the second video-to-audio. They are the same mask mechanism as the
    first frame, with every row set rather than one -- which is why the two modes need no machinery
    of their own beyond an encode. Giving both would leave nothing to sample, so it is refused.

    ``keyframe_latents`` holds a frame at ``keyframe_latent_index``, which must be non-zero -- index
    0 is what ``image_latents`` is for. It cannot overwrite grid tokens the way the first frame does,
    because the grid has one value per position and the generation needs that position too; instead
    the frame is *appended* to the sequence with coordinates naming where in time it belongs, and the
    same mask drives it. The appended rows are trimmed before anything unpacks the result. This is
    what makes a last frame (and, with ``image_latents``, first-to-last interpolation) possible.
    """
    if frozen_audio_latents is not None and audio_prefix_latents is not None:
        raise ValueError(
            "The soundtrack cannot be held whole and held at its opening at the same time. A "
            "continuation holds its opening; audio-to-video holds all of it."
        )
    if frozen_audio_latents is not None and frozen_video_latents is not None:
        raise ValueError(
            "Audio and video cannot both be held: that would leave nothing for the model to generate. "
            "Condition on one modality or the other."
        )
    # A held clip already covers frame 0, so the first-frame encode would be overwritten rather
    # than combined -- two different pictures asked for in the same rows. Holding a soundtrack
    # alongside a first frame is a different matter and stays allowed: those are separate streams.
    # A held clip covers every row, so an appended keyframe would be orphaned: the frozen branch
    # below replaces the whole sequence and the keyframe's rows go with it, leaving its coordinates
    # and token count pointing at tokens that are no longer there.
    if frozen_video_latents is not None and keyframe_latents is not None:
        raise ValueError(
            "A keyframe cannot be combined with a whole-clip video conditioning: the clip already "
            "supplies every frame. Wire one or the other."
        )
    if frozen_video_latents is not None and image_latents is not None:
        raise ValueError(
            "A first frame cannot be combined with a whole-clip video conditioning: the clip already "
            "supplies frame 0. Wire one or the other."
        )
    validate_canvas(height, width)
    validate_num_frames(num_frames)

    latent_frames, latent_height, latent_width = video_latent_shape(num_frames, height, width)
    audio_count = audio_latent_count(num_frames, fps)
    if audio_count < 1:
        raise ValueError(
            f"A {num_frames}-frame clip at {fps} fps is shorter than one audio latent "
            f"({1 / 25:.2f} s); generate a longer clip."
        )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    video_noise = randn_tensor(
        (1, LTX2_LATENT_CHANNELS, latent_frames, latent_height, latent_width),
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    audio_noise = randn_tensor(
        (1, LTX2_AUDIO_LATENT_CHANNELS, audio_count, LTX2_AUDIO_LATENT_MEL_BINS),
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    video_latents = pack_video_latents(video_noise)
    conditioning_mask: torch.Tensor | None = None
    clean_video_latents: torch.Tensor | None = None

    if image_latents is not None:
        anchor = _validated_anchor(
            image_latents,
            latent_frames,
            latent_height,
            latent_width,
            width,
            height,
            hint=(
                f"Re-run the conditioning node at {width}x{height}. If this is a continuation, the "
                f"held opening is longer than the clip being generated -- raise Frames, or lower the "
                f"conditioning node's context length."
            ),
        )
        if not 0.0 < conditioning_strength <= 1.0:
            raise ValueError(f"Image conditioning strength must be in (0, 1]; got {conditioning_strength}.")
        held = anchor.shape[2]
        clean = torch.zeros_like(video_noise)
        clean[:, :, :held] = anchor
        mask = torch.zeros((1, 1, latent_frames, latent_height, latent_width), dtype=torch.float32)
        mask[:, :, :held] = conditioning_strength

        clean_video_latents = pack_video_latents(clean)
        conditioning_mask = pack_video_latents(mask).squeeze(-1)
        video_latents = torch.lerp(video_latents, clean_video_latents, conditioning_mask.unsqueeze(-1))

    video_latents, conditioning_mask, clean_video_latents, keyframe_coords, keyframe_tokens = _append_keyframe(
        video_latents,
        conditioning_mask,
        clean_video_latents,
        keyframe_latents=keyframe_latents,
        keyframe_latent_index=keyframe_latent_index,
        keyframe_strength=keyframe_strength,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_frames=num_frames,
        width=width,
        height=height,
        fps=fps,
        generator=generator,
    )

    audio_latents = pack_audio_latents(audio_noise)
    audio_conditioning_mask: torch.Tensor | None = None
    clean_audio_latents: torch.Tensor | None = None

    if frozen_audio_latents is not None:
        expected_audio = (1, audio_count, LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_LATENT_MEL_BINS)
        if tuple(frozen_audio_latents.shape) != expected_audio:
            raise ValueError(
                f"The conditioning soundtrack is {tuple(frozen_audio_latents.shape)} but a "
                f"{num_frames}-frame clip at {fps:g} fps needs {expected_audio}. Derive the frame "
                f"count from the soundtrack rather than setting it separately."
            )
        clean_audio_latents = frozen_audio_latents.to(device="cpu", dtype=torch.float32)
        audio_conditioning_mask = torch.ones((1, audio_count), dtype=torch.float32)
        audio_latents = clean_audio_latents.clone()

    audio_latents, prefix_mask, prefix_clean = _hold_audio_prefix(audio_latents, audio_prefix_latents, audio_count, fps)
    if prefix_mask is not None:
        audio_conditioning_mask, clean_audio_latents = prefix_mask, prefix_clean

    if frozen_video_latents is not None:
        expected_video = (1, LTX2_LATENT_CHANNELS, latent_frames, latent_height, latent_width)
        if tuple(frozen_video_latents.shape) != expected_video:
            raise ValueError(
                f"The conditioning clip is {tuple(frozen_video_latents.shape)} but {width}x{height} "
                f"at {num_frames} frames needs {expected_video}."
            )
        clean_video_latents = pack_video_latents(frozen_video_latents.to(device="cpu", dtype=torch.float32))
        conditioning_mask = torch.ones((1, clean_video_latents.shape[1]), dtype=torch.float32)
        video_latents = clean_video_latents.clone()

    return LTX2DenoiseState(
        video_latents=video_latents,
        audio_latents=audio_latents,
        sigmas=build_sigmas(
            distilled=distilled,
            num_steps=num_steps,
            video_seq_len=video_sequence_length(num_frames, height, width),
        ),
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        audio_latents_count=audio_count,
        conditioning_mask=conditioning_mask,
        clean_video_latents=clean_video_latents,
        audio_conditioning_mask=audio_conditioning_mask,
        clean_audio_latents=clean_audio_latents,
        keyframe_coords=keyframe_coords,
        keyframe_tokens=keyframe_tokens,
        # LTX-2.5 samples its distilled schedule ancestrally and its shifted schedule
        # deterministically; eta is what selects between the two branches of one step.
        eta=LTX2_ANCESTRAL_ETA if distilled else 0.0,
        noise_seed=seed + LTX2_ANCESTRAL_NOISE_SEED_OFFSET,
    )


def build_refine_state(
    *,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    fps: float,
    seed: int,
    distilled: bool,
    num_steps: int,
    noise_scale: float,
    image_latents: torch.Tensor | None = None,
    conditioning_strength: float = 1.0,
    keyframe_latents: torch.Tensor | None = None,
    keyframe_latent_index: int | None = None,
    keyframe_strength: float = 1.0,
    audio_prefix_latents: torch.Tensor | None = None,
) -> LTX2DenoiseState:
    """The refine pass's state: stage one's result re-entered partway down the schedule.

    ``video_latents`` is the upsampled ``(1, 128, T, h, w)`` clip at the *second* stage's canvas,
    normalized; ``audio_latents`` is stage one's packed audio, which no upsampler touches. Both are
    noised back to the level the schedule is re-entered at -- ``x = (1 - sigma) * x0 + sigma * eps``,
    the forward process of the same rectified flow the step inverts -- because a partial schedule
    expects a sample at its first level, not a clean one.

    Audio is re-noised to the same level rather than carried through clean: the two modalities are
    denoised jointly and the transformer reads one pair of timesteps, so handing it clean audio
    beside a noised video would place the two streams at different points of the same trajectory.

    ``image_latents`` re-anchors a conditioned first frame, and has to be a *fresh* encode at this
    pass's canvas -- stage one's is half the size. Re-anchoring is not optional for image-to-video:
    the refine pass re-noises every token, frame 0 included, so at the released entry level about
    nine tenths of the anchored frame's signal is replaced by noise and nothing restores it. Without
    this the first frame would be regenerated from the prompt alone, and a two-stage run would
    quietly mean something different by "first frame" than a single-stage one. ``keyframe_latents`` is
    the same argument for a held last frame: appended tokens are not carried through the upsampler,
    so stage two has to be given the frame again or a two-stage run would end somewhere else than a
    single-stage one.
    """
    validate_canvas(height, width)
    validate_num_frames(num_frames)

    latent_frames, latent_height, latent_width = video_latent_shape(num_frames, height, width)
    expected = (1, LTX2_LATENT_CHANNELS, latent_frames, latent_height, latent_width)
    if tuple(video_latents.shape) != expected:
        raise ValueError(
            f"The refine pass was handed {tuple(video_latents.shape)} latents but {width}x{height} "
            f"at {num_frames} frames needs {expected}. Check the upsampler's scale against the "
            f"canvas the stages were planned at."
        )

    audio_count = audio_latent_count(num_frames, fps)
    # The packed audio row is one channel-by-mel-bin block, which happens to be the same width as a
    # video row; asserting the video channel count here would be the right number for the wrong
    # reason, and would stop being right the day either shape moved.
    audio_row = LTX2_AUDIO_LATENT_CHANNELS * LTX2_AUDIO_LATENT_MEL_BINS
    if tuple(audio_latents.shape) != (1, audio_count, audio_row):
        raise ValueError(
            f"The refine pass was handed {tuple(audio_latents.shape)} audio latents but this clip "
            f"needs {(1, audio_count, audio_row)}; both stages must run at one fps and frame count."
        )

    sigmas = build_refine_sigmas(
        distilled=distilled,
        refine_steps=num_steps,
        video_seq_len=video_sequence_length(num_frames, height, width),
        noise_scale=noise_scale,
    )
    sigma = sigmas[0].to(torch.float32)

    # Drawn on the CPU, so a refine is reproducible across devices, and offset from the base pass's
    # stream so the noise mixed back in is not the noise the clip was grown out of.
    generator = torch.Generator(device="cpu").manual_seed(seed + LTX2_REFINE_NOISE_SEED_OFFSET)
    packed_video = pack_video_latents(video_latents.to(device="cpu", dtype=torch.float32))
    audio = audio_latents.to(device="cpu", dtype=torch.float32)
    video_noise = randn_tensor(packed_video.shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32)
    audio_noise = randn_tensor(audio.shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32)

    video_latents = torch.lerp(packed_video, video_noise, sigma)
    conditioning_mask: torch.Tensor | None = None
    clean_video_latents: torch.Tensor | None = None

    if image_latents is not None:
        anchor = _validated_anchor(
            image_latents,
            latent_frames,
            latent_height,
            latent_width,
            width,
            height,
            # The likeliest mistake here is not a wrong canvas but the base pass's own encode, which
            # is half the size and passes every check a single-stage run would make.
            hint=f"Encode it at the refine canvas ({width}x{height}), not the base pass's.",
        )
        if not 0.0 < conditioning_strength <= 1.0:
            raise ValueError(f"Image conditioning strength must be in (0, 1]; got {conditioning_strength}.")

        held = anchor.shape[2]
        clean = torch.zeros((1, LTX2_LATENT_CHANNELS, latent_frames, latent_height, latent_width))
        clean[:, :, :held] = anchor
        mask = torch.zeros((1, 1, latent_frames, latent_height, latent_width), dtype=torch.float32)
        mask[:, :, :held] = conditioning_strength

        clean_video_latents = pack_video_latents(clean)
        conditioning_mask = pack_video_latents(mask).squeeze(-1)
        video_latents = torch.lerp(video_latents, clean_video_latents, conditioning_mask.unsqueeze(-1))

    video_latents, conditioning_mask, clean_video_latents, keyframe_coords, keyframe_tokens = _append_keyframe(
        video_latents,
        conditioning_mask,
        clean_video_latents,
        keyframe_latents=keyframe_latents,
        keyframe_latent_index=keyframe_latent_index,
        keyframe_strength=keyframe_strength,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_frames=num_frames,
        width=width,
        height=height,
        fps=fps,
        generator=generator,
    )

    # Re-held here for the same reason the video anchor is: this pass re-noises every row, so a
    # soundtrack opening held only in stage one is gone by stage two and the join's audio blend goes
    # back to fading in invented sound one overlap early.
    refined_audio, audio_prefix_mask, audio_prefix_clean = _hold_audio_prefix(
        torch.lerp(audio, audio_noise, sigma), audio_prefix_latents, audio_count, fps
    )

    return LTX2DenoiseState(
        video_latents=video_latents,
        audio_latents=refined_audio,
        audio_conditioning_mask=audio_prefix_mask,
        clean_audio_latents=audio_prefix_clean,
        sigmas=sigmas,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        audio_latents_count=audio_count,
        conditioning_mask=conditioning_mask,
        clean_video_latents=clean_video_latents,
        eta=LTX2_ANCESTRAL_ETA if distilled else 0.0,
        noise_seed=seed + LTX2_REFINE_NOISE_SEED_OFFSET + LTX2_ANCESTRAL_NOISE_SEED_OFFSET,
        keyframe_coords=keyframe_coords,
        keyframe_tokens=keyframe_tokens,
    )


@torch.no_grad()
def denoise(
    *,
    transformer: torch.nn.Module,
    state: LTX2DenoiseState,
    positive: LTX2ConditioningInfo,
    negative: LTX2ConditioningInfo | None,
    guidance: LTX2Guidance,
    fps: float,
    dtype: torch.dtype,
    device: torch.device,
    step_callback: Callable[[int, int, torch.Tensor], None] | None = None,
    is_canceled: Callable[[], bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the schedule and return the final packed, normalized ``(video, audio)`` latents."""
    passes = guidance.passes
    if PASS_UNCOND in passes and negative is None:
        raise ValueError("Classifier-free guidance needs negative conditioning, but none was provided.")

    video_latents = state.video_latents.to(device=device, dtype=torch.float32)
    audio_latents = state.audio_latents.to(device=device, dtype=torch.float32)
    sigmas = state.sigmas.to(device=device)

    conditioning_mask = clean_video_latents = None
    if state.conditioning_mask is not None:
        assert state.clean_video_latents is not None
        conditioning_mask = state.conditioning_mask.to(device=device, dtype=torch.float32)
        clean_video_latents = state.clean_video_latents.to(device=device, dtype=torch.float32)

    audio_mask = clean_audio_latents = None
    if state.audio_conditioning_mask is not None:
        assert state.clean_audio_latents is not None
        audio_mask = state.audio_conditioning_mask.to(device=device, dtype=torch.float32)
        clean_audio_latents = state.clean_audio_latents.to(device=device, dtype=torch.float32)

    # Three of the four passes read the *same* conditioning and differ only in their model flags,
    # so the device copies are made once per distinct conditioning rather than once per pass.
    positive_encoders = _encoder_inputs(positive, device=device, dtype=dtype)
    negative_encoders = (
        _encoder_inputs(negative, device=device, dtype=dtype) if PASS_UNCOND in passes and negative else None
    )
    encoders = {name: negative_encoders if name == PASS_UNCOND else positive_encoders for name in passes}

    # The RoPE coordinates are the audio/video alignment: both are expressed in seconds, so the
    # video grid has to be built at the clip's real frame rate.
    video_coords = transformer.rope.prepare_video_coords(
        1, state.latent_frames, state.latent_height, state.latent_width, device, fps=fps
    )
    if state.keyframe_coords is not None:
        # Appended tokens need appended positions, in the same order: the coordinates are what tell
        # the model *when* the held frame is, and without them it would read as more of frame 0.
        video_coords = torch.cat([video_coords, state.keyframe_coords.to(device=device)], dim=2)
    audio_coords = transformer.audio_rope.prepare_audio_coords(1, state.audio_latents_count, device)

    timestep_scale = float(transformer.config.timestep_scale_multiplier)
    stg_blocks = list(guidance.stg_blocks)
    noise_generator = torch.Generator(device="cpu").manual_seed(state.noise_seed)
    total_steps = state.num_steps

    with cancel_before_forward(transformer.transformer_blocks, is_canceled, device):
        for index in range(total_steps):
            sigma = sigmas[index]
            timestep = (sigma * timestep_scale).expand(1)
            # A conditioned token is presented as fully denoised; `sigma`/`audio_sigma` stay the
            # plain step value, which is what the prompt-AdaLN and the cross-modality gates read.
            video_timestep = timestep if conditioning_mask is None else timestep.unsqueeze(-1) * (1 - conditioning_mask)
            audio_timestep = timestep if audio_mask is None else timestep.unsqueeze(-1) * (1 - audio_mask)

            video_input = video_latents.to(dtype)
            audio_input = audio_latents.to(dtype)
            video_predictions: dict[LTX2GuidancePass, torch.Tensor] = {}
            audio_predictions: dict[LTX2GuidancePass, torch.Tensor] = {}
            for name in passes:
                video_velocity, audio_velocity = transformer(
                    hidden_states=video_input,
                    audio_hidden_states=audio_input,
                    timestep=video_timestep,
                    audio_timestep=audio_timestep,
                    sigma=timestep,
                    num_frames=state.latent_frames,
                    height=state.latent_height,
                    width=state.latent_width,
                    fps=fps,
                    audio_num_frames=state.audio_latents_count,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    isolate_modalities=name == PASS_MODALITY,
                    spatio_temporal_guidance_blocks=stg_blocks if name == PASS_STG else None,
                    use_cross_timestep=True,
                    return_dict=False,
                    **encoders[name],
                )
                # x0 = x - sigma * v, the space the guidance deltas and the step are written in,
                # at the step's own scalar sigma. A partially held token was shown a smaller
                # timestep than that, so its x0 is an approximation -- the same one the reference
                # condition path makes (`LTX2ConditionLoopAfterDenoiser` converts with the
                # scheduler's scalar sigma too), and the mask blend below corrects it.
                video_predictions[name] = video_latents - video_velocity.float() * sigma
                audio_predictions[name] = audio_latents - audio_velocity.float() * sigma

            video_x0 = guidance.combine_video(video_predictions)
            audio_x0 = guidance.combine_audio(audio_predictions)
            if conditioning_mask is not None:
                video_x0 = torch.lerp(video_x0, clean_video_latents, conditioning_mask.unsqueeze(-1))
            if audio_mask is not None:
                audio_x0 = torch.lerp(audio_x0, clean_audio_latents, audio_mask.unsqueeze(-1))

            if step_callback is not None:
                step_callback(index + 1, total_steps, video_x0)

            video_noise, audio_noise = _step_noise(state, video_latents, audio_latents, noise_generator, device)
            step = {"eta": state.eta, "s_noise": LTX2_ANCESTRAL_S_NOISE}
            video_latents = flow_step(video_latents, video_x0, sigmas, index, noise=video_noise, **step)
            audio_latents = flow_step(audio_latents, audio_x0, sigmas, index, noise=audio_noise, **step)
            if conditioning_mask is not None and state.eta > 0:
                # The ancestral branch renoises every token, anchors included, so they are put back.
                # The deterministic branch needs no restore: a step is a convex combination of the
                # token's own value and its (already blended) prediction, which for a fully clean
                # anchor is the anchor, and for a partial one is the interpolation the mask asks for.
                video_latents = torch.lerp(video_latents, clean_video_latents, conditioning_mask.unsqueeze(-1))
            if audio_mask is not None and state.eta > 0:
                audio_latents = torch.lerp(audio_latents, clean_audio_latents, audio_mask.unsqueeze(-1))

    # The keyframe rows are conditioning the model read, not picture it made, and they sit outside
    # the latent grid's geometry -- unpacking with them still attached would reshape garbage.
    return video_latents[:, : state.generated_tokens], audio_latents


def _encoder_inputs(
    conditioning: LTX2ConditioningInfo, *, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    return {
        "encoder_hidden_states": conditioning.video_embeds.to(device=device, dtype=dtype),
        "audio_encoder_hidden_states": conditioning.audio_embeds.to(device=device, dtype=dtype),
        "encoder_attention_mask": conditioning.attention_mask.to(device=device),
        "audio_encoder_attention_mask": conditioning.attention_mask.to(device=device),
    }


def _step_noise(
    state: LTX2DenoiseState,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """The ancestral step's re-injected noise, drawn video-first from one CPU generator.

    Drawn for the GENERATED grid, never for the appended keyframe rows, and then padded back out.
    Those rows are put straight back by the mask restore, so their noise is discarded either way --
    but drawing it would consume a different amount of the shared stream at every step, which
    shifts the audio noise and, from the next step on, the picture too. A last frame would then not
    change how a clip ends, it would change the clip: at a fixed seed on the distilled checkpoint
    (the only schedule that takes this branch) adding one regenerates everything.
    """
    if state.eta <= 0:
        return None, None
    generated = state.generated_tokens
    video_noise = randn_tensor(
        (video_latents.shape[0], generated, video_latents.shape[2]),
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    audio_noise = randn_tensor(
        audio_latents.shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32
    )
    if video_noise.shape[1] != video_latents.shape[1]:
        video_noise = torch.nn.functional.pad(video_noise, (0, 0, 0, video_latents.shape[1] - generated))

    return video_noise.to(device), audio_noise.to(device)


def preview_latent_frame(packed: torch.Tensor, state: LTX2DenoiseState) -> torch.Tensor:
    """The middle latent frame of a packed video tensor, as ``(1, 128, h, w)`` for the previewer.

    Sliced out of the packed rows rather than unpacked whole: the rows are frame-major, so one
    frame is one contiguous span, and unpacking the clip to keep a sixteenth of it would copy the
    whole thing once per step.
    """
    rows_per_frame = state.latent_height * state.latent_width
    start = (state.latent_frames // 2) * rows_per_frame
    frame = packed[:, start : start + rows_per_frame]

    return frame.transpose(1, 2).reshape(packed.shape[0], -1, state.latent_height, state.latent_width)
