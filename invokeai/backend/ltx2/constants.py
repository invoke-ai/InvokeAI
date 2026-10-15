"""The released LTX-2.5 generation recipe's constants.

Everything here is the released pipeline's value rather than a house default, from two upstream
sources that agree on the geometry:

- ``ltx-pipelines`` (Lightricks/LTX-2) ``utils/constants.py`` owns the per-generation
  ``PipelineParams``. A 2.5 checkpoint resolves to the 2.4 row, which inherits 2.3 and moves only
  the image CRF -- that is what fixes 30 steps, STG on block 28, the guidance scales and CRF 18
  for this architecture. Reading 2.0's defaults (40 steps, block 29, CRF 33) instead is the
  mistake the version table exists to prevent.
- ``diffusers.pipelines.ltx2.utils`` owns the distilled sigma lists, re-exported below rather
  than copied.

The two upstream copies of the default negative prompt are *not* identical: diffusers pins an
older commit whose text lacks the five leading dataset tags the current release opens with. The
release's text is the one reproduced here.
"""

from typing import Final

from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

# --- Video geometry ------------------------------------------------------------------------------

LTX2_LATENT_CHANNELS: Final = 128
LTX2_SPATIAL_COMPRESSION: Final = 32
LTX2_TEMPORAL_COMPRESSION: Final = 8

# The transformer patches the latent grid at 1x1x1, so a canvas only has to be a multiple of the
# VAE's spatial compression. ``packing`` asserts these against the loaded transformer's config.
LTX2_PATCH_SIZE: Final = 1
LTX2_PATCH_SIZE_T: Final = 1
LTX2_CANVAS_MULTIPLE: Final = LTX2_SPATIAL_COMPRESSION

# A two-stage run's *final* canvas has to halve onto that grid, because the base pass runs at half
# of it and the x2 latent upscaler doubles a latent grid exactly.
LTX2_TWO_STAGE_CANVAS_MULTIPLE: Final = LTX2_CANVAS_MULTIPLE * 2

# The causal VAE encodes the first frame on its own and every further group of 8, so a clip is
# 8k + 1 pixel frames.
LTX2_FRAME_MODULUS: Final = LTX2_TEMPORAL_COMPRESSION

# The longest clip the conditioning nodes will read. Nothing else bounds a clip's length --
# `validate_num_frames` only checks the grid -- so this is what stops a workflow handing one of them
# a ten-minute recording to decode. Mirrored by LTX2_NUM_FRAMES_MAX in the panel's dimensions.ts,
# which is where a generation's own frame count is bounded.
LTX2_NUM_FRAMES_MAX: Final = 481

LTX2_DEFAULT_FPS: Final = 24.0
LTX2_DEFAULT_NUM_FRAMES: Final = 121

# The decode's tile defaults, which the conditioning encode reuses: both run the same VAE over the
# same kind of clip, and a user who has to raise one usually has to raise the other.
LTX2_DEFAULT_TILE_SIZE: Final = 512
LTX2_DEFAULT_TEMPORAL_TILE: Final = 16

# --- Audio geometry ------------------------------------------------------------------------------

# The audio stream is a log-mel spectrogram at 16 kHz with a 160-sample hop (100 frames/s),
# compressed 4x in time and 4x over the 64 mel bins by the audio VAE: 25 latents/s of 8 channels
# over 16 mel bins, packed to one 128-wide row per latent.
LTX2_AUDIO_SAMPLING_RATE: Final = 16000
LTX2_AUDIO_HOP_LENGTH: Final = 160
LTX2_AUDIO_VAE_COMPRESSION: Final = 4
LTX2_AUDIO_MEL_BINS: Final = 64
LTX2_AUDIO_LATENT_MEL_BINS: Final = LTX2_AUDIO_MEL_BINS // LTX2_AUDIO_VAE_COMPRESSION
LTX2_AUDIO_LATENT_CHANNELS: Final = 8
LTX2_AUDIO_LATENTS_PER_SECOND: Final = LTX2_AUDIO_SAMPLING_RATE / LTX2_AUDIO_HOP_LENGTH / LTX2_AUDIO_VAE_COMPRESSION

# --- Sampling ------------------------------------------------------------------------------------

# Resolution-aware shift anchors: ``mu`` interpolates between ``base_shift`` at ``base_seq_len``
# packed video tokens and ``max_shift`` at ``max_seq_len``, and shifts the sigma schedule toward
# high noise for longer sequences.
LTX2_BASE_SEQ_LEN: Final = 1024
LTX2_MAX_SEQ_LEN: Final = 4096
LTX2_BASE_SHIFT: Final = 0.95
LTX2_MAX_SHIFT: Final = 2.05

# The guidance-distilled checkpoint's fixed schedule. The terminal 0 every schedule ends on is
# appended by ``sampling.build_sigmas``, so this is the 8 sampled levels only.
LTX2_DISTILLED_SIGMAS: Final[tuple[float, ...]] = tuple(DISTILLED_SIGMA_VALUES)
LTX2_DISTILLED_STEPS: Final = len(LTX2_DISTILLED_SIGMAS)

# Where the refine pass re-enters the schedule. Upstream publishes the distilled second stage as a
# list rather than a rule -- ``STAGE_2_DISTILLED_SIGMA_VALUES`` -- and it is exactly the tail of the
# distilled schedule from this level down, which is why the refine pass is built by truncating a
# checkpoint's own schedule instead of carrying a second one. The equality is pinned by a test.
LTX2_STAGE_2_NOISE_SCALE: Final = STAGE_2_DISTILLED_SIGMA_VALUES[0]

# LTX-2.5 samples the distilled schedule ancestrally (an SDE Euler step with full noise
# re-injection); the dev checkpoint's shifted schedule is sampled deterministically.
LTX2_ANCESTRAL_ETA: Final = 1.0
LTX2_ANCESTRAL_S_NOISE: Final = 1.0

# The ancestral loop's noise generator is seeded from the request seed plus this offset. Without
# it the loop's first draw would repeat the initial latent noise exactly: both are a standard
# normal at the same shape from a freshly seeded generator.
LTX2_ANCESTRAL_NOISE_SEED_OFFSET: Final = 10000

# And the refine pass draws from its own stream, for the same reason one step further on. Both
# stages are seeded from the request's seed so a run is reproducible, but the forward process the
# refine pass re-noises with -- ``x = (1 - sigma) * x0 + sigma * eps`` -- assumes ``eps`` is
# independent of ``x0``. Without an offset the refine's draw begins with exactly the values the base
# pass's initial noise was drawn from, in a different layout: the same numbers the clip was grown
# out of, mixed back into it.
LTX2_REFINE_NOISE_SEED_OFFSET: Final = 20000

LTX2_DEV_STEPS: Final = 30

# --- Guidance ------------------------------------------------------------------------------------

# Spatio-temporal guidance perturbs the self-attention of these blocks (2.3 moved them from 29).
LTX2_STG_BLOCKS: Final[tuple[int, ...]] = (28,)

LTX2_CFG_SCALE: Final = 3.0
LTX2_AUDIO_CFG_SCALE: Final = 7.0
LTX2_STG_SCALE: Final = 1.0
LTX2_MODALITY_SCALE: Final = 3.0
LTX2_GUIDANCE_RESCALE: Final = 0.7

# --- Conditioning --------------------------------------------------------------------------------

# A conditioning image is re-compressed as one H.264 intra frame at this CRF so it carries the
# compression artefacts the model was trained against. 2.5 moved it from 33.
LTX2_IMAGE_CRF: Final = 18

LTX2_MAX_SEQUENCE_LENGTH: Final = 1024

LTX2_DEFAULT_NEGATIVE_PROMPT: Final = (
    "has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, "
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)
