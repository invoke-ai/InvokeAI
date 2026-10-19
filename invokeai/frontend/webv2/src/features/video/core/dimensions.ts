import type {
  Ltx2TargetResolution,
  MiniMaxH3TargetResolution,
  VideoAspectRatioId,
  VideoConditioningClip,
  VideoReferenceImageDetail,
  WanTargetResolution,
} from './types';

/**
 * Mirror Wan _scale_and_snap, the H3 and LTX-2 canvas resolvers, and reference-image normalization. Return null
 * where backend math rejects input so panel validation can explain it.
 */

export interface VideoDimensions {
  width: number;
  height: number;
}

// Match Python half-to-even rounding; Math.round differs at exact .5 and would produce inconsistent canvas sizes.
const roundHalfToEven = (value: number): number => {
  const floor = Math.floor(value);
  const diff = value - floor;

  if (diff > 0.5) {
    return floor + 1;
  }

  if (diff < 0.5) {
    return floor;
  }

  return floor % 2 === 0 ? floor : floor + 1;
};

const snapToMultiple = (value: number, multiple: number): number =>
  Math.max(multiple, roundHalfToEven(value / multiple) * multiple);

/** Pixel-grid multiple = 2 (transformer patch) × VAE spatial scale. */
export const WAN_A14B_PIXEL_MULTIPLE = 16;
export const WAN_TI2V_PIXEL_MULTIPLE = 32;

/** Short-side pixel count for each Wan preset ("p" names the short dimension). */
export const WAN_TARGET_RESOLUTION_PX: Record<WanTargetResolution, number> = {
  '480p': 480,
  '720p': 720,
  '1080p': 1080,
};

/**
 * Scale the ratio to the target short side and snap to Wan's grid. Unlike the backend raw-size guard, accept small
 * positive ratio parts; reject nonfinite/nonpositive inputs.
 */
export const scaleAndSnapWanDimensions = (
  width: number,
  height: number,
  targetResolution: WanTargetResolution,
  multiple: number
): VideoDimensions | null => {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }

  const targetShortSide = WAN_TARGET_RESOLUTION_PX[targetResolution];
  const scale = targetShortSide / Math.min(width, height);

  return {
    height: snapToMultiple(height * scale, multiple),
    width: snapToMultiple(width * scale, multiple),
  };
};

export const MINIMAX_H3_SHORT_EDGE = 768;
export const MINIMAX_H3_MAX_PIXELS = 768 * 1344;
export const MINIMAX_H3_CANVAS_MULTIPLE = 32;
export const MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4;
export const MINIMAX_H3_MAX_ASPECT_RATIO = 4;

/**
 * H3 highres uses a 768 short edge and soft 768x1344 area cap before 32-pixel rounding; lowres caps the long edge
 * at 768. Reject ratios outside 1:4–4:1.
 */
export const resolveMiniMaxH3Canvas = (
  width: number,
  height: number,
  targetResolution: MiniMaxH3TargetResolution
): VideoDimensions | null => {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }

  const ratio = width / height;

  if (ratio < MINIMAX_H3_MIN_ASPECT_RATIO || ratio > MINIMAX_H3_MAX_ASPECT_RATIO) {
    return null;
  }

  let rawWidth: number;
  let rawHeight: number;

  if (targetResolution === '768 lowres') {
    // Long edge pinned to 768; area always sits far below the cap.
    if (ratio >= 1) {
      rawWidth = MINIMAX_H3_SHORT_EDGE;
      rawHeight = MINIMAX_H3_SHORT_EDGE / ratio;
    } else {
      rawWidth = MINIMAX_H3_SHORT_EDGE * ratio;
      rawHeight = MINIMAX_H3_SHORT_EDGE;
    }
  } else {
    if (ratio >= 1) {
      rawWidth = MINIMAX_H3_SHORT_EDGE * ratio;
      rawHeight = MINIMAX_H3_SHORT_EDGE;
    } else {
      rawWidth = MINIMAX_H3_SHORT_EDGE;
      rawHeight = MINIMAX_H3_SHORT_EDGE / ratio;
    }

    const area = rawWidth * rawHeight;

    if (area > MINIMAX_H3_MAX_PIXELS) {
      const scale = Math.sqrt(MINIMAX_H3_MAX_PIXELS / area);
      rawWidth *= scale;
      rawHeight *= scale;
    }
  }

  return {
    height: snapToMultiple(rawHeight, MINIMAX_H3_CANVAS_MULTIPLE),
    width: snapToMultiple(rawWidth, MINIMAX_H3_CANVAS_MULTIPLE),
  };
};

/** Upstream's reference-image rule: a constant short edge, whatever the generation size is. */
export const MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048;

/**
 * Pixels per packed row: H3 encodes at a 16x spatial compression and the transformer packs
 * 2x2 latent patches, so a 32x32 pixel block is one row.
 */
export const MINIMAX_H3_ROW_PIXELS = 32 * 32;

/**
 * Reference rows participate in every denoise step. max uses a 2048 short edge; match uses capped generation area.
 * Return null without valid geometry/target area.
 */
export const resolveMiniMaxH3ReferenceImage = (
  width: number,
  height: number,
  detail: VideoReferenceImageDetail,
  targetArea: number | null
): { dimensions: VideoDimensions; rows: number } | null => {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }

  let shortEdge = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE;

  if (detail === 'match') {
    if (targetArea === null || !Number.isFinite(targetArea) || targetArea <= 0) {
      return null;
    }
    const matched = Math.max(
      MINIMAX_H3_CANVAS_MULTIPLE,
      roundHalfToEven(Math.min(width, height) * Math.sqrt(targetArea / (width * height)))
    );

    shortEdge = Math.min(MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE, matched);
  }

  const scale = shortEdge / Math.min(width, height);
  const dimensions = {
    height: snapToMultiple(height * scale, MINIMAX_H3_CANVAS_MULTIPLE),
    width: snapToMultiple(width * scale, MINIMAX_H3_CANVAS_MULTIPLE),
  };

  // Both axes are multiples of 32, so this is exact.
  return { dimensions, rows: (dimensions.width * dimensions.height) / MINIMAX_H3_ROW_PIXELS };
};

export const LTX2_CANVAS_MULTIPLE = 32;

/**
 * A two-stage canvas is chosen on the doubled grid so that halving it -- which is what the base
 * pass runs at, the x2 latent upscaler doubling a latent grid exactly -- still lands on the plain
 * one. Mirrors `LTX2_TWO_STAGE_CANVAS_MULTIPLE` in `invokeai/backend/ltx2/constants.py`.
 */
export const LTX2_TWO_STAGE_CANVAS_MULTIPLE = LTX2_CANVAS_MULTIPLE * 2;

/** Short-side pixel count for each LTX-2 preset ("p" names the short dimension). */
export const LTX2_TARGET_RESOLUTION_PX: Record<Ltx2TargetResolution, number> = {
  '512p': 512,
  '704p': 704,
  '768p': 768,
  '1024p': 1024,
  '1536p': 1536,
};

/** The presets that run a base pass and then a refine pass over an upscaled latent. */
export const LTX2_TWO_STAGE_RESOLUTIONS: ReadonlySet<Ltx2TargetResolution> = new Set(['1024p', '1536p']);

export const isLtx2TwoStage = (targetResolution: Ltx2TargetResolution): boolean =>
  LTX2_TWO_STAGE_RESOLUTIONS.has(targetResolution);

/**
 * The LTX-2 canvas for an aspect ratio: the preset pins the SHORT edge, the long
 * edge follows the source's ratio, and both axes snap to the VAE's 32-pixel grid.
 * LTX-2 declares no aspect-ratio limit and no area cap, so only degenerate inputs
 * return null. Only the ratio of the inputs matters.
 */
export const resolveLtx2Canvas = (
  width: number,
  height: number,
  targetResolution: Ltx2TargetResolution
): VideoDimensions | null => {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return null;
  }

  const shortEdge = LTX2_TARGET_RESOLUTION_PX[targetResolution];
  const multiple = isLtx2TwoStage(targetResolution) ? LTX2_TWO_STAGE_CANVAS_MULTIPLE : LTX2_CANVAS_MULTIPLE;
  const ratio = width / height;
  const raw =
    ratio >= 1 ? { height: shortEdge, width: shortEdge * ratio } : { height: shortEdge / ratio, width: shortEdge };

  return {
    height: snapToMultiple(raw.height, multiple),
    width: snapToMultiple(raw.width, multiple),
  };
};

/**
 * The two canvases a two-stage run uses: the base pass's, and the final one the refine pass
 * produces. `base` is exactly half of `final` on both axes -- not a resize, but the x2 latent
 * upscaler's doubling read backwards, which is why `resolveLtx2Canvas` puts a two-stage canvas on
 * the 64 grid. Mirrors `base_canvas` in `invokeai/backend/ltx2/packing.py`.
 *
 * A single-stage preset returns the same canvas for both, so a caller can wire one shape.
 */
export const getLtx2StageCanvases = (
  width: number,
  height: number,
  targetResolution: Ltx2TargetResolution
): { base: VideoDimensions; final: VideoDimensions } | null => {
  const final = resolveLtx2Canvas(width, height, targetResolution);

  if (!final) {
    return null;
  }

  if (!isLtx2TwoStage(targetResolution)) {
    return { base: final, final };
  }

  return { base: { height: final.height / 2, width: final.width / 2 }, final };
};

// LTX-2's causal VAE encodes the first frame alone and then groups of 8, so
// (n - 1) % 8 == 0. 121 frames is 5 s at the model's 24 fps default, the length
// the released pipeline generates; 481 is 20 s, the longest clip the family
// generates. Time and VRAM grow roughly linearly with the frame count.
export const LTX2_NUM_FRAMES_MIN = 9;
export const LTX2_NUM_FRAMES_MAX = 481;
export const LTX2_NUM_FRAMES_STEP = 8;
export const LTX2_NUM_FRAMES_DEFAULT = 121;

// The seconds range the duration head was trained to regress over, mirroring
// LTX2_DURATION_MIN_SECONDS / _MAX_SECONDS in ltx2_duration.py. Outside it the head extrapolates,
// so these bound what the node will accept rather than merely defaulting it.
export const LTX2_DURATION_MIN_SECONDS = 1;
export const LTX2_DURATION_MAX_SECONDS = 20;

/**
 * The seconds range the duration head may choose a length from.
 *
 * Under auto duration the Frames value is a ceiling, not a length: it is the number the user sized
 * the run's memory for, and the head picks anything from the family's shortest clip up to it. Both
 * ends are converted at the run's own rate and clamped into the head's trained range.
 *
 * Null when that range is empty -- a ceiling at or under the floor (under a second, or under the
 * family's minimum) leaves the head nothing to choose, and the run uses the Frames value as set.
 */
export const ltx2AutoDurationBounds = (
  fps: number,
  framesMin: number,
  framesCeiling: number
): { maxSeconds: number; minSeconds: number } | null => {
  const clamp = (seconds: number) => Math.min(LTX2_DURATION_MAX_SECONDS, Math.max(LTX2_DURATION_MIN_SECONDS, seconds));
  const minSeconds = clamp(framesMin / fps);
  const maxSeconds = clamp(framesCeiling / fps);

  return maxSeconds > minSeconds ? { maxSeconds, minSeconds } : null;
};

/**
 * A frame count snapped *down* onto LTX-2's 8n + 1 grid. Mirrors `snap_num_frames_down` in
 * `invokeai/backend/ltx2/packing.py`: a clip supplies whatever it supplies, and the trailing
 * frames past the last whole group are dropped rather than padded -- padding would invent
 * picture, or silence, for the model to hold clean.
 */
export const snapLtx2FramesDown = (numFrames: number): number =>
  Math.max(0, Math.floor((Math.floor(numFrames) - 1) / LTX2_NUM_FRAMES_STEP)) * LTX2_NUM_FRAMES_STEP + 1;

/**
 * The frame count a conditioning clip will produce, as the panel can predict it. The graph wires
 * the conditioning node's own count into the denoise rather than trusting this, because the two
 * can differ: an audio track need not be exactly as long as the picture it came with, and the
 * gallery's frame count is itself duration x fps rounded.
 *
 * With the clip in the `audio` role the picture is generated at the panel's own frame rate, so
 * `fps` decides how many frames the soundtrack covers. In the `video` role the clip's own frames
 * are the generation, and `fps` is ignored.
 */
export const ltx2FramesForClip = (conditioning: VideoConditioningClip, fps: number): number => {
  if (conditioning.role === 'video') {
    return snapLtx2FramesDown(conditioning.clip.numFrames);
  }

  const seconds = conditioning.clip.fps > 0 ? conditioning.clip.numFrames / conditioning.clip.fps : 0;

  return snapLtx2FramesDown(Math.trunc(seconds * fps));
};

/**
 * Frames of the source an LTX-2 continuation opens with, held clean so the model can read the
 * clip's motion rather than just its last still. Mirrors `LTX2_DEFAULT_EXTEND_CONTEXT_FRAMES` in
 * `invokeai/app/invocations/ltx2/ltx2_extend_conditioning.py`, which is the node's own default.
 */
export const LTX2_EXTEND_CONTEXT_FRAMES = 17;

/**
 * What the join can afford to blend, in source pixels. `video_concat` buffers the crossfade at the
 * FIRST input's native resolution -- the trimmed source, not the generation canvas -- and refuses
 * anything over 512 MiB. Mirrors `MAX_TRANSITION_MEMORY_BYTES` and `_BLEND_WORKING_FRAMES` in
 * `invokeai/app/invocations/video_concat.py`, whose estimate is
 * `width * height * 3 * (transition_frames * 2 + 13)` for a crossfade.
 *
 * At LTX-2's 17-frame overlap that caps a source at ~3.8 megapixels: 2560x1440 fits with 3% to
 * spare, 4K needs 1115 MiB and does not. Without this the refusal lands in the join, after both
 * encodes, the transformer and the decode have already run, and neither remedy its message offers
 * is reachable from the panel.
 */
export const ltx2ExtendJoinFitsInMemory = (width: number, height: number, contextFrames: number): boolean =>
  width * height * 3 * (contextFrames * 2 + 13) <= 512 * 1024 * 1024;

/**
 * The widest context the join can blend for a given source, on the 8k + 1 grid.
 *
 * The memory ceiling is a property of the *source's* pixels, not the generation canvas, so it moves
 * with the clip the user picked: a 2560x1440 source affords 17 frames with 3% to spare while a 4K
 * one affords none. Exposed as a live bound rather than a fixed check because the control is now the
 * user's to drag — without it they could set a value that is refused only at enqueue, after both
 * encodes and the transformer have already run.
 *
 * Returns 0 when even the smallest usable context (9) does not fit, which is the panel's signal that
 * this source cannot be extended at all.
 */
export const ltx2MaxExtendContextFrames = (width: number, height: number): number => {
  for (
    let frames = snapLtx2FramesDown(LTX2_NUM_FRAMES_MAX);
    frames >= 1 + LTX2_NUM_FRAMES_STEP;
    frames -= LTX2_NUM_FRAMES_STEP
  ) {
    if (ltx2ExtendJoinFitsInMemory(width, height, frames)) {
      return frames;
    }
  }

  return 0;
};

/**
 * New material a continuation actually adds, in frames.
 *
 * The join emits `sum(inputs) - transition_frames * (n - 1)`, and the transition is the context, so
 * with two clips the source keeps its own length and the generation contributes `numFrames - context`.
 * Every frame of context is therefore a frame of new video given up — the trade the panel shows
 * beside the control, because Frames alone does not reveal it.
 */
export const ltx2NewFramesForExtend = (numFrames: number, contextFrames: number): number =>
  Math.max(0, numFrames - contextFrames);

export const LTX2_FPS_MIN = 1;
export const LTX2_FPS_MAX = 60;
export const LTX2_FPS_DEFAULT = 24;

/** The width/height parts of a preset ratio, for feeding the canvas resolvers. */
export const getVideoAspectRatioParts = (id: VideoAspectRatioId): VideoDimensions => {
  const [width = 1, height = 1] = id.split(':').map(Number);

  return { height, width };
};

/** The portrait/landscape mirror of a preset; every offered ratio has one. */
export const invertVideoAspectRatioId = (id: VideoAspectRatioId): VideoAspectRatioId => {
  const { width, height } = getVideoAspectRatioParts(id);

  return `${height}:${width}` as VideoAspectRatioId;
};

// Wan requires 4n+1 frames. The 81-frame training default supports best coherence; longer clips extend beyond the
// trained temporal range.
export const WAN_NUM_FRAMES_MIN = 5;
export const WAN_NUM_FRAMES_MAX = 161;
export const WAN_NUM_FRAMES_STEP = 4;
export const WAN_NUM_FRAMES_DEFAULT = 81;

export const WAN_FPS_MIN = 1;
export const WAN_FPS_MAX = 120;
export const WAN_FPS_DEFAULT = 16;

export const isValidWanNumFrames = (numFrames: number): boolean =>
  Number.isInteger(numFrames) && numFrames >= WAN_NUM_FRAMES_MIN && (numFrames - 1) % WAN_NUM_FRAMES_STEP === 0;

/** A frame count's grid, as the variant policies declare it. */
export interface VideoFramesGrid {
  min: number;
  max: number;
  step: number;
  defaultValue: number;
}

/**
 * The nearest frame count on a family's grid, clamped to its range. Ties round
 * UP: a count halfway between two grid points is as close to either, and
 * rounding a short request down toward the floor is the worse answer (it can
 * collapse a clip to the minimum). Matches `snap_num_frames` in
 * `invokeai/backend/ltx2/packing.py`.
 */
export const snapNumFramesToGrid = (grid: VideoFramesGrid, numFrames: number): number => {
  if (!Number.isFinite(numFrames)) {
    return grid.defaultValue;
  }

  const clamped = Math.min(grid.max, Math.max(grid.min, numFrames));

  return Math.floor((clamped - grid.min) / grid.step + 0.5) * grid.step + grid.min;
};

export const MINIMAX_H3_FPS = 24;

// Mirror H3's 17n+5 video frame choices from presets.py; exclude the five-frame still-image block.
export const MINIMAX_H3_NUM_FRAMES_CHOICES: readonly number[] = Array.from({ length: 16 }, (_, i) => 90 + i * 17);
export const MINIMAX_H3_NUM_FRAMES_DEFAULT = 124;

export const isValidMiniMaxH3NumFrames = (numFrames: number): boolean =>
  MINIMAX_H3_NUM_FRAMES_CHOICES.includes(numFrames);

/** A frame count's choice list, as the variant policies declare it. */
export interface VideoFramesChoices {
  choices: readonly number[];
  defaultValue: number;
}

/** The nearest offered frame count; the first of two equally near ones wins. */
export const snapNumFramesToChoices = (policy: VideoFramesChoices, numFrames: number): number => {
  if (!Number.isFinite(numFrames)) {
    return policy.defaultValue;
  }

  let best = policy.defaultValue;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const choice of policy.choices) {
    const distance = Math.abs(choice - numFrames);

    if (distance < bestDistance) {
      best = choice;
      bestDistance = distance;
    }
  }

  return best;
};

/** Clip length in seconds; matches the backend's `n / fps` labeling. */
export const getVideoDurationSeconds = (numFrames: number, fps: number): number | null =>
  Number.isFinite(numFrames) && Number.isFinite(fps) && fps > 0 && numFrames >= 0 ? numFrames / fps : null;

/**
 * The negative prompt LTX-2 was released with: a list of artifact and audio-defect tags its dev
 * checkpoint guides against at CFG 3. Mirrors `LTX2_DEFAULT_NEGATIVE_PROMPT` in
 * `invokeai/backend/ltx2/constants.py`, which is the text-encoder node's own default — the panel
 * seeds it so a fresh LTX-2 panel runs the released recipe rather than steering against nothing.
 */
export const LTX2_DEFAULT_NEGATIVE_PROMPT =
  'has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, ' +
  'blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, ' +
  'grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, ' +
  'deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, ' +
  'wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of ' +
  'field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent ' +
  'lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny ' +
  'valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, ' +
  'mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, ' +
  'off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward ' +
  'pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, ' +
  'inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.';
