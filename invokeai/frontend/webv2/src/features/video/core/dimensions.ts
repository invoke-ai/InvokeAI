import type {
  Ltx2TargetResolution,
  MiniMaxH3TargetResolution,
  VideoAspectRatioId,
  VideoConditioningClip,
  VideoReferenceImageDetail,
  WanTargetResolution,
} from './types';

/** Null where the backend's math rejects the input, so panel validation can explain it. */

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

/** Unlike the backend raw-size guard, small positive ratio parts are accepted. */
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

/** Highres: 768 short edge, soft 768x1344 area cap, 32-px rounding; lowres caps the long edge at 768. 1:4–4:1. */
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

/** 16x VAE compression and 2x2 latent patches make one 32x32 pixel block one row. */
export const MINIMAX_H3_ROW_PIXELS = 32 * 32;

/** Reference rows join every denoise step; `max` uses a 2048 short edge, `match` the capped generation area. */
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

/** Doubled grid so the half-size base pass still lands on the plain one; mirrors backend/ltx2/constants.py. */
export const LTX2_TWO_STAGE_CANVAS_MULTIPLE = LTX2_CANVAS_MULTIPLE * 2;

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

/** The preset pins the short edge; both axes snap to 32. No aspect or area limit: only degenerate inputs give null. */
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

/** `base` is exactly half of `final` (the x2 latent upscaler); mirrors `base_canvas` in backend/ltx2/packing.py. */
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

// The causal VAE encodes frame 1, then groups of 8: (n - 1) % 8 == 0. 121 is 5 s at 24 fps; 481 is 20 s.
export const LTX2_NUM_FRAMES_MIN = 9;
export const LTX2_NUM_FRAMES_MAX = 481;
export const LTX2_NUM_FRAMES_STEP = 8;
export const LTX2_NUM_FRAMES_DEFAULT = 121;

// Mirrors the duration head's trained range in ltx2_duration.py; the node accepts only these.
export const LTX2_DURATION_MIN_SECONDS = 1;
export const LTX2_DURATION_MAX_SECONDS = 20;

/** Frames is a ceiling under auto duration; null when the range is empty, leaving Frames as set. */
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

/** Mirrors `snap_num_frames_down` in ltx2/packing.py: trailing frames drop, since padding would invent content. */
export const snapLtx2FramesDown = (numFrames: number): number =>
  Math.max(0, Math.floor((Math.floor(numFrames) - 1) / LTX2_NUM_FRAMES_STEP)) * LTX2_NUM_FRAMES_STEP + 1;

/** A prediction; the graph wires the node's own count. In the `audio` role `fps` sets how many frames are covered. */
export const ltx2FramesForClip = (conditioning: VideoConditioningClip, fps: number): number => {
  if (conditioning.role === 'video') {
    return snapLtx2FramesDown(conditioning.clip.numFrames);
  }

  const seconds = conditioning.clip.fps > 0 ? conditioning.clip.numFrames / conditioning.clip.fps : 0;

  return snapLtx2FramesDown(Math.trunc(seconds * fps));
};

/** Mirrors `LTX2_DEFAULT_EXTEND_CONTEXT_FRAMES` in ltx2_extend_conditioning.py, the node's own default. */
export const LTX2_EXTEND_CONTEXT_FRAMES = 17;

/**
 * `video_concat` buffers the crossfade at the source's native resolution and refuses over 512 MiB (mirrors its
 * `MAX_TRANSITION_MEMORY_BYTES` estimate). Checked here, or the refusal lands after the whole run.
 */
export const ltx2ExtendJoinFitsInMemory = (width: number, height: number, contextFrames: number): boolean =>
  width * height * 3 * (contextFrames * 2 + 13) <= 512 * 1024 * 1024;

/** On the 8k + 1 grid; 0 when even the smallest usable context (9) does not fit. */
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

/** The join emits `sum(inputs) - transition * (n - 1)`, and the transition is the context. */
export const ltx2NewFramesForExtend = (numFrames: number, contextFrames: number): number =>
  Math.max(0, numFrames - contextFrames);

export const LTX2_FPS_MIN = 1;
export const LTX2_FPS_MAX = 60;
export const LTX2_FPS_DEFAULT = 24;

export const getVideoAspectRatioParts = (id: VideoAspectRatioId): VideoDimensions => {
  const [width = 1, height = 1] = id.split(':').map(Number);

  return { height, width };
};

/** The portrait/landscape mirror of a preset; every offered ratio has one. */
export const invertVideoAspectRatioId = (id: VideoAspectRatioId): VideoAspectRatioId => {
  const { width, height } = getVideoAspectRatioParts(id);

  return `${height}:${width}` as VideoAspectRatioId;
};

// Wan requires 4n+1 frames; 81 is the training default, and longer clips leave the trained range.
export const WAN_NUM_FRAMES_MIN = 5;
export const WAN_NUM_FRAMES_MAX = 161;
export const WAN_NUM_FRAMES_STEP = 4;
export const WAN_NUM_FRAMES_DEFAULT = 81;

export const WAN_FPS_MIN = 1;
export const WAN_FPS_MAX = 120;
export const WAN_FPS_DEFAULT = 16;

export const isValidWanNumFrames = (numFrames: number): boolean =>
  Number.isInteger(numFrames) && numFrames >= WAN_NUM_FRAMES_MIN && (numFrames - 1) % WAN_NUM_FRAMES_STEP === 0;

export interface VideoFramesGrid {
  min: number;
  max: number;
  step: number;
  defaultValue: number;
}

/** Ties round up: rounding a short request down can collapse it to the minimum. Matches `snap_num_frames`. */
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

/** Mirrors `LTX2_DEFAULT_NEGATIVE_PROMPT` in backend/ltx2/constants.py; seeded so LTX-2 runs the released recipe. */
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
