import type { MiniMaxH3TargetResolution, VideoAspectRatioId, WanTargetResolution } from './types';

/**
 * Client-side ports of the backend's video canvas math, so the panel can show
 * and validate the exact dimensions a graph will run at without extra nodes:
 *
 * - Wan: `_scale_and_snap` in `invokeai/app/invocations/wan_ideal_dimensions.py`
 *   ("nearest" rounding — the node default; the other modes are workflow-only).
 * - MiniMax H3: `resolve_canvas_size` in `invokeai/backend/minimax_h3/packing.py`
 *   and `resolve_lowres_canvas_size` in `invokeai/backend/minimax_h3/presets.py`.
 *
 * Where the backend raises, these return null: the panel falls back to defaults
 * and reports the problem through `getVideoValidationReasons` instead of throwing.
 */

export interface VideoDimensions {
  width: number;
  height: number;
}

// Python's round() (used by both backend implementations) is half-to-even;
// Math.round is half-up. The ports must agree with the backend on exact .5
// quotients (e.g. 720 / 32 = 22.5) or panel and workflow dims would differ.
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
 * Scale a source W×H so its shorter side equals the preset's pixel count, then
 * snap each dimension to the Wan pixel grid (nearest). Only the ratio of the
 * inputs matters, so aspect-ratio parts (16, 9) work as well as real pixels.
 * Null when the inputs are non-positive or non-finite.
 *
 * Deliberate divergence from the backend node: `_scale_and_snap` rejects
 * sources whose RAW long side is under one grid cell, which would also reject
 * pure ratio parts like (16, 9) on the ×32 grid. Since only the ratio matters
 * and scaling happens before snapping, tiny-but-well-formed inputs are
 * accepted here — the scaled long side is always ≥ the target short side.
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
 * The MiniMax H3 canvas for an aspect ratio. "768 highres" is the released
 * pipeline's policy: short edge 768, soft area cap of 768×1344, both axes then
 * rounded to the nearest multiple of 32 (so the final area may sit slightly
 * above the pre-rounding budget). "768 lowres" pins the LONG edge to 768
 * instead for cheaper preview renders. Only the ratio of the inputs matters.
 * Null when the inputs are degenerate or the ratio is outside H3's supported
 * 1:4 – 4:1 range.
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

// Wan's VAE compresses 4 pixel frames into 1 latent frame, so (n - 1) % 4 == 0.
// 81 frames (5 s at 16 fps) is the training default; the slider allows up to
// twice that, but coherence degrades past 81 as temporal RoPE leaves its
// training distribution — the docs recommend chaining extends instead.
export const WAN_NUM_FRAMES_MIN = 5;
export const WAN_NUM_FRAMES_MAX = 161;
export const WAN_NUM_FRAMES_STEP = 4;
export const WAN_NUM_FRAMES_DEFAULT = 81;

export const WAN_FPS_MIN = 1;
export const WAN_FPS_MAX = 120;
export const WAN_FPS_DEFAULT = 16;

export const isValidWanNumFrames = (numFrames: number): boolean =>
  Number.isInteger(numFrames) && numFrames >= WAN_NUM_FRAMES_MIN && (numFrames - 1) % WAN_NUM_FRAMES_STEP === 0;

export const snapWanNumFrames = (numFrames: number): number => {
  if (!Number.isFinite(numFrames)) {
    return WAN_NUM_FRAMES_DEFAULT;
  }

  const clamped = Math.min(WAN_NUM_FRAMES_MAX, Math.max(WAN_NUM_FRAMES_MIN, numFrames));

  return Math.round((clamped - 1) / WAN_NUM_FRAMES_STEP) * WAN_NUM_FRAMES_STEP + 1;
};

export const MINIMAX_H3_FPS = 24;

// The 17n + 5 grid points the H3 video VAE can encode, from the accepted
// 3.75 s floor to the 15 s ceiling. Mirrors MINIMAX_H3_VIDEO_FRAME_CHOICES in
// invokeai/backend/minimax_h3/presets.py; the 5-frame still-image block is
// deliberately absent — the panel generates video, not stills.
export const MINIMAX_H3_NUM_FRAMES_CHOICES: readonly number[] = Array.from({ length: 16 }, (_, i) => 90 + i * 17);
export const MINIMAX_H3_NUM_FRAMES_DEFAULT = 124;

export const isValidMiniMaxH3NumFrames = (numFrames: number): boolean =>
  MINIMAX_H3_NUM_FRAMES_CHOICES.includes(numFrames);

export const snapMiniMaxH3NumFrames = (numFrames: number): number => {
  if (!Number.isFinite(numFrames)) {
    return MINIMAX_H3_NUM_FRAMES_DEFAULT;
  }

  let best = MINIMAX_H3_NUM_FRAMES_DEFAULT;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const choice of MINIMAX_H3_NUM_FRAMES_CHOICES) {
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
