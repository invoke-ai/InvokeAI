/** A playback window in media seconds. */
export interface LoopRange {
  startSeconds: number;
  endSeconds: number;
}

/**
 * The part of `range` the media can loop: the end is clamped to a known duration and the start into `[0, end]`.
 * Collapsed or inverted ranges loop nothing, so they return null.
 */
export const clampLoopRange = (range: LoopRange, durationSeconds: number): LoopRange | null => {
  const endSeconds = durationSeconds > 0 ? Math.min(range.endSeconds, durationSeconds) : range.endSeconds;
  const startSeconds = Math.max(0, Math.min(range.startSeconds, endSeconds));

  return endSeconds > startSeconds ? { endSeconds, startSeconds } : null;
};

// Containers round durations; a frame that starts within this of the end does not count as a frame.
const FRAME_EPSILON = 1e-6;

/** Whole frames in the media, at least one. */
export const getFrameCount = (durationSeconds: number, fps: number): number =>
  Math.max(1, Math.floor(durationSeconds * fps + FRAME_EPSILON));

/** The frame shown at `timeSeconds`, clamped to the media's frames (the end of the clip shows the last one). */
export const getFrameAtTime = (timeSeconds: number, fps: number, frameCount: number): number =>
  Math.min(frameCount - 1, Math.max(0, Math.floor(timeSeconds * fps + FRAME_EPSILON)));

/** Seek target for a frame: the middle of its display interval, since seeking onto a boundary can decode a neighbor. */
export const getFrameTime = (frame: number, fps: number): number => (frame + 0.5) / fps;

/**
 * Scrubber steps without a frame rate. `pointer` is a thousandth of the clip, finer than a track's pixels, so a drag
 * lands where it points. `keyboard` is a hundredth, between 0.1 s (short clips stay fine-grained) and 5 s (an arrow
 * key on a long clip still moves visibly).
 */
export const getScrubSteps = (durationSeconds: number): { keyboard: number; pointer: number } => ({
  keyboard: Math.min(5, Math.max(0.1, durationSeconds / 100)),
  pointer: Math.max(0.001, durationSeconds / 1000),
});

/** `m:ss`, or `h:mm:ss` from an hour; negative and non-finite times read as zero. */
export const formatMediaTime = (seconds: number): string => {
  const total = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0;
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secondsPart = String(total % 60).padStart(2, '0');

  return hours > 0 ? `${hours}:${String(minutes).padStart(2, '0')}:${secondsPart}` : `${minutes}:${secondsPart}`;
};
