/**
 * Pure swipe physics for the preview carousel: gesture intent, finger-velocity estimation, edge resistance, and the
 * release decision. Offsets are stage-space pixels; negative moves the image left, revealing the next item.
 */

/** Travel that decides a touch; matches the touch drag sensor's tolerance, so a still-waiting drag yields on the same move. */
export const SWIPE_SLOP_PX = 10;
/** Release speed (px/ms) that commits regardless of distance. */
export const SWIPE_FLICK_VELOCITY = 0.4;
/** Fraction of the stage width past which a slow release commits. */
export const SWIPE_COMMIT_FRACTION = 0.4;
/** Only the last stretch of motion counts toward release velocity, so a pause before lifting is not a flick. */
const VELOCITY_WINDOW_MS = 100;
/** Settle speed floor (px/ms), so a slow or velocity-free release still finishes briskly. */
const MIN_SETTLE_VELOCITY = 1.5;
const MIN_SETTLE_MS = 90;
const MAX_SETTLE_MS = 320;
/** Share of the stage width an overscroll can approach at the gallery's ends. */
const RUBBER_BAND_LIMIT = 0.3;

export type SwipeDirection = -1 | 1;

export interface SwipeSample {
  time: number;
  x: number;
}

/** Returns +1 (next) or -1 (previous) for the item an offset reveals; zero reveals nothing. */
export const getRevealedDirection = (offset: number): SwipeDirection | 0 => (offset < 0 ? 1 : offset > 0 ? -1 : 0);

/** Horizontal intent once travel leaves the slop: claim sideways motion, release vertical motion, else keep waiting. */
export const classifySwipeIntent = (dx: number, dy: number): 'pending' | 'reject' | 'swipe' => {
  if (Math.hypot(dx, dy) <= SWIPE_SLOP_PX) {
    return 'pending';
  }

  return Math.abs(dx) > Math.abs(dy) ? 'swipe' : 'reject';
};

/** Appends a sample, discarding those older than the velocity window. */
export const recordSwipeSample = (samples: SwipeSample[], sample: SwipeSample): void => {
  samples.push(sample);

  while (samples.length > 2 && sample.time - samples[0]!.time > VELOCITY_WINDOW_MS) {
    samples.shift();
  }
};

/** Finger speed in px/ms over the recent window; zero when the finger rested at the end. */
export const getSwipeVelocity = (samples: readonly SwipeSample[], now: number): number => {
  const last = samples.at(-1);
  const first = samples.find((sample) => now - sample.time <= VELOCITY_WINDOW_MS);

  if (!last || !first || last === first || now - last.time > VELOCITY_WINDOW_MS) {
    return 0;
  }

  const elapsed = last.time - first.time;

  return elapsed > 0 ? (last.x - first.x) / elapsed : 0;
};

/** Diminishing overscroll: follows the finger at first, then approaches a fixed share of the width. */
export const rubberBand = (offset: number, width: number): number => {
  if (width <= 0) {
    return 0;
  }

  const limit = width * RUBBER_BAND_LIMIT;

  return Math.sign(offset) * limit * (1 - 1 / (Math.abs(offset) / limit + 1));
};

/** The finger travel that `rubberBand` shows as `displayed`, so a caught overscroll resumes without a jump. */
export const unrubberBand = (displayed: number, width: number): number => {
  const limit = width * RUBBER_BAND_LIMIT;
  const magnitude = Math.min(Math.abs(displayed), limit * 0.999);

  return width <= 0 ? 0 : (Math.sign(displayed) * limit * magnitude) / (limit - magnitude);
};

export type SwipeRelease =
  | { direction: SwipeDirection; durationMs: number; kind: 'commit' }
  | { durationMs: number; kind: 'cancel' };

/**
 * Decide a release: a flick toward an available neighbor, or a slow drag past the commit fraction, commits; anything
 * else, including a flick back toward the start, returns to rest. Settle time follows the finger's speed, so fast
 * flicks finish fast.
 */
export const resolveSwipeRelease = ({
  canNavigate,
  offset,
  velocity,
  width,
}: {
  canNavigate: (direction: SwipeDirection) => boolean;
  offset: number;
  velocity: number;
  width: number;
}): SwipeRelease => {
  const direction = getRevealedDirection(offset);
  const isFlick = Math.abs(velocity) >= SWIPE_FLICK_VELOCITY;
  const isFlickForward = isFlick && Math.sign(velocity) === Math.sign(offset);
  const isPastThreshold = width > 0 && Math.abs(offset) >= width * SWIPE_COMMIT_FRACTION;

  if (direction !== 0 && canNavigate(direction) && (isFlickForward || (!isFlick && isPastThreshold))) {
    return { direction, durationMs: getSettleDuration(width - Math.abs(offset), velocity), kind: 'commit' };
  }

  return { durationMs: getSettleDuration(Math.abs(offset), velocity), kind: 'cancel' };
};

const getSettleDuration = (distance: number, velocity: number): number =>
  Math.min(MAX_SETTLE_MS, Math.max(MIN_SETTLE_MS, distance / Math.max(Math.abs(velocity), MIN_SETTLE_VELOCITY)));

/** Ease-out quadratic: starts at twice the average speed, so a flick hands off without a visible stall. */
export const easeOutQuad = (progress: number): number => 1 - (1 - progress) * (1 - progress);
