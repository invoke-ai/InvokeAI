// @ts-check
import { interpolate } from '../../i18n.mjs';

// Position maths for the Canvas tour's before/after slider (`CanvasCompare.astro`). Positions are percentages of the
// image width, from the left edge: the before image shows left of the divider, the after image right of it.

/** One arrow-key press, in percent. */
export const KEY_STEP = 5;
/** One Page Up/Down press, in percent. */
export const PAGE_STEP = 20;

/**
 * Clamps a position to 0–100, rounded to a tenth of a percent so drags do not write long fractions into styles.
 * @param {number} value
 */
export const clampPercent = (value) => {
  if (!Number.isFinite(value)) return 0;
  return Math.round(Math.min(100, Math.max(0, value)) * 10) / 10;
};

/**
 * The position under a pointer, from its client x and the image's client rect.
 * @param {number} clientX
 * @param {{ left: number, width: number }} rect
 */
export const percentFromPointer = (clientX, rect) =>
  rect.width > 0 ? clampPercent(((clientX - rect.left) / rect.width) * 100) : 50;

/**
 * The position after a key press, or `null` when the key does not move the slider. Right and Up move the divider
 * right, matching a horizontal slider's value growing to the right.
 * @param {string} key a `KeyboardEvent.key`
 * @param {number} value the current position
 * @returns {number | null}
 */
export const percentFromKey = (key, value) => {
  switch (key) {
    case 'ArrowRight':
    case 'ArrowUp':
      return clampPercent(value + KEY_STEP);
    case 'ArrowLeft':
    case 'ArrowDown':
      return clampPercent(value - KEY_STEP);
    case 'PageUp':
      return clampPercent(value + PAGE_STEP);
    case 'PageDown':
      return clampPercent(value - PAGE_STEP);
    case 'Home':
      return 0;
    case 'End':
      return 100;
    default:
      return null;
  }
};

/** How far a touch or pen press moves before it counts as a scrub or a scroll, in CSS pixels. */
export const GESTURE_THRESHOLD = 6;

/**
 * What a touch or pen press on the image has become after moving `dx`, `dy` from where it started: still `pending`
 * inside the threshold, a horizontal `scrub` of the slider, or a vertical `scroll` of the page, which leaves the
 * slider alone. A press released while pending is a tap.
 * @param {number} dx
 * @param {number} dy
 * @returns {'pending' | 'scrub' | 'scroll'}
 */
export const touchGesture = (dx, dy) => {
  const x = Math.abs(dx);
  const y = Math.abs(dy);
  if (Math.max(x, y) < GESTURE_THRESHOLD) return 'pending';
  return x > y ? 'scrub' : 'scroll';
};

/**
 * The slider's spoken value: how much of each image shows.
 * @param {number} value
 * @param {string} template the `canvasCompare.valueText` UI string, with `{{before}}` and `{{after}}` percentages
 */
export const valueText = (value, template) => {
  const before = Math.round(clampPercent(value));
  return interpolate(template, { before, after: 100 - before });
};

/**
 * The hint's position at time `t` (0–1): out from `from` to `to` and back, eased in and out, so it ends where it
 * started.
 * @param {number} t
 * @param {number} from
 * @param {number} to
 */
export const nudgeAt = (t, from, to) => {
  const clamped = Math.min(1, Math.max(0, t));
  // 0 → 1 → 0 over the run, with zero velocity at both ends and at the turn.
  const swing = (1 - Math.cos(clamped * 2 * Math.PI)) / 2;
  return clampPercent(from + (to - from) * swing);
};
