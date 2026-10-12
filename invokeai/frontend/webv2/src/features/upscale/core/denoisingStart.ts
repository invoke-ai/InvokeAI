/**
 * Where on its schedule the second pass starts, from the Creativity slider.
 *
 * Creativity is defined by where FLUX enters: `flux_denoise` clips the schedule by *value*, so a
 * start of 0.499 enters at sigma 0.501 whatever the shift or step count. Z-Image and Anima clip by
 * *index* instead (`start_idx = int(denoising_start * steps)` on a shifted schedule), so the same
 * number enters far noisier there -- 0.499 starts Z-Image at sigma 0.78. Handing them the step whose
 * sigma is nearest FLUX's keeps one slider meaning one thing. This module goes away when those
 * nodes clip by value.
 */

/** The legacy formula: creativity -10..10 to a value-clipped start of 0.998..0. */
export const getUpscaleDenoisingStart = (creativity: number): number => ((creativity * -1 + 10) * 4.99) / 100;

/** `z_image_denoise._calculate_shift`, held at the fit's 4096-token end since the clamp (#378). */
const Z_IMAGE_SHIFT_MIN_SEQ_LEN = 256;
const Z_IMAGE_SHIFT_MAX_SEQ_LEN = 4096;
const Z_IMAGE_BASE_SHIFT = 0.5;
const Z_IMAGE_MAX_SHIFT = 1.15;

/** Z-Image's linear shift for a frame: one token per 16x16 pixels, as `z_image_denoise` patches it. */
export const zImageScheduleShift = ({ width, height }: { width: number; height: number }): number => {
  const seqLen = Math.min(Math.floor(width / 16) * Math.floor(height / 16), Z_IMAGE_SHIFT_MAX_SEQ_LEN);
  const slope = (Z_IMAGE_MAX_SHIFT - Z_IMAGE_BASE_SHIFT) / (Z_IMAGE_SHIFT_MAX_SEQ_LEN - Z_IMAGE_SHIFT_MIN_SEQ_LEN);

  return Math.exp(seqLen * slope + Z_IMAGE_BASE_SHIFT - slope * Z_IMAGE_SHIFT_MIN_SEQ_LEN);
};

/** `ANIMA_SHIFT`: Anima's schedule is shifted by a constant, not by resolution. */
export const ANIMA_SCHEDULE_SHIFT = 3;

/**
 * The start an index-clipping node needs to enter where a value-clipping one would.
 *
 * Both nodes build `sigma = shift * t / (1 + (shift - 1) * t)` at `t = 1 - i / steps` and begin at
 * `i = int(start * steps)`. This picks the `i` whose sigma is nearest `1 - valueStart` and aims at
 * the middle of that step, so float error in the node's product cannot floor it one step short.
 */
export const getIndexClippedDenoisingStart = (valueStart: number, steps: number, shift: number): number => {
  const target = 1 - valueStart;
  let entryIndex = 0;
  let entryDistance = Number.POSITIVE_INFINITY;

  // The last index would leave nothing to denoise, so it is never a candidate.
  for (let index = 0; index < steps; index += 1) {
    const t = 1 - index / steps;
    const distance = Math.abs((shift * t) / (1 + (shift - 1) * t) - target);

    if (distance < entryDistance) {
      entryIndex = index;
      entryDistance = distance;
    }
  }

  return entryIndex === 0 ? 0 : (entryIndex + 0.5) / steps;
};
