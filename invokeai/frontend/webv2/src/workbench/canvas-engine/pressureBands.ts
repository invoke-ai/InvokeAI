/**
 * Splits a pressure-sampled stroke into contiguous constant-alpha bands.
 *
 * ## Why bands, and why they must not overlap-blend
 *
 * `strokeSession` fills the whole stroke into a scratch surface at full alpha and composites
 * it into the layer exactly once at the stroke opacity. That single composite is what stops
 * overlapping segments *within one stroke* from darkening each other — a slow drag revisits
 * the same pixels dozens of times, and `source-over` at partial alpha would compound every
 * one of them into a dark blob.
 *
 * Pressure-dependent opacity breaks that model, because alpha now varies along the stroke and
 * cannot be one `globalAlpha` on one composite. The fix is to fill the stroke as a sequence of
 * bands, each a contiguous run of samples at one quantized alpha, and to paint each band into
 * the scratch as *replace* (`destination-out` the band's own path, then `source-over` it at the
 * band's alpha) rather than as a blend. Replace keeps the no-compounding guarantee: where two
 * bands overlap, the later one wins outright, which is also the behaviour a stroke wants — the
 * newer sample's pressure is the current one.
 *
 * Bands are quantized rather than per-sample because each band is a separate outline
 * computation and two extra canvas fills. Quantizing to {@link PRESSURE_ALPHA_STEPS} levels
 * keeps a normal drag to a handful of bands while staying below the ~2% alpha difference an
 * eye can pick out of a soft edge.
 *
 * Consecutive bands share a sample ({@link BAND_OVERLAP_SAMPLES}) so their outlines abut
 * instead of leaving a hairline gap where the pressure level changes.
 *
 * Zero React, zero import-time side effects.
 */

import type { StrokeSamplePoint } from '@workbench/canvas-engine/freehand';

/** Quantization levels for band alpha. */
export const PRESSURE_ALPHA_STEPS = 16;

/**
 * Samples a band repeats from the previous one. One is enough: `perfect-freehand` derives a
 * segment's outline from its endpoints, so sharing an endpoint makes the outlines meet.
 */
export const BAND_OVERLAP_SAMPLES = 1;

/**
 * Floor on band alpha. A zero-pressure sample would otherwise contribute an invisible band,
 * and some devices report 0 for the first sample of a stroke before pressure ramps up — which
 * would leave a gap at the stroke's start.
 */
export const MIN_PRESSURE_ALPHA = 1 / PRESSURE_ALPHA_STEPS;

/** A contiguous run of samples painted at one alpha. */
export interface PressureBand {
  /** Samples for this band's outline, including the shared sample from the previous band. */
  points: StrokeSamplePoint[];
  /** Quantized alpha in (0, 1]. */
  alpha: number;
}

/**
 * Quantizes pressure to a band alpha in (0, 1].
 *
 * Non-finite pressure floors to {@link MIN_PRESSURE_ALPHA} rather than propagating: clamping
 * alone would let NaN through to `globalAlpha`, where a non-finite value is ignored and the
 * band would silently paint at full alpha — the opposite of the lightest possible touch.
 */
export const toPressureAlpha = (pressure: number): number => {
  if (!Number.isFinite(pressure)) {
    return MIN_PRESSURE_ALPHA;
  }

  const clamped = Math.min(1, Math.max(0, pressure));
  const quantized = Math.round(clamped * PRESSURE_ALPHA_STEPS) / PRESSURE_ALPHA_STEPS;

  return Math.max(MIN_PRESSURE_ALPHA, quantized);
};

/**
 * Groups samples into constant-alpha bands, in stroke order.
 *
 * A single sample yields a single band — `perfect-freehand` renders a lone point as a dot, and
 * dropping it would make a tap invisible. An empty input yields no bands.
 */
export const getPressureBands = (points: readonly StrokeSamplePoint[]): PressureBand[] => {
  if (points.length === 0) {
    return [];
  }

  const bands: PressureBand[] = [];
  let current: PressureBand = { alpha: toPressureAlpha(points[0]!.pressure), points: [points[0]!] };

  for (let index = 1; index < points.length; index += 1) {
    const point = points[index]!;
    const alpha = toPressureAlpha(point.pressure);

    if (alpha === current.alpha) {
      current.points.push(point);
      continue;
    }

    // The level changed. Close the current band, then open the next one seeded with the
    // shared sample so the two outlines abut rather than leaving a seam.
    bands.push(current);
    const overlap = current.points.slice(-BAND_OVERLAP_SAMPLES);
    current = { alpha, points: [...overlap, point] };
  }

  bands.push(current);

  return bands;
};
