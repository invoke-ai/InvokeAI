/**
 * Pressure-opacity bands replace overlapping scratch pixels (destination-out then source-over), so newer pressure
 * wins without alpha compounding. Quantization to {@link PRESSURE_ALPHA_STEPS} limits outline/fill work; shared
 * {@link BAND_OVERLAP_SAMPLES} avoid seams. The completed scratch composites once at stroke opacity.
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
 * Quantizes pressure to (0,1]; nonfinite values use {@link MIN_PRESSURE_ALPHA}, avoiding canvas ignoring NaN and
 * painting fully opaque.
 */
export const toPressureAlpha = (pressure: number): number => {
  if (!Number.isFinite(pressure)) {
    return MIN_PRESSURE_ALPHA;
  }

  const clamped = Math.min(1, Math.max(0, pressure));
  const quantized = Math.round(clamped * PRESSURE_ALPHA_STEPS) / PRESSURE_ALPHA_STEPS;

  return Math.max(MIN_PRESSURE_ALPHA, quantized);
};

/** Groups samples by alpha in stroke order. Preserve a lone sample as a dot; empty input has no bands. */
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

    // Share the boundary sample when opening a band to avoid an outline seam.
    bands.push(current);
    const overlap = current.points.slice(-BAND_OVERLAP_SAMPLES);
    current = { alpha, points: [...overlap, point] };
  }

  bands.push(current);

  return bands;
};
