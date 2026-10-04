import { describe, expect, it } from 'vitest';

import type { StrokeSamplePoint } from './freehand';

import {
  BAND_OVERLAP_SAMPLES,
  getPressureBands,
  MIN_PRESSURE_ALPHA,
  PRESSURE_ALPHA_STEPS,
  toPressureAlpha,
} from './pressureBands';

const point = (x: number, pressure: number): StrokeSamplePoint => ({ pressure, x, y: 0 });

describe('toPressureAlpha', () => {
  it('quantizes to the configured step count', () => {
    expect(toPressureAlpha(1)).toBe(1);
    expect(toPressureAlpha(0.5)).toBe(0.5);
    // 0.53 snaps to the nearest 1/16 (0.5) rather than staying exact.
    expect(toPressureAlpha(0.53)).toBe(0.5);
    expect(toPressureAlpha(0.5 + 1 / PRESSURE_ALPHA_STEPS)).toBeCloseTo(0.5625, 6);
  });

  it('never returns zero, so a zero-pressure sample still paints', () => {
    // Devices commonly report 0 for a stroke's first sample; a 0-alpha band would leave a gap.
    expect(toPressureAlpha(0)).toBe(MIN_PRESSURE_ALPHA);
    expect(toPressureAlpha(0.001)).toBe(MIN_PRESSURE_ALPHA);
  });

  it('clamps out-of-range pressure instead of producing alpha outside (0, 1]', () => {
    expect(toPressureAlpha(-1)).toBe(MIN_PRESSURE_ALPHA);
    expect(toPressureAlpha(5)).toBe(1);
    expect(toPressureAlpha(Number.NaN)).toBe(MIN_PRESSURE_ALPHA);
  });
});

describe('getPressureBands', () => {
  it('returns nothing for an empty stroke', () => {
    expect(getPressureBands([])).toEqual([]);
  });

  it('keeps a single sample as one band, so a tap still paints a dot', () => {
    const bands = getPressureBands([point(0, 1)]);

    expect(bands).toHaveLength(1);
    expect(bands[0]).toMatchObject({ alpha: 1 });
    expect(bands[0]?.points).toHaveLength(1);
  });

  it('collapses a constant-pressure stroke into one band', () => {
    const bands = getPressureBands([point(0, 1), point(1, 1), point(2, 1)]);

    expect(bands).toHaveLength(1);
    expect(bands[0]?.points).toHaveLength(3);
  });

  it('collapses samples that quantize to the same level, keeping band count low', () => {
    // 0.50 / 0.51 / 0.52 all snap to 0.5, so they must not each become a band.
    const bands = getPressureBands([point(0, 0.5), point(1, 0.51), point(2, 0.52)]);

    expect(bands).toHaveLength(1);
    expect(bands[0]?.alpha).toBe(0.5);
  });

  it('splits on a level change and shares a sample so outlines abut', () => {
    const bands = getPressureBands([point(0, 1), point(1, 1), point(2, 0.5)]);

    expect(bands).toHaveLength(2);
    expect(bands[0]).toMatchObject({ alpha: 1 });
    expect(bands[1]).toMatchObject({ alpha: 0.5 });
    // The second band repeats the first band's last sample; without it the two outlines
    // would leave a hairline gap at the level change.
    expect(bands[1]?.points[0]).toBe(bands[0]?.points.at(-1));
    expect(bands[1]?.points).toHaveLength(BAND_OVERLAP_SAMPLES + 1);
  });

  it('preserves stroke order across many level changes', () => {
    const pressures = [1, 0.75, 0.5, 0.25, 0.5, 1];
    const bands = getPressureBands(pressures.map((pressure, index) => point(index, pressure)));

    expect(bands.map((band) => band.alpha)).toEqual([1, 0.75, 0.5, 0.25, 0.5, 1]);
  });

  it('covers every sample across the returned bands', () => {
    const points = [point(0, 1), point(1, 0.5), point(2, 0.5), point(3, 0.25)];
    const bands = getPressureBands(points);
    const covered = new Set(bands.flatMap((band) => band.points));

    for (const sample of points) {
      expect(covered.has(sample)).toBe(true);
    }
  });

  it('does not mutate the input samples', () => {
    const points = [point(0, 1), point(1, 0.5)];
    const snapshot = structuredClone(points);

    getPressureBands(points);

    expect(points).toEqual(snapshot);
  });
});
