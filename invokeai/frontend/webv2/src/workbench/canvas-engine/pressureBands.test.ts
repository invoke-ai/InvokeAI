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
  it('collapses a constant-pressure stroke into one band', () => {
    const bands = getPressureBands([point(0, 1), point(1, 1), point(2, 1)]);

    expect(bands).toHaveLength(1);
    expect(bands[0]?.points).toHaveLength(3);
  });

  it('splits on a level change and shares a sample so outlines abut', () => {
    const bands = getPressureBands([point(0, 1), point(1, 1), point(2, 0.5)]);

    expect(bands).toHaveLength(2);
    expect(bands[0]).toMatchObject({ alpha: 1 });
    expect(bands[1]).toMatchObject({ alpha: 0.5 });
    expect(bands[1]?.points[0]).toBe(bands[0]?.points.at(-1));
    expect(bands[1]?.points).toHaveLength(BAND_OVERLAP_SAMPLES + 1);
  });
});
