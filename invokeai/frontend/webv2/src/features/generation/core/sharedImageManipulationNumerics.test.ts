import { describe, expect, it } from 'vitest';

import fixtures from './__fixtures__/shared_image_manipulation_numerics.json';
import { oklabFromSrgb } from './colorConversion';
import { guidedFilter, type ScalarImage } from './guidedFilter';
import { edgeAwareBand } from './multiscale';

const scalarImage = (rows: number[][]): ScalarImage => ({
  width: rows[0].length,
  height: rows.length,
  data: new Float64Array(rows.flat()),
});

const rgbImage = (rows: number[][][]): { width: number; height: number; data: Uint8Array } => ({
  width: rows[0].length,
  height: rows.length,
  data: new Uint8Array(rows.flat(2)),
});

/** Compare compact buffers against independently derived float64 references, with no relative tolerance. */
const expectImageClose = (actual: ScalarImage, expected: number[][], atol: number) => {
  expect(actual.width).toBe(expected[0].length);
  expect(actual.height).toBe(expected.length);
  expect(actual.data).toHaveLength(expected.length * expected[0].length);
  expected.flat().forEach((value, i) => expect(Math.abs(actual.data[i] - value)).toBeLessThanOrEqual(atol));
};

describe('shared image manipulation numerical fixtures', () => {
  it('matches the independently derived guided-filter fixture and treats radius zero as identity', () => {
    const { guide, signal, radius, epsilon, expected, atol } = fixtures.guided_filter;
    expectImageClose(guidedFilter(scalarImage(guide), scalarImage(signal), radius, epsilon), expected, atol);

    const singleton = scalarImage([[0.375]]);
    const result = guidedFilter(scalarImage([[1]]), singleton, 0, epsilon);
    expect(Array.from(result.data)).toEqual([0.375]);
    expect(result.data).not.toBe(singleton.data);
  });

  it('keeps a constant guide numerically constant for near-zero regularization', () => {
    const guide = scalarImage(Array.from({ length: 3 }, () => Array(3).fill(0.1)));
    const signal = scalarImage([
      [0, 1, 0],
      [1, 0, 1],
      [0, 1, 0],
    ]);
    const expected = [
      [40 / 81, 41 / 81, 40 / 81],
      [41 / 81, 40 / 81, 41 / 81],
      [40 / 81, 41 / 81, 40 / 81],
    ];
    const actual = guidedFilter(guide, signal, 1, 1e-18);

    expect(Array.from(actual.data).every(Number.isFinite)).toBe(true);
    expectImageClose(actual, expected, 1e-10);
  });

  it('matches the edge-aware band fixture where the fine scale has radius zero', () => {
    const { lightness, inner_radius, outer_radius, epsilon, expected, atol } = fixtures.multiscale_band;
    expectImageClose(edgeAwareBand(scalarImage(lightness), inner_radius, outer_radius, epsilon), expected, atol);

    expect(() => edgeAwareBand(scalarImage(lightness), 1, 1, epsilon)).toThrow();
  });

  it('matches Oklab values for sRGB primaries and white, including black and channel endpoints', () => {
    const { rgb, expected_oklab, atol } = fixtures.color_adapter;
    const image = oklabFromSrgb(rgbImage(rgb));
    expect(image).toMatchObject({ width: expected_oklab[0].length, height: expected_oklab.length });
    expect(image.data).toHaveLength(expected_oklab.length * expected_oklab[0].length * 3);
    expected_oklab.flat(2).forEach((value, i) => expect(Math.abs(image.data[i] - value)).toBeLessThanOrEqual(atol));

    expect(Array.from(oklabFromSrgb({ width: 1, height: 1, data: new Uint8Array([0, 0, 0]) }).data)).toEqual([0, 0, 0]);
  });
});

describe('compact image buffer contract', () => {
  it('uses row-major scalar buffers and preserves radius-zero identity values', () => {
    const input = { width: 2, height: 2, data: new Float64Array([0.1, 0.2, 0.3, 0.4]) };
    const result = guidedFilter(input, input, 0, 0.1);

    expect(result).toMatchObject({ width: 2, height: 2 });
    expect(result.data).toBeInstanceOf(Float64Array);
    expect(Array.from(result.data)).toEqual(Array.from(input.data));
    expect(result.data).not.toBe(input.data);
  });

  it('stores RGB channels interleaved in row-major order and returns compact Oklab data', () => {
    const rgb = { width: 2, height: 1, data: new Uint8Array([255, 0, 0, 0, 255, 0]) };
    const result = oklabFromSrgb(rgb);

    expect(result).toMatchObject({ width: 2, height: 1 });
    expect(result.data).toBeInstanceOf(Float64Array);
    expect(result.data).toHaveLength(6);
    expect(Math.abs(result.data[0] - 0.62795536)).toBeLessThanOrEqual(0.00002);
    expect(Math.abs(result.data[3] - 0.86643961)).toBeLessThanOrEqual(0.00002);
  });

  it('returns compact bands and rejects malformed buffer dimensions', () => {
    const lightness = { width: 2, height: 2, data: new Float64Array([0.1, 0.2, 0.3, 0.4]) };
    const result = edgeAwareBand(lightness, 0, 1, 0.1);
    expect(result.data).toBeInstanceOf(Float64Array);
    expect(result.data).toHaveLength(4);
    expect(() => guidedFilter({ ...lightness, data: new Float64Array(3) }, lightness, 1, 0.1)).toThrow();
    expect(() => oklabFromSrgb({ width: 2, height: 1, data: new Uint8Array(5) })).toThrow();
  });
});
