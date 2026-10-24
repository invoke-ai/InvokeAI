import { oklabFromSrgb8, roundHalfEven, srgb8FromLinear } from '@workbench/canvas-engine/render/spatial/numerics';
import { describe, expect, it } from 'vitest';

import fixtures from './__fixtures__/shared_image_manipulation_numerics.json';

describe('roundHalfEven', () => {
  it.each([
    [2.5, 2],
    [3.5, 4],
    [-2.5, -2],
    [-3.5, -4],
    [0.5, 0],
    [2.4999999, 2],
    [2.5000001, 3],
  ])('rounds %d to %d as Python does', (value, expected) => {
    expect(roundHalfEven(value)).toBe(expected);
  });
});

describe('oklabFromSrgb8', () => {
  it('matches independently derived Oklab values for the sRGB primaries and white', () => {
    const { atol, expected_oklab, rgb } = fixtures.color_adapter;
    const out = new Float64Array(3);
    rgb.flat().forEach(([red, green, blue], pixel) => {
      oklabFromSrgb8(red, green, blue, out);
      expected_oklab.flat()[pixel].forEach((value, channel) => {
        expect(Math.abs(out[channel] - value)).toBeLessThanOrEqual(atol);
      });
    });
  });
});

describe('srgb8FromLinear', () => {
  const direct = (linear: number) => {
    const clamped = Math.min(1, Math.max(0, linear));
    const encoded = clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * clamped ** (1 / 2.4) - 0.055;
    return roundHalfEven(encoded * 255);
  };

  it('equals clamping, gamma encoding and half-even rounding, including next to every byte boundary', () => {
    const samples = [-1, -0, 0, 1e-300, 1, 1.5, Number.POSITIVE_INFINITY];
    for (let i = 0; i <= 200_000; i++) {
      samples.push(i / 200_000);
    }
    for (let byte = 1; byte < 256; byte++) {
      const encoded = (byte - 0.5) / 255;
      const boundary = encoded <= 0.04045 ? encoded / 12.92 : ((encoded + 0.055) / 1.055) ** 2.4;
      samples.push(boundary, boundary * (1 - 1e-15), boundary * (1 + 1e-15), boundary * (1 - 1e-9));
    }
    const mismatches = samples.filter((linear) => srgb8FromLinear(linear) !== direct(linear));
    expect(mismatches).toEqual([]);
  });
});
