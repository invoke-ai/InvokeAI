// SPDX-License-Identifier: Apache-2.0
/**
 * Fill color distance: OKLab plus alpha, max-norm, `d = max(|ΔL|, |Δa| / 0.4, |Δb| / 0.4, |Δα|)` on straight
 * (unpremultiplied) RGBA8, clamped to [0, 1] and quantized up to 0..255. Rounding up keeps 0 for identical pixels
 * only, so tolerance 0 selects exactly the seed color. A fully transparent pixel has no color: when either side is
 * transparent the distance is the alpha difference alone, so a fill of a transparent area stops at any paint.
 */

import { oklabFromSrgb8 } from '@workbench/canvas-engine/render/spatial/numerics';

/** OKLab a and b span about ±0.4 inside sRGB; dividing by it puts chroma on the same 0..1 scale as L and alpha. */
const CHROMA_SPAN = 0.4;
const LIGHTNESS_SCALE = 255;
const CHROMA_SCALE = 255 / CHROMA_SPAN;
/** Absorbs float noise in an exact integer alpha step before rounding up; far below any OKLab step between bytes. */
const ROUNDING_SLACK = 1e-7;

/** Direct-mapped memo of recent colors: flat and anti-aliased art repeats colors, so most pixels skip the cube roots. */
const MEMO_BITS = 12;
const MEMO_SIZE = 1 << MEMO_BITS;

/** Quantized distance (0..255) of the RGBA8 pixel at `offset` in `data` from the seed color. */
export type SeedColorDistance = (data: Uint8ClampedArray, offset: number) => number;

/**
 * Measures distances from the RGBA8 seed pixel at `seedOffset` in `seedData`. The returned function allocates
 * nothing per call; its memo is private to it, so one measurer serves one fill.
 */
export const createSeedColorDistance = (seedData: Uint8ClampedArray, seedOffset: number): SeedColorDistance => {
  const seedAlpha = seedData[seedOffset + 3];
  const lab = new Float64Array(3);
  oklabFromSrgb8(seedData[seedOffset], seedData[seedOffset + 1], seedData[seedOffset + 2], lab);
  const seedL = lab[0];
  const seedA = lab[1];
  const seedB = lab[2];
  const memoKeys = new Float64Array(MEMO_SIZE).fill(-1);
  const memoValues = new Uint8Array(MEMO_SIZE);

  return (data, offset) => {
    const red = data[offset];
    const green = data[offset + 1];
    const blue = data[offset + 2];
    const alpha = data[offset + 3];
    const alphaDelta = alpha > seedAlpha ? alpha - seedAlpha : seedAlpha - alpha;
    if (alpha === 0 || seedAlpha === 0) {
      return alphaDelta;
    }
    const key = (red | (green << 8) | (blue << 16) | (alpha << 24)) >>> 0;
    const slot = Math.imul(key, 0x9e3779b1) >>> (32 - MEMO_BITS);
    if (memoKeys[slot] === key) {
      return memoValues[slot];
    }
    oklabFromSrgb8(red, green, blue, lab);
    let scaled = alphaDelta;
    const lightness = Math.abs(lab[0] - seedL) * LIGHTNESS_SCALE;
    if (lightness > scaled) {
      scaled = lightness;
    }
    const a = Math.abs(lab[1] - seedA) * CHROMA_SCALE;
    if (a > scaled) {
      scaled = a;
    }
    const b = Math.abs(lab[2] - seedB) * CHROMA_SCALE;
    if (b > scaled) {
      scaled = b;
    }
    const distance = scaled >= 255 ? 255 : scaled <= ROUNDING_SLACK ? 0 : Math.ceil(scaled - ROUNDING_SLACK);
    memoKeys[slot] = key;
    memoValues[slot] = distance;
    return distance;
  };
};
