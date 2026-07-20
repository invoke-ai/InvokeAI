import type { CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import {
  adjustmentsKey,
  applyAdjustments,
  buildAdjustmentLuts,
  buildCurveLut,
  DEFAULT_ADJUSTMENTS,
  isIdentityAdjustments,
} from './adjustments';

/** Builds an ImageData-like object (node has no DOM ImageData; the shape is enough). */
const imageData = (pixels: number[]): ImageData =>
  ({ data: new Uint8ClampedArray(pixels), height: 1, width: pixels.length / 4 }) as ImageData;

describe('buildCurveLut', () => {
  it('is the identity for absent / empty / diagonal curves', () => {
    for (const pts of [
      undefined,
      [],
      [
        [0, 0],
        [255, 255],
      ],
    ] as const) {
      const lut = buildCurveLut(pts as never);
      expect(lut[0]).toBe(0);
      expect(lut[128]).toBe(128);
      expect(lut[255]).toBe(255);
    }
  });

  it('clamps values outside the first/last control point to the endpoints', () => {
    // A curve that starts at x=64 (y=0) and ends at x=192 (y=255): below 64 → 0, above 192 → 255.
    const lut = buildCurveLut([
      [64, 0],
      [192, 255],
    ]);
    expect(lut[0]).toBe(0);
    expect(lut[64]).toBe(0);
    expect(lut[192]).toBe(255);
    expect(lut[255]).toBe(255);
    // Midpoint between the two knots is roughly mid grey.
    expect(lut[128]).toBeGreaterThan(100);
    expect(lut[128]).toBeLessThan(160);
  });

  it('interpolates monotonically through interior points without overshoot', () => {
    const lut = buildCurveLut([
      [0, 0],
      [128, 200],
      [255, 255],
    ]);
    // The 128 knot is honoured.
    expect(lut[128]).toBe(200);
    // Monotonic non-decreasing, always within [0, 255].
    for (let i = 1; i < 256; i++) {
      expect(lut[i]).toBeGreaterThanOrEqual(lut[i - 1]);
      expect(lut[i]).toBeLessThanOrEqual(255);
      expect(lut[i]).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('buildAdjustmentLuts', () => {
  it('is the identity for default adjustments', () => {
    const { b, g, r } = buildAdjustmentLuts(DEFAULT_ADJUSTMENTS);
    for (let i = 0; i < 256; i++) {
      expect(r[i]).toBe(i);
      expect(g[i]).toBe(i);
      expect(b[i]).toBe(i);
    }
  });

  it('applies additive brightness and clamps', () => {
    const { r } = buildAdjustmentLuts({ brightness: 0.5, contrast: 0, saturation: 0 });
    // +0.5*255 ≈ +128 (rounded).
    expect(r[0]).toBe(128);
    expect(r[200]).toBe(255); // clamped
  });

  it('applies contrast about mid-grey', () => {
    const { r } = buildAdjustmentLuts({ brightness: 0, contrast: 1, saturation: 0 });
    // factor 2 about 128: 128 stays, 0 → -128 clamp 0, 255 → 382 clamp 255.
    expect(r[128]).toBe(128);
    expect(r[0]).toBe(0);
    expect(r[255]).toBe(255);
    expect(r[64]).toBe(0); // (64-128)*2+128 = 0
    expect(r[192]).toBe(255); // (192-128)*2+128 = 256 → clamp
  });

  it('composes curve → brightness → contrast', () => {
    // A curve mapping everything to 100, then +0 brightness, contrast 0 → all 100.
    const { r } = buildAdjustmentLuts({
      brightness: 0,
      contrast: 0,
      curves: {
        b: [
          [0, 0],
          [255, 255],
        ],
        g: [
          [0, 0],
          [255, 255],
        ],
        r: [
          [0, 100],
          [255, 100],
        ],
      },
      saturation: 0,
    });
    expect(r[0]).toBe(100);
    expect(r[255]).toBe(100);
  });
});

describe('isIdentityAdjustments / adjustmentsKey', () => {
  it('treats zeros + diagonal / absent curves as identity', () => {
    expect(isIdentityAdjustments(undefined)).toBe(true);
    expect(isIdentityAdjustments(DEFAULT_ADJUSTMENTS)).toBe(true);
    expect(
      isIdentityAdjustments({
        brightness: 0,
        contrast: 0,
        saturation: 0,
        curves: {
          b: [
            [0, 0],
            [255, 255],
          ],
          g: [
            [0, 0],
            [255, 255],
          ],
          r: [
            [0, 0],
            [255, 255],
          ],
        },
      })
    ).toBe(true);
    expect(adjustmentsKey(DEFAULT_ADJUSTMENTS)).toBe('identity');
  });

  it('is non-identity for any non-zero param or bent curve', () => {
    expect(isIdentityAdjustments({ brightness: 0.1, contrast: 0, saturation: 0 })).toBe(false);
    expect(
      isIdentityAdjustments({
        brightness: 0,
        contrast: 0,
        saturation: 0,
        curves: {
          b: [
            [0, 0],
            [255, 255],
          ],
          g: [
            [0, 0],
            [255, 255],
          ],
          r: [
            [0, 0],
            [128, 200],
            [255, 255],
          ],
        },
      })
    ).toBe(false);
  });

  it('produces distinct, stable keys per distinct adjustment', () => {
    const a: CanvasAdjustmentsContract = { brightness: 0.2, contrast: 0, saturation: 0 };
    const b: CanvasAdjustmentsContract = { brightness: 0.3, contrast: 0, saturation: 0 };
    expect(adjustmentsKey(a)).toBe(adjustmentsKey({ ...a }));
    expect(adjustmentsKey(a)).not.toBe(adjustmentsKey(b));
  });
});

describe('applyAdjustments', () => {
  it('is a no-op for identity adjustments', () => {
    const img = imageData([10, 20, 30, 255]);
    applyAdjustments(img, DEFAULT_ADJUSTMENTS);
    expect(Array.from(img.data)).toEqual([10, 20, 30, 255]);
  });

  it('never modifies alpha', () => {
    const img = imageData([10, 20, 30, 128]);
    applyAdjustments(img, { brightness: 0.5, contrast: 0, saturation: 0 });
    expect(img.data[3]).toBe(128);
  });

  it('brightens rgb', () => {
    const img = imageData([10, 20, 30, 255]);
    applyAdjustments(img, { brightness: 0.5, contrast: 0, saturation: 0 });
    expect(img.data[0]).toBe(138); // 10 + 128
    expect(img.data[1]).toBe(148);
    expect(img.data[2]).toBe(158);
  });

  it('fully desaturates to luma at saturation -1', () => {
    const img = imageData([200, 100, 50, 255]);
    applyAdjustments(img, { brightness: 0, contrast: 0, saturation: -1 });
    const luma = Math.round(0.299 * 200 + 0.587 * 100 + 0.114 * 50);
    expect(img.data[0]).toBe(luma);
    expect(img.data[1]).toBe(luma);
    expect(img.data[2]).toBe(luma);
  });
});
