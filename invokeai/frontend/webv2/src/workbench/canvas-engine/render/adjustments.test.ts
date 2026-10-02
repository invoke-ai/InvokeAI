import type { CanvasAdjustmentCurves, CanvasAdjustmentEntry } from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import {
  adjustmentsKey,
  applyAdjustments,
  buildCurveLut,
  compileAdjustments,
  isIdentityAdjustmentEntry,
  isIdentityAdjustments,
} from './adjustments';

/** Builds an ImageData-like object (node has no DOM ImageData; the shape is enough). */
const imageData = (pixels: number[]): ImageData =>
  ({ data: new Uint8ClampedArray(pixels), height: 1, width: pixels.length / 4 }) as ImageData;

type EntryOverrides = Partial<Pick<CanvasAdjustmentEntry, 'id' | 'isEnabled'>>;
const bc = (brightness: number, contrast: number, overrides: EntryOverrides = {}) =>
  ({ brightness, contrast, id: 'bc', isEnabled: true, type: 'brightness-contrast', ...overrides }) as const;
const hsl = (saturation: number, overrides: EntryOverrides = {}) =>
  ({ id: 'hsl', isEnabled: true, saturation, type: 'hsl', ...overrides }) as const;
const curves = (points: CanvasAdjustmentCurves) =>
  ({ curves: points, id: 'cv', isEnabled: true, type: 'curves' }) as const;
const levels = (
  inBlack: number,
  inWhite: number,
  gamma: number,
  outBlack: number,
  outWhite: number,
  overrides: EntryOverrides = {}
) =>
  ({ gamma, id: 'lv', inBlack, inWhite, isEnabled: true, outBlack, outWhite, type: 'levels', ...overrides }) as const;
const exposure = (stops: number, overrides: EntryOverrides = {}) =>
  ({ id: 'ex', isEnabled: true, stops, type: 'exposure', ...overrides }) as const;
const channelLevels = (channel: 'rgb' | 'r' | 'g' | 'b', outWhite: number) =>
  ({
    channel,
    gamma: 1,
    id: 'clv',
    inBlack: 0,
    inWhite: 255,
    isEnabled: true,
    outBlack: 0,
    outWhite,
    type: 'levels',
  }) as const;
const hue = (rotation: number, overrides: EntryOverrides = {}) =>
  ({ id: 'hue', isEnabled: true, rotation, type: 'hue', ...overrides }) as const;
const invert = (overrides: EntryOverrides = {}) =>
  ({ id: 'inv', isEnabled: true, type: 'invert', ...overrides }) as const;

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

describe('identity and keys', () => {
  it('treats zero values, diagonal/absent curves, disabled entries and empty stacks as identity', () => {
    expect(isIdentityAdjustments(undefined)).toBe(true);
    expect(isIdentityAdjustments([])).toBe(true);
    expect(isIdentityAdjustments([bc(0, 0), hsl(0), curves({})])).toBe(true);
    expect(
      isIdentityAdjustments([
        curves({
          r: [
            [0, 0],
            [255, 255],
          ],
        }),
      ])
    ).toBe(true);
    expect(isIdentityAdjustments([bc(0.5, 0, { isEnabled: false })])).toBe(true);
    expect(isIdentityAdjustments([bc(0.5, 0)])).toBe(false);
    expect(
      isIdentityAdjustmentEntry(
        curves({
          r: [
            [0, 0],
            [128, 200],
            [255, 255],
          ],
        })
      )
    ).toBe(false);
  });

  it('recognizes identity levels and hue; invert always contributes', () => {
    expect(isIdentityAdjustments([levels(0, 255, 1, 0, 255), hue(0), hue(360)])).toBe(true);
    expect(isIdentityAdjustmentEntry(levels(1, 255, 1, 0, 255))).toBe(false);
    expect(isIdentityAdjustmentEntry(levels(0, 255, 1.1, 0, 255))).toBe(false);
    expect(isIdentityAdjustmentEntry(hue(90))).toBe(false);
    expect(isIdentityAdjustmentEntry(invert())).toBe(false);
    expect(isIdentityAdjustments([invert({ isEnabled: false })])).toBe(true);
  });

  it('keys identify the pixel effect: stable across entry ids, sensitive to order and values', () => {
    expect(adjustmentsKey([bc(0, 0), hsl(0)])).toBe('identity');
    expect(adjustmentsKey([bc(0.2, 0)])).toBe(adjustmentsKey([bc(0.2, 0, { id: 'other' })]));
    expect(adjustmentsKey([bc(0.2, 0)])).not.toBe(adjustmentsKey([bc(0.3, 0)]));
    expect(adjustmentsKey([bc(0.2, 0), hsl(0.5)])).not.toBe(adjustmentsKey([hsl(0.5), bc(0.2, 0)]));
    // A disabled entry keys exactly like an absent one.
    expect(adjustmentsKey([bc(0.2, 0), hsl(0.5, { isEnabled: false })])).toBe(adjustmentsKey([bc(0.2, 0)]));
  });
});

describe('compileAdjustments', () => {
  it('folds adjacent per-channel entries into one LUT segment and keeps saturation apart', () => {
    const folded = compileAdjustments([
      curves({
        r: [
          [0, 0],
          [128, 200],
          [255, 255],
        ],
      }),
      bc(0.1, 0.2),
    ]);
    expect(folded.map((segment) => segment.kind)).toEqual(['lut']);

    const split = compileAdjustments([bc(0.1, 0), hsl(0.5), bc(0, 0.3)]);
    expect(split.map((segment) => segment.kind)).toEqual(['lut', 'saturation', 'lut']);
  });

  it('folds levels and invert into the LUT chain and keeps hue as a matrix segment', () => {
    const folded = compileAdjustments([levels(10, 240, 1.2, 0, 255), invert(), bc(0.1, 0)]);
    expect(folded.map((segment) => segment.kind)).toEqual(['lut']);
    const withHue = compileAdjustments([invert(), hue(90), levels(0, 200, 1, 0, 255)]);
    expect(withHue.map((segment) => segment.kind)).toEqual(['lut', 'matrix', 'lut']);
  });

  it('a folded pair produces the same pixels as sequential application', () => {
    const curve = curves({
      r: [
        [0, 0],
        [64, 160],
        [255, 255],
      ],
    });
    const sequential = imageData([10, 20, 30, 255, 200, 100, 50, 255]);
    applyAdjustments(sequential, [curve]);
    applyAdjustments(sequential, [bc(0.1, 0.4)]);
    const folded = imageData([10, 20, 30, 255, 200, 100, 50, 255]);
    applyAdjustments(folded, [curve, bc(0.1, 0.4)]);
    expect(Array.from(folded.data)).toEqual(Array.from(sequential.data));
  });
});

describe('exposure and channel-scoped levels', () => {
  it('treats zero stops as identity and any channel scope with default values as identity', () => {
    expect(isIdentityAdjustmentEntry(exposure(0))).toBe(true);
    expect(isIdentityAdjustmentEntry(exposure(0.5))).toBe(false);
    expect(isIdentityAdjustmentEntry(channelLevels('r', 255))).toBe(true);
  });

  it('applies exposure in linear light: +1 stop doubles linear energy', () => {
    // srgb 64 → linear ≈0.0513, doubled ≈0.1026 → srgb ≈90; srgb 188 → linear ≈0.503, doubled clamps → 255.
    const img = imageData([64, 188, 0, 255]);
    applyAdjustments(img, [exposure(1)]);
    expect(Math.abs(img.data[0]! - 90)).toBeLessThanOrEqual(1);
    expect(img.data[1]).toBe(255);
    expect(img.data[2]).toBe(0);
  });

  it('darkening by a stop then brightening by a stop returns close to the original midtones', () => {
    const img = imageData([128, 96, 160, 255]);
    applyAdjustments(img, [exposure(-1)]);
    applyAdjustments(img, [exposure(1)]);
    expect(Math.abs(img.data[0]! - 128)).toBeLessThanOrEqual(2);
    expect(Math.abs(img.data[1]! - 96)).toBeLessThanOrEqual(2);
    expect(Math.abs(img.data[2]! - 160)).toBeLessThanOrEqual(2);
  });

  it('a channel-scoped levels entry remaps only its channel', () => {
    const img = imageData([200, 200, 200, 255]);
    applyAdjustments(img, [channelLevels('g', 128)]);
    expect(img.data[0]).toBe(200);
    expect(img.data[1]).toBe(Math.round((200 / 255) * 128));
    expect(img.data[2]).toBe(200);
  });

  it('folds exposure and channel levels with neighbours into one LUT segment, bit-exact vs sequential', () => {
    const folded = compileAdjustments([exposure(0.5), channelLevels('b', 200), invert()]);
    expect(folded.map((segment) => segment.kind)).toEqual(['lut']);

    const sequential = imageData([10, 20, 30, 255, 200, 100, 50, 255]);
    applyAdjustments(sequential, [exposure(0.5)]);
    applyAdjustments(sequential, [channelLevels('b', 200)]);
    applyAdjustments(sequential, [invert()]);
    const combined = imageData([10, 20, 30, 255, 200, 100, 50, 255]);
    applyAdjustments(combined, [exposure(0.5), channelLevels('b', 200), invert()]);
    expect([...combined.data]).toEqual([...sequential.data]);
  });

  it('keys distinguish stops and channel scope but keep unscoped levels keys unchanged', () => {
    expect(adjustmentsKey([exposure(1)])).not.toBe(adjustmentsKey([exposure(2)]));
    expect(adjustmentsKey([levels(10, 240, 1, 0, 255)])).toBe(
      adjustmentsKey([{ ...levels(10, 240, 1, 0, 255), channel: 'rgb' }])
    );
    expect(adjustmentsKey([channelLevels('r', 128)])).not.toBe(adjustmentsKey([channelLevels('g', 128)]));
  });
});

describe('applyAdjustments', () => {
  it('is a no-op for an identity stack and never modifies alpha', () => {
    const img = imageData([10, 20, 30, 128]);
    applyAdjustments(img, [bc(0, 0), hsl(0)]);
    expect(Array.from(img.data)).toEqual([10, 20, 30, 128]);
    applyAdjustments(img, [bc(0.5, 0)]);
    expect(img.data[3]).toBe(128);
  });

  it('brightens rgb additively and clamps', () => {
    const img = imageData([10, 20, 30, 255]);
    applyAdjustments(img, [bc(0.5, 0)]);
    expect(img.data[0]).toBe(138); // 10 + 128
    expect(img.data[1]).toBe(148);
    expect(img.data[2]).toBe(158);
    const clamped = imageData([200, 0, 0, 255]);
    applyAdjustments(clamped, [bc(0.5, 0)]);
    expect(clamped.data[0]).toBe(255);
  });

  it('applies contrast about mid-grey', () => {
    const img = imageData([64, 128, 192, 255]);
    applyAdjustments(img, [bc(0, 1)]);
    expect(img.data[0]).toBe(0); // (64-128)*2+128
    expect(img.data[1]).toBe(128);
    expect(img.data[2]).toBe(255);
  });

  it('fully desaturates to luma at saturation -1', () => {
    const img = imageData([200, 100, 50, 255]);
    applyAdjustments(img, [hsl(-1)]);
    const luma = Math.round(0.299 * 200 + 0.587 * 100 + 0.114 * 50);
    expect(img.data[0]).toBe(luma);
    expect(img.data[1]).toBe(luma);
    expect(img.data[2]).toBe(luma);
  });

  it('inverts every channel exactly', () => {
    const img = imageData([0, 128, 255, 200]);
    applyAdjustments(img, [invert()]);
    expect(Array.from(img.data)).toEqual([255, 127, 0, 200]);
  });

  it('remaps through levels: input range, gamma midtones, output range', () => {
    // Input 64..192 stretched to full range: 64 → 0, 192 → 255, midpoint → 128.
    const stretch = imageData([64, 128, 192, 255]);
    applyAdjustments(stretch, [levels(64, 192, 1, 0, 255)]);
    expect(Array.from(stretch.data.slice(0, 3))).toEqual([0, 128, 255]);
    // Gamma 2 lifts midtones: 128 → 255 * (0.5)^(1/2) ≈ 180.
    const midtones = imageData([128, 0, 255, 255]);
    applyAdjustments(midtones, [levels(0, 255, 2, 0, 255)]);
    expect(midtones.data[0]).toBe(Math.round(255 * Math.sqrt(128 / 255)));
    expect(midtones.data[1]).toBe(0);
    expect(midtones.data[2]).toBe(255);
    // Output compression maps black to 50 and white to 200.
    const compress = imageData([0, 255, 128, 255]);
    applyAdjustments(compress, [levels(0, 255, 1, 50, 200)]);
    expect(compress.data[0]).toBe(50);
    expect(compress.data[1]).toBe(200);
  });

  it('hue rotation cycles the primaries and preserves grays', () => {
    // 120° sends red toward green (the standard hueRotate matrix dims it; green must dominate).
    const red = imageData([255, 0, 0, 255]);
    applyAdjustments(red, [hue(120)]);
    expect(red.data[0]).toBe(0);
    expect(red.data[2]).toBe(0);
    expect(red.data[1]).toBeGreaterThan(90);
    // A full ±360° is identity; gray is on the rotation axis and never moves.
    const gray = imageData([128, 128, 128, 255]);
    applyAdjustments(gray, [hue(90)]);
    expect(Array.from(gray.data)).toEqual([128, 128, 128, 255]);
    const wrapped = imageData([200, 100, 50, 255]);
    applyAdjustments(wrapped, [hue(360)]);
    expect(Array.from(wrapped.data)).toEqual([200, 100, 50, 255]);
  });

  it('entry order changes the pixels; a disabled entry renders exactly as a removed one', () => {
    const source = [180, 60, 40, 255];
    const bcFirst = imageData([...source]);
    applyAdjustments(bcFirst, [bc(0.3, 0.5), hsl(-0.8)]);
    const hslFirst = imageData([...source]);
    applyAdjustments(hslFirst, [hsl(-0.8), bc(0.3, 0.5)]);
    expect(Array.from(bcFirst.data)).not.toEqual(Array.from(hslFirst.data));

    const withDisabled = imageData([...source]);
    applyAdjustments(withDisabled, [bc(0.3, 0.5), hsl(-0.8, { isEnabled: false })]);
    const without = imageData([...source]);
    applyAdjustments(without, [bc(0.3, 0.5)]);
    expect(Array.from(withDisabled.data)).toEqual(Array.from(without.data));
  });
});
