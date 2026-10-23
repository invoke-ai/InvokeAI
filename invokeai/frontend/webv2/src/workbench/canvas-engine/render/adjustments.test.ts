import type { CanvasAdjustmentCurves, CanvasAdjustmentEntry } from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import {
  adjustmentsKey,
  applyAdjustments,
  buildCurveLut,
  compileAdjustments,
  isIdentityAdjustmentEntry,
  isIdentityAdjustments,
  isPositionDependentAdjustments,
  NOISE_FULL_SCALE_SIGMA,
  noiseGrainSpan,
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
type NoiseEntry = Extract<CanvasAdjustmentEntry, { type: 'noise' }>;
const noise = (amount: number, overrides: Partial<NoiseEntry> = {}): NoiseEntry => ({
  amount,
  distribution: 'gaussian',
  grain: 'monochrome',
  id: 'nz',
  isEnabled: true,
  response: 'flat',
  seed: 0x1234abcd,
  size: 1,
  type: 'noise',
  ...overrides,
});
const GRAINS = ['monochrome', 'color', 'chroma'] as const;
const RESPONSES = ['flat', 'midtones'] as const;

const ORIGIN = { x: 0, y: 0 };

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
    applyAdjustments(sequential, [curve], ORIGIN);
    applyAdjustments(sequential, [bc(0.1, 0.4)], ORIGIN);
    const folded = imageData([10, 20, 30, 255, 200, 100, 50, 255]);
    applyAdjustments(folded, [curve, bc(0.1, 0.4)], ORIGIN);
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
    applyAdjustments(img, [exposure(1)], ORIGIN);
    expect(Math.abs(img.data[0]! - 90)).toBeLessThanOrEqual(1);
    expect(img.data[1]).toBe(255);
    expect(img.data[2]).toBe(0);
  });

  it('darkening by a stop then brightening by a stop returns close to the original midtones', () => {
    const img = imageData([128, 96, 160, 255]);
    applyAdjustments(img, [exposure(-1)], ORIGIN);
    applyAdjustments(img, [exposure(1)], ORIGIN);
    expect(Math.abs(img.data[0]! - 128)).toBeLessThanOrEqual(2);
    expect(Math.abs(img.data[1]! - 96)).toBeLessThanOrEqual(2);
    expect(Math.abs(img.data[2]! - 160)).toBeLessThanOrEqual(2);
  });

  it('a channel-scoped levels entry remaps only its channel', () => {
    const img = imageData([200, 200, 200, 255]);
    applyAdjustments(img, [channelLevels('g', 128)], ORIGIN);
    expect(img.data[0]).toBe(200);
    expect(img.data[1]).toBe(Math.round((200 / 255) * 128));
    expect(img.data[2]).toBe(200);
  });

  it('folds exposure and channel levels with neighbours into one LUT segment, bit-exact vs sequential', () => {
    const folded = compileAdjustments([exposure(0.5), channelLevels('b', 200), invert()]);
    expect(folded.map((segment) => segment.kind)).toEqual(['lut']);

    const sequential = imageData([10, 20, 30, 255, 200, 100, 50, 255]);
    applyAdjustments(sequential, [exposure(0.5)], ORIGIN);
    applyAdjustments(sequential, [channelLevels('b', 200)], ORIGIN);
    applyAdjustments(sequential, [invert()], ORIGIN);
    const combined = imageData([10, 20, 30, 255, 200, 100, 50, 255]);
    applyAdjustments(combined, [exposure(0.5), channelLevels('b', 200), invert()], ORIGIN);
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
    applyAdjustments(img, [bc(0, 0), hsl(0)], ORIGIN);
    expect(Array.from(img.data)).toEqual([10, 20, 30, 128]);
    applyAdjustments(img, [bc(0.5, 0)], ORIGIN);
    expect(img.data[3]).toBe(128);
  });

  it('brightens rgb additively and clamps', () => {
    const img = imageData([10, 20, 30, 255]);
    applyAdjustments(img, [bc(0.5, 0)], ORIGIN);
    expect(img.data[0]).toBe(138); // 10 + 128
    expect(img.data[1]).toBe(148);
    expect(img.data[2]).toBe(158);
    const clamped = imageData([200, 0, 0, 255]);
    applyAdjustments(clamped, [bc(0.5, 0)], ORIGIN);
    expect(clamped.data[0]).toBe(255);
  });

  it('applies contrast about mid-grey', () => {
    const img = imageData([64, 128, 192, 255]);
    applyAdjustments(img, [bc(0, 1)], ORIGIN);
    expect(img.data[0]).toBe(0); // (64-128)*2+128
    expect(img.data[1]).toBe(128);
    expect(img.data[2]).toBe(255);
  });

  it('fully desaturates to luma at saturation -1', () => {
    const img = imageData([200, 100, 50, 255]);
    applyAdjustments(img, [hsl(-1)], ORIGIN);
    const luma = Math.round(0.299 * 200 + 0.587 * 100 + 0.114 * 50);
    expect(img.data[0]).toBe(luma);
    expect(img.data[1]).toBe(luma);
    expect(img.data[2]).toBe(luma);
  });

  it('inverts every channel exactly', () => {
    const img = imageData([0, 128, 255, 200]);
    applyAdjustments(img, [invert()], ORIGIN);
    expect(Array.from(img.data)).toEqual([255, 127, 0, 200]);
  });

  it('remaps through levels: input range, gamma midtones, output range', () => {
    // Input 64..192 stretched to full range: 64 → 0, 192 → 255, midpoint → 128.
    const stretch = imageData([64, 128, 192, 255]);
    applyAdjustments(stretch, [levels(64, 192, 1, 0, 255)], ORIGIN);
    expect(Array.from(stretch.data.slice(0, 3))).toEqual([0, 128, 255]);
    // Gamma 2 lifts midtones: 128 → 255 * (0.5)^(1/2) ≈ 180.
    const midtones = imageData([128, 0, 255, 255]);
    applyAdjustments(midtones, [levels(0, 255, 2, 0, 255)], ORIGIN);
    expect(midtones.data[0]).toBe(Math.round(255 * Math.sqrt(128 / 255)));
    expect(midtones.data[1]).toBe(0);
    expect(midtones.data[2]).toBe(255);
    // Output compression maps black to 50 and white to 200.
    const compress = imageData([0, 255, 128, 255]);
    applyAdjustments(compress, [levels(0, 255, 1, 50, 200)], ORIGIN);
    expect(compress.data[0]).toBe(50);
    expect(compress.data[1]).toBe(200);
  });

  it('hue rotation cycles the primaries and preserves grays', () => {
    // 120° sends red toward green (the standard hueRotate matrix dims it; green must dominate).
    const red = imageData([255, 0, 0, 255]);
    applyAdjustments(red, [hue(120)], ORIGIN);
    expect(red.data[0]).toBe(0);
    expect(red.data[2]).toBe(0);
    expect(red.data[1]).toBeGreaterThan(90);
    // A full ±360° is identity; gray is on the rotation axis and never moves.
    const gray = imageData([128, 128, 128, 255]);
    applyAdjustments(gray, [hue(90)], ORIGIN);
    expect(Array.from(gray.data)).toEqual([128, 128, 128, 255]);
    const wrapped = imageData([200, 100, 50, 255]);
    applyAdjustments(wrapped, [hue(360)], ORIGIN);
    expect(Array.from(wrapped.data)).toEqual([200, 100, 50, 255]);
  });

  it('entry order changes the pixels; a disabled entry renders exactly as a removed one', () => {
    const source = [180, 60, 40, 255];
    const bcFirst = imageData([...source]);
    applyAdjustments(bcFirst, [bc(0.3, 0.5), hsl(-0.8)], ORIGIN);
    const hslFirst = imageData([...source]);
    applyAdjustments(hslFirst, [hsl(-0.8), bc(0.3, 0.5)], ORIGIN);
    expect(Array.from(bcFirst.data)).not.toEqual(Array.from(hslFirst.data));

    const withDisabled = imageData([...source]);
    applyAdjustments(withDisabled, [bc(0.3, 0.5), hsl(-0.8, { isEnabled: false })], ORIGIN);
    const without = imageData([...source]);
    applyAdjustments(without, [bc(0.3, 0.5)], ORIGIN);
    expect(Array.from(withDisabled.data)).toEqual(Array.from(without.data));
  });
});

/** A `width`×`height` ImageData whose pixels `pixel(x, y)` fills. */
const imageOf = (width: number, height: number, pixel: (x: number, y: number) => readonly number[]): ImageData => {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      data.set(pixel(x, y), (y * width + x) * 4);
    }
  }
  return { data, height, width } as ImageData;
};
const midGray = (width: number, height: number): ImageData => imageOf(width, height, () => [128, 128, 128, 255]);
/** A busy source: every channel and alpha vary, including fully transparent pixels. */
const varied = (width: number, height: number): ImageData =>
  imageOf(width, height, (x, y) => [
    (x * 37 + y * 11) % 256,
    (x * 5 + y * 53) % 256,
    (x * y) % 256,
    (x + y) % 5 === 0 ? 0 : 255 - x,
  ]);
const crop = (image: ImageData, x: number, y: number, width: number, height: number): ImageData =>
  imageOf(width, height, (cx, cy) => {
    const i = ((y + cy) * image.width + x + cx) * 4;
    return Array.from(image.data.subarray(i, i + 4));
  });
/** The per-channel offsets noise added to a mid-gray image. */
const deltas = (image: ImageData): { r: number[]; g: number[]; b: number[] } => {
  const r: number[] = [];
  const g: number[] = [];
  const b: number[] = [];
  for (let i = 0; i < image.data.length; i += 4) {
    r.push(image.data[i]! - 128);
    g.push(image.data[i + 1]! - 128);
    b.push(image.data[i + 2]! - 128);
  }
  return { b, g, r };
};
const mean = (values: readonly number[]): number => values.reduce((sum, v) => sum + v, 0) / values.length;
const stddev = (values: readonly number[]): number => {
  const m = mean(values);
  return Math.sqrt(values.reduce((sum, v) => sum + (v - m) ** 2, 0) / values.length);
};
const correlation = (a: readonly number[], b: readonly number[]): number => {
  const ma = mean(a);
  const mb = mean(b);
  let cov = 0;
  for (let i = 0; i < a.length; i++) {
    cov += (a[i]! - ma) * (b[i]! - mb);
  }
  return cov / a.length / (stddev(a) * stddev(b));
};

describe('noise', () => {
  it("is identity at amount 0 and keys every parameter without changing other types' keys", () => {
    expect(isIdentityAdjustmentEntry(noise(0))).toBe(true);
    expect(isIdentityAdjustmentEntry(noise(0.01))).toBe(false);
    expect(isIdentityAdjustments([noise(0.5, { isEnabled: false })])).toBe(true);

    const base = adjustmentsKey([noise(0.1)]);
    const changes = [
      noise(0.2),
      noise(0.1, { distribution: 'uniform' }),
      noise(0.1, { grain: 'color' }),
      noise(0.1, { grain: 'chroma' }),
      noise(0.1, { response: 'midtones' }),
      noise(0.1, { size: 2.5 }),
      noise(0.1, { seed: 7 }),
    ];
    expect(new Set([base, ...changes.map((changed) => adjustmentsKey([changed]))]).size).toBe(changes.length + 1);
    expect(adjustmentsKey([noise(0.1, { id: 'other' })])).toBe(base);
    // Display caches key on these strings; the noise arm must leave the other types' keys as they were.
    expect(adjustmentsKey([bc(0.2, 0.1), exposure(1), invert(), hue(30), hsl(0.5)])).toBe(
      'bc:0.2,0.1||ex:1||inv||h:30||s:0.5'
    );
  });

  it('is its own segment: LUT neighbours fold on each side but never across it, keeping stack order', () => {
    const stack = [bc(0.1, 0.2), invert(), noise(0.2, { grain: 'color' }), levels(10, 240, 1.3, 0, 255), invert()];
    expect(compileAdjustments(stack).map((segment) => segment.kind)).toEqual(['lut', 'noise', 'lut']);

    const sequential = varied(16, 8);
    applyAdjustments(sequential, [bc(0.1, 0.2), invert()], ORIGIN);
    applyAdjustments(sequential, [noise(0.2, { grain: 'color' })], ORIGIN);
    applyAdjustments(sequential, [levels(10, 240, 1.3, 0, 255), invert()], ORIGIN);
    const combined = varied(16, 8);
    applyAdjustments(combined, stack, ORIGIN);
    expect(Array.from(combined.data)).toEqual(Array.from(sequential.data));
  });

  it('is region independent: a sub-rectangle adjusted at its own origin matches the whole-image pass', () => {
    const stack = [bc(0.1, 0.3), noise(0.15, { grain: 'color' }), hsl(-0.4), noise(0.05, { seed: 99 })];
    const origin = { x: -37, y: 12 };
    const whole = varied(40, 30);
    applyAdjustments(whole, stack, origin);

    const region = crop(varied(40, 30), 5, 7, 23, 9);
    applyAdjustments(region, stack, { x: origin.x + 5, y: origin.y + 7 });
    expect(Array.from(region.data)).toEqual(Array.from(crop(whole, 5, 7, 23, 9).data));
  });

  it.each(
    GRAINS.flatMap((grain) => RESPONSES.flatMap((response) => [1, 2.5, 8].map((size) => ({ grain, response, size }))))
  )(
    'is region independent for $grain grain, $response response, size $size, including regions thinner than the grain',
    (options) => {
      const stack = [bc(0.05, 0.2), noise(0.2, options), hsl(-0.3)];
      const origin = { x: -21, y: 9 };
      const whole = varied(48, 40);
      applyAdjustments(whole, stack, origin);
      // Sub-rectangles away from, touching and crossing the edges, down to a single row and column.
      for (const [x, y, width, height] of [
        [5, 7, 23, 9],
        [0, 0, 48, 1],
        [47, 0, 1, 40],
        [30, 31, 18, 9],
      ] as const) {
        const region = crop(varied(48, 40), x, y, width, height);
        applyAdjustments(region, stack, { x: origin.x + x, y: origin.y + y });
        expect(Array.from(region.data)).toEqual(Array.from(crop(whole, x, y, width, height).data));
      }
    }
  );

  it('keeps size 1 even grain bit-identical to the per-pixel kernel it replaced', () => {
    // Pinned from the kernel before grain size, tonal response and chroma existed.
    const pinned = {
      'color-gaussian': [
        96, 126, 156, 255, 104, 118, 156, 255, 120, 130, 122, 255, 134, 118, 174, 255, 84, 126, 150, 255, 114, 111, 151,
        255, 118, 135, 150, 255, 155, 121, 142, 255,
      ],
      'color-uniform': [
        95, 126, 158, 255, 102, 116, 158, 255, 120, 131, 129, 255, 135, 116, 171, 255, 82, 125, 152, 255, 116, 110, 154,
        255, 118, 137, 153, 255, 151, 119, 143, 255,
      ],
      'monochrome-gaussian': [
        96, 124, 146, 255, 104, 122, 144, 255, 120, 128, 150, 255, 134, 132, 154, 255, 84, 112, 124, 255, 114, 132, 144,
        255, 118, 126, 138, 255, 155, 153, 165, 255,
      ],
      'monochrome-uniform': [
        95, 123, 145, 255, 102, 120, 142, 255, 120, 128, 150, 255, 135, 133, 155, 255, 82, 110, 122, 255, 116, 134, 146,
        255, 118, 126, 138, 255, 151, 149, 161, 255,
      ],
    };
    for (const grain of ['monochrome', 'color'] as const) {
      for (const distribution of ['gaussian', 'uniform'] as const) {
        const image = imageOf(4, 2, (x, y) => [100 + x * 10, 128, 150 - y * 10, 255]);
        applyAdjustments(image, [noise(0.1, { distribution, grain })], { x: -3, y: 5 });
        expect(Array.from(image.data)).toEqual(pinned[`${grain}-${distribution}`]);
      }
    }
  });

  it('is deterministic per seed and changes pattern with the seed', () => {
    const first = midGray(32, 32);
    const again = midGray(32, 32);
    const reseeded = midGray(32, 32);
    applyAdjustments(first, [noise(0.1)], ORIGIN);
    applyAdjustments(again, [noise(0.1)], ORIGIN);
    applyAdjustments(reseeded, [noise(0.1, { seed: 0x1234abce })], ORIGIN);
    expect(Array.from(again.data)).toEqual(Array.from(first.data));
    const changed = deltas(first).r.filter((d, i) => d !== deltas(reseeded).r[i]).length;
    expect(changed).toBeGreaterThan(32 * 32 * 0.8);
  });

  it('shares one sample across channels when monochrome and samples each channel independently in color', () => {
    const mono = midGray(64, 64);
    applyAdjustments(mono, [noise(0.1)], ORIGIN);
    const monoDeltas = deltas(mono);
    expect(monoDeltas.g).toEqual(monoDeltas.r);
    expect(monoDeltas.b).toEqual(monoDeltas.r);

    const color = midGray(64, 64);
    applyAdjustments(color, [noise(0.1, { grain: 'color' })], ORIGIN);
    const { b, g, r } = deltas(color);
    expect(Math.abs(correlation(r, g))).toBeLessThan(0.05);
    expect(Math.abs(correlation(r, b))).toBeLessThan(0.05);
    expect(Math.abs(correlation(g, b))).toBeLessThan(0.05);
  });

  it.each(['gaussian', 'uniform'] as const)(
    "%s noise has zero mean, the amount's standard deviation, and no neighbour correlation",
    (distribution) => {
      const size = 256;
      const image = midGray(size, size);
      applyAdjustments(image, [noise(0.1, { distribution, grain: 'color' })], { x: -100, y: 3 });
      const sigma = 0.1 * NOISE_FULL_SCALE_SIGMA;
      for (const channel of Object.values(deltas(image))) {
        // 65,536 samples: the mean's standard error is ~0.05 and the deviation's ~0.4%.
        expect(Math.abs(mean(channel))).toBeLessThan(0.25);
        expect(stddev(channel) / sigma).toBeGreaterThan(0.98);
        expect(stddev(channel) / sigma).toBeLessThan(1.02);
        const right = channel.filter((_, i) => i % size !== size - 1);
        const rightNeighbour = channel.filter((_, i) => i % size !== 0);
        expect(Math.abs(correlation(right, rightNeighbour))).toBeLessThan(0.02);
        expect(Math.abs(correlation(channel.slice(0, -size), channel.slice(size)))).toBeLessThan(0.02);
      }
      const red = deltas(image).r;
      const beyondUniformBound = red.filter((d) => Math.abs(d) > Math.round(sigma * Math.sqrt(3))).length;
      if (distribution === 'uniform') {
        // Uniform with the same deviation spans ±σ√3.
        expect(beyondUniformBound).toBe(0);
      } else {
        // A gaussian puts ~7% of samples beyond ±1.73σ.
        expect(beyondUniformBound / red.length).toBeGreaterThan(0.05);
      }
    }
  );

  it.each(GRAINS.flatMap((grain) => [1, 4].map((size) => ({ grain, size }))))(
    'changes RGB only for $grain grain at size $size: alpha is preserved and fully transparent pixels are untouched',
    (options) => {
      const source = varied(32, 16);
      const image = varied(32, 16);
      applyAdjustments(image, [noise(1, { ...options, response: 'midtones' })], ORIGIN);
      let changed = 0;
      for (let i = 0; i < image.data.length; i += 4) {
        expect(image.data[i + 3]).toBe(source.data[i + 3]);
        if (source.data[i + 3] === 0) {
          expect(Array.from(image.data.subarray(i, i + 3))).toEqual(Array.from(source.data.subarray(i, i + 3)));
        } else if (image.data[i] !== source.data[i]) {
          changed += 1;
        }
      }
      expect(changed).toBeGreaterThan(0);
    }
  );
});

/** Rec. 709 luma's midtone weight 4·Y′·(1 − Y′) of a neutral grey, computed independently of the kernel. */
const greyWeight = (grey: number): number => 4 * (grey / 255) * (1 - grey / 255);

describe('noise tonal response', () => {
  it.each(GRAINS.flatMap((grain) => [1, 3].map((size) => ({ grain, size }))))(
    'midtones leaves pure black and white exactly and scales $grain grain by the tone at size $size',
    (options) => {
      const size = 128;
      const greys = [0, 64, 128, 255];
      const flat = imageOf(size, size * greys.length, (_, y) => {
        const grey = greys[Math.floor(y / size)]!;
        return [grey, grey, grey, 255];
      });
      const midtones = imageOf(flat.width, flat.height, (x, y) =>
        Array.from(flat.data.subarray((y * size + x) * 4, (y * size + x) * 4 + 4))
      );
      // Light enough that the 64-grey band never clips at black.
      applyAdjustments(flat, [noise(0.05, options)], ORIGIN);
      applyAdjustments(midtones, [noise(0.05, { ...options, response: 'midtones' })], ORIGIN);
      const band = (image: ImageData, index: number) => crop(image, 0, index * size, size, size);
      /** Deviation from the band's grey across all channels. */
      const spread = (image: ImageData, grey: number) => {
        const values: number[] = [];
        for (let i = 0; i < image.data.length; i += 4) {
          values.push(image.data[i]! - grey, image.data[i + 1]! - grey, image.data[i + 2]! - grey);
        }
        return Math.sqrt(values.reduce((sum, v) => sum + v * v, 0) / values.length);
      };

      expect(Array.from(band(midtones, 0).data)).toEqual(
        Array.from(
          band(
            imageOf(size, size, () => [0, 0, 0, 255]),
            0
          ).data
        )
      );
      expect(Array.from(band(midtones, 3).data)).toEqual(
        Array.from(imageOf(size, size, () => [255, 255, 255, 255]).data)
      );
      // Even response grains black and white too (it clips half of the samples there, but not all).
      expect(spread(band(flat, 0), 0)).toBeGreaterThan(0);
      expect(spread(band(flat, 3), 255)).toBeGreaterThan(0);
      for (const index of [1, 2]) {
        const grey = greys[index]!;
        const ratio = spread(band(midtones, index), grey) / spread(band(flat, index), grey);
        // Rounding to whole codes blurs the ratio slightly; mid-grey is full strength, 64 about three quarters.
        expect(ratio).toBeGreaterThan(greyWeight(grey) - 0.05);
        expect(ratio).toBeLessThan(greyWeight(grey) + 0.05);
      }
    }
  );
});

describe('noise grain size', () => {
  const field = (size: number, grain: 'monochrome' | 'color' = 'monochrome') => {
    const side = 384;
    const image = midGray(side, side);
    applyAdjustments(image, [noise(0.1, { grain, size })], { x: 1000, y: -77 });
    return { channels: deltas(image), side };
  };
  const rightNeighbourCorrelation = (channel: readonly number[], side: number) =>
    correlation(
      channel.filter((_, i) => i % side !== side - 1),
      channel.filter((_, i) => i % side !== 0)
    );
  const belowNeighbourCorrelation = (channel: readonly number[], side: number) =>
    correlation(channel.slice(0, -side), channel.slice(side));

  it.each([2, 4, 8])("keeps the amount's standard deviation and zero mean at size %d", (size) => {
    const { channels } = field(size, 'color');
    const sigma = 0.1 * NOISE_FULL_SCALE_SIGMA;
    for (const channel of Object.values(channels)) {
      // Larger grain means fewer independent samples: at size 8 the deviation's standard error is about 2.5%.
      expect(stddev(channel) / sigma).toBeGreaterThan(0.9);
      expect(stddev(channel) / sigma).toBeLessThan(1.1);
      expect(Math.abs(mean(channel))).toBeLessThan(1);
    }
    expect(Math.abs(correlation(channels.r, channels.g))).toBeLessThan(0.1);
  });

  it('correlates neighbours more as the grain grows, and not at all at size 1', () => {
    const correlations = [1, 2, 4, 8].map((size) => {
      const { channels, side } = field(size);
      return [rightNeighbourCorrelation(channels.r, side), belowNeighbourCorrelation(channels.r, side)] as const;
    });
    expect(Math.abs(correlations[0]![0])).toBeLessThan(0.02);
    expect(Math.abs(correlations[0]![1])).toBeLessThan(0.02);
    // Size 2 blurs by σ √3 / 2: taps 1, e^(−2/3), e^(−8/3), e⁻⁶ correlate adjacent pixels by about 0.71.
    expect(correlations[1]![0]).toBeGreaterThan(0.6);
    expect(correlations[1]![1]).toBeGreaterThan(0.6);
    for (let i = 2; i < correlations.length; i++) {
      expect(correlations[i]![0]).toBeGreaterThan(correlations[i - 1]![0]);
      expect(correlations[i]![1]).toBeGreaterThan(correlations[i - 1]![1]);
    }
  });

  it('runs sizes within one slider step of 1 as size 1, and the next step already correlates neighbours', () => {
    const one = midGray(64, 64);
    const barelyLarger = midGray(64, 64);
    applyAdjustments(one, [noise(0.1)], ORIGIN);
    applyAdjustments(barelyLarger, [noise(0.1, { size: 1.1 })], ORIGIN);
    expect(Array.from(barelyLarger.data)).toEqual(Array.from(one.data));

    // Size 1.2 blurs by σ ≈ 0.33: an outer tap of e^(−4.5) ≈ 0.011 correlates neighbours by about 0.02.
    const { channels, side } = field(1.2);
    expect(rightNeighbourCorrelation(channels.r, side)).toBeGreaterThan(0.01);
    expect(belowNeighbourCorrelation(channels.r, side)).toBeGreaterThan(0.01);
  });

  it('grain span equals the standard deviation box-averaging leaves, per block side', () => {
    const { channels, side } = field(8);
    const block = 32;
    const averages: number[] = [];
    for (let by = 0; by < side; by += block) {
      for (let bx = 0; bx < side; bx += block) {
        let sum = 0;
        for (let y = by; y < by + block; y++) {
          for (let x = bx; x < bx + block; x++) {
            sum += channels.r[y * side + x]!;
          }
        }
        averages.push(sum / (block * block));
      }
    }
    // 144 blocks: the measured ratio's standard error is about 6%.
    const ratio =
      Math.sqrt(averages.reduce((sum, v) => sum + v * v, 0) / averages.length) / (0.1 * NOISE_FULL_SCALE_SIGMA);
    expect(ratio / (noiseGrainSpan(8) / block)).toBeGreaterThan(0.75);
    expect(ratio / (noiseGrainSpan(8) / block)).toBeLessThan(1.25);
    expect(noiseGrainSpan(1)).toBe(1);
  });
});

/** OKLab [L, a, b] of an sRGB pixel, computed with the exact transfer curve and cube root. */
const oklab = (r: number, g: number, b: number): readonly [number, number, number] => {
  const linear = (v: number) => (v <= 0.04045 * 255 ? v / 255 / 12.92 : ((v / 255 + 0.055) / 1.055) ** 2.4);
  const [lr, lg, lb] = [linear(r), linear(g), linear(b)];
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
};
/** Each pixel's OKLab coordinates. */
const oklabPixels = (image: ImageData): (readonly [number, number, number])[] => {
  const pixels: (readonly [number, number, number])[] = [];
  for (let i = 0; i < image.data.length; i += 4) {
    pixels.push(oklab(image.data[i]!, image.data[i + 1]!, image.data[i + 2]!));
  }
  return pixels;
};

describe('chroma grain', () => {
  it.each([1, 3])('varies hue and colorfulness but keeps OKLab lightness at size %d', (size) => {
    // Muted midtones, so 5% grain stays inside the sRGB gamut and nothing clips.
    const fill = (x: number, y: number) => [100 + (x >> 1), 110 + (y >> 1), 140 - (x >> 1), 255];
    const source = oklabPixels(imageOf(96, 96, fill));
    const image = imageOf(96, 96, fill);
    applyAdjustments(image, [noise(0.05, { grain: 'chroma', size })], { x: 5, y: 5 });
    const adjusted = oklabPixels(image);
    const lightnessError = adjusted.map(([lightness], i) => Math.abs(lightness - source[i]![0]));
    const chromaShift = adjusted.map(([, a, b], i) => Math.hypot(a - source[i]![1], b - source[i]![2]));
    // Rounding each channel to a whole code moves lightness by at most about 0.003; the grain moves chroma by
    // σ√2 ≈ 0.023 on average.
    expect(Math.max(...lightnessError)).toBeLessThan(0.004);
    expect(mean(chromaShift)).toBeGreaterThan(0.01);
  });

  it('leaves neutral grey neutral on average, with colorfulness comparable to color grain', () => {
    const side = 256;
    const chroma = midGray(side, side);
    const color = midGray(side, side);
    applyAdjustments(chroma, [noise(0.1, { grain: 'chroma' })], ORIGIN);
    applyAdjustments(color, [noise(0.1, { grain: 'color' })], ORIGIN);
    const chromaLab = oklabPixels(chroma);
    const colorLab = oklabPixels(color);
    for (const axis of [1, 2] as const) {
      const chromaAxis = chromaLab.map((pixel) => pixel[axis]);
      // 65,536 samples of σ ≈ 0.032: the mean's standard error is about 0.0001; gamut clipping adds a little bias.
      expect(Math.abs(mean(chromaAxis))).toBeLessThan(0.002);
      // Chroma puts color grain's lightness spread onto a and b too, so it is somewhat more colorful.
      const ratio = stddev(chromaAxis) / stddev(colorLab.map((pixel) => pixel[axis]));
      expect(ratio).toBeGreaterThan(1);
      expect(ratio).toBeLessThan(1.6);
    }
  });
});

describe('isPositionDependentAdjustments', () => {
  it('is true only while a noise entry contributes', () => {
    expect(isPositionDependentAdjustments(undefined)).toBe(false);
    expect(isPositionDependentAdjustments([bc(0.2, 0), invert(), hue(30)])).toBe(false);
    expect(isPositionDependentAdjustments([bc(0.2, 0), noise(0.1)])).toBe(true);
    expect(isPositionDependentAdjustments([noise(0)])).toBe(false);
    expect(isPositionDependentAdjustments([noise(0.1, { isEnabled: false })])).toBe(false);
  });
});
