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
  perPixelAdjustments,
  toSpatialPlan,
} from './adjustments';

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

  it('interpolates monotonically through interior points without overshoot', () => {
    const lut = buildCurveLut([
      [0, 0],
      [128, 200],
      [255, 255],
    ]);
    expect(lut[128]).toBe(200);
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

  it('keys identify the pixel effect: stable across entry ids, sensitive to order and values', () => {
    expect(adjustmentsKey([bc(0, 0), hsl(0)])).toBe('identity');
    expect(adjustmentsKey([bc(0.2, 0)])).toBe(adjustmentsKey([bc(0.2, 0, { id: 'other' })]));
    expect(adjustmentsKey([bc(0.2, 0)])).not.toBe(adjustmentsKey([bc(0.3, 0)]));
    expect(adjustmentsKey([bc(0.2, 0), hsl(0.5)])).not.toBe(adjustmentsKey([hsl(0.5), bc(0.2, 0)]));
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
    const compress = imageData([0, 255, 128, 255]);
    applyAdjustments(compress, [levels(0, 255, 1, 50, 200)], ORIGIN);
    expect(compress.data[0]).toBe(50);
    expect(compress.data[1]).toBe(200);
  });

  it('hue rotation cycles the primaries and preserves grays', () => {
    const red = imageData([255, 0, 0, 255]);
    applyAdjustments(red, [hue(120)], ORIGIN);
    expect(red.data[0]).toBe(0);
    expect(red.data[2]).toBe(0);
    expect(red.data[1]).toBeGreaterThan(90);
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
    expect(adjustmentsKey([bc(0.2, 0.1), exposure(1), invert(), hue(30), hsl(0.5)])).toBe(
      'bc:0.2,0.1||ex:1||inv||h:30||s:0.5'
    );
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
});

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
      applyAdjustments(flat, [noise(0.05, options)], ORIGIN);
      applyAdjustments(midtones, [noise(0.05, { ...options, response: 'midtones' })], ORIGIN);
      const band = (image: ImageData, index: number) => crop(image, 0, index * size, size, size);
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
});

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
const oklabPixels = (image: ImageData): (readonly [number, number, number])[] => {
  const pixels: (readonly [number, number, number])[] = [];
  for (let i = 0; i < image.data.length; i += 4) {
    pixels.push(oklab(image.data[i]!, image.data[i + 1]!, image.data[i + 2]!));
  }
  return pixels;
};

describe('chroma grain', () => {
  it.each([1, 3])('varies hue and colorfulness but keeps OKLab lightness at size %d', (size) => {
    const fill = (x: number, y: number) => [100 + (x >> 1), 110 + (y >> 1), 140 - (x >> 1), 255];
    const source = oklabPixels(imageOf(96, 96, fill));
    const image = imageOf(96, 96, fill);
    applyAdjustments(image, [noise(0.05, { grain: 'chroma', size })], { x: 5, y: 5 });
    const adjusted = oklabPixels(image);
    const lightnessError = adjusted.map(([lightness], i) => Math.abs(lightness - source[i]![0]));
    const chromaShift = adjusted.map(([, a, b], i) => Math.hypot(a - source[i]![1], b - source[i]![2]));
    // Rounding each channel to a whole code moves lightness by at most about 0.003; the grain moves chroma by σ√2 ≈
    // 0.023 on average.
    expect(Math.max(...lightnessError)).toBeLessThan(0.004);
    expect(mean(chromaShift)).toBeGreaterThan(0.01);
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

const clarity = (amount: number, size = 40, overrides: EntryOverrides = {}) =>
  ({ amount, id: 'cl', isEnabled: true, size, type: 'clarity', ...overrides }) as const;
const texture = (amount: number, size = 8, overrides: EntryOverrides = {}) =>
  ({ amount, id: 'tx', isEnabled: true, size, type: 'texture', ...overrides }) as const;
const dehaze = (amount: number, overrides: EntryOverrides = {}) =>
  ({ amount, id: 'dh', isEnabled: true, type: 'dehaze', ...overrides }) as const;

describe('spatial entries', () => {
  it('are skipped by the per-pixel pass, which equals the stack without them', () => {
    const pixels = [40, 120, 200, 255, 230, 10, 90, 128];
    const stack = [clarity(80), bc(0.2, 0.1), dehaze(-40), hsl(-0.5), texture(50)];
    expect(compileAdjustments([clarity(80), texture(-50), dehaze(10)])).toEqual([]);
    expect(isIdentityAdjustments(perPixelAdjustments([clarity(80)]))).toBe(true);

    const withSpatial = imageData(pixels);
    applyAdjustments(withSpatial, stack, { x: 0, y: 0 });
    const without = imageData(pixels);
    applyAdjustments(without, [bc(0.2, 0.1), hsl(-0.5)], { x: 0, y: 0 });
    expect([...withSpatial.data]).toEqual([...without.data]);
    expect(perPixelAdjustments(stack)).toEqual([bc(0.2, 0.1), hsl(-0.5)]);
  });

  it('plan contributing entries as ordered runs, merging spatial neighbours across inert entries', () => {
    expect(toSpatialPlan([])).toEqual([]);
    expect(toSpatialPlan([bc(0.2, 0), hsl(0.1)])).toEqual([{ adjustments: [bc(0.2, 0), hsl(0.1)], kind: 'per-pixel' }]);
    expect(
      toSpatialPlan([
        bc(0.2, 0),
        clarity(30, 50),
        hsl(0.4, { isEnabled: false }),
        texture(-20, 6),
        exposure(0),
        dehaze(15),
        hsl(-0.3),
        clarity(0),
      ])
    ).toEqual([
      { adjustments: [bc(0.2, 0)], kind: 'per-pixel' },
      {
        kind: 'spatial',
        steps: [
          { amount: 30, kind: 'clarity', size: 50 },
          { amount: -20, kind: 'texture', size: 6 },
          { amount: 15, kind: 'dehaze' },
        ],
      },
      { adjustments: [hsl(-0.3)], kind: 'per-pixel' },
    ]);
    expect(toSpatialPlan([dehaze(10), bc(0.1, 0), dehaze(20)]).map((run) => run.kind)).toEqual([
      'spatial',
      'per-pixel',
      'spatial',
    ]);
  });
});
