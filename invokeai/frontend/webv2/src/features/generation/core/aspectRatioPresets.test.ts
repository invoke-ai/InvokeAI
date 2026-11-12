import { describe, expect, it } from 'vitest';

import {
  formatAspectRatio,
  formatAspectRatioTerms,
  getAspectRatioTerms,
  getAspectRatioTermsError,
  getLockedAspectRatioSettings,
  matchAspectRatio,
  MAX_ASPECT_RATIO_PRESETS,
  normalizeAspectRatioPresets,
} from './aspectRatioPresets';

const cinema = { height: 1, id: 'cinema', label: 'Cinema', width: 2.39 };

describe('normalizeAspectRatioPresets', () => {
  it('keeps valid ratios in order with trimmed labels, including decimal terms and the 1:8 and 8:1 bounds', () => {
    expect(
      normalizeAspectRatioPresets([
        { ...cinema, label: '  Cinema  ' },
        { height: 5, id: 'banner', label: 'Banner', width: 12 },
        { height: 8, id: 'tall', label: 'Tall', width: 1 },
        { height: 1, id: 'wide', label: 'Wide', width: 8 },
      ])
    ).toEqual([
      cinema,
      { height: 5, id: 'banner', label: 'Banner', width: 12 },
      { height: 8, id: 'tall', label: 'Tall', width: 1 },
      { height: 1, id: 'wide', label: 'Wide', width: 8 },
    ]);
  });

  it('drops malformed entries, out-of-range ratios, blank labels and repeated ids', () => {
    expect(
      normalizeAspectRatioPresets([
        null,
        'cinema',
        { height: 1, id: 'zero', label: 'Zero', width: 0 },
        { height: -2, id: 'negative', label: 'Negative', width: 3 },
        { height: 1, id: 'infinite', label: 'Infinite', width: Number.POSITIVE_INFINITY },
        { height: '1', id: 'text', label: 'Text', width: 2 },
        { height: 1, id: 'too-wide', label: 'Too wide', width: 9 },
        { height: 10, id: 'too-tall', label: 'Too tall', width: 1 },
        { height: 1, id: 'blank', label: '   ', width: 2 },
        { height: 1, id: '', label: 'No id', width: 2 },
        cinema,
        { ...cinema, label: 'Duplicate' },
      ])
    ).toEqual([cinema]);
    expect(normalizeAspectRatioPresets(undefined)).toEqual([]);
    expect(normalizeAspectRatioPresets({ cinema })).toEqual([]);
  });

  it('cuts a label to 40 characters', () => {
    const long = 'A ratio named at far greater length than the list can show';

    expect(normalizeAspectRatioPresets([{ ...cinema, label: long }])[0]?.label).toBe(
      'A ratio named at far greater length than'
    );
  });

  it('keeps the first ratios up to the cap', () => {
    const many = Array.from({ length: MAX_ASPECT_RATIO_PRESETS + 5 }, (_, index) => ({
      height: 1,
      id: `ratio-${index}`,
      label: `Ratio ${index}`,
      width: 2,
    }));

    const kept = normalizeAspectRatioPresets(many);

    expect(kept).toHaveLength(MAX_ASPECT_RATIO_PRESETS);
    expect(kept.at(-1)?.id).toBe(`ratio-${MAX_ASPECT_RATIO_PRESETS - 1}`);
  });
});

describe('getAspectRatioTermsError', () => {
  it('names a non-positive term before an out-of-range ratio', () => {
    expect(getAspectRatioTermsError(16, 9)).toBeNull();
    expect(getAspectRatioTermsError(0, 9)).toBe('terms');
    expect(getAspectRatioTermsError(Number.NaN, 9)).toBe('terms');
    expect(getAspectRatioTermsError(9, 1)).toBe('range');
    expect(getAspectRatioTermsError(1, 8.5)).toBe('range');
  });
});

describe('matchAspectRatio', () => {
  const widescreen = { height: 9, id: 'widescreen', label: 'Widescreen', width: 16 };

  it('names a locked ratio by the saved ratio equal to it, before a built-in equal to it', () => {
    expect(matchAspectRatio(16 / 9, [cinema, widescreen])).toEqual({ kind: 'preset', preset: widescreen });
    expect(matchAspectRatio(16 / 9, [cinema])).toEqual({ id: '16:9', kind: 'builtin' });
    expect(matchAspectRatio(2.39, [cinema])).toEqual({ kind: 'preset', preset: cinema });
  });

  it('tolerates float error but not a different ratio', () => {
    expect(matchAspectRatio(2.39 + 0.00005, [cinema])).toEqual({ kind: 'preset', preset: cinema });
    expect(matchAspectRatio(2.4, [cinema])).toBeNull();
    expect(matchAspectRatio(0, [cinema])).toBeNull();
  });
});

describe('getAspectRatioTerms', () => {
  it('finds small whole terms, so a size prefills the ratio it has', () => {
    expect(getAspectRatioTerms(1536 / 640)).toEqual({ height: 5, width: 12 });
    expect(getAspectRatioTerms(1024 / 576)).toEqual({ height: 9, width: 16 });
    expect(getAspectRatioTerms(640 / 1536)).toEqual({ height: 12, width: 5 });
    expect(getAspectRatioTerms(1)).toEqual({ height: 1, width: 1 });
  });

  it('falls back to the ratio against 1, to two decimals, when no small whole terms fit', () => {
    expect(getAspectRatioTerms(2.39)).toEqual({ height: 1, width: 2.39 });
    expect(getAspectRatioTerms(1 / Math.SQRT2)).toEqual({ height: 1.41, width: 1 });
    // 1224 × 512 is 153:64, past the 32 limit on the second term.
    expect(getAspectRatioTerms(1224 / 512)).toEqual({ height: 1, width: 2.39 });
  });
});

describe('formatAspectRatio', () => {
  it('writes the terms in the locale', () => {
    expect(formatAspectRatio(2.4, 'en')).toBe('12:5');
    expect(formatAspectRatio(2.39, 'en')).toBe('2.39:1');
    expect(formatAspectRatio(1 / Math.SQRT2, 'en')).toBe('1:1.41');
    expect(formatAspectRatio(2.39, 'de')).toBe('2,39:1');
    expect(formatAspectRatioTerms({ height: 1, width: 2.395 }, 'en')).toBe('2.4:1');
  });
});

describe('getLockedAspectRatioSettings', () => {
  it('stores a ratio equal to a built-in under its id and exact value, any other as locked Free', () => {
    expect(getLockedAspectRatioSettings(16 / 9 + 0.00001)).toEqual({
      aspectRatioId: '16:9',
      aspectRatioIsLocked: true,
      aspectRatioValue: 16 / 9,
    });
    expect(getLockedAspectRatioSettings(2.39)).toEqual({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: true,
      aspectRatioValue: 2.39,
    });
  });
});
