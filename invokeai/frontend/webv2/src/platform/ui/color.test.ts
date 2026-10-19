import { describe, expect, it } from 'vitest';

import { BLACK, formatHexColor, joinHexAlpha, normalizeHex, parseHexColor, splitHexAlpha } from './color';

describe('parseHexColor', () => {
  it('parses 6-digit hex as opaque', () => {
    expect(parseHexColor('#ff8000')).toEqual({ a: 1, b: 0, g: 128, r: 255 });
  });

  it('parses 8-digit hex, scaling alpha to [0, 1]', () => {
    expect(parseHexColor('#ff8000ff')).toEqual({ a: 1, b: 0, g: 128, r: 255 });
    expect(parseHexColor('#ff800000')).toEqual({ a: 0, b: 0, g: 128, r: 255 });
    expect(parseHexColor('#00000080').a).toBeCloseTo(128 / 255, 5);
  });

  it('expands 3- and 4-digit shorthand', () => {
    expect(parseHexColor('#f80')).toEqual({ a: 1, b: 0, g: 136, r: 255 });
    expect(parseHexColor('#f800')).toEqual({ a: 0, b: 0, g: 136, r: 255 });
  });

  it('accepts uppercase and a missing leading hash, and trims', () => {
    expect(parseHexColor('  FF8000  ')).toEqual({ a: 1, b: 0, g: 128, r: 255 });
    expect(parseHexColor('#FF8000')).toEqual(parseHexColor('#ff8000'));
  });

  it('falls back for garbage, wrong-length, and non-hex input', () => {
    // 5 and 7 digits are truncated/mistyped values, not shorthand.
    for (const value of ['', '#', 'rebeccapurple', '#ff', '#ff800', '#ff80000', '#ggghhh', 'var(--x)']) {
      expect(parseHexColor(value)).toEqual(BLACK);
    }
  });

  it('uses the supplied fallback', () => {
    const white = { a: 1, b: 255, g: 255, r: 255 };
    expect(parseHexColor('nope', white)).toEqual(white);
  });
});

describe('formatHexColor', () => {
  it('emits 6-digit hex by default and 8-digit under `alpha`', () => {
    const color = { a: 0.5, b: 0, g: 128, r: 255 };
    expect(formatHexColor(color)).toBe('#ff8000');
    expect(formatHexColor(color, { alpha: true })).toBe('#ff800080');
  });

  it('emits lowercase', () => {
    expect(formatHexColor({ a: 1, b: 255, g: 255, r: 255 })).toBe('#ffffff');
  });

  it('clamps out-of-range and non-finite channels', () => {
    expect(formatHexColor({ a: 2, b: -5, g: 999, r: 300 }, { alpha: true })).toBe('#ffff00ff');
    expect(formatHexColor({ a: Number.NaN, b: 0, g: 0, r: Number.NaN }, { alpha: true })).toBe('#000000ff');
  });

  it('rounds fractional channels', () => {
    expect(formatHexColor({ a: 1, b: 0, g: 0, r: 127.6 })).toBe('#800000');
  });
});

describe('round-trips', () => {
  it('preserves 6-digit hex', () => {
    for (const hex of ['#000000', '#ffffff', '#e07575', '#799ddb']) {
      expect(formatHexColor(parseHexColor(hex))).toBe(hex);
    }
  });

  it('preserves 8-digit hex', () => {
    for (const hex of ['#00000000', '#ffffffff', '#e0757580', '#799ddb01']) {
      expect(formatHexColor(parseHexColor(hex), { alpha: true })).toBe(hex);
    }
  });
});

describe('splitHexAlpha / joinHexAlpha', () => {
  it('splits an 8-digit hex into rgb and unit alpha', () => {
    expect(splitHexAlpha('#ff800080')).toEqual({ alpha: 128 / 255, hex: '#ff8000' });
  });

  it('treats a 6-digit hex as fully opaque', () => {
    expect(splitHexAlpha('#ff8000')).toEqual({ alpha: 1, hex: '#ff8000' });
  });

  it('joins back to 8-digit hex', () => {
    expect(joinHexAlpha('#ff8000', 1)).toBe('#ff8000ff');
    expect(joinHexAlpha('#ff8000', 0)).toBe('#ff800000');
  });

  it('clamps alpha when joining', () => {
    expect(joinHexAlpha('#ff8000', 5)).toBe('#ff8000ff');
    expect(joinHexAlpha('#ff8000', -1)).toBe('#ff800000');
  });

  it('survives a split/join round trip', () => {
    const original = '#12345678';
    const { alpha, hex } = splitHexAlpha(original);
    expect(joinHexAlpha(hex, alpha)).toBe(original);
  });

  it('ignores any alpha already on the rgb argument', () => {
    expect(joinHexAlpha('#ff800000', 1)).toBe('#ff8000ff');
  });
});

describe('normalizeHex', () => {
  it('lowercases and expands shorthand', () => {
    expect(normalizeHex('#FF8000')).toBe('#ff8000');
    // Shorthand doubles each nibble: `f80` → `ff8800`.
    expect(normalizeHex('F80')).toBe('#ff8800');
  });

  it('drops a fully-opaque alpha but keeps a partial one', () => {
    expect(normalizeHex('#ff8000ff')).toBe('#ff8000');
    expect(normalizeHex('#ff800080')).toBe('#ff800080');
  });

  it('returns the fallback verbatim for non-hex input', () => {
    expect(normalizeHex('transparent', 'transparent')).toBe('transparent');
    expect(normalizeHex('nope')).toBe('#000000');
  });

  it('makes case-different values comparable', () => {
    expect(normalizeHex('#FF0000')).toBe(normalizeHex('#ff0000'));
  });
});
