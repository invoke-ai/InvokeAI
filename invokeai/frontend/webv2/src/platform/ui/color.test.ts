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

  it('falls back for garbage, wrong-length, and non-hex input', () => {
    // 5 and 7 digits are truncated/mistyped values, not shorthand.
    for (const value of ['', '#', 'rebeccapurple', '#ff', '#ff800', '#ff80000', '#ggghhh', 'var(--x)']) {
      expect(parseHexColor(value)).toEqual(BLACK);
    }
  });
});

describe('formatHexColor', () => {
  it('emits 6-digit hex by default and 8-digit under `alpha`', () => {
    const color = { a: 0.5, b: 0, g: 128, r: 255 };
    expect(formatHexColor(color)).toBe('#ff8000');
    expect(formatHexColor(color, { alpha: true })).toBe('#ff800080');
  });

  it('clamps out-of-range and non-finite channels', () => {
    expect(formatHexColor({ a: 2, b: -5, g: 999, r: 300 }, { alpha: true })).toBe('#ffff00ff');
    expect(formatHexColor({ a: Number.NaN, b: 0, g: 0, r: Number.NaN }, { alpha: true })).toBe('#000000ff');
  });
});

describe('round-trips', () => {
  it.each([
    ['6-digit', ['#000000', '#ffffff', '#e07575', '#799ddb'], false],
    ['8-digit', ['#00000000', '#ffffffff', '#e0757580', '#799ddb01'], true],
  ])('preserves %s hex', (_label, hexes, alpha) => {
    for (const hex of hexes) {
      expect(formatHexColor(parseHexColor(hex), { alpha })).toBe(hex);
    }
  });
});

describe('splitHexAlpha / joinHexAlpha', () => {
  it.each([
    ['splits an 8-digit hex into rgb and unit alpha', '#ff800080', { alpha: 128 / 255, hex: '#ff8000' }],
    ['treats a 6-digit hex as fully opaque', '#ff8000', { alpha: 1, hex: '#ff8000' }],
  ])('%s', (_label, value, expected) => {
    expect(splitHexAlpha(value)).toEqual(expected);
  });

  it.each([
    [1, '#ff8000ff'],
    [0, '#ff800000'],
    [5, '#ff8000ff'],
    [-1, '#ff800000'],
  ])('joins alpha %s back to 8-digit hex, clamped, as %s', (alpha, expected) => {
    expect(joinHexAlpha('#ff8000', alpha)).toBe(expected);
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
});
