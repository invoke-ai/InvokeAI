import { describe, expect, it } from 'vitest';

import {
  areColorPairsEqual,
  CANVAS_ACTIVE_COLORS_KEY,
  CANVAS_ACTIVE_COLOR_TARGET_KEY,
  CANVAS_COLOR_PALETTE_KEY,
  DEFAULT_COLOR_PAIR,
  MAX_COLOR_PALETTE_SIZE,
  readActiveColorPair,
  readActiveColorTarget,
  readColorPalette,
  swapColorPair,
  withoutPaletteColor,
  withPairColor,
  withPaletteColor,
} from './colorPair';

describe('active color pair', () => {
  it('defaults to black on white with a foreground target', () => {
    expect(readActiveColorPair(undefined)).toEqual(DEFAULT_COLOR_PAIR);
    expect(readActiveColorPair({})).toEqual(DEFAULT_COLOR_PAIR);
    expect(readActiveColorTarget({})).toBe('foreground');
  });

  it('preserves an existing pair verbatim, so the read is idempotent', () => {
    const values = { [CANVAS_ACTIVE_COLORS_KEY]: { background: '#112233', foreground: '#abcdef' } };
    const first = readActiveColorPair(values);
    expect(first).toEqual({ background: '#112233', foreground: '#abcdef' });
    expect(readActiveColorPair({ [CANVAS_ACTIVE_COLORS_KEY]: first })).toEqual(first);
  });

  it('normalizes case and falls back per malformed field', () => {
    expect(readActiveColorPair({ [CANVAS_ACTIVE_COLORS_KEY]: { background: 12, foreground: '#ABCDEF' } })).toEqual({
      background: DEFAULT_COLOR_PAIR.background,
      foreground: '#abcdef',
    });
    expect(readActiveColorPair({ [CANVAS_ACTIVE_COLORS_KEY]: 'nope' })).toEqual(DEFAULT_COLOR_PAIR);
    expect(readActiveColorTarget({ [CANVAS_ACTIVE_COLOR_TARGET_KEY]: 'mask-tint' })).toBe('foreground');
    expect(readActiveColorTarget({ [CANVAS_ACTIVE_COLOR_TARGET_KEY]: 'background' })).toBe('background');
  });

  it('flattens alpha bytes and rejects non-hex strings, so the pair stays opaque paint', () => {
    expect(
      readActiveColorPair({ [CANVAS_ACTIVE_COLORS_KEY]: { background: 'transparent', foreground: '#ff000080' } })
    ).toEqual({ background: '#ffffff', foreground: '#ff0000' });
    expect(withPairColor(DEFAULT_COLOR_PAIR, 'foreground', '#00ff0080').foreground).toBe('#00ff00');
  });

  it('swaps and writes targets without touching the other slot', () => {
    const pair = { background: '#ffffff', foreground: '#ff0000' };
    expect(swapColorPair(pair)).toEqual({ background: '#ff0000', foreground: '#ffffff' });
    expect(withPairColor(pair, 'background', '#00FF00')).toEqual({ background: '#00ff00', foreground: '#ff0000' });
    expect(withPairColor(pair, 'foreground', 'garbage').foreground).toBe('#ff0000');
    expect(areColorPairsEqual(pair, { ...pair })).toBe(true);
  });
});

describe('project palette', () => {
  it('reads only well-formed unique colors up to the cap', () => {
    expect(readColorPalette(undefined)).toEqual([]);
    expect(readColorPalette({ [CANVAS_COLOR_PALETTE_KEY]: ['#FF0000', 7, '#ff0000', '#00ff00'] })).toEqual([
      '#ff0000',
      '#00ff00',
    ]);
    const oversized = Array.from(
      { length: MAX_COLOR_PALETTE_SIZE + 5 },
      (_, i) => `#${String(i).padStart(2, '0')}0000`
    );
    expect(readColorPalette({ [CANVAS_COLOR_PALETTE_KEY]: oversized })).toHaveLength(MAX_COLOR_PALETTE_SIZE);
  });

  it('appends new colors, keeps existing ones in place, and removes by value', () => {
    expect(withPaletteColor(['#111111'], '#222222')).toEqual(['#111111', '#222222']);
    expect(withPaletteColor(['#111111', '#222222'], '#111111')).toEqual(['#111111', '#222222']);
    expect(withoutPaletteColor(['#111111', '#222222'], '#111111')).toEqual(['#222222']);
  });

  it('salvages only parseable entries and refuses additions past the cap', () => {
    expect(readColorPalette({ [CANVAS_COLOR_PALETTE_KEY]: ['transparent', '#123', 12, '#ff000080'] })).toEqual([
      '#112233',
      '#ff0000',
    ]);
    const full = Array.from({ length: MAX_COLOR_PALETTE_SIZE }, (_, i) => `#${String(i).padStart(2, '0')}0000`);
    expect(withPaletteColor(full, '#abcdef')).toEqual(full);
    expect(withPaletteColor(['#111111'], 'garbage')).toEqual(['#111111']);
  });
});
