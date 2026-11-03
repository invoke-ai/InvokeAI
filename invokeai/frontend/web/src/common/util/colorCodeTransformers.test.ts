import { describe, expect, it } from 'vitest';

import { hexToRGBA, rgbaToHex } from './colorCodeTransformers';

describe('hexToRGBA', () => {
  it('parses a six-digit hex and uses the alpha passed in', () => {
    expect(hexToRGBA('#ff8000', 0.5)).toEqual({ r: 255, g: 128, b: 0, a: 0.5 });
    expect(hexToRGBA('ff8000', 1)).toEqual({ r: 255, g: 128, b: 0, a: 1 });
  });

  it('keeps the alpha of an eight-digit hex', () => {
    expect(hexToRGBA('#ff000080', 1)).toEqual({ r: 255, g: 0, b: 0, a: 128 / 255 });
    expect(hexToRGBA('#00ff0000', 1)).toEqual({ r: 0, g: 255, b: 0, a: 0 });
  });

  it('expands the three- and four-digit shorthands', () => {
    expect(hexToRGBA('#f80', 1)).toEqual({ r: 255, g: 136, b: 0, a: 1 });
    expect(hexToRGBA('#f808', 1)).toEqual({ r: 255, g: 136, b: 0, a: 136 / 255 });
  });

  it('round-trips the value rgbaToHex shows in the hex input', () => {
    const color = { r: 12, g: 34, b: 56, a: 0.4 };
    const parsed = hexToRGBA(rgbaToHex(color, true), 1);
    expect(parsed).toMatchObject({ r: 12, g: 34, b: 56 });
    expect(Math.round(parsed.a * 255)).toBe(Math.round(color.a * 255));
  });
});
