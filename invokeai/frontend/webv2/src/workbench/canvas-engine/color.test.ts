import { describe, expect, it } from 'vitest';

import { rgbaToHex } from './color';

describe('rgbaToHex', () => {
  it('formats channels as a lowercase #rrggbb string', () => {
    expect(rgbaToHex(0, 0, 0)).toBe('#000000');
    expect(rgbaToHex(255, 255, 255)).toBe('#ffffff');
    expect(rgbaToHex(255, 0, 0)).toBe('#ff0000');
    expect(rgbaToHex(18, 52, 86)).toBe('#123456');
  });

  it('pads single-digit hex bytes with a leading zero', () => {
    expect(rgbaToHex(1, 2, 3)).toBe('#010203');
  });

  it('rounds fractional channels to the nearest integer', () => {
    expect(rgbaToHex(127.6, 127.4, 0)).toBe('#807f00');
  });

  it('clamps out-of-range channels to [0, 255]', () => {
    expect(rgbaToHex(-10, 300, 128)).toBe('#00ff80');
  });
});
