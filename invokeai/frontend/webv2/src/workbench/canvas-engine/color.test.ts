import { describe, expect, it } from 'vitest';

import { rgbaToHex } from './color';

describe('rgbaToHex', () => {
  it('formats channels as a lowercase #rrggbb string', () => {
    expect(rgbaToHex(0, 0, 0)).toBe('#000000');
    expect(rgbaToHex(255, 255, 255)).toBe('#ffffff');
    expect(rgbaToHex(255, 0, 0)).toBe('#ff0000');
    expect(rgbaToHex(18, 52, 86)).toBe('#123456');
  });

  it('clamps out-of-range channels to [0, 255]', () => {
    expect(rgbaToHex(-10, 300, 128)).toBe('#00ff80');
  });
});
