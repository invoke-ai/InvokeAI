import { describe, expect, it } from 'vitest';

import { formatBrushSize, getBrushSizeKeyboardStep } from './BrushOptions';

describe('brush size control', () => {
  it.each([
    [0.1, 1, 0.01],
    [0.5, -1, 0.01],
    [1, -1, 0.01],
    [1, 1, 0.1],
    [10, -1, 0.1],
    [10, 1, 1],
    [100, -1, 1],
    [100, 1, 10],
  ] as const)(
    'uses a reversible human-sized keyboard step at %fpx in direction %i',
    (size, direction, expectedStep) => {
      expect(getBrushSizeKeyboardStep(size, direction)).toBe(expectedStep);
    }
  );

  it('formats fractional sizes without hiding precision or trailing zeroes', () => {
    expect(formatBrushSize(0.1)).toBe('0.1');
    expect(formatBrushSize(0.25)).toBe('0.25');
    expect(formatBrushSize(50)).toBe('50');
  });
});
