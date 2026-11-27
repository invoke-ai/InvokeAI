import { describe, expect, it } from 'vitest';

import { calculateLayerPosition, computeAlignedX } from './textRenderer';

describe('text alignment helpers', () => {
  it('computes x offsets for different alignments', () => {
    expect(computeAlignedX(50, 100, 'left', 4)).toBe(4);
    expect(computeAlignedX(50, 100, 'center', 4)).toBe(4 + (100 - 50) / 2);
    expect(computeAlignedX(50, 100, 'right', 4)).toBe(4 + (100 - 50));
  });

  it('calculates layer positions relative to anchor', () => {
    const anchor = { x: 200, y: 300 };
    expect(calculateLayerPosition(anchor, 'left', 100, 4)).toEqual({ x: 196, y: 296 });
    expect(calculateLayerPosition(anchor, 'center', 100, 4)).toEqual({ x: 146, y: 296 });
    expect(calculateLayerPosition(anchor, 'right', 100, 4)).toEqual({ x: 96, y: 296 });
  });

  it('uses top anchor regardless of line height', () => {
    const anchor = { x: 200, y: 300 };
    expect(calculateLayerPosition(anchor, 'left', 100, 4)).toEqual({ x: 196, y: 296 });
  });
});
