import { MAX_BRUSH_SIZE, MIN_BRUSH_SIZE } from '@workbench/canvas-engine/engineStores';
import { describe, expect, it } from 'vitest';

import { clampBrushSize, SIZE_STEP_FACTOR, stepBrushSize } from './paintConstants';

describe('clampBrushSize', () => {
  it('clamps to [MIN_BRUSH_SIZE, MAX_BRUSH_SIZE]', () => {
    expect(clampBrushSize(0)).toBe(MIN_BRUSH_SIZE);
    expect(clampBrushSize(-100)).toBe(MIN_BRUSH_SIZE);
    expect(clampBrushSize(1_000_000)).toBe(MAX_BRUSH_SIZE);
  });
});

describe('stepBrushSize', () => {
  it('grows by SIZE_STEP_FACTOR for direction +1', () => {
    expect(stepBrushSize(50, 1)).toBeCloseTo(50 * (1 + SIZE_STEP_FACTOR));
  });

  it('grows away from the new sub-pixel minimum', () => {
    expect(stepBrushSize(MIN_BRUSH_SIZE, 1)).toBe(0.11);
  });
});
