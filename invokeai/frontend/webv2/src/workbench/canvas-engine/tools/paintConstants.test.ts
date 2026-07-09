import { MAX_BRUSH_SIZE, MIN_BRUSH_SIZE } from '@workbench/canvas-engine/engineStores';
import { describe, expect, it } from 'vitest';

import { clampBrushSize, SIZE_STEP_FACTOR, stepBrushSize } from './paintConstants';

describe('clampBrushSize', () => {
  it('clamps to [MIN_BRUSH_SIZE, MAX_BRUSH_SIZE]', () => {
    expect(clampBrushSize(0)).toBe(MIN_BRUSH_SIZE);
    expect(clampBrushSize(-100)).toBe(MIN_BRUSH_SIZE);
    expect(clampBrushSize(1_000_000)).toBe(MAX_BRUSH_SIZE);
  });

  it('rounds to the nearest integer', () => {
    expect(clampBrushSize(50.6)).toBe(51);
    expect(clampBrushSize(50.4)).toBe(50);
  });
});

describe('stepBrushSize', () => {
  it('grows by SIZE_STEP_FACTOR for direction +1', () => {
    expect(stepBrushSize(50, 1)).toBe(Math.round(50 * (1 + SIZE_STEP_FACTOR)));
  });

  it('shrinks by SIZE_STEP_FACTOR for direction -1', () => {
    expect(stepBrushSize(50, -1)).toBe(Math.round(50 * (1 - SIZE_STEP_FACTOR)));
  });

  it('clamps growth at MAX_BRUSH_SIZE', () => {
    expect(stepBrushSize(MAX_BRUSH_SIZE, 1)).toBe(MAX_BRUSH_SIZE);
  });

  it('clamps shrink at MIN_BRUSH_SIZE', () => {
    expect(stepBrushSize(MIN_BRUSH_SIZE, -1)).toBe(MIN_BRUSH_SIZE);
  });

  it('is monotonic: repeated growth never decreases size', () => {
    let size = MIN_BRUSH_SIZE;
    for (let i = 0; i < 20; i++) {
      const next = stepBrushSize(size, 1);
      expect(next).toBeGreaterThanOrEqual(size);
      size = next;
    }
  });
});
