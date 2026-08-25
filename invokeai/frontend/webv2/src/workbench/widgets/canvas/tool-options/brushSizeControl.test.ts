import { MIN_BRUSH_SIZE } from '@workbench/canvas-engine/api';
import { describe, expect, it } from 'vitest';

import {
  BRUSH_SIZE_SLIDER_MAX,
  BRUSH_SIZE_SLIDER_MAX_SIZE,
  BRUSH_SIZE_SLIDER_MIN,
  BRUSH_SIZE_SLIDER_STEP,
  brushSizeToSliderPosition,
  formatBrushSize,
  getBrushSizeKeyboardStep,
  sliderPositionToBrushSize,
} from './BrushOptions';

describe('brush size control', () => {
  it('maps exact logarithmic slider boundaries', () => {
    expect(brushSizeToSliderPosition(MIN_BRUSH_SIZE)).toBe(BRUSH_SIZE_SLIDER_MIN);
    expect(brushSizeToSliderPosition(BRUSH_SIZE_SLIDER_MAX_SIZE)).toBe(BRUSH_SIZE_SLIDER_MAX);
    expect(sliderPositionToBrushSize(BRUSH_SIZE_SLIDER_MIN)).toBe(MIN_BRUSH_SIZE);
    expect(sliderPositionToBrushSize(BRUSH_SIZE_SLIDER_MAX)).toBe(BRUSH_SIZE_SLIDER_MAX_SIZE);
  });

  it('round-trips representative sub-pixel and normal sizes', () => {
    for (const size of [0.1, 0.25, 0.5, 1, 5, 50, 600]) {
      expect(sliderPositionToBrushSize(brushSizeToSliderPosition(size))).toBeCloseTo(size, 2);
    }
  });

  it('gives the 0.1px–1px range meaningful slider travel', () => {
    expect(brushSizeToSliderPosition(1) - brushSizeToSliderPosition(0.1)).toBeGreaterThan(200);
  });

  it('uses fine pointer steps while preserving the exact slider maximum', () => {
    expect(BRUSH_SIZE_SLIDER_MAX % BRUSH_SIZE_SLIDER_STEP).toBe(0);
    expect(sliderPositionToBrushSize(BRUSH_SIZE_SLIDER_MAX)).toBe(BRUSH_SIZE_SLIDER_MAX_SIZE);
  });

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
