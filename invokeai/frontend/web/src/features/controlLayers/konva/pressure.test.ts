import { describe, expect, it } from 'vitest';

import {
  getPressureStrokeRenderBounds,
  getPressureStrokeRenderOps,
  getShouldUsePressureForBrush,
  getShouldUsePressureForEraser,
} from './pressure';

describe('pressure helpers', () => {
  it('uses pressure for brush when width or opacity is enabled', () => {
    expect(getShouldUsePressureForBrush(false, false)).toBe(false);
    expect(getShouldUsePressureForBrush(true, false)).toBe(true);
    expect(getShouldUsePressureForBrush(false, true)).toBe(true);
  });

  it('uses pressure for eraser only when width is enabled', () => {
    expect(getShouldUsePressureForEraser(false)).toBe(false);
    expect(getShouldUsePressureForEraser(true)).toBe(true);
  });

  it('builds fixed-width opacity-sensitive render ops', () => {
    const ops = getPressureStrokeRenderOps({
      points: [10, 20, 0.25, 30, 40, 0.75],
      strokeWidth: 40,
      color: { r: 255, g: 255, b: 255, a: 0.8 },
      pressureAffectsWidth: false,
      pressureAffectsOpacity: true,
    });

    expect(ops).toEqual([
      {
        type: 'segment',
        from: { x: 10, y: 20 },
        to: { x: 30, y: 40 },
        width: 40,
        color: { r: 255, g: 255, b: 255, a: 0.4 },
      },
    ]);
  });

  it('builds width-sensitive render ops when opacity is disabled', () => {
    const ops = getPressureStrokeRenderOps({
      points: [10, 20, 0.25, 30, 40, 0.75],
      strokeWidth: 40,
      color: { r: 255, g: 255, b: 255, a: 0.8 },
      pressureAffectsWidth: true,
      pressureAffectsOpacity: false,
    });

    expect(ops).toEqual([
      {
        type: 'segment',
        from: { x: 10, y: 20 },
        to: { x: 30, y: 40 },
        width: 20,
        color: { r: 255, g: 255, b: 255, a: 0.8 },
      },
    ]);
  });

  it('builds a pressure-scaled dot for single-point strokes', () => {
    const ops = getPressureStrokeRenderOps({
      points: [10, 20, 0.25],
      strokeWidth: 40,
      color: { r: 255, g: 255, b: 255, a: 0.8 },
      pressureAffectsWidth: true,
      pressureAffectsOpacity: true,
    });

    expect(ops).toEqual([
      {
        type: 'dot',
        x: 10,
        y: 20,
        radius: 5,
        color: { r: 255, g: 255, b: 255, a: 0.2 },
      },
    ]);
  });

  it('computes render bounds for opacity-pressure strokes', () => {
    const bounds = getPressureStrokeRenderBounds([
      {
        type: 'segment',
        from: { x: 10, y: 20 },
        to: { x: 30, y: 40 },
        width: 40,
        color: { r: 255, g: 255, b: 255, a: 0.4 },
      },
    ]);

    expect(bounds).toEqual({
      x: -12,
      y: -2,
      width: 64,
      height: 64,
    });
  });
});
