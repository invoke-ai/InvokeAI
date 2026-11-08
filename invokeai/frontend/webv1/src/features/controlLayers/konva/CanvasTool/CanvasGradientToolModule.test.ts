import { describe, expect, it } from 'vitest';

import { buildGradientBufferState } from './gradientBufferState';

describe('CanvasGradientToolModule', () => {
  it('preserves source-atop for locked linear gradients', () => {
    const gradient = buildGradientBufferState({
      id: 'gradient:test',
      gradientType: 'linear',
      rect: { x: 0, y: 0, width: 128, height: 64 },
      start: { x: 10, y: 20 },
      end: { x: 10, y: 20 },
      clipCenter: { x: 10, y: 20 },
      clipRadius: 0,
      clipAngle: 0,
      clipEnabled: true,
      bboxRect: { x: 0, y: 0, width: 128, height: 64 },
      fgColor: { r: 255, g: 0, b: 0, a: 1 },
      bgColor: { r: 0, g: 0, b: 255, a: 0 },
      globalCompositeOperation: 'source-atop',
    });

    expect(gradient.globalCompositeOperation).toBe('source-atop');
    if (gradient.gradientType !== 'linear') {
      throw new Error('Expected a linear gradient');
    }
    expect(gradient.end).toEqual({ x: 11, y: 20 });
    expect(gradient.clipRadius).toBe(1);
  });

  it('preserves source-atop for locked radial gradients', () => {
    const gradient = buildGradientBufferState({
      id: 'gradient:test',
      gradientType: 'radial',
      rect: { x: 0, y: 0, width: 128, height: 64 },
      center: { x: 32, y: 24 },
      radius: 0,
      clipCenter: { x: 32, y: 24 },
      clipRadius: 0,
      clipEnabled: false,
      bboxRect: { x: 0, y: 0, width: 128, height: 64 },
      fgColor: { r: 255, g: 0, b: 0, a: 1 },
      bgColor: { r: 0, g: 0, b: 255, a: 0 },
      globalCompositeOperation: 'source-atop',
    });

    expect(gradient.globalCompositeOperation).toBe('source-atop');
    if (gradient.gradientType !== 'radial') {
      throw new Error('Expected a radial gradient');
    }
    expect(gradient.radius).toBe(1);
    expect(gradient.clipRadius).toBe(1);
  });
});
