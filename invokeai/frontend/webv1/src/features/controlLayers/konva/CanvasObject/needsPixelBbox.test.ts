import type { AnyObjectState } from 'features/controlLayers/konva/CanvasObject/types';
import { describe, expect, it } from 'vitest';

import { objectStateNeedsPixelBbox } from './needsPixelBbox';

describe('objectStateNeedsPixelBbox', () => {
  it('requires pixel bbox for source-atop shapes and gradients', () => {
    const rect = {
      id: 'rect',
      type: 'rect',
      rect: { x: 0, y: 0, width: 10, height: 10 },
      color: { r: 255, g: 0, b: 0, a: 1 },
      compositeOperation: 'source-atop',
      clip: null,
    } satisfies AnyObjectState;
    const oval = {
      ...rect,
      id: 'oval',
      type: 'oval',
    } satisfies AnyObjectState;
    const polygon = {
      id: 'polygon',
      type: 'polygon',
      points: [0, 0, 10, 0, 10, 10],
      color: { r: 255, g: 0, b: 0, a: 1 },
      compositeOperation: 'source-atop',
    } satisfies AnyObjectState;
    const gradient = {
      id: 'gradient',
      type: 'gradient',
      gradientType: 'linear',
      rect: { x: 0, y: 0, width: 10, height: 10 },
      start: { x: 0, y: 0 },
      end: { x: 10, y: 10 },
      clipCenter: { x: 0, y: 0 },
      clipRadius: 1,
      clipAngle: 0,
      clipEnabled: true,
      bboxRect: { x: 0, y: 0, width: 10, height: 10 },
      fgColor: { r: 255, g: 0, b: 0, a: 1 },
      bgColor: { r: 0, g: 0, b: 255, a: 1 },
      globalCompositeOperation: 'source-atop',
    } satisfies AnyObjectState;

    expect(objectStateNeedsPixelBbox(rect)).toBe(true);
    expect(objectStateNeedsPixelBbox(oval)).toBe(true);
    expect(objectStateNeedsPixelBbox(polygon)).toBe(true);
    expect(objectStateNeedsPixelBbox(gradient)).toBe(true);
  });

  it('does not require pixel bbox for normal unclipped additive shapes', () => {
    const rect = {
      id: 'rect',
      type: 'rect',
      rect: { x: 0, y: 0, width: 10, height: 10 },
      color: { r: 255, g: 0, b: 0, a: 1 },
      compositeOperation: 'source-over',
      clip: null,
    } satisfies AnyObjectState;
    const polygon = {
      id: 'polygon',
      type: 'polygon',
      points: [0, 0, 10, 0, 10, 10],
      color: { r: 255, g: 0, b: 0, a: 1 },
      compositeOperation: 'source-over',
    } satisfies AnyObjectState;

    expect(objectStateNeedsPixelBbox(rect)).toBe(false);
    expect(objectStateNeedsPixelBbox(polygon)).toBe(false);
  });
});
