import { describe, expect, it } from 'vitest';

import { panBy, pinchZoomAtPoints, wheelZoomAtPoint, zoomAtPoint } from './panZoom';

const clamp = (min: number, max: number) => (zoom: number) => Math.max(min, Math.min(max, zoom));

describe('panZoom', () => {
  it('keeps the anchored content point fixed when zoom changes', () => {
    const initial = { pan: { x: 30, y: -12 }, zoom: 1.5 };
    const anchor = { x: 200, y: 120 };
    const contentPoint = {
      x: (anchor.x - initial.pan.x) / initial.zoom,
      y: (anchor.y - initial.pan.y) / initial.zoom,
    };

    const next = zoomAtPoint(initial, 3, anchor, clamp(0.1, 20));

    expect(next.zoom).toBe(3);
    expect(next.zoom * contentPoint.x + next.pan.x).toBeCloseTo(anchor.x, 6);
    expect(next.zoom * contentPoint.y + next.pan.y).toBeCloseTo(anchor.y, 6);
  });

  it('applies the zoom constraint before calculating anchor translation', () => {
    const atMaximum = { pan: { x: -700, y: -525 }, zoom: 8 };

    const next = zoomAtPoint(atMaximum, 9.5, { x: 200, y: 150 }, clamp(1, 8));

    expect(next).toBe(atMaximum);
  });

  it('applies wheel sensitivity through the constrained transition', () => {
    const next = wheelZoomAtPoint(
      { pan: { x: 0, y: 0 }, zoom: 1 },
      -1,
      { x: 100, y: 50 },
      {
        constrainZoom: clamp(0.1, 20),
      }
    );

    // A one-unit wheel delta is a small continuous step, not a jump to a preset.
    expect(next.zoom).toBeGreaterThan(1);
    expect(next.zoom).toBeLessThan(1.01);
  });

  it('pans in screen space without changing zoom', () => {
    expect(panBy({ pan: { x: 5, y: 5 }, zoom: 2 }, { x: 10, y: -4 })).toEqual({
      pan: { x: 15, y: 1 },
      zoom: 2,
    });
  });

  it('scales a pinch by how far the fingers spread and follows their midpoint', () => {
    const start = { pan: { x: 0, y: 0 }, zoom: 1 };

    const next = pinchZoomAtPoints(
      start,
      { center: { x: 140, y: 60 }, distance: 200, startCenter: { x: 100, y: 50 }, startDistance: 100 },
      clamp(1, 8)
    );

    expect(next.zoom).toBe(2);
    expect(next.zoom * 100 + next.pan.x).toBeCloseTo(140, 6);
    expect(next.zoom * 50 + next.pan.y).toBeCloseTo(60, 6);
  });

  it('keeps panning with the fingers after the pinch is clamped at its zoom limit', () => {
    const start = { pan: { x: -100, y: -50 }, zoom: 4 };
    const gesture = { center: { x: 180, y: 90 }, startCenter: { x: 100, y: 50 }, startDistance: 100 };

    const atLimit = pinchZoomAtPoints(start, { ...gesture, distance: 200 }, clamp(1, 8));
    const pastLimit = pinchZoomAtPoints(start, { ...gesture, distance: 400 }, clamp(1, 8));

    expect(atLimit.zoom).toBe(8);
    expect(pastLimit.zoom).toBe(8);
    // Clamping zoom must still allow midpoint translation.
    expect(pastLimit.pan).toEqual(atLimit.pan);
    expect(pastLimit.zoom * 50 + pastLimit.pan.x).toBeCloseTo(180, 6);
  });

  it('leaves the transform untouched when the pinch has no measurable start', () => {
    const start = { pan: { x: 3, y: 7 }, zoom: 2 };

    expect(
      pinchZoomAtPoints(
        start,
        { center: { x: 10, y: 10 }, distance: 50, startCenter: { x: 0, y: 0 }, startDistance: 0 },
        clamp(1, 8)
      )
    ).toBe(start);
  });
});
