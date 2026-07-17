import { describe, expect, it } from 'vitest';

import { panBy, wheelZoomAtPoint, zoomAtPoint } from './panZoom';

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

  it('applies wheel sensitivity and optional snapping through the same constrained transition', () => {
    const next = wheelZoomAtPoint(
      { pan: { x: 0, y: 0 }, zoom: 1 },
      -1,
      { x: 100, y: 50 },
      {
        constrainZoom: clamp(0.1, 20),
        snapZoom: () => 1,
      }
    );

    expect(next).toEqual({ pan: { x: 0, y: 0 }, zoom: 1 });
  });

  it('pans in screen space without changing zoom', () => {
    expect(panBy({ pan: { x: 5, y: 5 }, zoom: 2 }, { x: 10, y: -4 })).toEqual({
      pan: { x: 15, y: 1 },
      zoom: 2,
    });
  });
});
