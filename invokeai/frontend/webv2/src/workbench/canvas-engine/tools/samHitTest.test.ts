import { identity, scale } from '@workbench/canvas-engine/math/mat2d';
import { describe, expect, it } from 'vitest';

import { clipPointToRect, clipRectToRect, moveSamBbox, rectFromPoints, resizeSamBbox, samHitTest } from './samHitTest';

const source = { height: 80, width: 100, x: 10, y: 20 };

describe('samHitTest', () => {
  it.each([1, 4])('keeps point and bbox handle hit regions screen-constant at %sx zoom', (zoom) => {
    const view = scale(identity(), zoom);
    const point = { x: 30, y: 40 };
    const bbox = { height: 30, width: 40, x: 40, y: 30 };

    expect(
      samHitTest({
        bbox,
        excludePoints: [],
        includePoints: [point],
        screenPoint: { x: point.x * zoom + 5, y: point.y * zoom },
        view,
      })
    ).toEqual({ index: 0, kind: 'point', label: 'include' });
    expect(
      samHitTest({
        bbox,
        excludePoints: [],
        includePoints: [],
        screenPoint: { x: bbox.x * zoom + 5, y: bbox.y * zoom },
        view,
      })
    ).toEqual({ handle: 'nw', kind: 'bbox-handle' });
  });

  it('prioritizes points, then handles, then the bbox body', () => {
    const bbox = { height: 30, width: 40, x: 40, y: 30 };
    const common = { bbox, excludePoints: [{ x: 40, y: 30 }], includePoints: [], view: identity() };

    expect(samHitTest({ ...common, screenPoint: { x: 40, y: 30 } })).toEqual({
      index: 0,
      kind: 'point',
      label: 'exclude',
    });
    expect(samHitTest({ ...common, excludePoints: [], screenPoint: { x: 80, y: 45 } })).toEqual({
      handle: 'e',
      kind: 'bbox-handle',
    });
    expect(samHitTest({ ...common, excludePoints: [], screenPoint: { x: 60, y: 45 } })).toEqual({
      kind: 'bbox-body',
    });
    expect(samHitTest({ ...common, excludePoints: [], screenPoint: { x: 95, y: 70 } })).toBeNull();
  });
});

describe('SAM bbox geometry', () => {
  it('normalizes dragged corners and clips to the source export bounds', () => {
    expect(rectFromPoints({ x: 120, y: 120 }, { x: 0, y: 0 }, source)).toEqual(source);
    expect(clipRectToRect({ height: 30, width: 30, x: 100, y: 80 }, source)).toEqual({
      height: 20,
      width: 10,
      x: 100,
      y: 80,
    });
  });

  it('clips points and bbox moves without changing bbox size', () => {
    expect(clipPointToRect({ x: -5, y: 200 }, source)).toEqual({ x: 10, y: 99 });
    expect(moveSamBbox({ height: 20, width: 30, x: 20, y: 30 }, 500, -500, source)).toEqual({
      height: 20,
      width: 30,
      x: 80,
      y: 20,
    });
  });

  it('resizes from screen-constant handles and clamps to source bounds and one document pixel', () => {
    expect(
      resizeSamBbox({
        bounds: source,
        delta: { x: 100, y: 100 },
        handle: 'nw',
        start: { height: 30, width: 40, x: 40, y: 30 },
      })
    ).toEqual({ height: 1, width: 1, x: 79, y: 59 });
    expect(
      resizeSamBbox({
        bounds: source,
        delta: { x: 100, y: 100 },
        handle: 'se',
        start: { height: 30, width: 40, x: 40, y: 30 },
      })
    ).toEqual({ height: 70, width: 70, x: 40, y: 30 });
  });
});
