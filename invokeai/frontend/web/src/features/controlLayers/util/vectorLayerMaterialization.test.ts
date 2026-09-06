import type { CanvasBezierPathState, CanvasBezierPointState, RgbaColor } from 'features/controlLayers/store/types';
import {
  buildClosedPathLassoObjects,
  buildClosedPathPolygonObjects,
  isFillableBezierPath,
} from 'features/controlLayers/util/vectorLayerMaterialization';
import { describe, expect, it } from 'vitest';

const makePoint = (x: number, y: number): CanvasBezierPointState => ({
  anchor: { x, y },
  inHandle: null,
  outHandle: null,
  type: 'corner',
});

const makePath = (id: string, isClosed: boolean, points: CanvasBezierPointState[]): CanvasBezierPathState => ({
  id,
  name: null,
  isClosed,
  points,
});

const closedPath = makePath('closed-path', true, [makePoint(0, 0), makePoint(20, 0), makePoint(10, 20)]);
const openPath = makePath('open-path', false, [makePoint(0, 0), makePoint(20, 0), makePoint(10, 20)]);
const invalidClosedPath = makePath('invalid-closed-path', true, [makePoint(0, 0), makePoint(20, 0)]);
const color: RgbaColor = { r: 12, g: 34, b: 56, a: 0.75 };

describe('vector layer materialization', () => {
  it('only considers closed paths with at least three points fillable', () => {
    expect(isFillableBezierPath(closedPath)).toBe(true);
    expect(isFillableBezierPath(openPath)).toBe(false);
    expect(isFillableBezierPath(invalidClosedPath)).toBe(false);
  });

  it('builds filled polygon objects only for fillable paths', () => {
    const objects = buildClosedPathPolygonObjects([openPath, invalidClosedPath, closedPath], color);

    expect(objects).toHaveLength(1);
    expect(objects[0]).toMatchObject({
      type: 'polygon',
      color,
      compositeOperation: 'source-over',
    });
    expect(objects[0]?.points.length).toBeGreaterThan(6);
    expect(objects[0]?.points.slice(0, 2)).toEqual(objects[0]?.points.slice(-2));
  });

  it('builds opaque lasso geometry only for fillable paths', () => {
    const objects = buildClosedPathLassoObjects([closedPath, openPath, invalidClosedPath]);

    expect(objects).toHaveLength(1);
    expect(objects[0]).toMatchObject({
      type: 'lasso',
      compositeOperation: 'source-over',
    });
    expect(objects[0]?.points.length).toBeGreaterThan(6);
    expect(objects[0]?.points.slice(0, 2)).toEqual(objects[0]?.points.slice(-2));
  });
});
