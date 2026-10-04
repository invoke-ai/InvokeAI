import type { CanvasBezierPathState, CanvasBezierPointState } from 'features/controlLayers/store/types';
import { describe, expect, it } from 'vitest';

import {
  canJoinVectorPathEndpoints,
  canSplitVectorPathAtPoints,
  deleteVectorPathPoints,
  joinVectorPathEndpoints,
  splitVectorPathAtPoint,
  splitVectorPathAtPoints,
} from './vectorPathTopology';

const buildPath = (id: string, anchors: Array<[number, number]>, isClosed = false): CanvasBezierPathState => ({
  id,
  name: null,
  isClosed,
  points: anchors.map(([x, y]) => ({
    anchor: { x, y },
    inHandle: null,
    outHandle: null,
    type: 'corner',
  })),
});

describe('vector path topology', () => {
  const expectConstrainedHandles = (point: CanvasBezierPointState) => {
    expect(point.inHandle).not.toBeNull();
    expect(point.outHandle).not.toBeNull();
    const incoming = { x: point.inHandle!.x - point.anchor.x, y: point.inHandle!.y - point.anchor.y };
    const outgoing = { x: point.outHandle!.x - point.anchor.x, y: point.outHandle!.y - point.anchor.y };
    expect(incoming.x * outgoing.y - incoming.y * outgoing.x).toBeCloseTo(0);
    expect(incoming.x * outgoing.x + incoming.y * outgoing.y).toBeLessThan(0);
    if (point.type === 'symmetric') {
      expect(incoming.x).toBeCloseTo(-outgoing.x);
      expect(incoming.y).toBeCloseTo(-outgoing.y);
    }
  };

  describe.each(['symmetric', 'smooth'] as const)('%s topology', (type) => {
    it.each([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ])('restores joining handles when connecting endpoint %s to %s', (sourceIndex, targetIndex) => {
      const source = buildPath('source', [
        [0, 0],
        [30, 10],
      ]);
      const target = buildPath('target', [
        [100, 50],
        [150, 0],
      ]);
      const sourcePoint = source.points[sourceIndex!]!;
      const targetPoint = target.points[targetIndex!]!;
      sourcePoint.type = type;
      targetPoint.type = type;
      const sourceHandle = { x: sourcePoint.anchor.x - 8, y: sourcePoint.anchor.y - 4 };
      const targetHandle = { x: targetPoint.anchor.x + 12, y: targetPoint.anchor.y - 6 };
      sourcePoint[sourceIndex === 0 ? 'outHandle' : 'inHandle'] = sourceHandle;
      targetPoint[targetIndex === 0 ? 'outHandle' : 'inHandle'] = targetHandle;

      const joined = joinVectorPathEndpoints(source, sourceIndex!, target, targetIndex!, false)!.path;
      const sourceEnd = joined.points[1]!;
      const targetStart = joined.points[2]!;
      expect(sourceEnd.type).toBe(type);
      expect(targetStart.type).toBe(type);
      expect(sourceEnd.inHandle).toEqual(sourceHandle);
      expect(targetStart.outHandle).toEqual(targetHandle);
      expectConstrainedHandles(sourceEnd);
      expectConstrainedHandles(targetStart);
      expect(sourcePoint[sourceIndex === 0 ? 'inHandle' : 'outHandle']).toBeNull();
      expect(targetPoint[targetIndex === 0 ? 'inHandle' : 'outHandle']).toBeNull();

      const deleted = deleteVectorPathPoints([joined], [{ pathId: joined.id, pointIndex: 2 }]);
      expectConstrainedHandles(deleted.paths[0]!.points[1]!);
    });

    it('restores both tangents when closing a path without welding', () => {
      const path = buildPath('path', [
        [0, 0],
        [100, 0],
        [100, 100],
      ]);
      path.points[0]!.type = type;
      path.points[0]!.outHandle = { x: 10, y: 5 };
      path.points[2]!.type = type;
      path.points[2]!.inHandle = { x: 100, y: 80 };
      const result = joinVectorPathEndpoints(path, 0, path, 2, false)!.path;
      expect(result.isClosed).toBe(true);
      expectConstrainedHandles(result.points[0]!);
      expectConstrainedHandles(result.points[2]!);
    });

    it('repairs the surviving neighbour of a deleted point in previously joined geometry', () => {
      const path = buildPath('path', [
        [0, 0],
        [30, 0],
        [60, 30],
        [90, 50],
        [120, 0],
      ]);
      path.points[1]!.type = type;
      path.points[1]!.inHandle = { x: 20, y: -5 };
      path.points[3]!.type = type;
      path.points[3]!.outHandle = { x: 100, y: 45 };
      const result = deleteVectorPathPoints([path], [{ pathId: path.id, pointIndex: 2 }]).paths[0]!;
      expectConstrainedHandles(result.points[1]!);
      expectConstrainedHandles(result.points[2]!);
      expect(result.points[1]!.inHandle).toEqual(path.points[1]!.inHandle);
      expect(result.points[2]!.outHandle).toEqual(path.points[3]!.outHandle);
      expect(result.points[0]).toEqual(path.points[0]);
      expect(path.points[1]!.outHandle).toBeNull();
    });
  });

  it.each(['corner', 'smooth', 'symmetric'] as const)('keeps the target type %s when welding paths', (type) => {
    const source = buildPath('source', [
      [0, 0],
      [20, 0],
    ]);
    const target = buildPath('target', [
      [22, 0],
      [50, 30],
    ]);
    source.points[1]!.inHandle = { x: 14, y: 0 };
    target.points[0]!.type = type;
    target.points[0]!.outHandle = { x: 22, y: 10 };
    const point = joinVectorPathEndpoints(source, 1, target, 0, true)!.path.points[1]!;
    expect(point.type).toBe(type);
    expect(point.anchor).toEqual(target.points[0]!.anchor);
    expect(point.outHandle).toEqual(target.points[0]!.outHandle);
    if (type === 'corner') {
      expect(point.inHandle).toEqual({ x: 16, y: 0 });
    } else {
      expectConstrainedHandles(point);
      if (type === 'smooth') {
        expect(Math.hypot(point.inHandle!.x - point.anchor.x, point.inHandle!.y - point.anchor.y)).toBeCloseTo(6);
      }
    }
  });

  it.each([0, 3])('keeps the last-selected tangent when welding one path onto endpoint %s', (targetIndex) => {
    const path = buildPath('path', [
      [0, 0],
      [100, 0],
      [100, 100],
      [1, 1],
    ]);
    path.points[0]!.type = 'symmetric';
    path.points[0]!.outHandle = { x: 10, y: 0 };
    path.points[3]!.type = 'symmetric';
    path.points[3]!.inHandle = { x: 1, y: 21 };
    const result = joinVectorPathEndpoints(path, 3 - targetIndex, path, targetIndex, true)!.path;
    const point = result.points[0]!;
    expect(point.type).toBe('symmetric');
    expectConstrainedHandles(point);
    const retainedHandle = targetIndex === 0 ? 'outHandle' : 'inHandle';
    expect(point[retainedHandle]).toEqual(path.points[targetIndex]![retainedHandle]);
  });

  it('preserves unequal Smooth handle lengths on a direct join', () => {
    const source = buildPath('source', [
      [0, 0],
      [20, 0],
    ]);
    const target = buildPath('target', [
      [50, 30],
      [80, 0],
    ]);
    source.points[1] = {
      anchor: { x: 20, y: 0 },
      type: 'smooth',
      inHandle: { x: 14, y: 0 },
      outHandle: { x: 32, y: 0 },
    };
    const result = joinVectorPathEndpoints(source, 1, target, 0, false)!.path;
    expect(result.points[1]).toEqual(source.points[1]);
  });

  it('normalizes closed-path neighbours across the seam after multiple deletions', () => {
    const path = buildPath(
      'path',
      [
        [0, 0],
        [30, 0],
        [60, 30],
        [30, 60],
        [0, 30],
      ],
      true
    );
    path.points[1]!.type = 'symmetric';
    path.points[1]!.outHandle = { x: 40, y: 5 };
    path.points[3]!.type = 'symmetric';
    path.points[3]!.inHandle = { x: 40, y: 55 };
    const result = deleteVectorPathPoints(
      [path],
      [
        { pathId: path.id, pointIndex: 0 },
        { pathId: path.id, pointIndex: 4 },
      ]
    ).paths[0]!;
    expect(result.isClosed).toBe(true);
    expectConstrainedHandles(result.points[0]!);
    expectConstrainedHandles(result.points[2]!);
  });

  it('deletes a two-point path when either point is deleted', () => {
    const path = buildPath('path', [
      [0, 0],
      [10, 0],
    ]);

    const result = deleteVectorPathPoints([path], [{ pathId: 'path', pointIndex: 0 }]);

    expect(result).toEqual({ paths: [], didDelete: true });
  });

  it('deletes a path when all of its points are selected', () => {
    const path = buildPath('path', [
      [0, 0],
      [10, 0],
      [20, 0],
    ]);

    const result = deleteVectorPathPoints(
      [path],
      path.points.map((_, pointIndex) => ({ pathId: path.id, pointIndex }))
    );

    expect(result).toEqual({ paths: [], didDelete: true });
  });

  it('deletes selected points when the remaining path is valid', () => {
    const path = buildPath('path', [
      [0, 0],
      [10, 0],
      [20, 0],
    ]);

    const result = deleteVectorPathPoints([path], [{ pathId: 'path', pointIndex: 1 }]);

    expect(result.paths[0]?.points.map((point) => point.anchor)).toEqual([
      { x: 0, y: 0 },
      { x: 20, y: 0 },
    ]);
    expect(result.didDelete).toBe(true);
    expect(path.points).toHaveLength(3);
  });

  it('allows splitting internal points of open paths and any point of closed paths', () => {
    const openPath = buildPath('open', [
      [0, 0],
      [10, 0],
      [20, 0],
    ]);
    const closedPath = buildPath(
      'closed',
      [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      true
    );

    expect(canSplitVectorPathAtPoints([openPath], [{ pathId: 'open', pointIndex: 1 }])).toBe(true);
    expect(canSplitVectorPathAtPoints([openPath], [{ pathId: 'open', pointIndex: 0 }])).toBe(false);
    expect(canSplitVectorPathAtPoints([closedPath], [{ pathId: 'closed', pointIndex: 0 }])).toBe(true);
  });

  it('allows joining exactly two endpoints of open paths', () => {
    const firstPath = buildPath('first', [
      [0, 0],
      [10, 0],
    ]);
    const secondPath = buildPath('second', [
      [20, 0],
      [30, 0],
    ]);

    expect(
      canJoinVectorPathEndpoints(
        [firstPath, secondPath],
        [
          { pathId: 'first', pointIndex: 1 },
          { pathId: 'second', pointIndex: 0 },
        ],
        12
      )
    ).toBe(true);
    expect(
      canJoinVectorPathEndpoints(
        [firstPath, secondPath],
        [
          { pathId: 'first', pointIndex: 0 },
          { pathId: 'first', pointIndex: 1 },
          { pathId: 'second', pointIndex: 0 },
        ],
        12
      )
    ).toBe(false);
  });

  it('opens a closed path at the selected point without changing its segment order', () => {
    const path = buildPath(
      'path',
      [
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
      ],
      true
    );

    const result = splitVectorPathAtPoint(path, 2, 'unused');

    expect(result?.paths).toHaveLength(1);
    expect(result?.paths[0]?.isClosed).toBe(false);
    expect(result?.paths[0]?.points.map((point) => point.anchor)).toEqual([
      { x: 10, y: 10 },
      { x: 0, y: 10 },
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
    ]);
  });

  it('splits an open path into two paths with duplicated endpoints', () => {
    const path = buildPath('path', [
      [0, 0],
      [10, 0],
      [20, 0],
    ]);

    const result = splitVectorPathAtPoint(path, 1, 'new-path');

    expect(result?.paths.map((candidate) => candidate.id)).toEqual(['path', 'new-path']);
    expect(result?.paths[0]?.points).toHaveLength(2);
    expect(result?.paths[1]?.points).toHaveLength(2);
    expect(result?.activePathId).toBe('new-path');
  });

  it('splits a closed path into open arcs at every selected point', () => {
    const path = buildPath(
      'path',
      [
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
      ],
      true
    );
    let nextId = 0;

    const result = splitVectorPathAtPoints(path, [0, 2], () => `new-path-${nextId++}`);

    expect(result?.paths.map((candidate) => candidate.points.map((point) => point.anchor))).toEqual([
      [
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 10, y: 10 },
      ],
      [
        { x: 10, y: 10 },
        { x: 0, y: 10 },
        { x: 0, y: 0 },
      ],
    ]);
    expect(result?.paths.every((candidate) => !candidate.isClosed)).toBe(true);
    expect(result?.activePathId).toBe('new-path-0');
  });

  it('splits an open path at every selected internal point', () => {
    const path = buildPath('path', [
      [0, 0],
      [10, 0],
      [20, 0],
      [30, 0],
      [40, 0],
    ]);
    let nextId = 0;

    const result = splitVectorPathAtPoints(path, [1, 3], () => `new-path-${nextId++}`);

    expect(result?.paths.map((candidate) => candidate.points.map((point) => point.anchor.x))).toEqual([
      [0, 10],
      [10, 20, 30],
      [30, 40],
    ]);
    expect(result?.activePathId).toBe('new-path-1');
  });

  it('welds nearby endpoints at the last selected endpoint', () => {
    const source = buildPath('source', [
      [0, 0],
      [10, 0],
    ]);
    const target = buildPath('target', [
      [12, 0],
      [20, 0],
    ]);

    const result = joinVectorPathEndpoints(source, 1, target, 0, true);

    expect(result?.path.points.map((point) => point.anchor)).toEqual([
      { x: 0, y: 0 },
      { x: 12, y: 0 },
      { x: 20, y: 0 },
    ]);
    expect(result?.activePointIndex).toBe(1);
  });

  it('connects distant endpoints directly without adding a point', () => {
    const source = buildPath('source', [
      [0, 0],
      [10, 0],
    ]);
    const target = buildPath('target', [
      [30, 0],
      [40, 0],
    ]);

    const result = joinVectorPathEndpoints(source, 1, target, 0, false);

    expect(result?.path.points.map((point) => point.anchor)).toEqual([
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 30, y: 0 },
      { x: 40, y: 0 },
    ]);
    expect(result?.activePointIndex).toBe(2);
  });

  it('reverses paths as needed before joining their selected endpoints', () => {
    const source = buildPath('source', [
      [0, 0],
      [10, 0],
    ]);
    const target = buildPath('target', [
      [30, 0],
      [40, 0],
    ]);

    const result = joinVectorPathEndpoints(source, 0, target, 1, false);

    expect(result?.path.points.map((point) => point.anchor)).toEqual([
      { x: 10, y: 0 },
      { x: 0, y: 0 },
      { x: 40, y: 0 },
      { x: 30, y: 0 },
    ]);
    expect(result?.activePointIndex).toBe(2);
  });

  it('closes distant endpoints of one path without adding a point', () => {
    const path = buildPath('path', [
      [0, 0],
      [20, 0],
      [20, 20],
    ]);

    const result = joinVectorPathEndpoints(path, 0, path, 2, false);

    expect(result?.path.isClosed).toBe(true);
    expect(result?.path.points).toEqual(path.points);
    expect(result?.activePointIndex).toBe(2);
  });

  it('welds nearby endpoints of one path at the last selected point', () => {
    const path = buildPath('path', [
      [0, 0],
      [20, 0],
      [20, 20],
      [1, 1],
    ]);

    const result = joinVectorPathEndpoints(path, 0, path, 3, true);

    expect(result?.path.isClosed).toBe(true);
    expect(result?.path.points).toHaveLength(3);
    expect(result?.path.points[0]?.anchor).toEqual({ x: 1, y: 1 });
    expect(result?.activePointIndex).toBe(0);
  });
});
