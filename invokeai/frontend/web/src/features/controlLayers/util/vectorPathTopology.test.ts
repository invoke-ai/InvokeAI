import type { CanvasBezierPathState } from 'features/controlLayers/store/types';
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

  it('connects distant endpoints through a new intermediate point', () => {
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
      { x: 20, y: 0 },
      { x: 30, y: 0 },
      { x: 40, y: 0 },
    ]);
    expect(result?.activePointIndex).toBe(3);
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
      { x: 20, y: 0 },
      { x: 40, y: 0 },
      { x: 30, y: 0 },
    ]);
  });

  it('closes distant endpoints of one path through an intermediate point', () => {
    const path = buildPath('path', [
      [0, 0],
      [20, 0],
      [20, 20],
    ]);

    const result = joinVectorPathEndpoints(path, 0, path, 2, false);

    expect(result?.path.isClosed).toBe(true);
    expect(result?.path.points.at(-1)?.anchor).toEqual({ x: 10, y: 10 });
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
