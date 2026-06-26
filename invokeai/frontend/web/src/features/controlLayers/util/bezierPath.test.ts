import { describe, expect, it } from 'vitest';

import { evaluateBezierSegment, findNearestBezierPathSegment, splitBezierSegmentAt } from './bezierPath';

describe('bezierPath utilities', () => {
  it('evaluates a linear segment as a straight interpolation', () => {
    const point = evaluateBezierSegment(
      { anchor: { x: 0, y: 0 }, inHandle: null, outHandle: null },
      { anchor: { x: 10, y: 0 }, inHandle: null, outHandle: null },
      0.5
    );

    expect(point).toEqual({ x: 5, y: 0 });
  });

  it('splits a linear segment into two linear segments', () => {
    const split = splitBezierSegmentAt(
      {
        anchor: { x: 0, y: 0 },
        inHandle: null,
        outHandle: null,
        type: 'corner',
      },
      {
        anchor: { x: 10, y: 0 },
        inHandle: null,
        outHandle: null,
        type: 'corner',
      },
      0.5
    );

    expect(split).toEqual({
      fromOutHandle: null,
      insertPoint: {
        anchor: { x: 5, y: 0 },
        inHandle: { x: 2.5, y: 0 },
        outHandle: { x: 7.5, y: 0 },
        type: 'smooth',
      },
      toInHandle: null,
    });
  });

  it('finds the nearest segment hit on an open path', () => {
    const hit = findNearestBezierPathSegment(
      [
        { anchor: { x: 0, y: 0 }, inHandle: null, outHandle: null },
        { anchor: { x: 10, y: 0 }, inHandle: null, outHandle: null },
      ],
      false,
      { x: 4, y: 2 }
    );

    expect(hit?.segmentIndex).toBe(0);
    expect(hit?.distance).toBeCloseTo(2, 1);
    expect(hit?.point.x).toBeCloseTo(4, 1);
    expect(hit?.point.y).toBeCloseTo(0, 1);
  });
});
