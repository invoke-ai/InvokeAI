import { describe, expect, it } from 'vitest';

import {
  evaluateBezierSegment,
  findNearestBezierPathSegment,
  setBezierPointHandle,
  setBezierPointType,
  splitBezierSegmentAt,
} from './bezierPath';

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

  it('moves corner handles independently', () => {
    const point = {
      anchor: { x: 0, y: 0 },
      inHandle: { x: -5, y: 0 },
      outHandle: { x: 5, y: 0 },
      type: 'corner' as const,
    };

    setBezierPointHandle(point, 'outHandle', { x: 0, y: 10 });

    expect(point.outHandle).toEqual({ x: 0, y: 10 });
    expect(point.inHandle).toEqual({ x: -5, y: 0 });
  });

  it('keeps smooth handles collinear without changing opposite handle length', () => {
    const point = {
      anchor: { x: 0, y: 0 },
      inHandle: { x: -20, y: 0 },
      outHandle: { x: 10, y: 0 },
      type: 'smooth' as const,
    };

    setBezierPointHandle(point, 'outHandle', { x: 0, y: 10 });

    expect(point.outHandle).toEqual({ x: 0, y: 10 });
    expect(point.inHandle?.x).toBeCloseTo(0);
    expect(point.inHandle?.y).toBeCloseTo(-20);
  });

  it('mirrors symmetric handles with equal length', () => {
    const point = {
      anchor: { x: 0, y: 0 },
      inHandle: { x: -20, y: 0 },
      outHandle: { x: 10, y: 0 },
      type: 'symmetric' as const,
    };

    setBezierPointHandle(point, 'outHandle', { x: 0, y: 10 });

    expect(point.outHandle).toEqual({ x: 0, y: 10 });
    expect(point.inHandle).toEqual({ x: 0, y: -10 });
  });

  it('keeps the assigned point type after moving a handle', () => {
    const point = {
      anchor: { x: 0, y: 0 },
      inHandle: { x: -20, y: 0 },
      outHandle: { x: 10, y: 0 },
      type: 'smooth' as const,
    };

    setBezierPointHandle(point, 'outHandle', { x: 0, y: 10 });

    expect(point.type).toBe('smooth');
  });

  it('stores the assigned point type on the point', () => {
    const point = {
      anchor: { x: 0, y: 0 },
      inHandle: { x: -20, y: 0 },
      outHandle: { x: 10, y: 0 },
      type: 'corner' as const,
    };

    setBezierPointType(point, 'symmetric', 'outHandle');

    expect(point.type).toBe('symmetric');
  });
});
