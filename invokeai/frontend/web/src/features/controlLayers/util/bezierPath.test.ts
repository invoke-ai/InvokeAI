import { describe, expect, it } from 'vitest';

import {
  evaluateBezierSegment,
  findNearestBezierPathSegment,
  fitPolylineToBezierPoints,
  getBezierPathHitSamplesPerSegment,
  getBezierPointPullHandleType,
  ovalToBezierPoints,
  rectToBezierPoints,
  setBezierPointHandle,
  setBezierPointType,
  smoothBezierPathPoints,
  splitBezierSegmentAt,
} from './bezierPath';

describe('bezierPath utilities', () => {
  it('does not fit a path without two distinct samples', () => {
    expect(fitPolylineToBezierPoints([{ x: 4, y: 8 }], 1)).toEqual([]);
    expect(
      fitPolylineToBezierPoints(
        [
          { x: 4, y: 8 },
          { x: 4, y: 8 },
        ],
        1
      )
    ).toEqual([]);
  });

  it('fits a two-point polyline to one cubic Bezier segment', () => {
    const points = fitPolylineToBezierPoints(
      [
        { x: 0, y: 0 },
        { x: 12, y: 0 },
      ],
      1
    );

    expect(points).toEqual([
      {
        anchor: { x: 0, y: 0 },
        inHandle: null,
        outHandle: { x: 4, y: 0 },
        type: 'smooth',
      },
      {
        anchor: { x: 12, y: 0 },
        inHandle: { x: 8, y: 0 },
        outHandle: null,
        type: 'smooth',
      },
    ]);
  });

  it('fits a sampled curve while preserving its endpoints', () => {
    const samples = Array.from({ length: 21 }, (_, index) => {
      const x = index * 5;
      return { x, y: Math.sin(index / 4) * 20 };
    });
    const points = fitPolylineToBezierPoints(samples, 2);

    expect(points[0]?.anchor).toEqual(samples[0]);
    expect(points.at(-1)?.anchor).toEqual(samples.at(-1));
    expect(points.length).toBeGreaterThanOrEqual(2);
    expect(points.length).toBeLessThan(samples.length);
    expect(
      points.every((point) =>
        [point.anchor, point.inHandle, point.outHandle]
          .filter((coordinate) => coordinate !== null)
          .every((coordinate) => Number.isFinite(coordinate.x) && Number.isFinite(coordinate.y))
      )
    ).toBe(true);
  });

  it('ignores duplicate adjacent samples when fitting a polyline', () => {
    const points = fitPolylineToBezierPoints(
      [
        { x: 0, y: 0 },
        { x: 0, y: 0 },
        { x: 10, y: 0 },
      ],
      1
    );

    expect(points.map((point) => point.anchor)).toEqual([
      { x: 0, y: 0 },
      { x: 10, y: 0 },
    ]);
  });

  it('converts a rectangle to four closed-path corner points', () => {
    expect(rectToBezierPoints({ x: 10, y: 20, width: 30, height: 40 })).toEqual([
      { anchor: { x: 10, y: 20 }, inHandle: null, outHandle: null, type: 'corner' },
      { anchor: { x: 40, y: 20 }, inHandle: null, outHandle: null, type: 'corner' },
      { anchor: { x: 40, y: 60 }, inHandle: null, outHandle: null, type: 'corner' },
      { anchor: { x: 10, y: 60 }, inHandle: null, outHandle: null, type: 'corner' },
    ]);
  });

  it('converts an oval to four symmetric Bezier points', () => {
    const points = ovalToBezierPoints({ x: 10, y: 20, width: 40, height: 20 });

    expect(points.map((point) => point.anchor)).toEqual([
      { x: 30, y: 20 },
      { x: 50, y: 30 },
      { x: 30, y: 40 },
      { x: 10, y: 30 },
    ]);
    expect(points.every((point) => point.type === 'symmetric')).toBe(true);
    expect(points[0]?.inHandle?.x).toBeCloseTo(18.954305);
    expect(points[0]?.outHandle?.x).toBeCloseTo(41.045695);
    expect(points[1]?.inHandle?.y).toBeCloseTo(24.477153);
    expect(points[1]?.outHandle?.y).toBeCloseTo(35.522847);
  });

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

  it('keeps curved path hit testing accurate at high zoom', () => {
    const from = {
      anchor: { x: 0, y: 0 },
      inHandle: null,
      outHandle: { x: 0, y: 1000 },
    };
    const to = {
      anchor: { x: 1000, y: 0 },
      inHandle: { x: 1000, y: 1000 },
      outHandle: null,
    };
    const stageScale = 20;
    const pointOnCurve = evaluateBezierSegment(from, to, 0.51);
    const hit = findNearestBezierPathSegment(
      [from, to],
      false,
      pointOnCurve,
      getBezierPathHitSamplesPerSegment(stageScale)
    );

    expect(hit?.distance).toBeLessThan(10 / stageScale);
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

  it('smooths an open polyline without moving its anchors', () => {
    const points = smoothBezierPathPoints(
      [
        { anchor: { x: 0, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
        { anchor: { x: 12, y: 12 }, inHandle: null, outHandle: null, type: 'corner' },
        { anchor: { x: 24, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
      ],
      false
    );

    expect(points.map((point) => point.anchor)).toEqual([
      { x: 0, y: 0 },
      { x: 12, y: 12 },
      { x: 24, y: 0 },
    ]);
    expect(points.map((point) => point.type)).toEqual(['smooth', 'smooth', 'smooth']);
    expect(points[0]?.inHandle).toBeNull();
    expect(points[0]?.outHandle?.x).toBeCloseTo(4);
    expect(points[0]?.outHandle?.y).toBeCloseTo(4);
    expect(points[1]?.inHandle?.x).toBeCloseTo(12 - 16 * (Math.sqrt(2) - 1));
    expect(points[1]?.inHandle?.y).toBeCloseTo(12);
    expect(points[1]?.outHandle?.x).toBeCloseTo(12 + 16 * (Math.sqrt(2) - 1));
    expect(points[1]?.outHandle?.y).toBeCloseTo(12);
    expect(points[2]?.inHandle?.x).toBeCloseTo(20);
    expect(points[2]?.inHandle?.y).toBeCloseTo(4);
    expect(points[2]?.outHandle).toBeNull();
  });

  it('wraps smoothing handles around a closed path', () => {
    const points = smoothBezierPathPoints(
      [
        { anchor: { x: 0, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
        { anchor: { x: 12, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
        { anchor: { x: 0, y: 12 }, inHandle: null, outHandle: null, type: 'corner' },
      ],
      true
    );

    expect(points[0]?.anchor).toEqual({ x: 0, y: 0 });
    expect(points[0]?.inHandle?.x).toBeCloseTo(-8 * (Math.sqrt(2) - 1));
    expect(points[0]?.inHandle?.y).toBeCloseTo(8 * (Math.sqrt(2) - 1));
    expect(points[0]?.outHandle?.x).toBeCloseTo(8 * (Math.sqrt(2) - 1));
    expect(points[0]?.outHandle?.y).toBeCloseTo(-8 * (Math.sqrt(2) - 1));
    expect(points[0]?.type).toBe('smooth');
    expect(points.every((point) => point.inHandle !== null && point.outHandle !== null)).toBe(true);
  });

  it('limits smooth handles independently on uneven adjacent segments', () => {
    const points = smoothBezierPathPoints(
      [
        { anchor: { x: 0, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
        { anchor: { x: 100, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
        { anchor: { x: 101, y: 1 }, inHandle: null, outHandle: null, type: 'corner' },
      ],
      false
    );
    const middlePoint = points[1];

    expect(middlePoint?.inHandle).not.toBeNull();
    expect(middlePoint?.outHandle).not.toBeNull();
    expect(Math.hypot(100 - (middlePoint?.inHandle?.x ?? 100), middlePoint?.inHandle?.y ?? 0)).toBeLessThanOrEqual(
      (100 * 2) / 3
    );
    expect(Math.hypot((middlePoint?.outHandle?.x ?? 100) - 100, middlePoint?.outHandle?.y ?? 0)).toBeLessThan(
      Math.SQRT2
    );
    const inVector = {
      x: 100 - (middlePoint?.inHandle?.x ?? 100),
      y: -(middlePoint?.inHandle?.y ?? 0),
    };
    const outVector = {
      x: (middlePoint?.outHandle?.x ?? 100) - 100,
      y: middlePoint?.outHandle?.y ?? 0,
    };
    expect(inVector.x * outVector.y - inVector.y * outVector.x).toBeCloseTo(0);
    expect(middlePoint?.type).toBe('smooth');
  });

  it('uses circular arc handles for a four-point circle', () => {
    const radius = 10;
    const circularControlPointRatio = (4 * (Math.sqrt(2) - 1)) / 3;
    const points = smoothBezierPathPoints(
      [
        { anchor: { x: 0, y: -radius }, inHandle: null, outHandle: null, type: 'corner' },
        { anchor: { x: radius, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
        { anchor: { x: 0, y: radius }, inHandle: null, outHandle: null, type: 'corner' },
        { anchor: { x: -radius, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
      ],
      true
    );

    expect(points[0]?.inHandle?.x).toBeCloseTo(-circularControlPointRatio * radius);
    expect(points[0]?.outHandle?.x).toBeCloseTo(circularControlPointRatio * radius);
    expect(points[0]?.inHandle?.y).toBeCloseTo(-radius);
    expect(points[0]?.outHandle?.y).toBeCloseTo(-radius);
  });

  it('preserves existing symmetric oval handles', () => {
    const oval = ovalToBezierPoints({ x: 10, y: 20, width: 80, height: 40 });

    expect(smoothBezierPathPoints(oval, true)).toEqual(oval);
  });

  it('chooses outgoing handle when pulling from the first point of an open path', () => {
    expect(
      getBezierPointPullHandleType([{ anchor: { x: 0, y: 0 } }, { anchor: { x: 10, y: 0 } }], false, 0, {
        x: -10,
        y: 0,
      })
    ).toBe('outHandle');
  });

  it('chooses incoming handle when pulling from the last point of an open path', () => {
    expect(
      getBezierPointPullHandleType([{ anchor: { x: 0, y: 0 } }, { anchor: { x: 10, y: 0 } }], false, 1, { x: 20, y: 0 })
    ).toBe('inHandle');
  });

  it('chooses the handle on the dragged side for a middle point', () => {
    const points = [{ anchor: { x: 0, y: 0 } }, { anchor: { x: 10, y: 0 } }, { anchor: { x: 20, y: 0 } }];

    expect(getBezierPointPullHandleType(points, false, 1, { x: 0, y: 0 })).toBe('inHandle');
    expect(getBezierPointPullHandleType(points, false, 1, { x: 20, y: 0 })).toBe('outHandle');
  });

  it('chooses the handle on the dragged side for a closed path point', () => {
    const points = [{ anchor: { x: 0, y: 0 } }, { anchor: { x: 10, y: 0 } }, { anchor: { x: 0, y: 10 } }];

    expect(getBezierPointPullHandleType(points, true, 0, { x: 0, y: 10 })).toBe('inHandle');
    expect(getBezierPointPullHandleType(points, true, 0, { x: 10, y: 0 })).toBe('outHandle');
  });
});
