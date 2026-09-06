import type { CanvasBezierPathState, RgbaColor } from 'features/controlLayers/store/types';
import { describe, expect, it } from 'vitest';

import { buildTaperedTracePoints, buildVectorTraceObject } from './vectorLayerTrace';

const color: RgbaColor = { r: 64, g: 128, b: 192, a: 1 };
const openPath: CanvasBezierPathState = {
  id: 'path',
  name: null,
  isClosed: false,
  points: [
    { anchor: { x: 0, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
    { anchor: { x: 40, y: 0 }, inHandle: null, outHandle: null, type: 'corner' },
  ],
};

describe('vector layer trace utilities', () => {
  it('tapers an open trace by physical distance from both ends', () => {
    const points = buildTaperedTracePoints(
      [
        { x: 0, y: 0 },
        { x: 20, y: 0 },
        { x: 40, y: 0 },
        { x: 60, y: 0 },
        { x: 80, y: 0 },
        { x: 100, y: 0 },
        { x: 120, y: 0 },
      ],
      10
    );
    const pressures = points.filter((_, index) => index % 3 === 2);

    expect(pressures[0]).toBe(0);
    expect(pressures[1]).toBeCloseTo(7 / 27);
    expect(pressures[2]).toBeCloseTo(20 / 27);
    expect(pressures[3]).toBe(1);
    expect(pressures[4]).toBeCloseTo(20 / 27);
    expect(pressures[5]).toBeCloseTo(7 / 27);
    expect(pressures[6]).toBe(0);
  });

  it('meets at full pressure in the middle of a short path', () => {
    const points = buildTaperedTracePoints(
      [
        { x: 0, y: 0 },
        { x: 5, y: 0 },
        { x: 10, y: 0 },
      ],
      10
    );
    const pressures = points.filter((_, index) => index % 3 === 2);

    expect(pressures).toEqual([0, 1, 0]);
  });

  it('returns full pressure for a zero-length trace', () => {
    expect(buildTaperedTracePoints([{ x: 4, y: 8 }], 10)).toEqual([4, 8, 1]);
  });

  it('builds a pressure-sensitive line when tapering an open path', () => {
    const object = buildVectorTraceObject(openPath, 10, color, true);

    expect(object?.type).toBe('brush_line_with_pressure');
    expect(object?.points[2]).toBe(0);
    expect(object?.points.at(-1)).toBe(0);
  });

  it('builds a constant-width line when tapering is disabled', () => {
    expect(buildVectorTraceObject(openPath, 10, color, false)?.type).toBe('brush_line');
  });

  it('does not taper a closed path', () => {
    expect(buildVectorTraceObject({ ...openPath, isClosed: true }, 10, color, true)?.type).toBe('brush_line');
  });
});
