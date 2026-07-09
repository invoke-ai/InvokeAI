import type { Rect } from '@workbench/canvas-engine/types';

import { describe, expect, it } from 'vitest';

import {
  clampZoom,
  constrainAspect,
  snapRectToGrid,
  snapToGrid,
  snapZoom,
  ZOOM_MAX,
  ZOOM_MIN,
  ZOOM_SNAP_CANDIDATES,
} from './snapping';

describe('snapZoom', () => {
  it('snaps to an exact candidate', () => {
    expect(snapZoom(1)).toBe(1);
    expect(snapZoom(0.25)).toBe(0.25);
  });

  it('snaps a value within tolerance of a candidate', () => {
    expect(snapZoom(1.02)).toBe(1);
    expect(snapZoom(0.98)).toBe(1);
  });

  it('passes through a value outside tolerance of any candidate', () => {
    expect(snapZoom(1.2)).toBe(1.2);
    expect(snapZoom(0.6)).toBe(0.6);
  });

  it('snaps at the tolerance boundary and passes through just beyond it', () => {
    const candidate = 2;
    const tolerance = candidate * 0.03;
    expect(snapZoom(candidate + tolerance * 0.99)).toBe(candidate);
    expect(snapZoom(candidate + tolerance * 1.5)).toBeCloseTo(candidate + tolerance * 1.5, 10);
  });

  it('supports custom candidates', () => {
    expect(snapZoom(1.01, [1, 10])).toBe(1);
    expect(snapZoom(5, [1, 10])).toBe(5);
  });

  it('exports the default candidates as a stable list', () => {
    expect(ZOOM_SNAP_CANDIDATES).toEqual([0.25, 0.33, 0.5, 0.67, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5]);
  });
});

describe('clampZoom', () => {
  it('clamps values below the minimum', () => {
    expect(clampZoom(0)).toBe(ZOOM_MIN);
    expect(clampZoom(-5)).toBe(ZOOM_MIN);
  });

  it('clamps values above the maximum', () => {
    expect(clampZoom(100)).toBe(ZOOM_MAX);
  });

  it('passes through in-range values', () => {
    expect(clampZoom(1)).toBe(1);
    expect(clampZoom(ZOOM_MIN)).toBe(ZOOM_MIN);
    expect(clampZoom(ZOOM_MAX)).toBe(ZOOM_MAX);
  });
});

describe('snapToGrid', () => {
  it('rounds to the nearest multiple of grid', () => {
    expect(snapToGrid(10, 8)).toBe(8);
    expect(snapToGrid(13, 8)).toBe(16);
    expect(snapToGrid(0, 16)).toBe(0);
  });

  it('handles negative values', () => {
    expect(snapToGrid(-10, 8)).toBe(-8);
  });

  it('returns the value unchanged for a non-positive grid', () => {
    expect(snapToGrid(13.3, 0)).toBe(13.3);
    expect(snapToGrid(13.3, -4)).toBe(13.3);
  });
});

describe('snapRectToGrid', () => {
  it('snaps position and the far edge to the grid, deriving size', () => {
    const r: Rect = { x: 3, y: 5, width: 30, height: 30 };
    // x: 3 -> 0, right: 33 -> 32 -> width 32
    // y: 5 -> 8, bottom: 35 -> 32 -> height 24
    expect(snapRectToGrid(r, 8)).toEqual({ x: 0, y: 8, width: 32, height: 24 });
  });

  it('is a no-op for a rect already aligned to the grid', () => {
    const r: Rect = { x: 16, y: 32, width: 64, height: 16 };
    expect(snapRectToGrid(r, 16)).toEqual(r);
  });
});

describe('constrainAspect', () => {
  const square: Rect = { x: 10, y: 10, width: 20, height: 20 };

  it('keeps the nw corner fixed', () => {
    const result = constrainAspect(square, 2, 'nw');
    expect(result).toEqual({ x: 10, y: 10, width: 40, height: 20 });
  });

  it('keeps the ne corner fixed', () => {
    const result = constrainAspect(square, 2, 'ne');
    // right edge (30) stays fixed, width grows to 40, so x = 30 - 40 = -10
    expect(result).toEqual({ x: -10, y: 10, width: 40, height: 20 });
  });

  it('keeps the sw corner fixed', () => {
    const result = constrainAspect(square, 0.5, 'sw');
    // bottom edge (30) stays fixed; height = width / aspect... width kept? currentAspect(1) < aspect? no aspect=0.5 <1
    // currentAspect(1) > aspect(0.5) -> keep width(20), height = width/aspect = 40
    expect(result).toEqual({ x: 10, y: -10, width: 20, height: 40 });
  });

  it('keeps the se corner fixed', () => {
    const result = constrainAspect(square, 2, 'se');
    expect(result).toEqual({ x: -10, y: 10, width: 40, height: 20 });
  });

  it('keeps the center fixed', () => {
    const result = constrainAspect(square, 2, 'center');
    expect(result).toEqual({ x: 0, y: 10, width: 40, height: 20 });
  });

  it('is a no-op-ish passthrough for a non-positive aspect or empty rect', () => {
    expect(constrainAspect(square, 0, 'nw')).toEqual(square);
    expect(constrainAspect(square, -1, 'nw')).toEqual(square);
    const empty: Rect = { x: 0, y: 0, width: 0, height: 10 };
    expect(constrainAspect(empty, 2, 'nw')).toEqual(empty);
  });

  it('leaves an already-matching aspect rect unchanged in size', () => {
    const wide: Rect = { x: 0, y: 0, width: 40, height: 20 };
    const result = constrainAspect(wide, 2, 'center');
    expect(result.width).toBeCloseTo(40, 8);
    expect(result.height).toBeCloseTo(20, 8);
  });
});
