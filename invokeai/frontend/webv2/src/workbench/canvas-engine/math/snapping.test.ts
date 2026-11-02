import type { Rect } from '@workbench/canvas-engine/types';

import { describe, expect, it } from 'vitest';

import { clampZoom, snapMovedPoint, snapRectToGrid, snapToGrid, ZOOM_MAX, ZOOM_MIN, ZOOM_PRESETS } from './snapping';

describe('ZOOM_PRESETS', () => {
  it('exports the default candidates as a stable list', () => {
    expect(ZOOM_PRESETS).toEqual([0.25, 0.33, 0.5, 0.67, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5]);
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

describe('snapMovedPoint', () => {
  it('snaps the resulting position rather than the delta', () => {
    // An off-grid origin seats ONTO the grid: 3+21=24, 5+15=20 -> 24.
    expect(snapMovedPoint({ x: 3, y: 5 }, { x: 21, y: 15 }, 8)).toEqual({ x: 24, y: 24 });
  });

  it('passes an axis with a zero delta through untouched', () => {
    // Shift-locked (or simply unmoved) axes must not drift onto a grid line.
    expect(snapMovedPoint({ x: 3, y: 5 }, { x: 21, y: 0 }, 8)).toEqual({ x: 24, y: 5 });
    expect(snapMovedPoint({ x: 3, y: 5 }, { x: 0, y: 15 }, 8)).toEqual({ x: 3, y: 24 });
  });

  it('is a plain translation for a non-positive grid', () => {
    expect(snapMovedPoint({ x: 3, y: 5 }, { x: 21, y: 15 }, 0)).toEqual({ x: 24, y: 20 });
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
