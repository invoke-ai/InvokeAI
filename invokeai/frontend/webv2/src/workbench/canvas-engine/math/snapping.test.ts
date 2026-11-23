import type { Rect } from '@workbench/canvas-engine/types';

import { describe, expect, it } from 'vitest';

import { clampZoom, snapMovedPoint, snapRectToGrid, snapToGrid, ZOOM_MIN } from './snapping';

describe('clampZoom', () => {
  it('clamps values below the minimum', () => {
    expect(clampZoom(0)).toBe(ZOOM_MIN);
    expect(clampZoom(-5)).toBe(ZOOM_MIN);
  });
});

describe('snapToGrid', () => {
  it('rounds to the nearest multiple of grid', () => {
    expect(snapToGrid(10, 8)).toBe(8);
    expect(snapToGrid(13, 8)).toBe(16);
    expect(snapToGrid(0, 16)).toBe(0);
  });
});

describe('snapMovedPoint', () => {
  it('snaps the resulting position rather than the delta', () => {
    // An off-grid origin seats ONTO the grid: 3+21=24, 5+15=20 -> 24.
    expect(snapMovedPoint({ x: 3, y: 5 }, { x: 21, y: 15 }, 8)).toEqual({ x: 24, y: 24 });
  });
});

describe('snapRectToGrid', () => {
  it('snaps position and the far edge to the grid, deriving size', () => {
    const r: Rect = { x: 3, y: 5, width: 30, height: 30 };
    // x: 3 -> 0, right: 33 -> 32 -> width 32 y: 5 -> 8, bottom: 35 -> 32 -> height 24
    expect(snapRectToGrid(r, 8)).toEqual({ x: 0, y: 8, width: 32, height: 24 });
  });
});
