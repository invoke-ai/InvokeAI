import type { Rect } from '@workbench/canvas-engine/types';

import { describe, expect, it } from 'vitest';

import {
  BBOX_HANDLE_HIT_PX,
  bboxEquals,
  bboxHandleAt,
  bboxHandlePoint,
  bboxTargetAt,
  constrainBboxToRatio,
  isInsideRect,
  moveBbox,
  resizeBbox,
  roundBbox,
} from './bboxHitTest';

const rect = (x: number, y: number, width: number, height: number): Rect => ({ height, width, x, y });

describe('bboxHandlePoint', () => {
  it('places the eight handles on corners and edge midpoints', () => {
    const r = rect(0, 0, 100, 80);
    expect(bboxHandlePoint(r, 'nw')).toEqual({ x: 0, y: 0 });
    expect(bboxHandlePoint(r, 'ne')).toEqual({ x: 100, y: 0 });
    expect(bboxHandlePoint(r, 'se')).toEqual({ x: 100, y: 80 });
    expect(bboxHandlePoint(r, 'sw')).toEqual({ x: 0, y: 80 });
    expect(bboxHandlePoint(r, 'n')).toEqual({ x: 50, y: 0 });
    expect(bboxHandlePoint(r, 'e')).toEqual({ x: 100, y: 40 });
    expect(bboxHandlePoint(r, 's')).toEqual({ x: 50, y: 80 });
    expect(bboxHandlePoint(r, 'w')).toEqual({ x: 0, y: 40 });
  });
});

describe('bboxHandleAt (screen-space hit-testing)', () => {
  it('returns the handle whose grab area contains the point', () => {
    const screenRect = rect(0, 0, 100, 100);
    expect(bboxHandleAt(screenRect, { x: 0, y: 0 })).toBe('nw');
    expect(bboxHandleAt(screenRect, { x: 100, y: 100 })).toBe('se');
    expect(bboxHandleAt(screenRect, { x: 50, y: 0 })).toBe('n');
  });

  it('tolerates a few pixels of slack around the handle', () => {
    const screenRect = rect(0, 0, 100, 100);
    const slack = BBOX_HANDLE_HIT_PX / 2 - 1;
    expect(bboxHandleAt(screenRect, { x: slack, y: slack })).toBe('nw');
    // Just outside the grab area → no handle.
    expect(bboxHandleAt(screenRect, { x: BBOX_HANDLE_HIT_PX, y: BBOX_HANDLE_HIT_PX })).toBeNull();
  });

  it('keeps the grab area a constant screen size across zooms', () => {
    // Same doc bbox {0,0,100,100} projected at two zooms. A point 5px from the
    // top-left corner in SCREEN space hits 'nw' regardless of the frame's on-screen size.
    const atZoom1 = rect(0, 0, 100, 100);
    const atZoom4 = rect(0, 0, 400, 400);
    expect(bboxHandleAt(atZoom1, { x: 5, y: 5 })).toBe('nw');
    expect(bboxHandleAt(atZoom4, { x: 5, y: 5 })).toBe('nw');
    // The midpoint of the top edge moves with zoom: 50 at zoom1, 200 at zoom4.
    expect(bboxHandleAt(atZoom1, { x: 50, y: 0 })).toBe('n');
    // At zoom4 the top-edge midpoint lives at 200, so 50 grabs nothing.
    expect(bboxHandleAt(atZoom4, { x: 50, y: 0 })).toBeNull();
    expect(bboxHandleAt(atZoom4, { x: 200, y: 0 })).toBe('n');
  });

  it('prefers a corner over an overlapping edge midpoint on a tiny frame', () => {
    // At a 10px frame the edge midpoints sit within the corner grab areas; corners win.
    const tiny = rect(0, 0, 10, 10);
    expect(bboxHandleAt(tiny, { x: 0, y: 5 })).toBe('nw');
  });
});

describe('bboxTargetAt', () => {
  const screenRect = rect(10, 10, 100, 100);
  it('returns a handle when over one', () => {
    expect(bboxTargetAt(screenRect, { x: 10, y: 10 })).toBe('nw');
  });
  it("returns 'move' inside the frame away from handles", () => {
    expect(bboxTargetAt(screenRect, { x: 60, y: 60 })).toBe('move');
  });
  it('returns null outside the frame (bbox is never deselected)', () => {
    expect(bboxTargetAt(screenRect, { x: 200, y: 200 })).toBeNull();
  });
});

describe('isInsideRect', () => {
  it('includes the border', () => {
    expect(isInsideRect(rect(0, 0, 10, 10), { x: 0, y: 10 })).toBe(true);
    expect(isInsideRect(rect(0, 0, 10, 10), { x: 11, y: 5 })).toBe(false);
  });
});

describe('resizeBbox — free (unconstrained)', () => {
  const start = rect(0, 0, 100, 100);

  it('grows the SE corner and snaps the moved edges to the grid', () => {
    const out = resizeBbox({ constrain: false, dx: 10, dy: 10, grid: 8, handle: 'se', ratio: 1, snap: true, start });
    // right = snap(110) = 112, bottom = snap(110) = 112, anchored at (0,0).
    expect(out).toEqual(rect(0, 0, 112, 112));
  });

  it('drags the NW corner (moves origin) and snaps', () => {
    const out = resizeBbox({ constrain: false, dx: -10, dy: -10, grid: 8, handle: 'nw', ratio: 1, snap: true, start });
    // left = snap(-10) = -8, top = -8; right/bottom fixed at 100.
    expect(out).toEqual(rect(-8, -8, 108, 108));
  });

  it('resizes a single edge, leaving the perpendicular axis untouched', () => {
    const out = resizeBbox({ constrain: false, dx: 13, dy: 0, grid: 8, handle: 'e', ratio: 1, snap: true, start });
    expect(out).toEqual(rect(0, 0, 112, 100));
  });

  it('bypasses snapping when snap is false (alt held)', () => {
    const out = resizeBbox({ constrain: false, dx: 13, dy: 0, grid: 8, handle: 'e', ratio: 1, snap: false, start });
    expect(out).toEqual(rect(0, 0, 113, 100));
  });

  it('clamps to a minimum of one grid cell when collapsing an edge', () => {
    const out = resizeBbox({ constrain: false, dx: -500, dy: 0, grid: 8, handle: 'e', ratio: 1, snap: true, start });
    expect(out).toEqual(rect(0, 0, 8, 100));
  });

  it('never goes below 1px when the grid is sub-pixel', () => {
    const out = resizeBbox({ constrain: false, dx: -500, dy: 0, grid: 0, handle: 'e', ratio: 1, snap: false, start });
    expect(out.width).toBe(1);
  });
});

describe('resizeBbox — aspect constrained', () => {
  const start = rect(0, 0, 100, 100);

  it('preserves the ratio on a corner drag (anchored at the opposite corner)', () => {
    const out = resizeBbox({ constrain: true, dx: 100, dy: 0, grid: 8, handle: 'se', ratio: 2, snap: false, start });
    // Pointer widens to 200; ratio 2:1 → height 100. Anchor NW stays at (0,0).
    expect(out).toEqual(rect(0, 0, 200, 100));
  });

  it('re-derives the free axis on an edge drag, keeping the center fixed', () => {
    const out = resizeBbox({ constrain: true, dx: 100, dy: 0, grid: 8, handle: 'e', ratio: 2, snap: false, start });
    // width 200 → height 100 (ratio 2), centered on the original vertical center (50).
    expect(out).toEqual(rect(0, 0, 200, 100));
  });

  it('grid-aligns the driven dimension while keeping the ratio exact', () => {
    const out = resizeBbox({ constrain: true, dx: 13, dy: 13, grid: 8, handle: 'se', ratio: 1, snap: true, start });
    // se raw ~113 → snap width 112, height = width/ratio = 112.
    expect(out.width).toBe(112);
    expect(out.height).toBe(112);
  });
});

describe('moveBbox', () => {
  it('translates and snaps the origin to the grid', () => {
    expect(moveBbox(rect(10, 10, 100, 100), 5, 5, 8, true)).toEqual(rect(16, 16, 100, 100));
  });
  it('bypasses snap when snap is false', () => {
    expect(moveBbox(rect(10, 10, 100, 100), 5, 5, 8, false)).toEqual(rect(15, 15, 100, 100));
  });
});

describe('constrainBboxToRatio', () => {
  it('re-fits to the ratio around the center and snaps the width', () => {
    const out = constrainBboxToRatio(rect(0, 0, 100, 100), 2, 8);
    expect(out.width / out.height).toBeCloseTo(2, 5);
    expect(out.width % 8).toBe(0);
    // Center preserved.
    expect(out.x + out.width / 2).toBeCloseTo(50, 5);
    expect(out.y + out.height / 2).toBeCloseTo(50, 5);
  });
});

describe('roundBbox / bboxEquals', () => {
  it('rounds to integer document px with a ≥1 size', () => {
    expect(roundBbox(rect(1.4, 2.6, 0.2, 10.5))).toEqual(rect(1, 3, 1, 11));
  });
  it('compares equality after rounding', () => {
    expect(bboxEquals(rect(0, 0, 100, 100), rect(0.2, -0.1, 100.4, 99.6))).toBe(true);
    expect(bboxEquals(rect(0, 0, 100, 100), rect(0, 0, 108, 100))).toBe(false);
  });
});
