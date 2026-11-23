import type { Rect } from '@workbench/canvas-engine/types';

import { describe, expect, it } from 'vitest';

import type { ResizeBboxParams } from './bboxHitTest';

import {
  BBOX_HANDLES,
  bboxHandleAt,
  bboxHandlePoint,
  bboxTargetAt,
  moveBbox,
  resizeBbox,
  roundBbox,
} from './bboxHitTest';

const rect = (x: number, y: number, width: number, height: number): Rect => ({ height, width, x, y });

const resize = (p: Omit<ResizeBboxParams, 'symmetric'> & { symmetric?: boolean }): Rect =>
  resizeBbox({ symmetric: false, ...p });

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
});

describe('bboxTargetAt', () => {
  const screenRect = rect(10, 10, 100, 100);
  it('returns a handle when over one', () => {
    expect(bboxTargetAt(screenRect, { x: 10, y: 10 })).toBe('nw');
  });
});

describe('resizeBbox — free (unconstrained)', () => {
  const start = rect(0, 0, 100, 100);

  it('grows the SE corner and snaps the moved edges to the grid', () => {
    const out = resize({ constrain: false, dx: 10, dy: 10, grid: 8, handle: 'se', ratio: 1, snap: true, start });
    // right = snap(110) = 112, bottom = snap(110) = 112, anchored at (0,0).
    expect(out).toEqual(rect(0, 0, 112, 112));
  });

  it('clamps to a minimum of one grid cell when collapsing an edge', () => {
    const out = resize({ constrain: false, dx: -500, dy: 0, grid: 8, handle: 'e', ratio: 1, snap: true, start });
    expect(out).toEqual(rect(0, 0, 8, 100));
  });
});

describe('resizeBbox — aspect constrained', () => {
  const start = rect(0, 0, 100, 100);

  it('preserves the ratio on a corner drag (anchored at the opposite corner)', () => {
    const out = resize({ constrain: true, dx: 100, dy: 0, grid: 8, handle: 'se', ratio: 2, snap: false, start });
    // Pointer widens to 200; ratio 2:1 → height 100. Anchor NW stays at (0,0).
    expect(out).toEqual(rect(0, 0, 200, 100));
  });
});

describe('resizeBbox — symmetric (mirrored across the center)', () => {
  const start = rect(0, 0, 100, 100);

  it('moves the opposite edge by the same amount, holding the center', () => {
    const out = resize({
      constrain: false,
      dx: 10,
      dy: 0,
      grid: 8,
      handle: 'e',
      ratio: 1,
      snap: false,
      start,
      symmetric: true,
    });
    // East edge follows the pointer to 110, west mirrors to -10; center stays at 50.
    expect(out).toEqual(rect(-10, 0, 120, 100));
    expect(out.x + out.width / 2).toBe(50);
  });

  it('pixel-snaps a half-pixel symmetric resize without shifting the center', () => {
    const pixelStart = rect(16, 16, 96, 96);
    const out = resize({
      constrain: false,
      dx: 0.5,
      dy: 0,
      grid: 8,
      handle: 'e',
      ratio: 1,
      snap: false,
      start: pixelStart,
      symmetric: true,
    });
    expect(out).toEqual(rect(15, 16, 98, 96));
    expect(out.x + out.width / 2).toBe(64);

    const shrinking = resize({
      constrain: false,
      dx: -0.5,
      dy: 0,
      grid: 8,
      handle: 'e',
      ratio: 1,
      snap: false,
      start: pixelStart,
      symmetric: true,
    });
    expect(shrinking).toEqual(pixelStart);

    const minimum = resize({
      constrain: false,
      dx: 1,
      dy: 0,
      grid: 0,
      handle: 'w',
      ratio: 1,
      snap: false,
      start: rect(0, 0, 2, 2),
      symmetric: true,
    });
    expect(minimum).toEqual(rect(0, 0, 2, 2));
  });

  it('keeps bypassed resize geometry pixel-aligned across every handle and constraint mode', () => {
    const pixelStart = rect(-21, 13, 95, 96);
    const startRight = pixelStart.x + pixelStart.width;
    const startBottom = pixelStart.y + pixelStart.height;
    const centerX = pixelStart.x + pixelStart.width / 2;
    const centerY = pixelStart.y + pixelStart.height / 2;

    for (const handle of BBOX_HANDLES) {
      for (const constrain of [false, true]) {
        for (const symmetric of [false, true]) {
          const out = resize({
            constrain,
            dx: 13.4,
            dy: -7.6,
            grid: 8,
            handle,
            ratio: 1.7,
            snap: false,
            start: pixelStart,
            symmetric,
          });

          expect([out.x, out.y, out.width, out.height].every(Number.isInteger)).toBe(true);
          if (symmetric) {
            expect(out.x + out.width / 2).toBe(centerX);
            expect(out.y + out.height / 2).toBe(centerY);
          } else {
            if (handle.includes('w')) {
              expect(out.x + out.width).toBe(startRight);
            }
            if (handle.includes('e')) {
              expect(out.x).toBe(pixelStart.x);
            }
            if (handle.includes('n')) {
              expect(out.y + out.height).toBe(startBottom);
            }
            if (handle.includes('s')) {
              expect(out.y).toBe(pixelStart.y);
            }
          }
        }
      }
    }
  });
});

describe('moveBbox', () => {
  it('translates and snaps the origin to the grid', () => {
    expect(moveBbox(rect(10, 10, 100, 100), 5, 5, 8, true)).toEqual(rect(16, 16, 100, 100));
  });
});

describe('roundBbox / bboxEquals', () => {
  it('rounds to integer document px with a ≥1 size', () => {
    expect(roundBbox(rect(1.4, 2.6, 0.2, 10.5))).toEqual(rect(1, 3, 1, 11));
  });
});
