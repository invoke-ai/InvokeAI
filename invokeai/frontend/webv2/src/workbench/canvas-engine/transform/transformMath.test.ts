import { applyToPoint } from '@workbench/canvas-engine/math/mat2d';
import { describe, expect, it } from 'vitest';

import type { LayerTransform, TransformRect } from './transformMath';

import {
  applyMove,
  applyRotate,
  applyScale,
  bakeMatrix,
  IDENTITY_TRANSFORM,
  layerTransformMatrix,
  localHandlePoint,
  MIN_ABS_SCALE,
  resizeCursorForHandle,
  ROTATE_SNAP_RAD,
  snapRotation,
  TRANSFORM_ROTATE_NUB_PX,
  transformOverlayGeometry,
  transformTargetAt,
} from './transformMath';

/** The screen-space tip of the drawn rotation nub for `transform` at `zoom`. */
const nubTip = (transform: LayerTransform, sz: TransformRect, zoom: number): { x: number; y: number } => {
  const geo = transformOverlayGeometry(transform, sz);
  const toScreen = (p: { x: number; y: number }) => ({ x: zoom * p.x, y: zoom * p.y });
  const a = toScreen(geo.rotationAnchor);
  const c = toScreen(geo.center);
  const dx = a.x - c.x;
  const dy = a.y - c.y;
  const len = Math.hypot(dx, dy) || 1;
  return { x: a.x + (dx / len) * TRANSFORM_ROTATE_NUB_PX, y: a.y + (dy / len) * TRANSFORM_ROTATE_NUB_PX };
};

const identity: LayerTransform = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };
const rect: TransformRect = { height: 100, width: 100, x: 0, y: 0 };

/** A zoom-only screen projection (no rotation/flip), matching the real viewport. */
const scaleToScreen =
  (zoom: number, panX = 0, panY = 0) =>
  (p: { x: number; y: number }) => ({ x: zoom * p.x + panX, y: zoom * p.y + panY });

const identityScreen = scaleToScreen(1);

const close = (a: number, b: number, eps = 1e-6): boolean => Math.abs(a - b) <= eps;

describe('localHandlePoint', () => {
  it('places corners and edge midpoints on the native rect', () => {
    expect(localHandlePoint(rect, 'nw')).toEqual({ x: 0, y: 0 });
    expect(localHandlePoint(rect, 'se')).toEqual({ x: 100, y: 100 });
    expect(localHandlePoint(rect, 'e')).toEqual({ x: 100, y: 50 });
    expect(localHandlePoint(rect, 'n')).toEqual({ x: 50, y: 0 });
  });

  it('honors an off-origin content rect (content-sized paint layers)', () => {
    // A paint layer whose bitmap sits at offset (10, 20): every handle shifts
    // by the origin — the frame must wrap the pixels, not [0,w]×[0,h].
    const offOrigin: TransformRect = { height: 40, width: 60, x: 10, y: 20 };
    expect(localHandlePoint(offOrigin, 'nw')).toEqual({ x: 10, y: 20 });
    expect(localHandlePoint(offOrigin, 'se')).toEqual({ x: 70, y: 60 });
    expect(localHandlePoint(offOrigin, 'e')).toEqual({ x: 70, y: 40 });
  });
});

describe('off-origin content rect (content-sized layers)', () => {
  const offOrigin: TransformRect = { height: 40, width: 60, x: 10, y: 20 };

  it('transformOverlayGeometry wraps the off-origin rect at identity', () => {
    const geo = transformOverlayGeometry(identity, offOrigin);
    expect(geo.corners).toEqual([
      { x: 10, y: 20 },
      { x: 70, y: 20 },
      { x: 70, y: 60 },
      { x: 10, y: 60 },
    ]);
    expect(geo.center).toEqual({ x: 40, y: 40 });
  });

  it('transformTargetAt hit-tests the shifted frame (interior + corner)', () => {
    const inside = transformTargetAt({
      point: { x: 40, y: 40 },
      rect: offOrigin,
      toScreen: identityScreen,
      transform: identity,
    });
    expect(inside).toEqual({ kind: 'move' });
    const corner = transformTargetAt({
      point: { x: 70, y: 60 },
      rect: offOrigin,
      toScreen: identityScreen,
      transform: identity,
    });
    expect(corner).toEqual({ handle: 'se', kind: 'scale' });
    // A point far from the shifted frame (outside its interior, handles, and
    // rotate zones) is NOT a target, even though an (incorrect) origin-anchored
    // frame would sit in that direction.
    const stale = transformTargetAt({
      point: { x: -40, y: -40 },
      rect: offOrigin,
      toScreen: identityScreen,
      transform: identity,
    });
    expect(stale).toBeNull();
  });

  it('applyScale keeps the opposite (off-origin) anchor fixed in document space', () => {
    // Drag the se handle from (70,60) to (130,100): 2× in both axes, anchored at nw.
    const next = applyScale({
      alt: false,
      handle: 'se',
      pointerDoc: { x: 130, y: 100 },
      rect: offOrigin,
      shift: false,
      start: identity,
      startPointerDoc: { x: 70, y: 60 },
    });
    expect(next.scaleX).toBeCloseTo(2, 6);
    expect(next.scaleY).toBeCloseTo(2, 6);
    // The nw corner (local 10,20) must not move: T = anchorDoc − S·anchorLocal.
    const anchorAfter = applyToPoint(layerTransformMatrix(next), { x: 10, y: 20 });
    expect(anchorAfter.x).toBeCloseTo(10, 6);
    expect(anchorAfter.y).toBeCloseTo(20, 6);
  });

  it('applyRotate pivots about the off-origin rect center', () => {
    const next = applyRotate({
      pointerDoc: { x: 40, y: 100 },
      rect: offOrigin,
      shift: false,
      start: identity,
      // Start due east of the center (40,40); end due south → +90°.
      startPointerDoc: { x: 100, y: 40 },
    });
    expect(next.rotation).toBeCloseTo(Math.PI / 2, 6);
    const centerAfter = applyToPoint(layerTransformMatrix(next), { x: 40, y: 40 });
    expect(centerAfter.x).toBeCloseTo(40, 6);
    expect(centerAfter.y).toBeCloseTo(40, 6);
  });
});

describe('transformTargetAt', () => {
  it('hits a corner scale handle at zoom 1', () => {
    const target = transformTargetAt({
      point: { x: 100, y: 100 },
      rect,
      toScreen: identityScreen,
      transform: identity,
    });
    expect(target).toEqual({ handle: 'se', kind: 'scale' });
  });

  it('reports the interior as move', () => {
    const target = transformTargetAt({ point: { x: 50, y: 50 }, rect, toScreen: identityScreen, transform: identity });
    expect(target).toEqual({ kind: 'move' });
  });

  it('reports a rotate zone just outside a corner', () => {
    const target = transformTargetAt({
      point: { x: 112, y: 112 },
      rect,
      toScreen: identityScreen,
      transform: identity,
    });
    expect(target).toEqual({ handle: 'se', kind: 'rotate' });
  });

  it('hits the rotation nub above the top edge (drawn but previously not hit-testable)', () => {
    // The nub tip sits `TRANSFORM_ROTATE_NUB_PX` above the top-edge midpoint —
    // outside the frame polygon and away from every corner, so before the fix it
    // fell through to `null` (misread by the tool as an off-frame press → reset).
    const target = transformTargetAt({
      point: nubTip(identity, rect, 1),
      rect,
      toScreen: identityScreen,
      transform: identity,
    });
    expect(target).toEqual({ handle: 'n', kind: 'rotate' });
  });

  it('hits the rotation nub of a rotated + scaled layer at a non-1 zoom', () => {
    const transform: LayerTransform = { rotation: 0.4, scaleX: 1.6, scaleY: 1.3, x: 30, y: 20 };
    const target = transformTargetAt({
      point: nubTip(transform, rect, 2),
      rect,
      toScreen: scaleToScreen(2),
      transform,
    });
    expect(target).toEqual({ handle: 'n', kind: 'rotate' });
  });

  it('keeps the interior a move below the nub anchor (nub never steals an interior click)', () => {
    // Inside the top edge, clear of the `'n'` scale handle: the polygon wins so a
    // near-top interior click stays a move, never a rotate.
    const target = transformTargetAt({ point: { x: 50, y: 20 }, rect, toScreen: identityScreen, transform: identity });
    expect(target).toEqual({ kind: 'move' });
  });

  it('returns null far outside the frame', () => {
    const target = transformTargetAt({
      point: { x: 400, y: 400 },
      rect,
      toScreen: identityScreen,
      transform: identity,
    });
    expect(target).toBeNull();
  });

  it('keeps a constant screen-space grab area under zoom (handle at scaled corner)', () => {
    // At zoom 2 the se corner projects to (200,200); a point 3px away still hits.
    const target = transformTargetAt({
      point: { x: 202, y: 199 },
      rect,
      toScreen: scaleToScreen(2),
      transform: identity,
    });
    expect(target).toEqual({ handle: 'se', kind: 'scale' });
  });

  it('hit-tests handles through the layer rotation', () => {
    // Rotate 90° about origin: local se (100,100) → doc. Handle tracks the rotated corner.
    const rotated: LayerTransform = { rotation: Math.PI / 2, scaleX: 1, scaleY: 1, x: 0, y: 0 };
    const seDoc = applyToPoint(layerTransformMatrix(rotated), localHandlePoint(rect, 'se'));
    const target = transformTargetAt({ point: seDoc, rect, toScreen: identityScreen, transform: rotated });
    expect(target).toEqual({ handle: 'se', kind: 'scale' });
  });
});

describe('resizeCursorForHandle', () => {
  it('gives axis cursors for an unrotated layer', () => {
    expect(resizeCursorForHandle(identity, 'e')).toBe('ew-resize');
    expect(resizeCursorForHandle(identity, 'w')).toBe('ew-resize');
    expect(resizeCursorForHandle(identity, 'n')).toBe('ns-resize');
    expect(resizeCursorForHandle(identity, 's')).toBe('ns-resize');
  });

  it('gives diagonal cursors for corners', () => {
    // se outward normal (1,1) → screen y-down diagonal → nwse.
    expect(resizeCursorForHandle(identity, 'se')).toBe('nwse-resize');
    expect(resizeCursorForHandle(identity, 'ne')).toBe('nesw-resize');
  });

  it('rotates the cursor with the layer (45° turns an edge into a diagonal)', () => {
    const rotated: LayerTransform = { rotation: Math.PI / 4, scaleX: 1, scaleY: 1, x: 0, y: 0 };
    // The n edge normal (0,-1) rotated 45° becomes a diagonal → nesw.
    expect(resizeCursorForHandle(rotated, 'n')).toBe('nesw-resize');
    // A 90° rotation swaps the axes: the e handle now reads vertical.
    const quarter: LayerTransform = { rotation: Math.PI / 2, scaleX: 1, scaleY: 1, x: 0, y: 0 };
    expect(resizeCursorForHandle(quarter, 'e')).toBe('ns-resize');
  });
});

describe('applyScale: about the opposite handle', () => {
  it('scales a corner about the opposite corner, tracking the pointer', () => {
    const next = applyScale({
      alt: false,
      handle: 'se',
      pointerDoc: { x: 150, y: 100 },
      shift: false,
      rect,
      start: identity,
      startPointerDoc: { x: 100, y: 100 },
    });
    // nw anchor fixed at (0,0); width doubled in x → scaleX 1.5, scaleY unchanged.
    expect(close(next.scaleX, 1.5)).toBe(true);
    expect(close(next.scaleY, 1)).toBe(true);
    expect(close(next.x, 0)).toBe(true);
    expect(close(next.y, 0)).toBe(true);
  });

  it('scales an edge handle on one axis only', () => {
    const next = applyScale({
      alt: false,
      handle: 'e',
      pointerDoc: { x: 150, y: 999 },
      shift: false,
      rect,
      start: identity,
      startPointerDoc: { x: 100, y: 50 },
    });
    expect(close(next.scaleX, 1.5)).toBe(true);
    expect(close(next.scaleY, 1)).toBe(true);
  });

  it('keeps the opposite corner fixed in document space', () => {
    const start: LayerTransform = { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 20 };
    const anchorBefore = applyToPoint(layerTransformMatrix(start), localHandlePoint(rect, 'nw'));
    const next = applyScale({
      alt: false,
      handle: 'se',
      pointerDoc: { x: 200, y: 200 },
      shift: false,
      rect,
      start,
      startPointerDoc: { x: 110, y: 120 },
    });
    const anchorAfter = applyToPoint(layerTransformMatrix(next), localHandlePoint(rect, 'nw'));
    expect(close(anchorAfter.x, anchorBefore.x)).toBe(true);
    expect(close(anchorAfter.y, anchorBefore.y)).toBe(true);
  });
});

describe('applyScale: modifiers', () => {
  it('alt scales about the center (center stays fixed)', () => {
    const centerBefore = applyToPoint(layerTransformMatrix(identity), { x: 50, y: 50 });
    const next = applyScale({
      alt: true,
      handle: 'se',
      pointerDoc: { x: 150, y: 150 },
      shift: false,
      rect,
      start: identity,
      startPointerDoc: { x: 100, y: 100 },
    });
    const centerAfter = applyToPoint(layerTransformMatrix(next), { x: 50, y: 50 });
    expect(close(centerAfter.x, centerBefore.x)).toBe(true);
    expect(close(centerAfter.y, centerBefore.y)).toBe(true);
  });

  it('shift keeps a uniform aspect on a corner drag', () => {
    const next = applyScale({
      alt: false,
      handle: 'se',
      // Pointer moves mostly in x; uniform factor follows the larger axis.
      pointerDoc: { x: 200, y: 110 },
      shift: true,
      rect,
      start: identity,
      startPointerDoc: { x: 100, y: 100 },
    });
    expect(close(next.scaleX, next.scaleY)).toBe(true);
    expect(next.scaleX).toBeGreaterThan(1);
  });
});

describe('applyScale: flip and clamp', () => {
  it('allows a flip through the anchor (negative scale, no NaN)', () => {
    const next = applyScale({
      alt: false,
      handle: 'e',
      // Drag the east edge past the west anchor → negative scaleX.
      pointerDoc: { x: -50, y: 50 },
      shift: false,
      rect,
      start: identity,
      startPointerDoc: { x: 100, y: 50 },
    });
    expect(next.scaleX).toBeLessThan(0);
    expect(Number.isNaN(next.scaleX)).toBe(false);
  });

  it('clamps a collapsed axis to the minimum magnitude', () => {
    const next = applyScale({
      alt: false,
      handle: 'e',
      // Drag the east edge exactly onto the west anchor → zero width.
      pointerDoc: { x: 0, y: 50 },
      shift: false,
      rect,
      start: identity,
      startPointerDoc: { x: 100, y: 50 },
    });
    expect(Math.abs(next.scaleX)).toBeGreaterThanOrEqual(MIN_ABS_SCALE - 1e-9);
    expect(Number.isFinite(next.x)).toBe(true);
  });
});

describe('applyRotate', () => {
  it('rotates about the center by the swept angle', () => {
    const next = applyRotate({
      pointerDoc: { x: 50, y: 100 }, // 90° CCW around center from the start ray
      shift: false,
      rect,
      start: identity,
      startPointerDoc: { x: 100, y: 50 },
    });
    // start ray points +x; end ray points +y (down) → +90° in canvas space.
    expect(close(next.rotation, Math.PI / 2)).toBe(true);
    // Center stays fixed.
    const center = applyToPoint(layerTransformMatrix(next), { x: 50, y: 50 });
    expect(close(center.x, 50)).toBe(true);
    expect(close(center.y, 50)).toBe(true);
  });

  it('snaps to 15° increments under shift', () => {
    const next = applyRotate({
      pointerDoc: { x: 100, y: 62 }, // a small angle just above 0
      shift: true,
      rect,
      start: identity,
      startPointerDoc: { x: 100, y: 50 },
    });
    const remainder = Math.abs(next.rotation % ROTATE_SNAP_RAD);
    expect(remainder < 1e-9 || Math.abs(remainder - ROTATE_SNAP_RAD) < 1e-9).toBe(true);
  });
});

describe('snapRotation', () => {
  it('rounds to the nearest 15°', () => {
    expect(close(snapRotation(ROTATE_SNAP_RAD * 0.4), 0)).toBe(true);
    expect(close(snapRotation(ROTATE_SNAP_RAD * 0.6), ROTATE_SNAP_RAD)).toBe(true);
  });
});

describe('applyMove', () => {
  it('translates by the document delta', () => {
    expect(applyMove(identity, { x: 5, y: -3 }, false)).toMatchObject({ x: 5, y: -3, rotation: 0, scaleX: 1 });
  });
  it('constrains to the dominant axis under shift', () => {
    expect(applyMove(identity, { x: 8, y: 2 }, true)).toMatchObject({ x: 8, y: 0 });
    expect(applyMove(identity, { x: 1, y: -9 }, true)).toMatchObject({ x: 0, y: -9 });
  });
});

describe('bakeMatrix', () => {
  it('equals the layer matrix so bake-then-draw ≡ transform-then-draw', () => {
    const transform: LayerTransform = { rotation: Math.PI / 6, scaleX: 1.5, scaleY: 0.8, x: 12, y: 34 };
    const bake = bakeMatrix(transform);
    const direct = layerTransformMatrix(transform);
    // For any local point, drawing the baked surface at identity reproduces the
    // transformed point exactly.
    for (const q of [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
      { x: 40, y: 70 },
    ]) {
      const viaBake = applyToPoint(bake, q); // baked surface then identity draw
      const viaTransform = applyToPoint(direct, q);
      expect(close(viaBake.x, viaTransform.x)).toBe(true);
      expect(close(viaBake.y, viaTransform.y)).toBe(true);
    }
  });
});

describe('transformOverlayGeometry', () => {
  it('returns rotated corners, eight handles, center, and the top anchor', () => {
    const geo = transformOverlayGeometry(identity, rect);
    expect(geo.corners).toHaveLength(4);
    expect(geo.handles).toHaveLength(8);
    expect(geo.center).toEqual({ x: 50, y: 50 });
    expect(geo.rotationAnchor).toEqual({ x: 50, y: 0 });
  });
});

describe('IDENTITY_TRANSFORM', () => {
  it('is a true identity', () => {
    expect(IDENTITY_TRANSFORM).toEqual({ rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 });
  });
});
