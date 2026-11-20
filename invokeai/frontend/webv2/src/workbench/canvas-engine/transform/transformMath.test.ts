import { applyToPoint } from '@workbench/canvas-engine/math/mat2d';
import { describe, expect, it } from 'vitest';

import type { LayerTransform, TransformRect } from './transformMath';

import {
  applyMove,
  applyRotate,
  applyScale,
  bakeMatrix,
  composeLayerTransforms,
  IDENTITY_TRANSFORM,
  layerTransformMatrix,
  localHandlePoint,
  resizeCursorForHandle,
  ROTATE_SNAP_RAD,
  TRANSFORM_ROTATE_NUB_PX,
  transformOverlayGeometry,
  transformTargetAt,
} from './transformMath';

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
});

describe('off-origin content rect (content-sized layers)', () => {
  const offOrigin: TransformRect = { height: 40, width: 60, x: 10, y: 20 };

  it('applyScale keeps the opposite (off-origin) anchor fixed in document space', () => {
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

  it('hits the rotation nub above the top edge (drawn but previously not hit-testable)', () => {
    const target = transformTargetAt({
      point: nubTip(identity, rect, 1),
      rect,
      toScreen: identityScreen,
      transform: identity,
    });
    expect(target).toEqual({ handle: 'n', kind: 'rotate' });
  });

  it('hit-tests handles through the layer rotation', () => {
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
    expect(close(next.scaleX, 1.5)).toBe(true);
    expect(close(next.scaleY, 1)).toBe(true);
    expect(close(next.x, 0)).toBe(true);
    expect(close(next.y, 0)).toBe(true);
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
      pointerDoc: { x: -50, y: 50 },
      shift: false,
      rect,
      start: identity,
      startPointerDoc: { x: 100, y: 50 },
    });
    expect(next.scaleX).toBeLessThan(0);
    expect(Number.isNaN(next.scaleX)).toBe(false);
  });
});

describe('applyRotate', () => {
  it('rotates about the center by the swept angle', () => {
    const next = applyRotate({
      pointerDoc: { x: 50, y: 100 },
      shift: false,
      rect,
      start: identity,
      startPointerDoc: { x: 100, y: 50 },
    });
    expect(close(next.rotation, Math.PI / 2)).toBe(true);
    const center = applyToPoint(layerTransformMatrix(next), { x: 50, y: 50 });
    expect(close(center.x, 50)).toBe(true);
    expect(close(center.y, 50)).toBe(true);
  });

  it('snaps to 15° increments under shift', () => {
    const next = applyRotate({
      pointerDoc: { x: 100, y: 62 },
      shift: true,
      rect,
      start: identity,
      startPointerDoc: { x: 100, y: 50 },
    });
    const remainder = Math.abs(next.rotation % ROTATE_SNAP_RAD);
    expect(remainder < 1e-9 || Math.abs(remainder - ROTATE_SNAP_RAD) < 1e-9).toBe(true);
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

describe('applyMove: grid snapping', () => {
  const offGrid: LayerTransform = { rotation: 0, scaleX: 1, scaleY: 1, x: 3, y: 5 };

  it('snaps the resulting position, not the delta', () => {
    // 3+21=24 (already a multiple), 5+15=20 → 24 (nearest, ties round up).
    expect(applyMove(offGrid, { x: 21, y: 15 }, false, 8)).toMatchObject({ x: 24, y: 24 });
  });
});

describe('applyScale: grid snapping', () => {
  const scaleAt = (grid: number, pointerDoc: { x: number; y: number }): LayerTransform =>
    applyScale({
      alt: false,
      grid,
      handle: 'se',
      pointerDoc,
      shift: false,
      rect,
      start: identity,
      startPointerDoc: { x: 100, y: 100 },
    });

  it('lands the dragged handle on a grid multiple', () => {
    // se handle starts at doc (100,100); a 53px x-drag targets 153 → snaps to 152.
    const next = scaleAt(8, { x: 153, y: 100 });
    const handleDoc = applyToPoint(layerTransformMatrix(next), localHandlePoint(rect, 'se'));
    expect(close(handleDoc.x, 152)).toBe(true);
    expect(close(handleDoc.y, 100)).toBe(true);
    expect(close(next.scaleX, 1.52)).toBe(true);
  });
});

describe('bakeMatrix', () => {
  it('equals the layer matrix so bake-then-draw ≡ transform-then-draw', () => {
    const transform: LayerTransform = { rotation: Math.PI / 6, scaleX: 1.5, scaleY: 0.8, x: 12, y: 34 };
    const bake = bakeMatrix(transform);
    const direct = layerTransformMatrix(transform);
    for (const q of [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 100, y: 100 },
      { x: 40, y: 70 },
    ]) {
      const viaBake = applyToPoint(bake, q);
      const viaTransform = applyToPoint(direct, q);
      expect(close(viaBake.x, viaTransform.x)).toBe(true);
      expect(close(viaBake.y, viaTransform.y)).toBe(true);
    }
  });
});

describe('composeLayerTransforms', () => {
  it('is neutral with the identity on either side', () => {
    const t = { rotation: 0.4, scaleX: 2, scaleY: 0.5, x: 10, y: -3 };
    for (const composed of [
      composeLayerTransforms(IDENTITY_TRANSFORM, t),
      composeLayerTransforms(t, IDENTITY_TRANSFORM),
    ]) {
      expect(composed!.rotation).toBeCloseTo(t.rotation, 12);
      expect(composed!.scaleX).toBeCloseTo(t.scaleX, 12);
      expect(composed!.scaleY).toBeCloseTo(t.scaleY, 12);
      expect(composed!.x).toBeCloseTo(t.x, 12);
      expect(composed!.y).toBeCloseTo(t.y, 12);
    }
  });

  it('scales the inner translation by the outer transform and adds rotations', () => {
    expect(
      composeLayerTransforms(
        { rotation: 0, scaleX: 2, scaleY: 3, x: 5, y: 7 },
        { rotation: 0, scaleX: 1, scaleY: 1, x: 10, y: 10 }
      )
    ).toEqual({ rotation: 0, scaleX: 2, scaleY: 3, x: 25, y: 37 });
    const rotated = composeLayerTransforms(
      { rotation: 0.3, scaleX: 1.5, scaleY: 1.5, x: 0, y: 0 },
      { rotation: 0.2, scaleX: 2, scaleY: 1, x: 0, y: 0 }
    );
    expect(rotated!.rotation).toBeCloseTo(0.5, 12);
    expect(rotated!.scaleX).toBeCloseTo(3, 12);
    expect(rotated!.scaleY).toBeCloseTo(1.5, 12);
  });

  it('keeps a flip a flip instead of turning it into a half-turn', () => {
    const flip = { rotation: 0, scaleX: -1, scaleY: 1, x: 0, y: 0 };
    const composed = composeLayerTransforms(IDENTITY_TRANSFORM, flip);
    expect(composed!.scaleX).toBe(-1);
    expect(composed!.scaleY).toBe(1);
    expect(composed!.rotation).toBeCloseTo(0, 12);
  });

  it('refuses a composite that shears', () => {
    expect(
      composeLayerTransforms(
        { rotation: 0, scaleX: 2, scaleY: 1, x: 0, y: 0 },
        { rotation: Math.PI / 4, scaleX: 1, scaleY: 1, x: 0, y: 0 }
      )
    ).toBeNull();
  });
});
