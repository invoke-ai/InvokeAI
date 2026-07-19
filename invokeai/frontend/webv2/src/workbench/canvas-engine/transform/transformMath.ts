/**
 * Pure transform math for the transform tool: handle geometry, screen-space
 * hit-testing over a rotated/scaled layer, per-handle rotation-aware cursors, and
 * the scale-about-anchor / rotate / move transform derivations.
 *
 * A layer's raster cache holds unscaled pixels in the layer's LOCAL space; the
 * compositor draws it through `fromTRS(transform)` (translate·rotate·scale). The
 * transform tool edits that {@link LayerTransform}. All handle positions are
 * computed in LOCAL space, mapped to document space via the layer matrix, then
 * projected to SCREEN space by the caller-supplied `toScreen` so grab areas stay
 * a constant pixel size regardless of zoom (mirrors `bboxHitTest.ts`, generalized
 * to a rotated frame). Every derivation returns a fresh transform from a captured
 * `start` transform plus a pointer delta — negative scale (a flip past the anchor)
 * is allowed and kept correct; `|scale|` is clamped to {@link MIN_ABS_SCALE} so a
 * collapsed axis never produces a singular/NaN matrix.
 *
 * Zero React, zero DOM, zero import-time side effects.
 */

import type { CanvasLayerBaseContract } from '@workbench/canvas-engine/contracts';
import type { Mat2d, Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint, fromTRS } from '@workbench/canvas-engine/math/mat2d';

/** The editable layer transform (translate + per-axis scale + rotation in radians). */
export type LayerTransform = CanvasLayerBaseContract['transform'];

/**
 * The layer's content rectangle in its LOCAL (untransformed) space — the extent
 * being transformed. Content-sized layers can sit off-origin (a paint layer's
 * persisted offset, negative for strokes up/left of the origin), so the frame,
 * handles, and scale/rotate anchors are all derived from this rect, never from
 * an assumed `[0,w]×[0,h]`.
 */
export interface TransformRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** The eight scale handles: four corners + four edge midpoints. */
export type TransformHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

/**
 * What a pointer landed on over the transform frame: a scale handle, a corner
 * rotate zone (just outside a corner), or the interior (`'move'`).
 */
export type TransformTarget =
  | { kind: 'scale'; handle: TransformHandle }
  | { kind: 'rotate'; handle: TransformHandle }
  | { kind: 'move' };

/** Handles in hit-test priority order: corners first (they win over adjacent edges). */
export const TRANSFORM_HANDLES: readonly TransformHandle[] = ['nw', 'ne', 'se', 'sw', 'n', 'e', 's', 'w'];

/** The four corner handles (also the rotate-zone anchors). */
export const CORNER_HANDLES: readonly TransformHandle[] = ['nw', 'ne', 'se', 'sw'];

/** Side length (screen px) of a scale handle's square grab area. */
export const TRANSFORM_HANDLE_HIT_PX = 14;

/** Radius (screen px) of a corner's rotate zone, measured from the corner outward. */
export const TRANSFORM_ROTATE_ZONE_PX = 22;

/**
 * Screen-px length of the rotation-indicator nub the overlay draws past the top
 * edge midpoint (the `'n'` handle), outward away from the center. Shared with the
 * overlay renderer so the drawn nub and its hit region stay in lock-step.
 */
export const TRANSFORM_ROTATE_NUB_PX = 18;

/** Screen-px grab radius around the rotation nub/knob (the capsule half-width). */
export const TRANSFORM_ROTATE_NUB_HIT_PX = 11;

/** Rotation snap increment under shift: 15°. */
export const ROTATE_SNAP_RAD = Math.PI / 12;

/** Smallest permitted |scaleX|/|scaleY| — keeps the layer matrix invertible. */
export const MIN_ABS_SCALE = 0.01;

/** The corner/edge opposite a handle (the fixed anchor for a non-alt scale). */
const OPPOSITE: Record<TransformHandle, TransformHandle> = {
  e: 'w',
  n: 's',
  ne: 'sw',
  nw: 'se',
  s: 'n',
  se: 'nw',
  sw: 'ne',
  w: 'e',
};

const hasWest = (h: TransformHandle): boolean => h === 'nw' || h === 'w' || h === 'sw';
const hasEast = (h: TransformHandle): boolean => h === 'ne' || h === 'e' || h === 'se';
const hasNorth = (h: TransformHandle): boolean => h === 'nw' || h === 'n' || h === 'ne';
const hasSouth = (h: TransformHandle): boolean => h === 'sw' || h === 's' || h === 'se';

/** True for the four corner handles (two-letter ids). */
export const isCornerHandle = (h: TransformHandle): boolean => h.length === 2;

/** A handle's position on the layer's local content rect (off-origin aware). */
export const localHandlePoint = (rect: TransformRect, handle: TransformHandle): Vec2 => ({
  x: rect.x + (hasWest(handle) ? 0 : hasEast(handle) ? rect.width : rect.width / 2),
  y: rect.y + (hasNorth(handle) ? 0 : hasSouth(handle) ? rect.height : rect.height / 2),
});

/** The local content rect's center (the rotation pivot / alt-scale anchor). */
const localCenter = (rect: TransformRect): Vec2 => ({
  x: rect.x + rect.width / 2,
  y: rect.y + rect.height / 2,
});

/** The layer's local→document affine matrix from its transform. */
export const layerTransformMatrix = (t: LayerTransform): Mat2d =>
  fromTRS({ x: t.x, y: t.y }, t.rotation, t.scaleX, t.scaleY);

/** The rotation matrix for `rad` (linear part only). */
const rotationMat = (rad: number): { a: number; b: number; c: number; d: number } => {
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return { a: cos, b: sin, c: -sin, d: cos };
};

/** Applies a rotation matrix's linear part to a vector. */
const applyLinear = (m: { a: number; b: number; c: number; d: number }, v: Vec2): Vec2 => ({
  x: m.a * v.x + m.c * v.y,
  y: m.b * v.x + m.d * v.y,
});

/** A handle's outward normal in LOCAL space (y-down, matching document/screen). */
const localOutward = (handle: TransformHandle): Vec2 => ({
  x: hasWest(handle) ? -1 : hasEast(handle) ? 1 : 0,
  y: hasNorth(handle) ? -1 : hasSouth(handle) ? 1 : 0,
});

/**
 * Geometry the overlay needs to draw the transform frame, in DOCUMENT space
 * (the overlay projects each point through the view). `handles` is ordered to
 * match {@link TRANSFORM_HANDLES}.
 */
export interface TransformOverlayGeometry {
  /** The four rotated-rect corners (nw, ne, se, sw) for the bounds polygon. */
  corners: Vec2[];
  /** The eight scale-handle positions (order = {@link TRANSFORM_HANDLES}). */
  handles: Vec2[];
  /** The layer center (rotation pivot). */
  center: Vec2;
  /** The top edge midpoint (root of the rotation indicator nub). */
  rotationAnchor: Vec2;
}

/** Document-space geometry for the transform overlay of a layer at `transform`. */
export const transformOverlayGeometry = (transform: LayerTransform, rect: TransformRect): TransformOverlayGeometry => {
  const m = layerTransformMatrix(transform);
  return {
    center: applyToPoint(m, localCenter(rect)),
    corners: (['nw', 'ne', 'se', 'sw'] as const).map((h) => applyToPoint(m, localHandlePoint(rect, h))),
    handles: TRANSFORM_HANDLES.map((h) => applyToPoint(m, localHandlePoint(rect, h))),
    rotationAnchor: applyToPoint(m, localHandlePoint(rect, 'n')),
  };
};

/** True when `pt` is inside the convex polygon `poly` (consistent-winding cross test). */
const pointInConvexPolygon = (poly: readonly Vec2[], pt: Vec2): boolean => {
  let pos = false;
  let neg = false;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i]!;
    const b = poly[(i + 1) % poly.length]!;
    const cross = (b.x - a.x) * (pt.y - a.y) - (b.y - a.y) * (pt.x - a.x);
    if (cross > 1e-9) {
      pos = true;
    } else if (cross < -1e-9) {
      neg = true;
    }
    if (pos && neg) {
      return false;
    }
  }
  return true;
};

/** Distance (screen px) from `p` to the segment `a`→`b`. */
const distanceToSegment = (p: Vec2, a: Vec2, b: Vec2): number => {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const len2 = abx * abx + aby * aby;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((p.x - a.x) * abx + (p.y - a.y) * aby) / len2)) : 0;
  return Math.hypot(p.x - (a.x + t * abx), p.y - (a.y + t * aby));
};

/** Parameters for {@link transformTargetAt}. */
export interface TransformHitTestParams {
  transform: LayerTransform;
  rect: TransformRect;
  /** Projects a document-space point to screen space (CSS px). */
  toScreen: (docPoint: Vec2) => Vec2;
  /** The screen-space pointer position (CSS px). */
  point: Vec2;
  /** Scale-handle grab-area side length (screen px). */
  handleHitPx?: number;
  /** Corner rotate-zone radius (screen px). */
  rotateZonePx?: number;
  /** Rotation-nub grab radius (screen px). */
  rotateNubHitPx?: number;
}

/**
 * What a screen-space `point` targets on the transform frame: a scale handle
 * (grab area first, corners before edges), a corner rotate zone (outside the
 * frame, near a corner), the interior (`'move'`), or `null`.
 */
export const transformTargetAt = (params: TransformHitTestParams): TransformTarget | null => {
  const { point, rect, toScreen, transform } = params;
  const half = (params.handleHitPx ?? TRANSFORM_HANDLE_HIT_PX) / 2;
  const m = layerTransformMatrix(transform);
  const screenOf = (handle: TransformHandle): Vec2 => toScreen(applyToPoint(m, localHandlePoint(rect, handle)));

  for (const handle of TRANSFORM_HANDLES) {
    const s = screenOf(handle);
    if (Math.abs(point.x - s.x) <= half && Math.abs(point.y - s.y) <= half) {
      return { handle, kind: 'scale' };
    }
  }

  const screenCorners = CORNER_HANDLES.map(screenOf);
  if (pointInConvexPolygon(screenCorners, point)) {
    return { kind: 'move' };
  }

  // Rotation nub: a capsule from the top-edge (`'n'`) midpoint outward (away
  // from center) by `TRANSFORM_ROTATE_NUB_PX`, matching the drawn nub. Tested
  // after the interior polygon (so an interior click stays a move) and before
  // the corner rotate zones (disjoint regions — a top-middle nub vs corners).
  // Without this, a click on the visible nub falls through to `null`, which the
  // tool misreads as an off-frame press — re-opening the session (resetting its
  // live transform) instead of starting a rotation.
  const anchorScreen = screenOf('n');
  const centerScreen = toScreen(applyToPoint(m, { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 }));
  const outX = anchorScreen.x - centerScreen.x;
  const outY = anchorScreen.y - centerScreen.y;
  const outLen = Math.hypot(outX, outY) || 1;
  const nubTip: Vec2 = {
    x: anchorScreen.x + (outX / outLen) * TRANSFORM_ROTATE_NUB_PX,
    y: anchorScreen.y + (outY / outLen) * TRANSFORM_ROTATE_NUB_PX,
  };
  if (distanceToSegment(point, anchorScreen, nubTip) <= (params.rotateNubHitPx ?? TRANSFORM_ROTATE_NUB_HIT_PX)) {
    return { handle: 'n', kind: 'rotate' };
  }

  const zone = params.rotateZonePx ?? TRANSFORM_ROTATE_ZONE_PX;
  for (let i = 0; i < CORNER_HANDLES.length; i++) {
    const s = screenCorners[i]!;
    if (Math.hypot(point.x - s.x, point.y - s.y) <= zone) {
      return { handle: CORNER_HANDLES[i]!, kind: 'rotate' };
    }
  }
  return null;
};

/** The four resize cursors, indexed by the quantized 45° bucket of the outward normal. */
const RESIZE_CURSORS = ['ew-resize', 'nesw-resize', 'ns-resize', 'nwse-resize'] as const;

/**
 * The resize cursor for a scale handle, accounting for the layer's rotation:
 * the handle's outward normal is rotated/scaled by the layer matrix, then the
 * resulting screen-space angle is quantized to one of the four resize
 * orientations. (The view adds only a uniform scale — no rotation/flip — so the
 * document-space angle equals the screen-space angle.)
 */
export const resizeCursorForHandle = (transform: LayerTransform, handle: TransformHandle): string => {
  const m = layerTransformMatrix(transform);
  const dir = applyLinear(m, localOutward(handle));
  // Screen space is y-down; negate y to get a conventional (y-up) math angle.
  let deg = (Math.atan2(-dir.y, dir.x) * 180) / Math.PI;
  deg = ((deg % 360) + 360) % 360;
  deg %= 180; // opposite directions share a resize cursor
  const bucket = Math.round(deg / 45) % 4;
  return RESIZE_CURSORS[bucket]!;
};

/** Clamps |s| to at least {@link MIN_ABS_SCALE}, preserving sign (0 → +MIN). */
const clampScale = (s: number): number => {
  if (!Number.isFinite(s) || s === 0) {
    return MIN_ABS_SCALE;
  }
  const sign = s < 0 ? -1 : 1;
  return Math.abs(s) < MIN_ABS_SCALE ? sign * MIN_ABS_SCALE : s;
};

/** Applies shift axis-constraint to a document-space delta (dominant axis wins). */
export const constrainMoveDelta = (delta: Vec2, shift: boolean): Vec2 => {
  if (!shift) {
    return delta;
  }
  return Math.abs(delta.x) >= Math.abs(delta.y) ? { x: delta.x, y: 0 } : { x: 0, y: delta.y };
};

/** Translates `start` by a document-space pointer delta (shift = axis constrain). */
export const applyMove = (start: LayerTransform, deltaDoc: Vec2, shift: boolean): LayerTransform => {
  const d = constrainMoveDelta(deltaDoc, shift);
  return { ...start, x: start.x + d.x, y: start.y + d.y };
};

/** Parameters for {@link applyScale}. */
export interface ScaleParams {
  start: LayerTransform;
  rect: TransformRect;
  handle: TransformHandle;
  /** Pointer position (document space) at gesture start. */
  startPointerDoc: Vec2;
  /** Current pointer position (document space). */
  pointerDoc: Vec2;
  /** Preserve aspect ratio (corner handles only). */
  shift: boolean;
  /** Scale about the layer center instead of the opposite handle. */
  alt: boolean;
}

/**
 * Scales `start` by dragging `handle`, anchored at the opposite handle (or the
 * center under `alt`). The grabbed handle tracks the pointer; edge handles scale
 * one axis, corners scale both (`shift` forces a uniform, aspect-preserving
 * factor). Dragging past the anchor flips the axis (negative scale), which is
 * kept correct; each axis is clamped to {@link MIN_ABS_SCALE}.
 */
export const applyScale = (p: ScaleParams): LayerTransform => {
  const { alt, handle, rect, shift, start } = p;
  const m = layerTransformMatrix(start);
  const handleLocal = localHandlePoint(rect, handle);
  const anchorLocal = alt ? localCenter(rect) : localHandlePoint(rect, OPPOSITE[handle]);
  const anchorDoc = applyToPoint(m, anchorLocal);
  const handleDoc = applyToPoint(m, handleLocal);
  const targetDoc: Vec2 = {
    x: handleDoc.x + (p.pointerDoc.x - p.startPointerDoc.x),
    y: handleDoc.y + (p.pointerDoc.y - p.startPointerDoc.y),
  };

  // Un-rotate the anchor→target vector into the layer's scaled-local frame.
  const cos = Math.cos(start.rotation);
  const sin = Math.sin(start.rotation);
  const dx = targetDoc.x - anchorDoc.x;
  const dy = targetDoc.y - anchorDoc.y;
  const vx = cos * dx + sin * dy;
  const vy = -sin * dx + cos * dy;

  const dlx = handleLocal.x - anchorLocal.x;
  const dly = handleLocal.y - anchorLocal.y;
  let sx = dlx !== 0 ? vx / dlx : start.scaleX;
  let sy = dly !== 0 ? vy / dly : start.scaleY;

  if (shift && isCornerHandle(handle)) {
    const fx = start.scaleX !== 0 ? sx / start.scaleX : 1;
    const fy = start.scaleY !== 0 ? sy / start.scaleY : 1;
    const f = Math.abs(fx) >= Math.abs(fy) ? fx : fy;
    sx = start.scaleX * f;
    sy = start.scaleY * f;
  }

  sx = clampScale(sx);
  sy = clampScale(sy);

  // Keep the anchor fixed: T = anchorDoc − R·(sx·anchorLocal.x, sy·anchorLocal.y).
  const rs = applyLinear(rotationMat(start.rotation), { x: sx * anchorLocal.x, y: sy * anchorLocal.y });
  return {
    rotation: start.rotation,
    scaleX: sx,
    scaleY: sy,
    x: anchorDoc.x - rs.x,
    y: anchorDoc.y - rs.y,
  };
};

/** Snaps a radian angle to the nearest {@link ROTATE_SNAP_RAD} increment. */
export const snapRotation = (rad: number): number => Math.round(rad / ROTATE_SNAP_RAD) * ROTATE_SNAP_RAD;

/** Parameters for {@link applyRotate}. */
export interface RotateParams {
  start: LayerTransform;
  rect: TransformRect;
  startPointerDoc: Vec2;
  pointerDoc: Vec2;
  /** Snap the resulting absolute rotation to 15° increments. */
  shift: boolean;
}

/**
 * Rotates `start` about the layer center by the angle the pointer sweeps around
 * that center since gesture start. `shift` snaps the resulting absolute rotation
 * to 15° increments. Scale is unchanged.
 */
export const applyRotate = (p: RotateParams): LayerTransform => {
  const { rect, start } = p;
  const m = layerTransformMatrix(start);
  const centerLocal = localCenter(rect);
  const center = applyToPoint(m, centerLocal);
  const a0 = Math.atan2(p.startPointerDoc.y - center.y, p.startPointerDoc.x - center.x);
  const a1 = Math.atan2(p.pointerDoc.y - center.y, p.pointerDoc.x - center.x);
  let rotation = start.rotation + (a1 - a0);
  if (p.shift) {
    rotation = snapRotation(rotation);
  }
  // Keep the center fixed: T = center − R(rotation)·(sx·cx, sy·cy).
  const rs = applyLinear(rotationMat(rotation), {
    x: start.scaleX * centerLocal.x,
    y: start.scaleY * centerLocal.y,
  });
  return {
    rotation,
    scaleX: start.scaleX,
    scaleY: start.scaleY,
    x: center.x - rs.x,
    y: center.y - rs.y,
  };
};

/**
 * The matrix that renders a layer's local (cache) pixels into a NEW document-space
 * surface so that the result, drawn at identity, reproduces the layer as if drawn
 * through `transform` — i.e. `fromTRS(transform)`. Used by the paint-layer bake:
 * old pixels are drawn through this matrix, the layer transform is then reset to
 * identity, and the composite is unchanged. Kept as a named helper so the bake
 * site reads intentionally and composes with `math/mat2d`.
 */
export const bakeMatrix = (transform: LayerTransform): Mat2d => layerTransformMatrix(transform);

/** The identity transform (a reset layer transform after a bake). */
export const IDENTITY_TRANSFORM: LayerTransform = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };
