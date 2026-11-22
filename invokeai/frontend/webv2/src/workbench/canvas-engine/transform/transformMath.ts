/** Derive fresh transforms from gesture-start values; negative scales are flips. */

import type { CanvasLayerBaseContract } from '@workbench/canvas-engine/contracts';
import type { Mat2d, Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint, fromTRS, multiply } from '@workbench/canvas-engine/math/mat2d';
import { snapMovedPoint } from '@workbench/canvas-engine/math/snapping';

export type LayerTransform = CanvasLayerBaseContract['transform'];

/** Frames and anchors derive from local content rects (negative offsets included), never origin-zero bounds. */
export interface TransformRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type TransformHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

export type TransformAnchor = TransformHandle | 'center';

export type TransformTarget =
  | { kind: 'scale'; handle: TransformHandle }
  | { kind: 'rotate'; handle: TransformHandle }
  | { kind: 'move' };

/** Handles in hit-test priority order: corners first (they win over adjacent edges). */
export const TRANSFORM_HANDLES: readonly TransformHandle[] = ['nw', 'ne', 'se', 'sw', 'n', 'e', 's', 'w'];

export const CORNER_HANDLES: readonly TransformHandle[] = ['nw', 'ne', 'se', 'sw'];

/** Screen px. */
export const TRANSFORM_HANDLE_HIT_PX = 14;

/** Screen px, from the corner outward. */
export const TRANSFORM_ROTATE_ZONE_PX = 22;

/** Screen px; shared by drawing and hit testing. */
export const TRANSFORM_ROTATE_NUB_PX = 18;

/** Screen px. */
export const TRANSFORM_ROTATE_NUB_HIT_PX = 11;

export const ROTATE_SNAP_RAD = Math.PI / 12;

/** Smallest permitted |scaleX|/|scaleY| — keeps the layer matrix invertible. */
export const MIN_ABS_SCALE = 0.01;

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

const hasWest = (h: TransformAnchor): boolean => h === 'nw' || h === 'w' || h === 'sw';
const hasEast = (h: TransformAnchor): boolean => h === 'ne' || h === 'e' || h === 'se';
const hasNorth = (h: TransformAnchor): boolean => h === 'nw' || h === 'n' || h === 'ne';
const hasSouth = (h: TransformAnchor): boolean => h === 'sw' || h === 's' || h === 'se';

export const isCornerHandle = (h: TransformHandle): boolean => h.length === 2;

export const localHandlePoint = (rect: TransformRect, handle: TransformAnchor): Vec2 => ({
  x: rect.x + (hasWest(handle) ? 0 : hasEast(handle) ? rect.width : rect.width / 2),
  y: rect.y + (hasNorth(handle) ? 0 : hasSouth(handle) ? rect.height : rect.height / 2),
});

const localCenter = (rect: TransformRect): Vec2 => ({
  x: rect.x + rect.width / 2,
  y: rect.y + rect.height / 2,
});

export const layerTransformMatrix = (t: LayerTransform): Mat2d =>
  fromTRS({ x: t.x, y: t.y }, t.rotation, t.scaleX, t.scaleY);

const rotationMat = (rad: number): { a: number; b: number; c: number; d: number } => {
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return { a: cos, b: sin, c: -sin, d: cos };
};

const applyLinear = (m: { a: number; b: number; c: number; d: number }, v: Vec2): Vec2 => ({
  x: m.a * v.x + m.c * v.y,
  y: m.b * v.x + m.d * v.y,
});

type LinearTransform = Pick<LayerTransform, 'rotation' | 'scaleX' | 'scaleY'>;

/** The translation that puts `local` at `doc`: T = doc − R·(sx·local.x, sy·local.y). */
export const moveLocalPointTo = (linear: LinearTransform, local: Vec2, doc: Vec2): LayerTransform => {
  const rs = applyLinear(rotationMat(linear.rotation), { x: linear.scaleX * local.x, y: linear.scaleY * local.y });
  return { rotation: linear.rotation, scaleX: linear.scaleX, scaleY: linear.scaleY, x: doc.x - rs.x, y: doc.y - rs.y };
};

export const layerPointToDocument = (transform: LayerTransform, local: Vec2): Vec2 =>
  applyToPoint(layerTransformMatrix(transform), local);

export const retransformAbout = (start: LayerTransform, local: Vec2, next: Partial<LinearTransform>): LayerTransform =>
  moveLocalPointTo({ ...start, ...next }, local, layerPointToDocument(start, local));

/** Mirrors in document space about the content centre, so the angle negates and one local axis flips. */
export const flipTransform = (
  transform: LayerTransform,
  rect: TransformRect,
  direction: 'horizontal' | 'vertical'
): LayerTransform => {
  const center = localCenter(rect);
  return moveLocalPointTo(
    {
      rotation: -transform.rotation,
      scaleX: direction === 'horizontal' ? -transform.scaleX : transform.scaleX,
      scaleY: direction === 'vertical' ? -transform.scaleY : transform.scaleY,
    },
    center,
    layerPointToDocument(transform, center)
  );
};

/** The layer's own box in document pixels (not its rotated bounds). */
export const transformedSize = (transform: LayerTransform, rect: TransformRect): { width: number; height: number } => ({
  height: rect.height * Math.abs(transform.scaleY),
  width: rect.width * Math.abs(transform.scaleX),
});

/** A handle's outward normal in LOCAL space (y-down, matching document/screen). */
const localOutward = (handle: TransformHandle): Vec2 => ({
  x: hasWest(handle) ? -1 : hasEast(handle) ? 1 : 0,
  y: hasNorth(handle) ? -1 : hasSouth(handle) ? 1 : 0,
});

/** Document space; `handles` follows {@link TRANSFORM_HANDLES}. */
export interface TransformOverlayGeometry {
  corners: Vec2[];
  handles: Vec2[];
  center: Vec2;
  rotationAnchor: Vec2;
}

export const transformOverlayGeometry = (transform: LayerTransform, rect: TransformRect): TransformOverlayGeometry => {
  const m = layerTransformMatrix(transform);
  return {
    center: applyToPoint(m, localCenter(rect)),
    corners: (['nw', 'ne', 'se', 'sw'] as const).map((h) => applyToPoint(m, localHandlePoint(rect, h))),
    handles: TRANSFORM_HANDLES.map((h) => applyToPoint(m, localHandlePoint(rect, h))),
    rotationAnchor: applyToPoint(m, localHandlePoint(rect, 'n')),
  };
};

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

const distanceToSegment = (p: Vec2, a: Vec2, b: Vec2): number => {
  const abx = b.x - a.x;
  const aby = b.y - a.y;
  const len2 = abx * abx + aby * aby;
  const t = len2 > 0 ? Math.max(0, Math.min(1, ((p.x - a.x) * abx + (p.y - a.y) * aby) / len2)) : 0;
  return Math.hypot(p.x - (a.x + t * abx), p.y - (a.y + t * aby));
};

export interface TransformHitTestParams {
  transform: LayerTransform;
  rect: TransformRect;
  toScreen: (docPoint: Vec2) => Vec2;
  point: Vec2;
  handleHitPx?: number;
  rotateZonePx?: number;
  rotateNubHitPx?: number;
}

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

  // The nub is tested after the interior and before corner rotate zones; a miss would reset the session.
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

const RESIZE_CURSORS = ['ew-resize', 'nesw-resize', 'ns-resize', 'nwse-resize'] as const;

/** A uniform nonrotating viewport scale preserves the document angle. */
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

const clampScale = (s: number): number => {
  if (!Number.isFinite(s) || s === 0) {
    return MIN_ABS_SCALE;
  }
  const sign = s < 0 ? -1 : 1;
  return Math.abs(s) < MIN_ABS_SCALE ? sign * MIN_ABS_SCALE : s;
};

export const scaleForLength = (length: number, rectLength: number, current: number): number =>
  (current < 0 ? -1 : 1) * Math.max(MIN_ABS_SCALE, length / rectLength);

export const scaleAxis = (
  start: LayerTransform,
  axis: 'x' | 'y',
  scale: number,
  linked: boolean
): Pick<LayerTransform, 'scaleX' | 'scaleY'> => {
  const next = clampScale(scale);
  const [own, other] = axis === 'x' ? [start.scaleX, start.scaleY] : [start.scaleY, start.scaleX];
  const linkedOther = linked ? clampScale(other * Math.abs(next / own)) : other;
  return axis === 'x' ? { scaleX: next, scaleY: linkedOther } : { scaleX: linkedOther, scaleY: next };
};

export const constrainMoveDelta = (delta: Vec2, shift: boolean): Vec2 => {
  if (!shift) {
    return delta;
  }
  return Math.abs(delta.x) >= Math.abs(delta.y) ? { x: delta.x, y: 0 } : { x: 0, y: delta.y };
};

export const applyMove = (start: LayerTransform, deltaDoc: Vec2, shift: boolean, grid = 0): LayerTransform => ({
  ...start,
  ...snapMovedPoint(start, constrainMoveDelta(deltaDoc, shift), grid),
});

export interface ScaleParams {
  start: LayerTransform;
  rect: TransformRect;
  handle: TransformHandle;
  startPointerDoc: Vec2;
  pointerDoc: Vec2;
  constrain: boolean;
  alt: boolean;
  grid?: number;
}

/** Free edges affect one axis; flips allowed down to the minimum absolute scale. */
export const applyScale = (p: ScaleParams): LayerTransform => {
  const { alt, constrain, handle, rect, start } = p;
  const m = layerTransformMatrix(start);
  const handleLocal = localHandlePoint(rect, handle);
  const anchorLocal = alt ? localCenter(rect) : localHandlePoint(rect, OPPOSITE[handle]);
  const anchorDoc = applyToPoint(m, anchorLocal);
  const handleDoc = applyToPoint(m, handleLocal);
  // An axis the pointer never moved along keeps its scale.
  const targetDoc = snapMovedPoint(
    handleDoc,
    { x: p.pointerDoc.x - p.startPointerDoc.x, y: p.pointerDoc.y - p.startPointerDoc.y },
    p.grid ?? 0
  );

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

  if (constrain) {
    const fx = start.scaleX !== 0 ? sx / start.scaleX : 1;
    const fy = start.scaleY !== 0 ? sy / start.scaleY : 1;
    if (isCornerHandle(handle)) {
      const f = Math.abs(fx) >= Math.abs(fy) ? fx : fy;
      sx = start.scaleX * f;
      sy = start.scaleY * f;
    } else if (dlx !== 0) {
      sy = start.scaleY * Math.abs(fx);
    } else {
      sx = start.scaleX * Math.abs(fy);
    }
  }

  return moveLocalPointTo(
    { rotation: start.rotation, scaleX: clampScale(sx), scaleY: clampScale(sy) },
    anchorLocal,
    anchorDoc
  );
};

export const snapRotation = (rad: number): number => Math.round(rad / ROTATE_SNAP_RAD) * ROTATE_SNAP_RAD;

export interface RotateParams {
  start: LayerTransform;
  rect: TransformRect;
  startPointerDoc: Vec2;
  pointerDoc: Vec2;
  shift: boolean;
}

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
  return moveLocalPointTo({ rotation, scaleX: start.scaleX, scaleY: start.scaleY }, centerLocal, center);
};

export const bakeMatrix = (transform: LayerTransform): Mat2d => layerTransformMatrix(transform);

export const IDENTITY_TRANSFORM: LayerTransform = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };

const SHEAR_TOLERANCE = 1e-9;

/** Null when the composite shears or collapses an axis, which a TRS transform cannot express. */
export const composeLayerTransforms = (outer: LayerTransform, inner: LayerTransform): LayerTransform | null => {
  const { a, b, c, d, e, f } = multiply(layerTransformMatrix(outer), layerTransformMatrix(inner));
  const xAxis = Math.hypot(a, b);
  const yAxis = Math.hypot(c, d);
  if (xAxis === 0 || yAxis === 0 || Math.abs(a * c + b * d) > SHEAR_TOLERANCE * xAxis * yAxis) {
    return null;
  }
  const scaleX = Math.sign(outer.scaleX * inner.scaleX) < 0 ? -xAxis : xAxis;
  return {
    rotation: Math.atan2(b / scaleX, a / scaleX),
    scaleX,
    scaleY: (a * d - b * c) / scaleX,
    x: e,
    y: f,
  };
};
