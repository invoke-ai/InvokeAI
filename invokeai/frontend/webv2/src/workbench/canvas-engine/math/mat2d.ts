/** Immutable 2D affine math matching canvas transforms: x' = a*x + c*y + e; y' = b*x + d*y + f. */

import type { Mat2d, Vec2 } from '@workbench/canvas-engine/types';

/** Returns the identity matrix. */
export const identity = (): Mat2d => ({ a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });

/** Returns a*b: apply b first, then a, matching DOMMatrix multiplication. */
export const multiply = (a: Mat2d, b: Mat2d): Mat2d => ({
  a: a.a * b.a + a.c * b.b,
  b: a.b * b.a + a.d * b.b,
  c: a.a * b.c + a.c * b.d,
  d: a.b * b.c + a.d * b.d,
  e: a.a * b.e + a.c * b.f + a.e,
  f: a.b * b.e + a.d * b.f + a.f,
});

/** Inverts a matrix. Returns `null` if the matrix is singular (determinant ~0). */
export const invert = (m: Mat2d): Mat2d | null => {
  const det = m.a * m.d - m.b * m.c;
  if (Math.abs(det) < 1e-12) {
    return null;
  }
  const invDet = 1 / det;
  const a = m.d * invDet;
  const b = -m.b * invDet;
  const c = -m.c * invDet;
  const d = m.a * invDet;
  const e = -(a * m.e + c * m.f);
  const f = -(b * m.e + d * m.f);
  return { a, b, c, d, e, f };
};

/** Returns a matrix translated by `v` (post-multiplies a translation). */
export const translate = (m: Mat2d, v: Vec2): Mat2d => multiply(m, { a: 1, b: 0, c: 0, d: 1, e: v.x, f: v.y });

/** Returns a matrix scaled by `sx`/`sy` (defaults `sy` to `sx` for uniform scale). */
export const scale = (m: Mat2d, sx: number, sy: number = sx): Mat2d =>
  multiply(m, { a: sx, b: 0, c: 0, d: sy, e: 0, f: 0 });

/** Returns a matrix rotated by `rad` radians (positive = clockwise in canvas space). */
export const rotate = (m: Mat2d, rad: number): Mat2d => {
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return multiply(m, { a: cos, b: sin, c: -sin, d: cos, e: 0, f: 0 });
};

/** Applies a matrix to a point, returning the transformed point. */
export const applyToPoint = (m: Mat2d, p: Vec2): Vec2 => ({
  x: m.a * p.x + m.c * p.y + m.e,
  y: m.b * p.x + m.d * p.y + m.f,
});

/** Composes translate*rotate*scale, applying local scale first, then rotation and translation. */
export const fromTRS = (translation: Vec2, rotationRad: number, scaleX: number, scaleY: number = scaleX): Mat2d => {
  let m = identity();
  m = translate(m, translation);
  m = rotate(m, rotationRad);
  m = scale(m, scaleX, scaleY);
  return m;
};

/**
 * Approximate uniform scale from the geometric mean of transformed unit-axis lengths; exact for uniform scale
 * without shear.
 */
export const getScale = (m: Mat2d): number => {
  const scaleX = Math.hypot(m.a, m.b);
  const scaleY = Math.hypot(m.c, m.d);
  return Math.sqrt(scaleX * scaleY);
};
