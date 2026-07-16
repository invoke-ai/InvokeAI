/**
 * Pure raster-adjustment math (no DOM, no engine, no React).
 *
 * Turns a {@link CanvasAdjustmentsContract} into per-channel 256-entry lookup
 * tables and applies them (plus saturation) to raw `ImageData`. Kept pure and
 * allocation-conscious so it can drive BOTH the non-destructive display cache
 * (`adjustedSurfaceCache.ts`) and the generation composite
 * (`export/compositeForGeneration.ts`), and so its correctness is exhaustively
 * node-testable without any canvas.
 *
 * ## Composition order (documented, tested)
 *
 * Per channel, a single LUT is built as `contrast(brightness(curve(i)))`:
 *  1. **Curve** — the per-channel monotone-cubic curve remaps the raw value.
 *  2. **Brightness** — additive offset (`+ brightness * 255`).
 *  3. **Contrast** — linear scale about mid-grey (`(v - 128) * (1 + contrast) + 128`).
 * The LUT is applied to R/G/B; alpha is never touched. **Saturation** is applied
 * last, per pixel, on the post-LUT values (a luminance lerp), since it mixes
 * channels and cannot be expressed as an independent per-channel LUT.
 *
 * All scalar params are in `[-1, 1]` with `0` = identity; curve control points
 * are `[input, output]` pairs in `[0, 255]`.
 */

import type { CanvasAdjustmentsContract } from '@workbench/types';

/** The identity adjustment (no brightness/contrast/saturation, no curves). */
export const DEFAULT_ADJUSTMENTS: CanvasAdjustmentsContract = { brightness: 0, contrast: 0, saturation: 0 };

const LUT_SIZE = 256;

/** ITU-R BT.601 luma weights, matching legacy grayscale/lightness math. */
const LUMA_R = 0.299;
const LUMA_G = 0.587;
const LUMA_B = 0.114;

const clamp255 = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : v);

type CurvePoints = readonly (readonly [number, number])[];

/** True when a channel's curve points describe the identity mapping (absent, or exactly 0→0 / 255→255). */
const isIdentityCurve = (points: CurvePoints | undefined): boolean => {
  if (!points || points.length === 0) {
    return true;
  }
  // Any 2-point curve that is exactly the diagonal is identity.
  if (points.length === 2) {
    const sorted = [...points].sort((a, b) => a[0] - b[0]);
    const [p0, p1] = sorted;
    return p0[0] === 0 && p0[1] === 0 && p1[0] === 255 && p1[1] === 255;
  }
  return false;
};

/**
 * Builds a 256-entry LUT that maps input → output through the channel's curve
 * control points using monotone-cubic (Fritsch–Carlson) interpolation, so the
 * result never overshoots between points. Fewer than two points → identity.
 * Values before the first / after the last point are clamped to that point's
 * output (flat extension).
 */
export const buildCurveLut = (points: CurvePoints | undefined): Uint8ClampedArray => {
  const lut = new Uint8ClampedArray(LUT_SIZE);
  if (isIdentityCurve(points)) {
    for (let i = 0; i < LUT_SIZE; i++) {
      lut[i] = i;
    }
    return lut;
  }

  // Clean + sort + dedupe by x (keep the last y for a duplicated x).
  const byX = new Map<number, number>();
  for (const [x, y] of points as CurvePoints) {
    byX.set(clamp255(Math.round(x)), clamp255(y));
  }
  const xs = [...byX.keys()].sort((a, b) => a - b);
  if (xs.length < 2) {
    for (let i = 0; i < LUT_SIZE; i++) {
      lut[i] = i;
    }
    return lut;
  }
  const ys = xs.map((x) => byX.get(x) as number);

  const n = xs.length;
  // Secant slopes between consecutive points.
  const delta: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    const dx = xs[i + 1] - xs[i];
    delta.push(dx === 0 ? 0 : (ys[i + 1] - ys[i]) / dx);
  }
  // Fritsch–Carlson tangents (m) enforcing monotonicity.
  const m: number[] = Array.from({ length: n }, () => 0);
  m[0] = delta[0];
  m[n - 1] = delta[n - 2];
  for (let i = 1; i < n - 1; i++) {
    if (delta[i - 1] * delta[i] <= 0) {
      m[i] = 0;
    } else {
      m[i] = (delta[i - 1] + delta[i]) / 2;
    }
  }
  for (let i = 0; i < n - 1; i++) {
    if (delta[i] === 0) {
      m[i] = 0;
      m[i + 1] = 0;
      continue;
    }
    const a = m[i] / delta[i];
    const b = m[i + 1] / delta[i];
    const h = Math.hypot(a, b);
    if (h > 3) {
      const t = 3 / h;
      m[i] = t * a * delta[i];
      m[i + 1] = t * b * delta[i];
    }
  }

  let seg = 0;
  for (let i = 0; i < LUT_SIZE; i++) {
    if (i <= xs[0]) {
      lut[i] = ys[0];
      continue;
    }
    if (i >= xs[n - 1]) {
      lut[i] = ys[n - 1];
      continue;
    }
    while (seg < n - 2 && i > xs[seg + 1]) {
      seg += 1;
    }
    const x0 = xs[seg];
    const x1 = xs[seg + 1];
    const hSeg = x1 - x0;
    const t = (i - x0) / hSeg;
    const t2 = t * t;
    const t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;
    const value = h00 * ys[seg] + h10 * hSeg * m[seg] + h01 * ys[seg + 1] + h11 * hSeg * m[seg + 1];
    lut[i] = clamp255(Math.round(value));
  }
  return lut;
};

/**
 * Builds the composed per-channel LUTs for `adjustments`:
 * `contrast(brightness(curve(i)))`, clamped to `[0, 255]`. Brightness/contrast are
 * shared across channels; the curve is per-channel.
 */
export const buildAdjustmentLuts = (
  adjustments: CanvasAdjustmentsContract
): { r: Uint8ClampedArray; g: Uint8ClampedArray; b: Uint8ClampedArray } => {
  const brightnessOffset = (adjustments.brightness ?? 0) * 255;
  const contrastFactor = 1 + (adjustments.contrast ?? 0);
  const curves = adjustments.curves;

  const compose = (curveLut: Uint8ClampedArray): Uint8ClampedArray => {
    const out = new Uint8ClampedArray(LUT_SIZE);
    for (let i = 0; i < LUT_SIZE; i++) {
      const curved = curveLut[i];
      const brightened = curved + brightnessOffset;
      const contrasted = (brightened - 128) * contrastFactor + 128;
      out[i] = clamp255(Math.round(contrasted));
    }
    return out;
  };

  return {
    b: compose(buildCurveLut(curves?.b)),
    g: compose(buildCurveLut(curves?.g)),
    r: compose(buildCurveLut(curves?.r)),
  };
};

/** True when `adjustments` is a no-op (identity brightness/contrast/saturation + identity curves). */
export const isIdentityAdjustments = (adjustments: CanvasAdjustmentsContract | undefined): boolean => {
  if (!adjustments) {
    return true;
  }
  if ((adjustments.brightness ?? 0) !== 0 || (adjustments.contrast ?? 0) !== 0 || (adjustments.saturation ?? 0) !== 0) {
    return false;
  }
  const { curves } = adjustments;
  if (!curves) {
    return true;
  }
  return isIdentityCurve(curves.r) && isIdentityCurve(curves.g) && isIdentityCurve(curves.b);
};

/** A deterministic cache key fully identifying an adjustment's pixel effect. */
export const adjustmentsKey = (adjustments: CanvasAdjustmentsContract | undefined): string => {
  if (isIdentityAdjustments(adjustments)) {
    return 'identity';
  }
  const a = adjustments as CanvasAdjustmentsContract;
  const curveKey = (pts: CurvePoints | undefined): string =>
    pts && pts.length > 0 ? pts.map(([x, y]) => `${x},${y}`).join(';') : '-';
  return [
    `b${a.brightness ?? 0}`,
    `c${a.contrast ?? 0}`,
    `s${a.saturation ?? 0}`,
    `r:${curveKey(a.curves?.r)}`,
    `g:${curveKey(a.curves?.g)}`,
    `bl:${curveKey(a.curves?.b)}`,
  ].join('|');
};

/**
 * Applies `adjustments` to `imageData` IN PLACE: the composed per-channel LUTs
 * remap R/G/B, then saturation lerps each channel toward its luma. Alpha is never
 * modified. A no-op for identity adjustments.
 */
export const applyAdjustments = (imageData: ImageData, adjustments: CanvasAdjustmentsContract | undefined): void => {
  if (isIdentityAdjustments(adjustments)) {
    return;
  }
  const a = adjustments as CanvasAdjustmentsContract;
  const { b: lutB, g: lutG, r: lutR } = buildAdjustmentLuts(a);
  const sat = 1 + (a.saturation ?? 0);
  const applySaturation = sat !== 1;
  const { data } = imageData;
  for (let i = 0; i + 3 < data.length; i += 4) {
    let r = lutR[data[i]];
    let g = lutG[data[i + 1]];
    let b = lutB[data[i + 2]];
    if (applySaturation) {
      const lum = LUMA_R * r + LUMA_G * g + LUMA_B * b;
      r = clamp255(Math.round(lum + (r - lum) * sat));
      g = clamp255(Math.round(lum + (g - lum) * sat));
      b = clamp255(Math.round(lum + (b - lum) * sat));
    }
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
  }
};
