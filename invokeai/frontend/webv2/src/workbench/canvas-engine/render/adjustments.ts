import type { CanvasAdjustmentEntry, CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';

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
 * Builds a 256-entry monotone-cubic Fritsch–Carlson LUT without overshoot. Fewer than two points gives identity;
 * outside endpoints uses flat extension.
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

/** True when `entry` cannot change a pixel, regardless of enablement. */
export const isIdentityAdjustmentEntry = (entry: CanvasAdjustmentEntry): boolean => {
  switch (entry.type) {
    case 'brightness-contrast':
      return entry.brightness === 0 && entry.contrast === 0;
    case 'exposure':
      return entry.stops === 0;
    case 'levels':
      return (
        entry.inBlack === 0 &&
        entry.inWhite === 255 &&
        entry.gamma === 1 &&
        entry.outBlack === 0 &&
        entry.outWhite === 255
      );
    case 'hsl':
      return entry.saturation === 0;
    case 'hue':
      return entry.rotation % 360 === 0;
    case 'invert':
      return false;
    case 'curves':
      return isIdentityCurve(entry.curves.r) && isIdentityCurve(entry.curves.g) && isIdentityCurve(entry.curves.b);
  }
};

const contributes = (entry: CanvasAdjustmentEntry): boolean => entry.isEnabled && !isIdentityAdjustmentEntry(entry);

/** True when the stack is a no-op: empty, or only disabled/identity entries. */
export const isIdentityAdjustments = (adjustments: CanvasAdjustmentsContract | undefined): boolean =>
  !adjustments || adjustments.every((entry) => !contributes(entry));

/**
 * One step of the compiled pixel pass. Per-channel entries fold into LUT
 * segments (adjacent ones compose into ONE lut); saturation and hue are
 * cross-channel and stand alone.
 */
export type CompiledAdjustmentSegment =
  | {
      readonly kind: 'lut';
      readonly r: Uint8ClampedArray;
      readonly g: Uint8ClampedArray;
      readonly b: Uint8ClampedArray;
    }
  | { readonly kind: 'saturation'; readonly factor: number }
  | { readonly kind: 'matrix'; readonly m: readonly number[] };

const brightnessContrastLut = (brightness: number, contrast: number): Uint8ClampedArray => {
  const offset = brightness * 255;
  const factor = 1 + contrast;
  const lut = new Uint8ClampedArray(LUT_SIZE);
  for (let i = 0; i < LUT_SIZE; i++) {
    lut[i] = clamp255(Math.round((i + offset - 128) * factor + 128));
  }
  return lut;
};

const levelsLut = (entry: Extract<CanvasAdjustmentEntry, { type: 'levels' }>): Uint8ClampedArray => {
  const inSpan = Math.max(1, entry.inWhite - entry.inBlack);
  const outSpan = entry.outWhite - entry.outBlack;
  const exponent = 1 / Math.max(0.01, entry.gamma);
  const lut = new Uint8ClampedArray(LUT_SIZE);
  for (let i = 0; i < LUT_SIZE; i++) {
    const normalized = Math.min(1, Math.max(0, (i - entry.inBlack) / inSpan));
    lut[i] = clamp255(Math.round(entry.outBlack + Math.pow(normalized, exponent) * outSpan));
  }
  return lut;
};

const invertLut = (): Uint8ClampedArray => {
  const lut = new Uint8ClampedArray(LUT_SIZE);
  for (let i = 0; i < LUT_SIZE; i++) {
    lut[i] = 255 - i;
  }
  return lut;
};

const srgbToLinear = (v: number): number => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
const linearToSrgb = (v: number): number => (v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);

/** True exposure: 2^stops in linear light, not a gamma-space brightness multiply. */
const exposureLut = (stops: number): Uint8ClampedArray => {
  const factor = Math.pow(2, stops);
  const lut = new Uint8ClampedArray(LUT_SIZE);
  for (let i = 0; i < LUT_SIZE; i++) {
    lut[i] = clamp255(Math.round(linearToSrgb(Math.min(1, srgbToLinear(i / 255) * factor)) * 255));
  }
  return lut;
};

const identityLut = (): Uint8ClampedArray => {
  const lut = new Uint8ClampedArray(LUT_SIZE);
  for (let i = 0; i < LUT_SIZE; i++) {
    lut[i] = i;
  }
  return lut;
};

/** The SVG feColorMatrix `hueRotate` matrix: luminance-preserving rotation around the gray axis. */
const hueRotateMatrix = (degrees: number): number[] => {
  const radians = (degrees * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return [
    0.213 + cos * 0.787 - sin * 0.213,
    0.715 - cos * 0.715 - sin * 0.715,
    0.072 - cos * 0.072 + sin * 0.928,
    0.213 - cos * 0.213 + sin * 0.143,
    0.715 + cos * 0.285 + sin * 0.14,
    0.072 - cos * 0.072 - sin * 0.283,
    0.213 - cos * 0.213 - sin * 0.787,
    0.715 - cos * 0.715 + sin * 0.715,
    0.072 + cos * 0.928 + sin * 0.072,
  ];
};

/** `outer` applied after `inner`, folded into one table. */
const composeLut = (inner: Uint8ClampedArray, outer: Uint8ClampedArray): Uint8ClampedArray => {
  const lut = new Uint8ClampedArray(LUT_SIZE);
  for (let i = 0; i < LUT_SIZE; i++) {
    lut[i] = outer[inner[i]];
  }
  return lut;
};

const entryLuts = (
  entry: Extract<CanvasAdjustmentEntry, { type: 'brightness-contrast' | 'exposure' | 'levels' | 'invert' | 'curves' }>
): { r: Uint8ClampedArray; g: Uint8ClampedArray; b: Uint8ClampedArray } => {
  if (entry.type === 'curves') {
    return { b: buildCurveLut(entry.curves.b), g: buildCurveLut(entry.curves.g), r: buildCurveLut(entry.curves.r) };
  }
  if (entry.type === 'levels') {
    const lut = levelsLut(entry);
    const channel = entry.channel ?? 'rgb';
    if (channel === 'rgb') {
      return { b: lut, g: lut, r: lut };
    }
    return {
      b: channel === 'b' ? lut : identityLut(),
      g: channel === 'g' ? lut : identityLut(),
      r: channel === 'r' ? lut : identityLut(),
    };
  }
  const lut =
    entry.type === 'brightness-contrast'
      ? brightnessContrastLut(entry.brightness, entry.contrast)
      : entry.type === 'exposure'
        ? exposureLut(entry.stops)
        : invertLut();
  return { b: lut, g: lut, r: lut };
};

/**
 * Compiles enabled nonidentity entries in order, folding adjacent per-channel operations into LUT trios. Empty
 * means identity.
 */
export const compileAdjustments = (adjustments: CanvasAdjustmentsContract | undefined): CompiledAdjustmentSegment[] => {
  const segments: CompiledAdjustmentSegment[] = [];
  for (const entry of adjustments ?? []) {
    if (!contributes(entry)) {
      continue;
    }
    if (entry.type === 'hsl') {
      segments.push({ factor: 1 + entry.saturation, kind: 'saturation' });
      continue;
    }
    if (entry.type === 'hue') {
      segments.push({ kind: 'matrix', m: hueRotateMatrix(entry.rotation) });
      continue;
    }
    const luts = entryLuts(entry);
    const last = segments[segments.length - 1];
    if (last?.kind === 'lut') {
      segments[segments.length - 1] = {
        b: composeLut(last.b, luts.b),
        g: composeLut(last.g, luts.g),
        kind: 'lut',
        r: composeLut(last.r, luts.r),
      };
    } else {
      segments.push({ ...luts, kind: 'lut' });
    }
  }
  return segments;
};

/** A deterministic cache key fully identifying the stack's pixel effect; entry ids do not affect it. */
export const adjustmentsKey = (adjustments: CanvasAdjustmentsContract | undefined): string => {
  const active = (adjustments ?? []).filter(contributes);
  if (active.length === 0) {
    return 'identity';
  }
  const curveKey = (pts: CurvePoints | undefined): string =>
    pts && pts.length > 0 ? pts.map(([x, y]) => `${x},${y}`).join(';') : '-';
  return active
    .map((entry) => {
      switch (entry.type) {
        case 'brightness-contrast':
          return `bc:${entry.brightness},${entry.contrast}`;
        case 'exposure':
          return `ex:${entry.stops}`;
        case 'levels':
          // Unscoped entries keep their pre-`channel` key so caches survive.
          return `lv:${entry.channel && entry.channel !== 'rgb' ? `${entry.channel}:` : ''}${entry.inBlack},${entry.inWhite},${entry.gamma},${entry.outBlack},${entry.outWhite}`;
        case 'hsl':
          return `s:${entry.saturation}`;
        case 'hue':
          return `h:${entry.rotation}`;
        case 'invert':
          return 'inv';
        case 'curves':
          return `cv:${curveKey(entry.curves.r)}|${curveKey(entry.curves.g)}|${curveKey(entry.curves.b)}`;
      }
    })
    .join('||');
};

/** Applies compiled LUT/saturation segments in one in-place pixel pass, preserving alpha. Identity is a no-op. */
export const applyAdjustments = (imageData: ImageData, adjustments: CanvasAdjustmentsContract | undefined): void => {
  const segments = compileAdjustments(adjustments);
  if (segments.length === 0) {
    return;
  }
  const { data } = imageData;
  for (let i = 0; i + 3 < data.length; i += 4) {
    let r = data[i];
    let g = data[i + 1];
    let b = data[i + 2];
    for (const segment of segments) {
      if (segment.kind === 'lut') {
        r = segment.r[r];
        g = segment.g[g];
        b = segment.b[b];
      } else if (segment.kind === 'saturation') {
        const lum = LUMA_R * r + LUMA_G * g + LUMA_B * b;
        r = clamp255(Math.round(lum + (r - lum) * segment.factor));
        g = clamp255(Math.round(lum + (g - lum) * segment.factor));
        b = clamp255(Math.round(lum + (b - lum) * segment.factor));
      } else {
        const { m } = segment;
        const nr = m[0] * r + m[1] * g + m[2] * b;
        const ng = m[3] * r + m[4] * g + m[5] * b;
        const nb = m[6] * r + m[7] * g + m[8] * b;
        r = clamp255(Math.round(nr));
        g = clamp255(Math.round(ng));
        b = clamp255(Math.round(nb));
      }
    }
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
  }
};
