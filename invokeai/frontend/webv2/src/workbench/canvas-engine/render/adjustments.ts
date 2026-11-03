import type { CanvasAdjustmentEntry, CanvasAdjustmentsContract } from '@workbench/canvas-engine/contracts';
import type { SpatialAdjustmentStep } from '@workbench/canvas-engine/render/spatial/spatialSteps';

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
    case 'noise':
    case 'clarity':
    case 'texture':
    case 'dehaze':
      return entry.amount === 0;
    case 'curves':
      return isIdentityCurve(entry.curves.r) && isIdentityCurve(entry.curves.g) && isIdentityCurve(entry.curves.b);
  }
};

const contributes = (entry: CanvasAdjustmentEntry): boolean => entry.isEnabled && !isIdentityAdjustmentEntry(entry);

/** True when the stack is a no-op: empty, or only disabled/identity entries. */
export const isIdentityAdjustments = (adjustments: CanvasAdjustmentsContract | undefined): boolean =>
  !adjustments || adjustments.every((entry) => !contributes(entry));

/**
 * True when the stack's output depends on pixel position (contributing noise). Such a layer stack must be applied to
 * layer-local pixels before resampling on every path, so resampling shapes its grain exactly as the display does.
 */
export const isPositionDependentAdjustments = (adjustments: CanvasAdjustmentsContract | undefined): boolean =>
  adjustments?.some((entry) => entry.type === 'noise' && contributes(entry)) ?? false;

type SpatialAdjustmentEntry = Extract<CanvasAdjustmentEntry, { type: 'clarity' | 'texture' | 'dehaze' }>;

/**
 * Spatial entries read a neighborhood across the owner's whole surface (its alpha bounds), so they are neither
 * region independent nor resample-commuting. `compileAdjustments` and `applyAdjustments` skip them: they run in the
 * spatial worker on the owner's full source.
 */
export const isSpatialAdjustmentEntry = (entry: CanvasAdjustmentEntry): entry is SpatialAdjustmentEntry =>
  entry.type === 'clarity' || entry.type === 'texture' || entry.type === 'dehaze';

/** True when any contributing entry is spatial. */
export const hasSpatialAdjustments = (adjustments: CanvasAdjustmentsContract | undefined): boolean =>
  adjustments?.some((entry) => isSpatialAdjustmentEntry(entry) && contributes(entry)) ?? false;

/** The stack without its spatial entries: what `applyAdjustments` evaluates, and the display's interim. */
export const perPixelAdjustments = (adjustments: CanvasAdjustmentsContract | undefined): CanvasAdjustmentsContract =>
  (adjustments ?? []).filter((entry) => !isSpatialAdjustmentEntry(entry));

/** One run of a {@link toSpatialPlan}: per-pixel entries for `applyAdjustments`, or steps for one spatial job. */
export type SpatialPlanRun =
  | { readonly kind: 'per-pixel'; readonly adjustments: CanvasAdjustmentsContract }
  | { readonly kind: 'spatial'; readonly steps: readonly SpatialAdjustmentStep[] };

const spatialStep = (entry: SpatialAdjustmentEntry): SpatialAdjustmentStep =>
  entry.type === 'dehaze'
    ? { amount: entry.amount, kind: 'dehaze' }
    : { amount: entry.amount, kind: entry.type, size: entry.size };

/**
 * Splits the contributing entries into ordered alternating runs, so evaluating them in turn equals the stack in list
 * order. Consecutive spatial entries share one run (one worker job); empty means identity.
 */
export const toSpatialPlan = (adjustments: CanvasAdjustmentsContract | undefined): SpatialPlanRun[] => {
  const runs: SpatialPlanRun[] = [];
  for (const entry of adjustments ?? []) {
    if (!contributes(entry)) {
      continue;
    }
    const last = runs[runs.length - 1];
    if (isSpatialAdjustmentEntry(entry)) {
      if (last?.kind === 'spatial') {
        runs[runs.length - 1] = { kind: 'spatial', steps: [...last.steps, spatialStep(entry)] };
      } else {
        runs.push({ kind: 'spatial', steps: [spatialStep(entry)] });
      }
    } else if (last?.kind === 'per-pixel') {
      runs[runs.length - 1] = { adjustments: [...last.adjustments, entry], kind: 'per-pixel' };
    } else {
      runs.push({ adjustments: [entry], kind: 'per-pixel' });
    }
  }
  return runs;
};

/**
 * One step of the compiled pixel pass. Per-channel entries fold into LUT
 * segments (adjacent ones compose into ONE lut); saturation and hue are
 * cross-channel and stand alone. Noise depends on pixel position, so it is its
 * own pass and LUTs never fold across it.
 */
export type CompiledAdjustmentSegment =
  | {
      readonly kind: 'lut';
      readonly r: Uint8ClampedArray;
      readonly g: Uint8ClampedArray;
      readonly b: Uint8ClampedArray;
    }
  | { readonly kind: 'saturation'; readonly factor: number }
  | { readonly kind: 'matrix'; readonly m: readonly number[] }
  | {
      readonly kind: 'noise';
      readonly grain: NoiseEntry['grain'];
      readonly midtones: boolean;
      /** Each of the table's equiprobable quantiles scaled to the entry's σ: channel units, or OKLab a/b for chroma. */
      readonly samples: Float64Array;
      /** The grain kernel's taps 0…R (see `grainKernel`); null at size 1, where every pixel is its own sample. */
      readonly kernel: Float64Array | null;
      readonly seed: number;
    };

type NoiseEntry = Extract<CanvasAdjustmentEntry, { type: 'noise' }>;

/** Standard deviation at `amount` 1, in 0–255 channel units (see the noise entry's contract). */
export const NOISE_FULL_SCALE_SIGMA = 128;
/** Chroma grain's OKLab a and b standard deviation at `amount` 1 (see the noise entry's contract). */
export const NOISE_CHROMA_FULL_SCALE_SIGMA = 0.32;

const NOISE_TABLE_BITS = 10;
const NOISE_TABLE_SIZE = 1 << NOISE_TABLE_BITS;
const NOISE_TABLE_MASK = NOISE_TABLE_SIZE - 1;

/** Acklam's approximation of the standard normal inverse CDF for `p` in (0, 0.5] (relative error < 1.2e-9). */
const inverseNormalCdf = (p: number): number => {
  const c0 = -7.784894002430293e-3;
  const c1 = -3.223964580411365e-1;
  const c2 = -2.400758277161838;
  const c3 = -2.549732539343734;
  const c4 = 4.374664141464968;
  const c5 = 2.938163982698783;
  const d0 = 7.784695709041462e-3;
  const d1 = 3.224671290700398e-1;
  const d2 = 2.445134137142996;
  const d3 = 3.754408661907416;
  if (p < 0.02425) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  const numerator =
    (((((-3.969683028665376e1 * r + 2.209460984245205e2) * r - 2.759285104469687e2) * r + 1.38357751867269e2) * r -
      3.066479806614716e1) *
      r +
      2.506628277459239) *
    q;
  const denominator =
    ((((-5.447609879822406e1 * r + 1.615858368580409e2) * r - 1.556989798598866e2) * r + 6.680131188771972e1) * r -
      1.328068155288572e1) *
      r +
    1;
  return numerator / denominator;
};

/**
 * `NOISE_TABLE_SIZE` equiprobable quantiles of a zero-mean, unit-variance distribution. A uniform table index then
 * samples the distribution with no per-pixel `log`/`sqrt`. Mirrored so the mean is exactly zero, and rescaled so
 * the discrete variance is exactly one; uniform spans ±√3 to match gaussian's standard deviation.
 */
const buildUnitNoiseTable = (quantile: (p: number) => number): Float64Array => {
  const table = new Float64Array(NOISE_TABLE_SIZE);
  let sumSquares = 0;
  for (let i = 0; i < NOISE_TABLE_SIZE / 2; i++) {
    const value = quantile((i + 0.5) / NOISE_TABLE_SIZE);
    table[i] = value;
    table[NOISE_TABLE_SIZE - 1 - i] = -value;
    sumSquares += 2 * value * value;
  }
  const scale = 1 / Math.sqrt(sumSquares / NOISE_TABLE_SIZE);
  for (let i = 0; i < NOISE_TABLE_SIZE; i++) {
    table[i] *= scale;
  }
  return table;
};

// Built on first use: module scope stays free of evaluation work.
const unitNoiseTables: Partial<Record<'gaussian' | 'uniform', Float64Array>> = {};
const unitNoiseTable = (distribution: 'gaussian' | 'uniform'): Float64Array =>
  (unitNoiseTables[distribution] ??= buildUnitNoiseTable(
    distribution === 'gaussian' ? inverseNormalCdf : (p) => Math.sqrt(3) * (2 * p - 1)
  ));

/** Below this, a kernel's outer taps change no 8-bit pixel visibly, so it runs as white noise at a third of the cost. */
const GRAIN_MIN_OUTER_TAP = 1e-3;

/**
 * Taps 0…R of the symmetric, separable grain kernel for `size`: a Gaussian of σ `√(size² − 1) / 2` pixels truncated
 * at 3σ, scaled so the full kernel's squared taps sum to one. Blurring unit-variance white noise along both axes then
 * keeps unit variance exactly, so size never changes strength. The σ makes a grain about `size` pixels across: a
 * pixel itself spans one, and blurring adds 4σ² to its squared width. Null at size 1 and wherever the first outer tap
 * would stay under `GRAIN_MIN_OUTER_TAP` (sizes under about 1.14, within one 0.1 slider step of 1).
 */
const grainKernel = (size: number): Float64Array | null => {
  if (size <= 1) {
    return null;
  }
  const sigma = Math.sqrt(size * size - 1) / 2;
  if (Math.exp(-1 / (2 * sigma * sigma)) < GRAIN_MIN_OUTER_TAP) {
    return null;
  }
  const taps = new Float64Array(Math.ceil(3 * sigma) + 1);
  let sumSquares = 0;
  for (let j = 0; j < taps.length; j++) {
    taps[j] = Math.exp((-j * j) / (2 * sigma * sigma));
    sumSquares += (j === 0 ? 1 : 2) * taps[j] * taps[j];
  }
  const scale = 1 / Math.sqrt(sumSquares);
  for (let j = 0; j < taps.length; j++) {
    taps[j] *= scale;
  }
  return taps;
};

/**
 * The grain's linear span in pixels, `(Σ taps)²` of its normalized kernel (1 at size 1). Box-averaging the grain over
 * B×B pixels, with B much larger than the grain, leaves its standard deviation multiplied by `span / B`.
 */
export const noiseGrainSpan = (size: number): number => {
  const kernel = grainKernel(size);
  if (!kernel) {
    return 1;
  }
  let sum = kernel[0];
  for (let j = 1; j < kernel.length; j++) {
    sum += 2 * kernel[j];
  }
  return sum * sum;
};

const noiseSegment = (entry: NoiseEntry): CompiledAdjustmentSegment => {
  const unit = unitNoiseTable(entry.distribution);
  const sigma = entry.amount * (entry.grain === 'chroma' ? NOISE_CHROMA_FULL_SCALE_SIGMA : NOISE_FULL_SCALE_SIGMA);
  const samples = new Float64Array(NOISE_TABLE_SIZE);
  for (let i = 0; i < NOISE_TABLE_SIZE; i++) {
    samples[i] = sigma * unit[i];
  }
  return {
    grain: entry.grain,
    kernel: grainKernel(entry.size),
    kind: 'noise',
    midtones: entry.response === 'midtones',
    samples,
    seed: entry.seed >>> 0,
  };
};

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
 * Compiles enabled nonidentity per-pixel entries in order (spatial entries are skipped), folding adjacent per-channel
 * operations into LUT trios. Empty means identity.
 */
export const compileAdjustments = (adjustments: CanvasAdjustmentsContract | undefined): CompiledAdjustmentSegment[] => {
  const segments: CompiledAdjustmentSegment[] = [];
  for (const entry of adjustments ?? []) {
    if (!contributes(entry) || isSpatialAdjustmentEntry(entry)) {
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
    if (entry.type === 'noise') {
      segments.push(noiseSegment(entry));
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
        case 'noise':
          return `nz:${entry.distribution},${entry.grain},${entry.response},${entry.size},${entry.amount},${entry.seed}`;
        case 'clarity':
          return `cl:${entry.amount},${entry.size}`;
        case 'texture':
          return `tx:${entry.amount},${entry.size}`;
        case 'dehaze':
          return `dh:${entry.amount}`;
        case 'curves':
          return `cv:${curveKey(entry.curves.r)}|${curveKey(entry.curves.g)}|${curveKey(entry.curves.b)}`;
      }
    })
    .join('||');
};

/** Integer pixel coordinate of an ImageData's top-left pixel in its adjustment stack's reference space. */
export interface AdjustmentOrigin {
  readonly x: number;
  readonly y: number;
}

/** One pixel-major pass over consecutive position-independent segments. */
const applyColorSegments = (data: Uint8ClampedArray, segments: readonly CompiledAdjustmentSegment[]): void => {
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
      } else if (segment.kind === 'matrix') {
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

/** Chris Wellons' `lowbias32`: a bijective, well-avalanched 32-bit integer finalizer. */
const hash32 = (value: number): number => {
  let h = value ^ (value >>> 16);
  h = Math.imul(h, 0x7feb352d);
  h ^= h >>> 15;
  h = Math.imul(h, 0x846ca68b);
  return h ^ (h >>> 16);
};

const NOISE_ROW_STRIDE = 0x9e3779b9;
const NOISE_STEP_SALT = 0x632be5ab;

/**
 * Row `y`'s Weyl sequence: a start hashed from its coordinate and an odd step hashed from the start. A per-row step
 * keeps rows from being shifted copies of each other; pixel `x` hashes `start + x × step`.
 */
const noiseRowStart = (seedKey: number, y: number): number => hash32((seedKey + Math.imul(y, NOISE_ROW_STRIDE)) | 0);
const noiseRowStep = (rowStart: number): number => hash32(rowStart ^ NOISE_STEP_SALT) | 1;

/**
 * Fills `out` with white table samples keyed by `(seed, x, y)` for `count` pixels of row `y` from column `x0`. One
 * hash feeds every channel from disjoint 10-bit fields.
 */
const fillWhiteRow = (
  out: readonly Float64Array[],
  samples: Float64Array,
  seedKey: number,
  x0: number,
  y: number,
  count: number
): void => {
  const rowStart = noiseRowStart(seedKey, y);
  const step = noiseRowStep(rowStart);
  let key = (rowStart + Math.imul(x0, step)) | 0;
  const [c0, c1, c2] = out;
  if (c2) {
    for (let x = 0; x < count; x++, key = (key + step) | 0) {
      const h = hash32(key);
      c0[x] = samples[h & NOISE_TABLE_MASK];
      c1[x] = samples[(h >>> NOISE_TABLE_BITS) & NOISE_TABLE_MASK];
      c2[x] = samples[(h >>> (2 * NOISE_TABLE_BITS)) & NOISE_TABLE_MASK];
    }
  } else if (c1) {
    for (let x = 0; x < count; x++, key = (key + step) | 0) {
      const h = hash32(key);
      c0[x] = samples[h & NOISE_TABLE_MASK];
      c1[x] = samples[(h >>> NOISE_TABLE_BITS) & NOISE_TABLE_MASK];
    }
  } else {
    for (let x = 0; x < count; x++, key = (key + step) | 0) {
      c0[x] = samples[hash32(key) & NOISE_TABLE_MASK];
    }
  }
};

/**
 * Streams the region's noise rows top to bottom, `emit(row, noise)` receiving each row's per-channel values. At size 1
 * they are the white samples; above it, the separable kernel blurs the white field, horizontally into a ring of 2R + 1
 * rows and then vertically. White samples past the region's edges come from the same hash, never from pixels, and
 * every output sums the same samples in the same order wherever the region starts, so the field stays a pure
 * function of the coordinate. Memory is O(R × width), never a padded copy of the region.
 */
const streamNoiseRows = (
  width: number,
  height: number,
  originX: number,
  originY: number,
  segment: Extract<CompiledAdjustmentSegment, { kind: 'noise' }>,
  channels: number,
  emit: (row: number, noise: readonly Float64Array[]) => void
): void => {
  const { kernel, samples } = segment;
  const seedKey = hash32(segment.seed);
  const noise = Array.from({ length: channels }, () => new Float64Array(width));
  if (!kernel) {
    for (let row = 0; row < height; row++) {
      fillWhiteRow(noise, samples, seedKey, originX, originY + row, width);
      emit(row, noise);
    }
    return;
  }
  const radius = kernel.length - 1;
  const span = 2 * radius + 1;
  const center = kernel[0];
  const white = Array.from({ length: channels }, () => new Float64Array(width + 2 * radius));
  const ring = Array.from({ length: channels }, () => new Float64Array(span * width));
  for (let row = 0; row < height + 2 * radius; row++) {
    fillWhiteRow(white, samples, seedKey, originX - radius, originY - radius + row, width + 2 * radius);
    const slot = (row % span) * width;
    for (let c = 0; c < channels; c++) {
      const source = white[c];
      const target = ring[c];
      for (let x = 0; x < width; x++) {
        target[slot + x] = center * source[x + radius];
      }
      for (let j = 1; j <= radius; j++) {
        const tap = kernel[j];
        for (let x = 0, left = radius - j, right = radius + j; x < width; x++, left++, right++) {
          target[slot + x] += tap * (source[left] + source[right]);
        }
      }
    }
    if (row < 2 * radius) {
      continue;
    }
    // The ring now holds the 2R + 1 filtered rows around output row `row − 2R`.
    const middle = ((row - radius) % span) * width;
    for (let c = 0; c < channels; c++) {
      const source = ring[c];
      const target = noise[c];
      for (let x = 0; x < width; x++) {
        target[x] = center * source[middle + x];
      }
      for (let k = 1; k <= radius; k++) {
        const tap = kernel[k];
        const above = ((row - radius - k) % span) * width;
        const below = ((row - radius + k) % span) * width;
        for (let x = 0; x < width; x++) {
          target[x] += tap * (source[above + x] + source[below + x]);
        }
      }
    }
    emit(row - 2 * radius, noise);
  }
};

/** 255 × 10000: Rec. 709 luma weights in ten-thousandths summed over 0–255 channels, so white sums exactly to it. */
const LUMA_SCALE = 2_550_000;
const MIDTONE_NORM = 4 / (LUMA_SCALE * LUMA_SCALE);

/** 4·Y′·(1 − Y′) of a pixel's Rec. 709 luma Y′: 1 at mid-grey, exactly 0 at black and white (integer luma sum). */
const midtoneWeight = (r: number, g: number, b: number): number => {
  const luma = 2126 * r + 7152 * g + 722 * b;
  return luma * (LUMA_SCALE - luma) * MIDTONE_NORM;
};

/** Linear-light encoding table resolution: dark codes sit about five entries apart, so rounding stays exact or ±1. */
const LINEAR_TABLE_MAX = 16383;

interface ChromaTables {
  /** sRGB code → linear light. */
  readonly toLinear: Float64Array;
  /** Linear light × `LINEAR_TABLE_MAX`, rounded → sRGB code. */
  readonly toSrgb: Uint8Array;
  /** One float32 and its bits, the cube root's scratch. */
  readonly bits: Float32Array;
  readonly word: Uint32Array;
}

// Built on first use: module scope stays free of evaluation work.
let chromaTables: ChromaTables | null = null;
const getChromaTables = (): ChromaTables => {
  if (!chromaTables) {
    const toLinear = new Float64Array(LUT_SIZE);
    for (let i = 0; i < LUT_SIZE; i++) {
      toLinear[i] = srgbToLinear(i / 255);
    }
    const toSrgb = new Uint8Array(LINEAR_TABLE_MAX + 1);
    for (let i = 0; i <= LINEAR_TABLE_MAX; i++) {
      toSrgb[i] = Math.round(linearToSrgb(i / LINEAR_TABLE_MAX) * 255);
    }
    const bits = new Float32Array(1);
    chromaTables = { bits, toLinear, toSrgb, word: new Uint32Array(bits.buffer) };
  }
  return chromaTables;
};

const encodeSrgb = (toSrgb: Uint8Array, linear: number): number =>
  linear <= 0 ? 0 : linear >= 1 ? 255 : toSrgb[(linear * LINEAR_TABLE_MAX + 0.5) | 0];

/**
 * Cube root of a non-negative value: the float32 exponent-divide guess (within ~4%) and one Halley step, relative
 * error < 3·10⁻⁵ (far below one 8-bit code) at about a third of `Math.cbrt`'s cost.
 */
const fastCbrt = (value: number, bits: Float32Array, word: Uint32Array): number => {
  bits[0] = value;
  word[0] = ((word[0] / 3) | 0) + 709_921_077;
  const guess = bits[0];
  const cube = guess * guess * guess;
  return (guess * (cube + 2 * value)) / (2 * cube + value);
};

/**
 * Adds OKLab a/b noise to one row, keeping OKLab L. Adding (0, δa, δb) in OKLab adds M₂⁻¹·(0, δa, δb) to the
 * cube-rooted LMS, so L, a and b are never materialized. Out-of-gamut results clamp in linear light.
 */
const applyChromaRow = (
  data: Uint8ClampedArray,
  start: number,
  width: number,
  noiseA: Float64Array,
  noiseB: Float64Array,
  midtones: boolean,
  tables: ChromaTables
): void => {
  const { bits, toLinear, toSrgb, word } = tables;
  for (let x = 0, i = start; x < width; x++, i += 4) {
    if (data[i + 3] === 0) {
      continue;
    }
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const weight = midtones ? midtoneWeight(r, g, b) : 1;
    if (weight === 0) {
      continue;
    }
    const da = noiseA[x] * weight;
    const db = noiseB[x] * weight;
    const lr = toLinear[r];
    const lg = toLinear[g];
    const lb = toLinear[b];
    const l =
      fastCbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb, bits, word) +
      0.3963377774 * da +
      0.2158037573 * db;
    const m =
      fastCbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb, bits, word) -
      0.1055613458 * da -
      0.0638541728 * db;
    const s =
      fastCbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb, bits, word) -
      0.0894841775 * da -
      1.291485548 * db;
    const l3 = l * l * l;
    const m3 = m * m * m;
    const s3 = s * s * s;
    data[i] = encodeSrgb(toSrgb, 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3);
    data[i + 1] = encodeSrgb(toSrgb, -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3);
    data[i + 2] = encodeSrgb(toSrgb, -0.0041960863 * l3 - 0.7034186147 * m3 + 1.707614701 * s3);
  }
};

/** Adds monochrome or color noise to one row; fractional values round in the clamped store. */
const applyRgbRow = (
  data: Uint8ClampedArray,
  start: number,
  width: number,
  noise: readonly Float64Array[],
  midtones: boolean
): void => {
  const [nr, ng = nr, nb = nr] = noise;
  for (let x = 0, i = start; x < width; x++, i += 4) {
    if (data[i + 3] === 0) {
      continue;
    }
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    // One luma weight for all channels keeps neutral pixels neutral on average.
    const weight = midtones ? midtoneWeight(r, g, b) : 1;
    data[i] = r + nr[x] * weight;
    data[i + 1] = g + ng[x] * weight;
    data[i + 2] = b + nb[x] * weight;
  }
};

/**
 * Size 1 monochrome or color grain straight from the hash, the most common case. Flat response adds integer offsets;
 * midtones scales the samples by the pixel's weight and rounds in the clamped store. Monochrome reuses the first
 * 10-bit field for every channel.
 */
const applyPixelRgbNoise = (
  data: Uint8ClampedArray,
  width: number,
  height: number,
  originX: number,
  originY: number,
  segment: Extract<CompiledAdjustmentSegment, { kind: 'noise' }>
): void => {
  const { midtones, samples } = segment;
  const monochrome = segment.grain === 'monochrome';
  // Pixels are integers, so rounding a flat offset equals rounding the noisy value; midtones never reads these.
  const offsets = midtones ? null : Int16Array.from(samples, Math.round);
  const seedKey = hash32(segment.seed);
  for (let row = 0, i = 0; row < height; row++) {
    const rowStart = noiseRowStart(seedKey, originY + row);
    const step = noiseRowStep(rowStart);
    let key = (rowStart + Math.imul(originX, step)) | 0;
    const rowEnd = i + width * 4;
    if (midtones && monochrome) {
      for (; i < rowEnd; i += 4, key = (key + step) | 0) {
        if (data[i + 3] === 0) {
          continue;
        }
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const offset = samples[hash32(key) & NOISE_TABLE_MASK] * midtoneWeight(r, g, b);
        data[i] = r + offset;
        data[i + 1] = g + offset;
        data[i + 2] = b + offset;
      }
    } else if (midtones) {
      for (; i < rowEnd; i += 4, key = (key + step) | 0) {
        if (data[i + 3] === 0) {
          continue;
        }
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        // One luma weight for all channels keeps neutral pixels neutral on average.
        const weight = midtoneWeight(r, g, b);
        const h = hash32(key);
        data[i] = r + samples[h & NOISE_TABLE_MASK] * weight;
        data[i + 1] = g + samples[(h >>> NOISE_TABLE_BITS) & NOISE_TABLE_MASK] * weight;
        data[i + 2] = b + samples[(h >>> (2 * NOISE_TABLE_BITS)) & NOISE_TABLE_MASK] * weight;
      }
    } else if (monochrome) {
      for (; i < rowEnd; i += 4, key = (key + step) | 0) {
        if (data[i + 3] === 0) {
          continue;
        }
        const offset = offsets![hash32(key) & NOISE_TABLE_MASK];
        // Uint8ClampedArray stores clamp to 0–255.
        data[i] += offset;
        data[i + 1] += offset;
        data[i + 2] += offset;
      }
    } else {
      for (; i < rowEnd; i += 4, key = (key + step) | 0) {
        if (data[i + 3] === 0) {
          continue;
        }
        const h = hash32(key);
        data[i] += offsets![h & NOISE_TABLE_MASK];
        data[i + 1] += offsets![(h >>> NOISE_TABLE_BITS) & NOISE_TABLE_MASK];
        data[i + 2] += offsets![(h >>> (2 * NOISE_TABLE_BITS)) & NOISE_TABLE_MASK];
      }
    }
  }
};

/** Adds the segment's grain keyed by `(seed, x, y)` in reference space; alpha and fully transparent pixels stay. */
const applyNoise = (
  data: Uint8ClampedArray,
  width: number,
  height: number,
  origin: AdjustmentOrigin,
  segment: Extract<CompiledAdjustmentSegment, { kind: 'noise' }>
): void => {
  const originX = Math.round(origin.x);
  const originY = Math.round(origin.y);
  const { grain, midtones } = segment;
  if (grain !== 'chroma' && !segment.kernel) {
    applyPixelRgbNoise(data, width, height, originX, originY, segment);
    return;
  }
  if (grain === 'chroma') {
    const tables = getChromaTables();
    streamNoiseRows(width, height, originX, originY, segment, 2, (row, noise) =>
      applyChromaRow(data, row * width * 4, width, noise[0], noise[1], midtones, tables)
    );
    return;
  }
  streamNoiseRows(width, height, originX, originY, segment, grain === 'color' ? 3 : 1, (row, noise) =>
    applyRgbRow(data, row * width * 4, width, noise, midtones)
  );
};

/** Unpremultiplied RGBA8 pixels, row-major: an `ImageData` or a worker result viewed the same way. */
export interface AdjustablePixels {
  readonly data: Uint8ClampedArray;
  readonly width: number;
  readonly height: number;
}

/**
 * Applies the compiled stack in place, preserving alpha; identity is a no-op. `origin` places the pixels in the
 * stack's reference space. Spatial entries are skipped (see {@link isSpatialAdjustmentEntry}); every entry applied is
 * region independent: adjusting a sub-rectangle with its matching origin yields exactly the pixels a whole-surface
 * pass would there.
 */
export const applyAdjustments = (
  imageData: AdjustablePixels,
  adjustments: CanvasAdjustmentsContract | undefined,
  origin: AdjustmentOrigin
): void => {
  const segments = compileAdjustments(adjustments);
  if (segments.length === 0) {
    return;
  }
  const { data, height, width } = imageData;
  let runStart = 0;
  for (let index = 0; index <= segments.length; index++) {
    const segment = segments[index];
    if (segment && segment.kind !== 'noise') {
      continue;
    }
    if (index > runStart) {
      applyColorSegments(data, runStart === 0 && !segment ? segments : segments.slice(runStart, index));
    }
    if (segment) {
      applyNoise(data, width, height, origin, segment);
    }
    runStart = index + 1;
  }
};
