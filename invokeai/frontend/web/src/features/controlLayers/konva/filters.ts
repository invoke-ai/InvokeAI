/**
 * Konva filters
 * https://konvajs.org/docs/filters/Custom_Filter.html
 */

/**
 * Calculates the lightness (HSL) of a given pixel and sets the alpha channel to that value.
 * This is useful for edge maps and other masks, to make the black areas transparent.
 * @param imageData The image data to apply the filter to
 */
export const LightnessToAlphaFilter = (imageData: ImageData): void => {
  const len = imageData.data.length / 4;
  for (let i = 0; i < len; i++) {
    const r = imageData.data[i * 4 + 0] as number;
    const g = imageData.data[i * 4 + 1] as number;
    const b = imageData.data[i * 4 + 2] as number;
    const a = imageData.data[i * 4 + 3] as number;
    const cMin = Math.min(r, g, b);
    const cMax = Math.max(r, g, b);
    imageData.data[i * 4 + 3] = Math.min(a, (cMin + cMax) / 2);
  }
};

// Utility clamp
const clamp = (v: number, min: number, max: number) => (v < min ? min : v > max ? max : v);

type SimpleAdjustParams = {
  brightness: number; // -1..1 (additive)
  contrast: number; // -1..1 (scale around 128)
  saturation: number; // -1..1
  temperature: number; // -1..1 (blue<->yellow approx)
  tint: number; // -1..1 (green<->magenta approx)
  sharpness: number; // -1..1 (light unsharp mask)
};

/**
 * Per-layer simple adjustments filter (brightness, contrast, saturation, temp, tint, sharpness).
 *
 * Parameters are read from the Konva node attr `adjustmentsSimple` set by the adapter.
 */
type KonvaFilterThis = { getAttr?: (key: string) => unknown };
export const AdjustmentsSimpleFilter = function (this: KonvaFilterThis, imageData: ImageData): void {
  const params = (this?.getAttr?.('adjustmentsSimple') as SimpleAdjustParams | undefined) ?? null;
  if (!params) {
    return;
  }

  const { brightness, contrast, saturation, temperature, tint, sharpness } = params;

  const data = imageData.data;
  const len = data.length / 4;
  const width = (imageData as ImageData & { width: number }).width ?? 0;
  const height = (imageData as ImageData & { height: number }).height ?? 0;

  // Precompute factors
  const brightnessShift = brightness * 255; // additive shift
  const contrastFactor = 1 + contrast; // scale around 128

  // Temperature/Tint multipliers
  const tempK = 0.5;
  const tintK = 0.5;
  const rTempMul = 1 + temperature * tempK;
  const bTempMul = 1 - temperature * tempK;
  // Tint: green <-> magenta. Positive = magenta (R/B up, G down). Negative = green (G up, R/B down).
  const t = clamp(tint, -1, 1) * tintK;
  const mag = Math.abs(t);
  const rTintMul = t >= 0 ? 1 + mag : 1 - mag;
  const gTintMul = t >= 0 ? 1 - mag : 1 + mag;
  const bTintMul = t >= 0 ? 1 + mag : 1 - mag;

  // Saturation matrix (HSL-based approximation via luma coefficients)
  const lumaR = 0.2126;
  const lumaG = 0.7152;
  const lumaB = 0.0722;
  const S = 1 + saturation; // 0..2
  const m00 = lumaR * (1 - S) + S;
  const m01 = lumaG * (1 - S);
  const m02 = lumaB * (1 - S);
  const m10 = lumaR * (1 - S);
  const m11 = lumaG * (1 - S) + S;
  const m12 = lumaB * (1 - S);
  const m20 = lumaR * (1 - S);
  const m21 = lumaG * (1 - S);
  const m22 = lumaB * (1 - S) + S;

  // First pass: apply per-pixel color adjustments (excluding sharpness)
  for (let i = 0; i < len; i++) {
    const idx = i * 4;
    let r = data[idx + 0] as number;
    let g = data[idx + 1] as number;
    let b = data[idx + 2] as number;
    const a = data[idx + 3] as number;

    // Brightness (additive)
    r = r + brightnessShift;
    g = g + brightnessShift;
    b = b + brightnessShift;

    // Contrast around mid-point 128
    r = (r - 128) * contrastFactor + 128;
    g = (g - 128) * contrastFactor + 128;
    b = (b - 128) * contrastFactor + 128;

    // Temperature (R/B axis) and Tint (G vs Magenta)
    r = r * rTempMul * rTintMul;
    g = g * gTintMul;
    b = b * bTempMul * bTintMul;

    // Saturation via matrix
    const r2 = r * m00 + g * m01 + b * m02;
    const g2 = r * m10 + g * m11 + b * m12;
    const b2 = r * m20 + g * m21 + b * m22;

    data[idx + 0] = clamp(r2, 0, 255);
    data[idx + 1] = clamp(g2, 0, 255);
    data[idx + 2] = clamp(b2, 0, 255);
    data[idx + 3] = a;
  }

  // Optional sharpen (simple unsharp mask with 3x3 kernel)
  if (Math.abs(sharpness) > 1e-3 && width > 2 && height > 2) {
    const src = new Uint8ClampedArray(data); // copy of modified data
    const a = Math.max(-1, Math.min(1, sharpness)) * 0.5; // amount
    const center = 1 + 4 * a;
    const neighbor = -a;
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        for (let c = 0; c < 3; c++) {
          const centerPx = src[idx + c] ?? 0;
          const leftPx = src[idx - 4 + c] ?? 0;
          const rightPx = src[idx + 4 + c] ?? 0;
          const topPx = src[idx - width * 4 + c] ?? 0;
          const bottomPx = src[idx + width * 4 + c] ?? 0;
          const v = centerPx * center + leftPx * neighbor + rightPx * neighbor + topPx * neighbor + bottomPx * neighbor;
          data[idx + c] = clamp(v, 0, 255);
        }
        // preserve alpha
      }
    }
  }
};

// Build a 256-length LUT from 0..255 control points (linear interpolation for v1)
export const buildCurveLUT = (points: Array<[number, number]>): number[] => {
  if (!points || points.length === 0) {
    return Array.from({ length: 256 }, (_, i) => i);
  }
  const pts = points
    .map(([x, y]) => [clamp(Math.round(x), 0, 255), clamp(Math.round(y), 0, 255)] as [number, number])
    .sort((a, b) => a[0] - b[0]);
  if ((pts[0]?.[0] ?? 0) !== 0) {
    pts.unshift([0, pts[0]?.[1] ?? 0]);
  }
  const last = pts[pts.length - 1];
  if ((last?.[0] ?? 255) !== 255) {
    pts.push([255, last?.[1] ?? 255]);
  }
  const lut = new Array<number>(256);
  let j = 0;
  for (let x = 0; x <= 255; x++) {
    while (j < pts.length - 2 && x > (pts[j + 1]?.[0] ?? 255)) {
      j++;
    }
    const p0 = pts[j] ?? [0, 0];
    const p1 = pts[j + 1] ?? [255, 255];
    const [x0, y0] = p0;
    const [x1, y1] = p1;
    const t = x1 === x0 ? 0 : (x - x0) / (x1 - x0);
    const y = y0 + (y1 - y0) * t;
    lut[x] = clamp(Math.round(y), 0, 255);
  }
  return lut;
};

type CurvesAdjustParams = {
  master: number[];
  r: number[];
  g: number[];
  b: number[];
};

// Curves filter: apply master curve, then per-channel curves
export const AdjustmentsCurvesFilter = function (this: KonvaFilterThis, imageData: ImageData): void {
  const params = (this?.getAttr?.('adjustmentsCurves') as CurvesAdjustParams | undefined) ?? null;
  if (!params) {
    return;
  }
  const { master, r, g, b } = params;
  if (!master || !r || !g || !b) {
    return;
  }
  const data = imageData.data;
  const len = data.length / 4;
  for (let i = 0; i < len; i++) {
    const idx = i * 4;
    const r0 = data[idx + 0] as number;
    const g0 = data[idx + 1] as number;
    const b0 = data[idx + 2] as number;
    const rm = master[r0] ?? r0;
    const gm = master[g0] ?? g0;
    const bm = master[b0] ?? b0;
    data[idx + 0] = clamp(r[rm] ?? rm, 0, 255);
    data[idx + 1] = clamp(g[gm] ?? gm, 0, 255);
    data[idx + 2] = clamp(b[bm] ?? bm, 0, 255);
  }
};
