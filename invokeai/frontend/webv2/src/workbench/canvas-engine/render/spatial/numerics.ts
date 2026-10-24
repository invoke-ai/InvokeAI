// SPDX-License-Identifier: Apache-2.0 AND MIT
/**
 * Scalar ports of the backend's sRGB, Oklab and gamut-clip conversions
 * (`invokeai/backend/image_util/color_conversion.py`, `color.py`), with its constants, so canvas previews match
 * the nodes.
 *
 * The gamut clip (maximum-saturation polynomial, cusp, gamut intersection and adaptive lightness anchor) ports
 * Björn Ottosson's reference implementation, https://bottosson.github.io/posts/gamutclipping/, under this notice:
 *
 * Copyright (c) 2021 Björn Ottosson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of
 * the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

const SRGB_TO_LINEAR_THRESHOLD = 0.0404482362771082;
const LINEAR_TO_SRGB_THRESHOLD = 0.0031308;
/** torch.finfo(torch.float32).max, the backend's "no intersection" step. */
const MAX_FLOAT32 = 3.4028234663852886e38;
const GAMUT_CLIP_ALPHA = 0.05;
const GAMUT_CLIP_MIN_CHROMA = 0.00001;
/** The backend's `_GAMUT_TOLERANCE`: smaller overshoots are conversion noise and are clamped, not clipped. */
const GAMUT_TOLERANCE = 1e-5;

/** Python's `round()` and `torch.round`: ties go to the even integer. */
export const roundHalfEven = (value: number): number => {
  const rounded = Math.round(value);
  return rounded - value === 0.5 && rounded % 2 !== 0 ? rounded - 1 : rounded;
};

const buildLinearFromSrgb8 = (): Float64Array => {
  const table = new Float64Array(256);
  for (let value = 0; value < 256; value++) {
    const normalized = value / 255;
    table[value] = normalized <= SRGB_TO_LINEAR_THRESHOLD ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
  }
  return table;
};

const LINEAR_FROM_SRGB8 = buildLinearFromSrgb8();

/** The backend's 8-bit encoding: clamp to [0, 1], gamma-encode, scale and round half to even. */
const srgb8FromLinearDirect = (linear: number): number => {
  const clamped = linear < 0 ? 0 : linear > 1 ? 1 : linear;
  const encoded = clamped <= LINEAR_TO_SRGB_THRESHOLD ? clamped * 12.92 : 1.055 * clamped ** (1 / 2.4) - 0.055;
  return roundHalfEven(encoded * 255);
};

const SRGB8_BUCKETS = 4096;

/**
 * Exact replacement for the per-channel power in {@link srgb8FromLinearDirect}: `thresholds[k]` is the smallest
 * double that encodes to byte k or above (found by bisection over the direct encoding, which is monotone), and each
 * of the uniform buckets records the byte at its lower edge, leaving at most a couple of threshold steps per lookup.
 */
const buildSrgb8Encoding = (): { readonly thresholds: Float64Array; readonly bucketBytes: Uint8Array } => {
  const thresholds = new Float64Array(257);
  thresholds[256] = Number.POSITIVE_INFINITY;
  for (let byte = 1; byte < 256; byte++) {
    let below = 0;
    let above = 1;
    for (;;) {
      const middle = (below + above) / 2;
      if (middle === below || middle === above) {
        break;
      }
      if (srgb8FromLinearDirect(middle) >= byte) {
        above = middle;
      } else {
        below = middle;
      }
    }
    thresholds[byte] = above;
  }
  const bucketBytes = new Uint8Array(SRGB8_BUCKETS);
  for (let bucket = 0, byte = 0; bucket < SRGB8_BUCKETS; bucket++) {
    while (bucket / SRGB8_BUCKETS >= thresholds[byte + 1]) {
      byte++;
    }
    bucketBytes[bucket] = byte;
  }
  return { bucketBytes, thresholds };
};

const SRGB8_ENCODING = buildSrgb8Encoding();

/** Encodes one linear channel to the byte the backend stores for it. */
export const srgb8FromLinear = (linear: number): number => {
  if (!(linear > 0)) {
    return 0;
  }
  if (linear >= 1) {
    return 255;
  }
  const { bucketBytes, thresholds } = SRGB8_ENCODING;
  let byte = bucketBytes[Math.floor(linear * SRGB8_BUCKETS)];
  while (linear >= thresholds[byte + 1]) {
    byte++;
  }
  return byte;
};

/** Writes Oklab (L, a, b) for linear sRGB into `out`. */
export const oklabFromLinearSrgb = (red: number, green: number, blue: number, out: Float64Array): void => {
  const l = Math.cbrt(0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue);
  const m = Math.cbrt(0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue);
  const s = Math.cbrt(0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue);
  out[0] = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
  out[1] = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
  out[2] = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
};

/** Writes Oklab (L, a, b) for byte-valued sRGB into `out`. */
export const oklabFromSrgb8 = (red: number, green: number, blue: number, out: Float64Array): void => {
  oklabFromLinearSrgb(LINEAR_FROM_SRGB8[red], LINEAR_FROM_SRGB8[green], LINEAR_FROM_SRGB8[blue], out);
};

/** Writes unclamped linear sRGB for Oklab (L, a, b) into `out`. */
export const linearSrgbFromOklab = (lightness: number, a: number, b: number, out: Float64Array): void => {
  const lRoot = lightness + 0.3963377774 * a + 0.2158037573 * b;
  const mRoot = lightness - 0.1055613458 * a - 0.0638541728 * b;
  const sRoot = lightness - 0.0894841775 * a - 1.291485548 * b;
  const l = lRoot * lRoot * lRoot;
  const m = mRoot * mRoot * mRoot;
  const s = sRoot * sRoot * sRoot;
  out[0] = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  out[1] = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  out[2] = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s;
};

/**
 * Maximum saturation (C / L) inside sRGB along the unit hue (a, b), refined with one Halley step, as
 * `_max_srgb_saturation_tensor(steps=1)`.
 */
const maxSrgbSaturation = (a: number, b: number): number => {
  // Polynomial fit per limiting channel (red, green, else blue) and that channel's LMS-to-linear-sRGB row.
  let saturation: number;
  let wl: number;
  let wm: number;
  let ws: number;
  if (-1.88170328 * a - 0.80936493 * b > 1) {
    saturation = 1.19086277 + 1.76576728 * a + 0.59662641 * b + 0.75515197 * a * a + 0.56771245 * a * b;
    wl = 4.0767416621;
    wm = -3.3077115913;
    ws = 0.2309699292;
  } else if (1.81444104 * a - 1.19445276 * b > 1) {
    saturation = 0.73956515 - 0.45954494 * a + 0.08285427 * b + 0.1254107 * a * a + 0.14503204 * a * b;
    wl = -1.2684380046;
    wm = 2.6097574011;
    ws = -0.3413193965;
  } else {
    saturation = 1.35733652 - 0.00915799 * a - 1.1513021 * b - 0.50559606 * a * a + 0.00692167 * a * b;
    wl = -0.0041960863;
    wm = -0.7034186147;
    ws = 1.707614701;
  }
  const kl = 0.3963377774 * a + 0.2158037573 * b;
  const km = -0.1055613458 * a - 0.0638541728 * b;
  const ks = -0.0894841775 * a - 1.291485548 * b;
  const lRoot = 1 + saturation * kl;
  const mRoot = 1 + saturation * km;
  const sRoot = 1 + saturation * ks;
  const f = wl * lRoot * lRoot * lRoot + wm * mRoot * mRoot * mRoot + ws * sRoot * sRoot * sRoot;
  const f1 = wl * 3 * kl * lRoot * lRoot + wm * 3 * km * mRoot * mRoot + ws * 3 * ks * sRoot * sRoot;
  const f2 = wl * 6 * kl * kl * lRoot + wm * 6 * km * km * mRoot + ws * 6 * ks * ks * sRoot;
  return saturation - (f * f1) / (f1 * f1 - 0.5 * f * f2);
};

/** Newton-Halley step length towards one channel's gamut boundary; MAX_FLOAT32 when it moves away. */
const channelStep = (value: number, first: number, second: number): number => {
  const u = first / (first * first - 0.5 * value * second);
  return u < 0 ? MAX_FLOAT32 : -value * u;
};

const scratch = new Float64Array(3);

/**
 * Projects out-of-gamut linear sRGB in `rgb` towards an adaptive lightness anchor, as the backend's restore does:
 * `gamut_clip_tensor` (alpha 0.05, one refinement step) in float64, applied only to colors more than
 * {@link GAMUT_TOLERANCE} outside [0, 1]. The clip moves colors that sit on the gamut boundary, so the tolerance
 * keeps float32 round-trip noise on an in-gamut color from reaching it. Other colors are left unchanged.
 */
export const clipToSrgbGamut = (rgb: Float64Array): void => {
  const red = rgb[0];
  const green = rgb[1];
  const blue = rgb[2];
  if (!(Math.max(red, green, blue) > 1 + GAMUT_TOLERANCE || Math.min(red, green, blue) < -GAMUT_TOLERANCE)) {
    return;
  }
  oklabFromLinearSrgb(red, green, blue, scratch);
  const lightness = scratch[0];
  let chroma = Math.sqrt(scratch[1] * scratch[1] + scratch[2] * scratch[2]);
  if (chroma < GAMUT_CLIP_MIN_CHROMA) {
    chroma = GAMUT_CLIP_MIN_CHROMA;
  }
  const unitA = scratch[1] / chroma;
  const unitB = scratch[2] / chroma;
  const lightnessOffset = lightness - 0.5;
  const absOffset = Math.abs(lightnessOffset);
  const e1 = absOffset + chroma * GAMUT_CLIP_ALPHA + 0.5;
  const anchor = (Math.sign(lightnessOffset) * (e1 - Math.sqrt(e1 * e1 - 2 * absOffset)) + 1) * 0.5;

  // Cusp of the hue slice: the most saturated sRGB color along (unitA, unitB).
  const cuspSaturation = maxSrgbSaturation(unitA, unitB);
  linearSrgbFromOklab(1, cuspSaturation * unitA, cuspSaturation * unitB, scratch);
  const cuspLightness = (1 / Math.max(scratch[0], scratch[1], scratch[2])) ** (1 / 3);
  const cuspChroma = cuspLightness * cuspSaturation;

  // Intersection of the line from (lightness, chroma) to (anchor, 0) with the gamut boundary.
  const upper = (lightness - anchor) * cuspChroma - (cuspLightness - anchor) * chroma > 0;
  let t = upper
    ? (cuspChroma * (anchor - 1)) / (chroma * (cuspLightness - 1) + cuspChroma * (anchor - lightness))
    : (cuspChroma * anchor) / (chroma * cuspLightness + cuspChroma * (anchor - lightness));
  if (upper) {
    const kl = 0.3963377774 * unitA + 0.2158037573 * unitB;
    const km = -0.1055613458 * unitA - 0.0638541728 * unitB;
    const ks = -0.0894841775 * unitA - 1.291485548 * unitB;
    const dLightness = lightness - anchor;
    const lDt = kl * chroma + dLightness;
    const mDt = km * chroma + dLightness;
    const sDt = ks * chroma + dLightness;
    const stepLightness = anchor * (1 - t) + t * lightness;
    const stepChroma = t * chroma;
    const lRoot = kl * stepChroma + stepLightness;
    const mRoot = km * stepChroma + stepLightness;
    const sRoot = ks * stepChroma + stepLightness;
    const l = lRoot * lRoot * lRoot;
    const m = mRoot * mRoot * mRoot;
    const s = sRoot * sRoot * sRoot;
    const l1 = lRoot * lRoot * lDt * 3;
    const m1 = mRoot * mRoot * mDt * 3;
    const s1 = sRoot * sRoot * sDt * 3;
    const l2 = lDt * lDt * lRoot * 6;
    const m2 = mDt * mDt * mRoot * 6;
    const s2 = sDt * sDt * sRoot * 6;
    const redStep = channelStep(
      4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s - 1,
      4.0767416621 * l1 - 3.3077115913 * m1 + 0.2309699292 * s1,
      4.0767416621 * l2 - 3.3077115913 * m2 + 0.2309699292 * s2
    );
    const greenStep = channelStep(
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s - 1,
      -1.2684380046 * l1 + 2.6097574011 * m1 - 0.3413193965 * s1,
      -1.2684380046 * l2 + 2.6097574011 * m2 - 0.3413193965 * s2
    );
    const blueStep = channelStep(
      -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s - 1,
      -0.0041960863 * l1 - 0.7034186147 * m1 + 1.707614701 * s1,
      -0.0041960863 * l2 - 0.7034186147 * m2 + 1.707614701 * s2
    );
    t += Math.min(redStep, greenStep, blueStep);
  }
  const clippedChroma = t * chroma;
  linearSrgbFromOklab(anchor * (1 - t) + t * lightness, clippedChroma * unitA, clippedChroma * unitB, rgb);
};
