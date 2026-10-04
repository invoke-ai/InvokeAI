/** Preserve hue/saturation for achromatic colors. Triangle weights: hue=s·v, white=v·(1−s), black=1−v. */

import type { RgbaColor } from './color';

import { formatHexColor, parseHexColor } from './color';

/** `h` in degrees `[0, 360)`; `s`/`v` in `[0, 1]`. */
export interface HsvColor {
  h: number;
  s: number;
  v: number;
}

export interface Point {
  x: number;
  y: number;
}

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));

const normalizeHue = (degrees: number): number => {
  const wrapped = degrees % 360;
  return wrapped < 0 ? wrapped + 360 : wrapped;
};

export const rgbToHsv = ({ b, g, r }: RgbaColor): HsvColor => {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;
  let h = 0;
  if (delta > 0) {
    if (max === rn) {
      h = 60 * (((gn - bn) / delta) % 6);
    } else if (max === gn) {
      h = 60 * ((bn - rn) / delta + 2);
    } else {
      h = 60 * ((rn - gn) / delta + 4);
    }
  }
  return { h: normalizeHue(h), s: max === 0 ? 0 : delta / max, v: max };
};

export const hsvToRgb = ({ h, s, v }: HsvColor): RgbaColor => {
  const hue = normalizeHue(h);
  const c = clamp01(v) * clamp01(s);
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
  const m = clamp01(v) - c;
  const [rn, gn, bn] =
    hue < 60
      ? [c, x, 0]
      : hue < 120
        ? [x, c, 0]
        : hue < 180
          ? [0, c, x]
          : hue < 240
            ? [0, x, c]
            : hue < 300
              ? [x, 0, c]
              : [c, 0, x];
  return { a: 1, b: Math.round((bn + m) * 255), g: Math.round((gn + m) * 255), r: Math.round((rn + m) * 255) };
};

/** `h` in degrees `[0, 360)`; `s`/`l` in `[0, 1]`. */
export interface HslColor {
  h: number;
  s: number;
  l: number;
}

export const rgbToHsl = (color: RgbaColor): HslColor => {
  const { h, s, v } = rgbToHsv(color);
  const l = v * (1 - s / 2);
  return { h, l, s: l === 0 || l === 1 ? 0 : (v - l) / Math.min(l, 1 - l) };
};

export const hslToRgb = ({ h, l, s }: HslColor): RgbaColor => {
  const v = clamp01(l) + clamp01(s) * Math.min(clamp01(l), 1 - clamp01(l));
  return hsvToRgb({ h, s: v === 0 ? 0 : 2 * (1 - clamp01(l) / v), v });
};

/** Preserve previous hue for greys and saturation at black; hex cannot encode them. */
export const hexToHsv = (hex: string, previous?: HsvColor): HsvColor => {
  const parsed = rgbToHsv(parseHexColor(hex));
  if (!previous) {
    return parsed;
  }
  return {
    h: parsed.s === 0 ? previous.h : parsed.h,
    s: parsed.v === 0 ? previous.s : parsed.s,
    v: parsed.v,
  };
};

export const hsvToHex = (hsv: HsvColor): string => formatHexColor(hsvToRgb(hsv));

/** Screen-space angle (radians, y-down) of a hue on the ring; 0° hue sits at the right. */
export const hueToWheelAngle = (hue: number): number => (normalizeHue(hue) * Math.PI) / 180;

export const wheelAngleToHue = (angle: number): number => normalizeHue((angle * 180) / Math.PI);

export const pointToWheelAngle = ({ x, y }: Point): number => {
  const angle = Math.atan2(y, x);
  return angle < 0 ? angle + 2 * Math.PI : angle;
};

/** The triangle's corners for `hue` on a ring of `radius`: pure hue, then white, then black. */
export const triangleCorners = (hue: number, radius: number): [Point, Point, Point] => {
  const corner = (offsetDegrees: number): Point => {
    const angle = hueToWheelAngle(hue + offsetDegrees);
    return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
  };
  return [corner(0), corner(120), corner(240)];
};

/** Where `{s, v}` sits inside the hue's triangle (center-relative coordinates). */
export const hsvToTrianglePoint = (hsv: HsvColor, radius: number): Point => {
  const [hueCorner, whiteCorner, blackCorner] = triangleCorners(hsv.h, radius);
  const v = clamp01(hsv.v);
  const s = clamp01(hsv.s);
  const weightHue = s * v;
  const weightWhite = v * (1 - s);
  const weightBlack = 1 - v;
  return {
    x: hueCorner.x * weightHue + whiteCorner.x * weightWhite + blackCorner.x * weightBlack,
    y: hueCorner.y * weightHue + whiteCorner.y * weightWhite + blackCorner.y * weightBlack,
  };
};

/** Clamp outside points by flooring and renormalizing barycentric weights, keeping drags inside the triangle. */
export const trianglePointToHsv = (point: Point, hue: number, radius: number): HsvColor => {
  const [hueCorner, whiteCorner, blackCorner] = triangleCorners(hue, radius);
  // Solve the barycentric weights for `point` against the three corners.
  const v0x = whiteCorner.x - hueCorner.x;
  const v0y = whiteCorner.y - hueCorner.y;
  const v1x = blackCorner.x - hueCorner.x;
  const v1y = blackCorner.y - hueCorner.y;
  const v2x = point.x - hueCorner.x;
  const v2y = point.y - hueCorner.y;
  const denominator = v0x * v1y - v1x * v0y;
  let weightWhite = denominator === 0 ? 0 : (v2x * v1y - v1x * v2y) / denominator;
  let weightBlack = denominator === 0 ? 0 : (v0x * v2y - v2x * v0y) / denominator;
  let weightHue = 1 - weightWhite - weightBlack;
  weightHue = Math.max(0, weightHue);
  weightWhite = Math.max(0, weightWhite);
  weightBlack = Math.max(0, weightBlack);
  const total = weightHue + weightWhite + weightBlack;
  weightHue /= total;
  weightWhite /= total;
  weightBlack /= total;
  const v = 1 - weightBlack;
  return { h: normalizeHue(hue), s: v === 0 ? 0 : weightHue / v, v };
};
