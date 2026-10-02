import { describe, expect, it } from 'vitest';

import {
  hexToHsv,
  hslToRgb,
  hsvToHex,
  hsvToRgb,
  hsvToTrianglePoint,
  hueToWheelAngle,
  pointToWheelAngle,
  rgbToHsl,
  rgbToHsv,
  triangleCorners,
  trianglePointToHsv,
  wheelAngleToHue,
} from './hsv';

const rgba = (r: number, g: number, b: number) => ({ a: 1, b, g, r });

describe('rgb↔hsv', () => {
  it('converts the primaries and greys', () => {
    expect(rgbToHsv(rgba(255, 0, 0))).toEqual({ h: 0, s: 1, v: 1 });
    expect(rgbToHsv(rgba(0, 255, 0))).toEqual({ h: 120, s: 1, v: 1 });
    expect(rgbToHsv(rgba(0, 0, 255))).toEqual({ h: 240, s: 1, v: 1 });
    expect(rgbToHsv(rgba(255, 255, 255))).toEqual({ h: 0, s: 0, v: 1 });
    expect(rgbToHsv(rgba(0, 0, 0))).toEqual({ h: 0, s: 0, v: 0 });
  });

  it('round-trips a sweep of colors within one channel step', () => {
    for (let h = 0; h < 360; h += 30) {
      for (const s of [0.2, 0.6, 1]) {
        for (const v of [0.2, 0.6, 1]) {
          const rgb = hsvToRgb({ h, s, v });
          const back = rgbToHsv(rgb);
          const hueDelta = Math.min(Math.abs(back.h - h), 360 - Math.abs(back.h - h));
          expect(hueDelta).toBeLessThan(1.5);
          expect(back.s).toBeCloseTo(s, 1);
          expect(back.v).toBeCloseTo(v, 1);
        }
      }
    }
  });

  it('keeps the previous hue for greys and the previous saturation at black', () => {
    expect(hexToHsv('#808080', { h: 200, s: 1, v: 0.4 }).h).toBe(200);
    expect(hexToHsv('#000000', { h: 200, s: 0.7, v: 0.4 })).toEqual({ h: 200, s: 0.7, v: 0 });
    expect(hexToHsv('#ff0000', { h: 200, s: 0.7, v: 0.4 })).toEqual({ h: 0, s: 1, v: 1 });
    expect(hexToHsv('#00ff00')).toEqual({ h: 120, s: 1, v: 1 });
    expect(hsvToHex({ h: 120, s: 1, v: 1 })).toBe('#00ff00');
  });
});

describe('rgb↔hsl', () => {
  it('converts landmarks and round-trips a sweep', () => {
    expect(rgbToHsl(rgba(255, 0, 0))).toEqual({ h: 0, l: 0.5, s: 1 });
    expect(rgbToHsl(rgba(255, 255, 255))).toEqual({ h: 0, l: 1, s: 0 });
    expect(rgbToHsl(rgba(0, 0, 0))).toEqual({ h: 0, l: 0, s: 0 });
    for (let h = 0; h < 360; h += 60) {
      for (const s of [0.25, 0.75]) {
        for (const l of [0.2, 0.5, 0.8]) {
          const back = rgbToHsl(hslToRgb({ h, l, s }));
          expect(back.l).toBeCloseTo(l, 1);
          expect(back.s).toBeCloseTo(s, 1);
        }
      }
    }
  });
});

describe('wheel geometry', () => {
  it('maps hue to ring angle and back', () => {
    for (const hue of [0, 45, 120, 359]) {
      expect(wheelAngleToHue(hueToWheelAngle(hue))).toBeCloseTo(hue, 5);
    }
    expect(pointToWheelAngle({ x: 1, y: 0 })).toBeCloseTo(0, 5);
    expect(wheelAngleToHue(pointToWheelAngle({ x: 0, y: 1 }))).toBeCloseTo(90, 5);
  });

  it('puts the corners at the hue, white, and black positions', () => {
    const [hueCorner, whiteCorner, blackCorner] = triangleCorners(0, 100);
    expect(hueCorner.x).toBeCloseTo(100, 5);
    expect(hueCorner.y).toBeCloseTo(0, 5);
    expect(whiteCorner.x).toBeCloseTo(-50, 5);
    expect(blackCorner.x).toBeCloseTo(-50, 5);
    expect(whiteCorner.y).toBeCloseTo(86.6, 1);
    expect(blackCorner.y).toBeCloseTo(-86.6, 1);
  });

  it('round-trips s/v through the triangle at several hues', () => {
    for (const h of [0, 80, 200, 310]) {
      for (const s of [0, 0.3, 0.7, 1]) {
        for (const v of [0.1, 0.5, 1]) {
          const back = trianglePointToHsv(hsvToTrianglePoint({ h, s, v }, 90), h, 90);
          expect(back.v).toBeCloseTo(v, 5);
          expect(back.s).toBeCloseTo(v === 0 ? 0 : s, 5);
        }
      }
    }
  });

  it('clamps points outside the triangle onto it', () => {
    const clamped = trianglePointToHsv({ x: 500, y: 0 }, 0, 90);
    expect(clamped.s).toBeGreaterThanOrEqual(0);
    expect(clamped.s).toBeLessThanOrEqual(1);
    expect(clamped.v).toBeGreaterThanOrEqual(0);
    expect(clamped.v).toBeLessThanOrEqual(1);
    // Far past the hue corner lands on the pure hue.
    expect(clamped).toMatchObject({ h: 0, s: 1, v: 1 });
  });
});
