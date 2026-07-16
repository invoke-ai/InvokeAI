import { describe, expect, it } from 'vitest';

import { applyToPoint, fromTRS, getScale, identity, invert, multiply, rotate, scale, translate } from './mat2d';

describe('identity', () => {
  it('returns the identity matrix', () => {
    expect(identity()).toEqual({ a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 });
  });

  it('applied to a point returns the same point', () => {
    expect(applyToPoint(identity(), { x: 5, y: -3 })).toEqual({ x: 5, y: -3 });
  });
});

describe('translate', () => {
  it('shifts points by the translation vector', () => {
    const m = translate(identity(), { x: 10, y: 20 });
    expect(applyToPoint(m, { x: 0, y: 0 })).toEqual({ x: 10, y: 20 });
    expect(applyToPoint(m, { x: 5, y: 5 })).toEqual({ x: 15, y: 25 });
  });
});

describe('scale', () => {
  it('scales points uniformly when sy is omitted', () => {
    const m = scale(identity(), 2);
    expect(applyToPoint(m, { x: 3, y: 4 })).toEqual({ x: 6, y: 8 });
  });

  it('scales points non-uniformly when sy is given', () => {
    const m = scale(identity(), 2, 3);
    expect(applyToPoint(m, { x: 1, y: 1 })).toEqual({ x: 2, y: 3 });
  });
});

describe('rotate', () => {
  it('rotates a point 90 degrees clockwise', () => {
    const m = rotate(identity(), Math.PI / 2);
    const p = applyToPoint(m, { x: 1, y: 0 });
    expect(p.x).toBeCloseTo(0, 10);
    expect(p.y).toBeCloseTo(1, 10);
  });

  it('rotating by 0 radians is a no-op', () => {
    const m = rotate(identity(), 0);
    expect(applyToPoint(m, { x: 7, y: -2 })).toEqual({ x: 7, y: -2 });
  });
});

describe('multiply', () => {
  it('composes transforms so the right operand applies first', () => {
    // translate(10,0) * scale(2) applied to (1,1): scale first -> (2,2), then translate -> (12, 2)
    const t = translate(identity(), { x: 10, y: 0 });
    const s = scale(identity(), 2);
    const m = multiply(t, s);
    expect(applyToPoint(m, { x: 1, y: 1 })).toEqual({ x: 12, y: 2 });
  });

  it('multiplying by identity is a no-op on either side', () => {
    const m = translate(scale(identity(), 3, 4), { x: 1, y: 2 });
    expect(multiply(m, identity())).toEqual(m);
    expect(multiply(identity(), m)).toEqual(m);
  });
});

describe('invert', () => {
  it('round-trips applyToPoint through a matrix and its inverse', () => {
    const m = fromTRS({ x: 5, y: -3 }, Math.PI / 6, 2, 0.5);
    const inv = invert(m);
    expect(inv).not.toBeNull();
    const p = { x: 13, y: -4 };
    const roundTripped = applyToPoint(inv as NonNullable<typeof inv>, applyToPoint(m, p));
    expect(roundTripped.x).toBeCloseTo(p.x, 8);
    expect(roundTripped.y).toBeCloseTo(p.y, 8);
  });

  it('returns null for a singular matrix', () => {
    // Zero determinant: a*d - b*c = 0
    expect(invert({ a: 1, b: 2, c: 2, d: 4, e: 0, f: 0 })).toBeNull();
    expect(invert({ a: 0, b: 0, c: 0, d: 0, e: 1, f: 1 })).toBeNull();
  });

  it('inverting identity returns identity', () => {
    const inv = invert(identity());
    expect(inv).not.toBeNull();
    const nonNullInv = inv as NonNullable<typeof inv>;
    expect(nonNullInv.a).toBeCloseTo(1, 10);
    expect(nonNullInv.b).toBeCloseTo(0, 10);
    expect(nonNullInv.c).toBeCloseTo(0, 10);
    expect(nonNullInv.d).toBeCloseTo(1, 10);
    expect(nonNullInv.e).toBeCloseTo(0, 10);
    expect(nonNullInv.f).toBeCloseTo(0, 10);
  });
});

describe('fromTRS', () => {
  it('matches manual translate . rotate . scale composition', () => {
    const translation = { x: 4, y: -2 };
    const rad = 0.7;
    const sx = 1.5;
    const sy = 2.5;

    const t = translate(identity(), translation);
    const r = rotate(identity(), rad);
    const s = scale(identity(), sx, sy);
    const manual = multiply(multiply(t, r), s);
    const composed = fromTRS(translation, rad, sx, sy);

    expect(composed.a).toBeCloseTo(manual.a, 10);
    expect(composed.b).toBeCloseTo(manual.b, 10);
    expect(composed.c).toBeCloseTo(manual.c, 10);
    expect(composed.d).toBeCloseTo(manual.d, 10);
    expect(composed.e).toBeCloseTo(manual.e, 10);
    expect(composed.f).toBeCloseTo(manual.f, 10);
  });

  it('defaults scaleY to scaleX for uniform scale', () => {
    const composed = fromTRS({ x: 0, y: 0 }, 0, 3, 3);
    const defaulted = fromTRS({ x: 0, y: 0 }, 0, 3);
    expect(defaulted).toEqual(composed);
  });

  it('with no rotation/scale reduces to pure translation', () => {
    expect(fromTRS({ x: 8, y: 9 }, 0, 1, 1)).toEqual(translate(identity(), { x: 8, y: 9 }));
  });
});

describe('getScale', () => {
  it('returns 1 for the identity matrix', () => {
    expect(getScale(identity())).toBeCloseTo(1, 10);
  });

  it('returns the scale factor for a uniformly scaled matrix', () => {
    expect(getScale(scale(identity(), 3))).toBeCloseTo(3, 10);
  });

  it('returns the geometric mean for a non-uniformly scaled matrix', () => {
    expect(getScale(scale(identity(), 2, 8))).toBeCloseTo(4, 10);
  });

  it('is unaffected by translation', () => {
    const m = translate(scale(identity(), 2), { x: 100, y: -50 });
    expect(getScale(m)).toBeCloseTo(2, 10);
  });

  it('is unaffected by rotation for a uniformly scaled matrix', () => {
    const m = rotate(scale(identity(), 5), 1.234);
    expect(getScale(m)).toBeCloseTo(5, 8);
  });
});
