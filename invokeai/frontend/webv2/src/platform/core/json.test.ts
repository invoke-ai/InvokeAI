import { describe, expect, it } from 'vitest';

import { areJsonValuesStructurallyEqual } from './json';

describe('areJsonValuesStructurallyEqual', () => {
  it.each([
    [null, null, true],
    [true, true, true],
    [true, false, false],
    [3, 3, true],
    [3, 4, false],
    ['prompt', 'prompt', true],
    ['prompt', 'other', false],
  ])('compares primitive values (%j, %j)', (left, right, expected) => {
    expect(areJsonValuesStructurallyEqual(left, right)).toBe(expected);
  });

  it('compares arrays recursively and in order', () => {
    expect(areJsonValuesStructurallyEqual([1, { nested: ['a', null] }], [1, { nested: ['a', null] }])).toBe(true);
    expect(areJsonValuesStructurallyEqual([1, 2], [2, 1])).toBe(false);
    expect(areJsonValuesStructurallyEqual([1], [1, 2])).toBe(false);
  });

  it('compares object values independently of key insertion order', () => {
    expect(areJsonValuesStructurallyEqual({ a: 1, nested: { b: true } }, { nested: { b: true }, a: 1 })).toBe(true);
    expect(areJsonValuesStructurallyEqual({ a: 1 }, { a: 2 })).toBe(false);
  });

  it('distinguishes a missing key from a present key', () => {
    expect(areJsonValuesStructurallyEqual({ value: null }, {})).toBe(false);
    expect(areJsonValuesStructurallyEqual({}, { value: null })).toBe(false);
  });

  it('rejects distinct non-JSON objects while preserving object identity', () => {
    const same = new Date(0);

    expect(areJsonValuesStructurallyEqual(new Date(0), new Date(0))).toBe(false);
    expect(areJsonValuesStructurallyEqual(same, same)).toBe(true);
    expect(
      areJsonValuesStructurallyEqual(
        () => undefined,
        () => undefined
      )
    ).toBe(false);
  });
});
