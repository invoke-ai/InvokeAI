import { describe, expect, it } from 'vitest';

import { constrainNumber } from './constrainNumber';

expect.addEqualityTesters([
  function (a, b) {
    if (typeof a === 'number' && typeof b === 'number') {
      return a === b;
    }
  },
]);

describe('constrainNumber', () => {
  // Default constraints to be used in tests
  const defaultConstraints = { min: 0, max: 10, step: 1 };

  it('should keep values within range', () => {
    expect(constrainNumber(5, defaultConstraints)).toEqual(5);
    expect(constrainNumber(-5, defaultConstraints)).toEqual(0);
    expect(constrainNumber(15, defaultConstraints)).toEqual(10);
  });

  it('should round to nearest multiple', () => {
    const constraints = { min: 0, max: 10, step: 2 };
    expect(constrainNumber(1, constraints)).toEqual(2);
    expect(constrainNumber(2, constraints)).toEqual(2);
    expect(constrainNumber(3, constraints)).toEqual(4);
    expect(constrainNumber(9, constraints)).toEqual(10);
    expect(constrainNumber(11, constraints)).toEqual(10);
  });

  it('should always prefer to round to a multiple rather than the nearest value within the min and max', () => {
    const constraints = { min: 0, max: 10, step: 3 };
    expect(constrainNumber(1, constraints)).toEqual(0);
    expect(constrainNumber(2, constraints)).toEqual(3);
    expect(constrainNumber(3, constraints)).toEqual(3);
    expect(constrainNumber(4, constraints)).toEqual(3);
    expect(constrainNumber(7, constraints)).toEqual(6);
    expect(constrainNumber(8, constraints)).toEqual(9);
    expect(constrainNumber(9, constraints)).toEqual(9);

    expect(constrainNumber(12, { min: 7, max: 12, step: 5 })).toEqual(10);
    expect(constrainNumber(13, { min: 7, max: 12, step: 5 })).toEqual(10);
    expect(constrainNumber(14, { min: 7, max: 12, step: 5 })).toEqual(10);

    expect(constrainNumber(3, { min: 7, max: 12, step: 5 })).toEqual(10);
    expect(constrainNumber(4, { min: 7, max: 12, step: 5 })).toEqual(10);
    expect(constrainNumber(5, { min: 7, max: 12, step: 5 })).toEqual(10);

    expect(constrainNumber(42, { min: 43, max: 81, step: 8 })).toEqual(48);
  });

  it('should handle negative multiples', () => {
    const constraints = { min: -10, max: 10, step: 3 };
    expect(constrainNumber(-9, constraints)).toEqual(-9);
    expect(constrainNumber(-8, constraints)).toEqual(-9);
    expect(constrainNumber(-7, constraints)).toEqual(-6);
    expect(constrainNumber(-3, constraints)).toEqual(-3);
    expect(constrainNumber(-2, constraints)).toEqual(-3);
    expect(constrainNumber(-1, constraints)).toEqual(0);
    expect(constrainNumber(0, constraints)).toEqual(0);
    expect(constrainNumber(1, constraints)).toEqual(0);
    expect(constrainNumber(2, constraints)).toEqual(3);
    expect(constrainNumber(3, constraints)).toEqual(3);
    expect(constrainNumber(4, constraints)).toEqual(3);
    expect(constrainNumber(7, constraints)).toEqual(6);
    expect(constrainNumber(8, constraints)).toEqual(9);
    expect(constrainNumber(9, constraints)).toEqual(9);
  });

  it('should respect boundaries when rounding', () => {
    const constraints = { min: 0, max: 10, step: 4 };
    // Value at 9 would normally round to 8
    expect(constrainNumber(9, constraints)).toEqual(8);
    // Value at 11 would normally round to 12, but max is 10
    expect(constrainNumber(11, constraints)).toEqual(8);
  });

  it('should handle decimal multiples', () => {
    const constraints = { min: 0, max: 1, step: 0.25 };
    expect(constrainNumber(0.3, constraints)).toEqual(0.25);
    expect(constrainNumber(0.87, constraints)).toEqual(0.75);
    expect(constrainNumber(0.88, constraints)).toEqual(1.0);
    expect(constrainNumber(0.13, constraints)).toEqual(0.25);
  });

  it('should apply overrides correctly', () => {
    // Override min
    expect(constrainNumber(2, defaultConstraints, { min: 5 })).toEqual(5);

    // Override max
    expect(constrainNumber(8, defaultConstraints, { max: 7 })).toEqual(7);

    // Override multipleOf
    expect(constrainNumber(4.7, defaultConstraints, { step: 2 })).toEqual(4);

    // Override all
    expect(constrainNumber(15, defaultConstraints, { min: 5, max: 20, step: 5 })).toEqual(15);
  });

  it('should handle edge cases', () => {
    // Value exactly at min
    expect(constrainNumber(0, defaultConstraints)).toEqual(0);

    // Value exactly at max
    expect(constrainNumber(10, defaultConstraints)).toEqual(10);

    // multipleOf larger than range
    const narrowConstraints = { min: 5, max: 7, step: 5 };
    expect(constrainNumber(6, narrowConstraints)).toEqual(5);
  });
});
