import { describe, expect, it } from 'vitest';

import { constrainNumber } from './constrainNumber';

describe('constrainNumber', () => {
  // Default constraints to be used in tests
  const defaultConstraints = { min: 0, max: 10, multipleOf: 1 };

  it('should keep values within range', () => {
    expect(constrainNumber(5, defaultConstraints)).toBe(5);
    expect(constrainNumber(-5, defaultConstraints)).toBe(0);
    expect(constrainNumber(15, defaultConstraints)).toBe(10);
  });

  it('should round to nearest multiple', () => {
    const constraints = { min: 0, max: 10, multipleOf: 2 };
    expect(constrainNumber(1, constraints)).toBe(2);
    expect(constrainNumber(2, constraints)).toBe(2);
    expect(constrainNumber(3, constraints)).toBe(4);
    expect(constrainNumber(9, constraints)).toBe(10);
    expect(constrainNumber(11, constraints)).toBe(10);
  });

  it('should handle negative multiples', () => {
    const constraints = { min: -10, max: 10, multipleOf: 3 };
    expect(constrainNumber(-9, constraints)).toBe(-9);
    expect(constrainNumber(-8, constraints)).toBe(-9);
    expect(constrainNumber(-7, constraints)).toBe(-6);
    expect(constrainNumber(-3, constraints)).toBe(-3);
    expect(constrainNumber(-2, constraints)).toBe(-3);
    // In JS, -0 !== +0... :)
    expect(constrainNumber(-1, constraints)).toBe(0);
    expect(constrainNumber(0, constraints)).toBe(0);
    expect(constrainNumber(1, constraints)).toBe(0);
    expect(constrainNumber(2, constraints)).toBe(3);
    expect(constrainNumber(3, constraints)).toBe(3);
    expect(constrainNumber(4, constraints)).toBe(3);
    expect(constrainNumber(7, constraints)).toBe(6);
    expect(constrainNumber(8, constraints)).toBe(9);
    expect(constrainNumber(9, constraints)).toBe(9);
  });

  it('should respect boundaries when rounding', () => {
    const constraints = { min: 0, max: 10, multipleOf: 4 };
    // Value at 9 would normally round to 8
    expect(constrainNumber(9, constraints)).toBe(8);
    // Value at 11 would normally round to 12, but max is 10
    expect(constrainNumber(11, constraints)).toBe(10);
  });

  it('should handle decimal multiples', () => {
    const constraints = { min: 0, max: 1, multipleOf: 0.25 };
    expect(constrainNumber(0.3, constraints)).toBe(0.25);
    expect(constrainNumber(0.87, constraints)).toBe(0.75);
    expect(constrainNumber(0.88, constraints)).toBe(1.0);
    expect(constrainNumber(0.13, constraints)).toBe(0.25);
  });

  it('should apply overrides correctly', () => {
    // Override min
    expect(constrainNumber(2, defaultConstraints, { min: 5 })).toBe(5);

    // Override max
    expect(constrainNumber(8, defaultConstraints, { max: 7 })).toBe(7);

    // Override multipleOf
    expect(constrainNumber(4.7, defaultConstraints, { multipleOf: 2 })).toBe(4);

    // Override all
    expect(constrainNumber(15, defaultConstraints, { min: 5, max: 20, multipleOf: 5 })).toBe(15);
  });

  it('should handle edge cases', () => {
    // Value exactly at min
    expect(constrainNumber(0, defaultConstraints)).toBe(0);

    // Value exactly at max
    expect(constrainNumber(10, defaultConstraints)).toBe(10);

    // multipleOf larger than range
    const narrowConstraints = { min: 5, max: 7, multipleOf: 5 };
    expect(constrainNumber(6, narrowConstraints)).toBe(5);
  });
});
