import { NUMPY_RAND_MAX } from 'app/constants';
import { describe, expect, it } from 'vitest';

import { generateSeeds } from './generateSeeds';

describe('generateSeeds', () => {
  it('returns consecutive seeds from the start seed', () => {
    expect(generateSeeds({ count: 3, start: 42 })).toEqual([42, 43, 44]);
  });

  it('keeps the maximum seed as the first seed', () => {
    expect(generateSeeds({ count: 1, start: NUMPY_RAND_MAX })).toEqual([NUMPY_RAND_MAX]);
  });

  it('wraps to the minimum only after the maximum seed', () => {
    expect(generateSeeds({ count: 3, start: NUMPY_RAND_MAX - 1 })).toEqual([NUMPY_RAND_MAX - 1, NUMPY_RAND_MAX, 0]);
  });

  it('wraps within a custom range', () => {
    expect(generateSeeds({ count: 4, start: 9, min: 5, max: 10 })).toEqual([9, 10, 5, 6]);
  });

  it('draws a random start within the range when no start is given', () => {
    for (let i = 0; i < 50; i++) {
      const seeds = generateSeeds({ count: 3, min: 5, max: 7 });
      expect(seeds).toHaveLength(3);
      for (const seed of seeds) {
        expect(seed).toBeGreaterThanOrEqual(5);
        expect(seed).toBeLessThanOrEqual(7);
      }
    }
  });
});
