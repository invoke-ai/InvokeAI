import { describe, expect, it } from 'vitest';

import {
  getGridSize,
  getIsSizeOptimal,
  getIsSizeTooLarge,
  getIsSizeTooSmall,
  getOptimalDimension,
} from './optimalDimension';

describe('getOptimalDimension', () => {
  it('returns 512 for sd-1', () => {
    expect(getOptimalDimension('sd-1')).toBe(512);
  });

  it('returns 512 for sd-2', () => {
    expect(getOptimalDimension('sd-2')).toBe(512);
  });

  it('returns 1024 for qwen-image', () => {
    expect(getOptimalDimension('qwen-image')).toBe(1024);
  });

  it('returns 1024 for flux', () => {
    expect(getOptimalDimension('flux')).toBe(1024);
  });

  it('returns 1024 for sdxl', () => {
    expect(getOptimalDimension('sdxl')).toBe(1024);
  });

  it('returns 1024 for z-image', () => {
    expect(getOptimalDimension('z-image')).toBe(1024);
  });

  it('returns 1024 for null/undefined', () => {
    expect(getOptimalDimension(null)).toBe(1024);
    expect(getOptimalDimension(undefined)).toBe(1024);
  });
});

describe('getGridSize', () => {
  it('returns 16 for qwen-image', () => {
    expect(getGridSize('qwen-image')).toBe(16);
  });

  it('returns 16 for flux', () => {
    expect(getGridSize('flux')).toBe(16);
  });

  it('returns 16 for z-image', () => {
    expect(getGridSize('z-image')).toBe(16);
  });

  it('returns 32 for cogview4', () => {
    expect(getGridSize('cogview4')).toBe(32);
  });

  it('returns 8 for sd-1', () => {
    expect(getGridSize('sd-1')).toBe(8);
  });

  it('returns 8 for sdxl', () => {
    expect(getGridSize('sdxl')).toBe(8);
  });

  it('returns 8 for null/undefined', () => {
    expect(getGridSize(null)).toBe(8);
    expect(getGridSize(undefined)).toBe(8);
  });
});

describe('getIsSizeOptimal', () => {
  it('returns true for dimensions near optimal area for qwen-image (1024x1024)', () => {
    expect(getIsSizeOptimal(1024, 1024, 'qwen-image')).toBe(true);
  });

  it('returns true for non-square dimensions within 20% of optimal area', () => {
    // 896x1152 = 1,032,192 vs optimal 1,048,576 (~1.6% diff)
    expect(getIsSizeOptimal(896, 1152, 'qwen-image')).toBe(true);
  });

  it('returns false for dimensions too small (< 80% of optimal area)', () => {
    // 512x512 = 262,144 vs optimal 1,048,576 (~75% too small)
    expect(getIsSizeOptimal(512, 512, 'qwen-image')).toBe(false);
  });

  it('returns false for dimensions too large (> 120% of optimal area)', () => {
    // 2048x2048 = 4,194,304 vs optimal 1,048,576 (~300% too large)
    expect(getIsSizeOptimal(2048, 2048, 'qwen-image')).toBe(false);
  });

  it('returns true for sd-1 at 512x512', () => {
    expect(getIsSizeOptimal(512, 512, 'sd-1')).toBe(true);
  });

  it('returns false for sd-1 at 1024x1024 (too large)', () => {
    expect(getIsSizeOptimal(1024, 1024, 'sd-1')).toBe(false);
  });
});

describe('getIsSizeTooSmall', () => {
  it('returns true when area is below 80% of optimal', () => {
    expect(getIsSizeTooSmall(400, 400, 1024)).toBe(true);
  });

  it('returns false when area is at or above 80% of optimal', () => {
    expect(getIsSizeTooSmall(920, 920, 1024)).toBe(false);
  });
});

describe('getIsSizeTooLarge', () => {
  it('returns true when area exceeds 120% of optimal', () => {
    expect(getIsSizeTooLarge(1200, 1200, 1024)).toBe(true);
  });

  it('returns false when area is at or below 120% of optimal', () => {
    expect(getIsSizeTooLarge(1100, 1024, 1024)).toBe(false);
  });
});
