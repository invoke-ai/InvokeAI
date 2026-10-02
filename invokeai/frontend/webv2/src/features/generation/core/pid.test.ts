import { describe, expect, it } from 'vitest';

import {
  clampPidSteps,
  getIsPidActive,
  getIsPidSupportedBase,
  getPidDecoderBaseForMainBase,
  getPidDimensionOverrides,
  getPidGenerationSize,
  getPidScale,
  isPidMode,
  PID_SCALE,
} from './pid';

describe('getPidDecoderBaseForMainBase', () => {
  it('maps each supported base to its own decoder base', () => {
    expect(getPidDecoderBaseForMainBase('flux')).toBe('flux');
    expect(getPidDecoderBaseForMainBase('flux2')).toBe('flux2');
    expect(getPidDecoderBaseForMainBase('sd-3')).toBe('sd-3');
    expect(getPidDecoderBaseForMainBase('sdxl')).toBe('sdxl');
    expect(getPidDecoderBaseForMainBase('qwen-image')).toBe('qwen-image');
  });

  it('maps Z-Image to the FLUX decoder, which it shares a VAE with', () => {
    // Z-Image ships no PiD checkpoints of its own; it reuses FLUX's 16-channel decoder.
    expect(getPidDecoderBaseForMainBase('z-image')).toBe('flux');
  });

  it('returns null for bases with no PiD support', () => {
    expect(getPidDecoderBaseForMainBase('krea-2')).toBeNull();
    expect(getPidDecoderBaseForMainBase('wan')).toBeNull();
    expect(getPidDecoderBaseForMainBase('sd-1')).toBeNull();
    expect(getPidDecoderBaseForMainBase('anima')).toBeNull();
    expect(getPidDecoderBaseForMainBase(null)).toBeNull();
    expect(getPidDecoderBaseForMainBase(undefined)).toBeNull();
  });
});

describe('getIsPidSupportedBase / getIsPidActive', () => {
  it('reports support independently of the mode', () => {
    expect(getIsPidSupportedBase('flux')).toBe(true);
    expect(getIsPidSupportedBase('krea-2')).toBe(false);
  });

  it('is inactive while off, and on an unsupported base whatever the mode', () => {
    expect(getIsPidActive('off', 'flux')).toBe(false);
    expect(getIsPidActive('fit', 'krea-2')).toBe(false);
    expect(getIsPidActive('native', 'krea-2')).toBe(false);
  });

  it('is active for a supported base in fit and native', () => {
    expect(getIsPidActive('fit', 'flux')).toBe(true);
    expect(getIsPidActive('native', 'z-image')).toBe(true);
  });
});

describe('getPidScale', () => {
  it('only scales in native mode', () => {
    expect(getPidScale('off')).toBe(1);
    expect(getPidScale('fit')).toBe(1);
    expect(getPidScale('native')).toBe(PID_SCALE);
  });
});

describe('getPidGenerationSize', () => {
  it('leaves the size alone when off or fitting', () => {
    expect(getPidGenerationSize({ height: 1024, width: 1024 }, 'off', 16)).toEqual({ height: 1024, width: 1024 });
    expect(getPidGenerationSize({ height: 1024, width: 1024 }, 'fit', 16)).toEqual({ height: 1024, width: 1024 });
  });

  it('generates at a quarter of the requested size in native mode', () => {
    // 2048 is the native optimum; a quarter of it is FLUX's own 512 optimum.
    expect(getPidGenerationSize({ height: 2048, width: 2048 }, 'native', 16)).toEqual({ height: 512, width: 512 });
  });

  it('keeps a non-square native target proportional', () => {
    expect(getPidGenerationSize({ height: 1024, width: 2048 }, 'native', 16)).toEqual({ height: 256, width: 512 });
  });

  it('floors a hand-entered size onto the model grid', () => {
    // 2044 / 4 = 511, which is not a multiple of 16, so it floors to 496.
    expect(getPidGenerationSize({ height: 2044, width: 2044 }, 'native', 16)).toEqual({ height: 496, width: 496 });
  });

  it('never floors below one grid step', () => {
    expect(getPidGenerationSize({ height: 16, width: 16 }, 'native', 16)).toEqual({ height: 16, width: 16 });
  });
});

describe('getPidDimensionOverrides', () => {
  it('leaves the model rules alone when PiD is off or fitting', () => {
    expect(getPidDimensionOverrides('off', 'flux', 16, 1024)).toEqual({ grid: 16, optimalSide: 1024 });
    // Fit generates at the requested size, so the model's own grid still applies.
    expect(getPidDimensionOverrides('fit', 'flux', 16, 1024)).toEqual({ grid: 16, optimalSide: 1024 });
  });

  it('scales the grid and reports PiD’s own optimum in native mode', () => {
    // grid * 4 keeps requested / 4 on the model's grid; 2048 is 512 -> 4x.
    expect(getPidDimensionOverrides('native', 'flux', 16, 1024)).toEqual({ grid: 64, optimalSide: 2048 });
    expect(getPidDimensionOverrides('native', 'sdxl', 8, 1024)).toEqual({ grid: 32, optimalSide: 2048 });
  });

  it('leaves an unsupported base alone even in native mode', () => {
    expect(getPidDimensionOverrides('native', 'krea-2', 16, 1024)).toEqual({ grid: 16, optimalSide: 1024 });
  });

  it('produces a grid that divides the native optimum exactly', () => {
    for (const [base, grid] of [
      ['flux', 16],
      ['sdxl', 8],
    ] as const) {
      const overrides = getPidDimensionOverrides('native', base, grid, 1024);

      expect(overrides.optimalSide % overrides.grid).toBe(0);
      // And the resulting generation size lands on the model's own grid.
      expect(
        getPidGenerationSize({ height: overrides.optimalSide, width: overrides.optimalSide }, 'native', grid)
      ).toEqual({ height: overrides.optimalSide / 4, width: overrides.optimalSide / 4 });
    }
  });
});

describe('isPidMode / clampPidSteps', () => {
  it('accepts only the three modes', () => {
    expect(isPidMode('off')).toBe(true);
    expect(isPidMode('fit')).toBe(true);
    expect(isPidMode('native')).toBe(true);
    expect(isPidMode('4x')).toBe(false);
    expect(isPidMode(null)).toBe(false);
    expect(isPidMode(4)).toBe(false);
  });

  it('clamps and rounds steps into range', () => {
    expect(clampPidSteps(4)).toBe(4);
    expect(clampPidSteps(0)).toBe(1);
    expect(clampPidSteps(-3)).toBe(1);
    expect(clampPidSteps(99)).toBe(16);
    expect(clampPidSteps(4.6)).toBe(5);
  });
});
