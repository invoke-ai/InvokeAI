import { describe, expect, it } from 'vitest';

import type { MainModelConfig } from './types';

import { getActiveRefiner, getRefinerDenoisingSplit, isRefinerModel } from './sdxlRefiner';

const sdxl: MainModelConfig = { base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' };
const refiner: MainModelConfig = { base: 'sdxl-refiner', key: 'refiner', name: 'Refiner', type: 'main' };

describe('sdxlRefiner', () => {
  it('activates a refiner only for a refiner-family model on an SDXL main', () => {
    expect(getActiveRefiner({ refinerModel: refiner }, sdxl)).toBe(refiner);
    expect(getActiveRefiner({ refinerModel: refiner }, { ...sdxl, base: 'sd-1' })).toBeNull();
    expect(getActiveRefiner({ refinerModel: sdxl }, sdxl)).toBeNull();
    expect(getActiveRefiner({ refinerModel: null }, sdxl)).toBeNull();
    expect(isRefinerModel({ base: 'sdxl-refiner', type: 'lora' })).toBe(false);
  });

  it('ends the base pass at the hand-over and never starts it later than that', () => {
    expect(getRefinerDenoisingSplit(0.8)).toEqual({ denoising_end: 0.8, denoising_start: 0 });
    expect(getRefinerDenoisingSplit(0.8, 0.3)).toEqual({ denoising_end: 0.8, denoising_start: 0.3 });
    expect(getRefinerDenoisingSplit(0.8, 0.95)).toEqual({ denoising_end: 0.8, denoising_start: 0.8 });
  });
});
