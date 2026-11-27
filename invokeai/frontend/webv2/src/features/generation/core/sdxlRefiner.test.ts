import { describe, expect, it } from 'vitest';

import type { MainModelConfig } from './types';

import { getActiveRefiner, getBaseDenoisingRange, getRefinerDenoisingSplit, isRefinerModel } from './sdxlRefiner';

const sdxl: MainModelConfig = { base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' };
const refiner: MainModelConfig = { base: 'sdxl-refiner', key: 'refiner', name: 'Refiner', type: 'main' };
const withRefiner = { refinerModel: refiner, refinerStart: 0.8 };

describe('sdxlRefiner', () => {
  it('activates a refiner only for a refiner-family model on an SDXL main', () => {
    expect(getActiveRefiner(withRefiner, sdxl)).toBe(refiner);
    expect(getActiveRefiner(withRefiner, { ...sdxl, base: 'sd-1' })).toBeNull();
    expect(getActiveRefiner({ ...withRefiner, refinerModel: sdxl }, sdxl)).toBeNull();
    expect(getActiveRefiner({ ...withRefiner, refinerModel: null }, sdxl)).toBeNull();
    expect(isRefinerModel({ base: 'sdxl-refiner', type: 'lora' })).toBe(false);
  });

  it('treats a start of 1 as off, since the refiner would have no schedule left', () => {
    expect(getActiveRefiner({ ...withRefiner, refinerStart: 1 }, sdxl)).toBeNull();
    expect(getActiveRefiner({ ...withRefiner, refinerStart: 0 }, sdxl)).toBe(refiner);
  });

  it('ends the base pass at the hand-over and never starts it later than that', () => {
    expect(getRefinerDenoisingSplit(0.8)).toEqual({ denoising_end: 0.8, denoising_start: 0 });
    expect(getRefinerDenoisingSplit(0.8, 0.3)).toEqual({ denoising_end: 0.8, denoising_start: 0.3 });
    expect(getRefinerDenoisingSplit(0.8, 0.95)).toEqual({ denoising_end: 0.8, denoising_start: 0.8 });
  });

  it('owns the base range for every caller: a strength-derived start never restores the full schedule', () => {
    expect(getBaseDenoisingRange(withRefiner, sdxl, 0)).toEqual({ denoising_end: 0.8, denoising_start: 0 });
    expect(getBaseDenoisingRange(withRefiner, sdxl, 0.4)).toEqual({ denoising_end: 0.8, denoising_start: 0.4 });
    expect(getBaseDenoisingRange(withRefiner, sdxl, 0.95)).toEqual({ denoising_end: 0.8, denoising_start: 0.8 });
    expect(getBaseDenoisingRange({ ...withRefiner, refinerModel: null }, sdxl, 0.4)).toEqual({
      denoising_end: 1,
      denoising_start: 0.4,
    });
    expect(getBaseDenoisingRange({ ...withRefiner, refinerStart: 1 }, sdxl, 0.4)).toEqual({
      denoising_end: 1,
      denoising_start: 0.4,
    });
  });
});
