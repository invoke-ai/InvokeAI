import { describe, expect, it } from 'vitest';

import { DEFAULT_MODEL_GRID, gridSizeForModelBase } from './bboxGrid';

describe('gridSizeForModelBase', () => {
  it('maps every 16px generation base to a 16px canvas grid', () => {
    for (const base of ['flux', 'flux2', 'sd-3', 'qwen-image', 'z-image', 'ideogram-4', 'krea-2', 'wan']) {
      expect(gridSizeForModelBase(base)).toBe(16);
    }
  });

  it('maps cogview4 to a 32px grid', () => {
    expect(gridSizeForModelBase('cogview4')).toBe(32);
  });

  it('maps sd/sdxl and unknown bases to the default 8px grid', () => {
    for (const base of ['sd-1', 'sd-2', 'sdxl', 'anima', 'mystery-model']) {
      expect(gridSizeForModelBase(base)).toBe(8);
    }
  });

  it('falls back to the default grid for null/undefined', () => {
    expect(gridSizeForModelBase(null)).toBe(DEFAULT_MODEL_GRID);
    expect(gridSizeForModelBase(undefined)).toBe(DEFAULT_MODEL_GRID);
    expect(DEFAULT_MODEL_GRID).toBe(8);
  });
});
