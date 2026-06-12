import { describe, expect, it } from 'vitest';

import { calculateNewSize, deriveAspectRatioId, isGenerateSettings, normalizeGenerateSettings } from './settings';

/** The persisted widget-value shape from before aspect ratio / VAE / seamless / CLIP skip landed. */
const legacyStoredValues = {
  batchCount: 2,
  cfgRescaleMultiplier: 0,
  cfgScale: 7,
  height: 768,
  modelKey: 'legacy-model',
  negativePrompt: 'blurry',
  positivePrompt: 'a castle',
  scheduler: 'euler_a',
  seed: 123,
  shouldRandomizeSeed: false,
  steps: 30,
  width: 512,
};

describe('normalizeGenerateSettings', () => {
  it('upgrades legacy persisted values without losing the core fields', () => {
    const normalized = normalizeGenerateSettings(legacyStoredValues);

    expect(normalized).not.toBeNull();
    expect(normalized?.positivePrompt).toBe('a castle');
    expect(normalized?.width).toBe(512);
    expect(normalized?.height).toBe(768);
    expect(normalized?.aspectRatioId).toBe('2:3');
    expect(normalized?.aspectRatioIsLocked).toBe(false);
    expect(normalized?.clipSkip).toBe(0);
    expect(normalized?.seamlessXAxis).toBe(false);
    expect(normalized?.vae).toBeNull();
    expect(normalized?.vaePrecision).toBe('fp32');
    expect(normalized && isGenerateSettings(normalized)).toBe(true);
  });

  it('rejects values missing core fields', () => {
    expect(normalizeGenerateSettings({})).toBeNull();
    expect(normalizeGenerateSettings({ ...legacyStoredValues, seed: Number.NaN })).toBeNull();
    expect(normalizeGenerateSettings({ ...legacyStoredValues, positivePrompt: undefined })).toBeNull();
    expect(normalizeGenerateSettings(null)).toBeNull();
  });

  it('drops malformed values for the newer fields back to defaults', () => {
    const normalized = normalizeGenerateSettings({
      ...legacyStoredValues,
      aspectRatioId: 'bogus',
      vae: { key: 'k', name: 'n', type: 'main' },
      vaePrecision: 'fp64',
    });

    expect(normalized?.aspectRatioId).toBe('2:3');
    expect(normalized?.vae).toBeNull();
    expect(normalized?.vaePrecision).toBe('fp32');
  });
});

describe('dimension helpers', () => {
  it('derives the closest preset aspect ratio', () => {
    expect(deriveAspectRatioId(1024, 1024)).toBe('1:1');
    expect(deriveAspectRatioId(1536, 1024)).toBe('3:2');
    expect(deriveAspectRatioId(1000, 770)).toBe('Free');
  });

  it('fits a ratio into a pixel area on the dimension grid', () => {
    const { height, width } = calculateNewSize(1, 1024 * 1024);

    expect(width).toBe(1024);
    expect(height).toBe(1024);

    const wide = calculateNewSize(16 / 9, 1024 * 1024);

    expect(wide.width % 8).toBe(0);
    expect(wide.height % 8).toBe(0);
    expect(wide.width / wide.height).toBeCloseTo(16 / 9, 1);
  });
});
