import { describe, expect, it } from 'vitest';

import type { ModelBase, ModelConfig } from './types';

import {
  getModelBaseColorPalette,
  getModelBaseInfo,
  getModelBaseLabel,
  isConvertibleToDiffusers,
  KNOWN_MODEL_BASES,
  MODEL_BASES,
} from './baseIdentity';

const createModel = (overrides: Partial<ModelConfig>): ModelConfig => ({
  base: 'sdxl',
  file_size: 1024,
  format: 'checkpoint',
  hash: 'hash',
  key: 'key',
  name: 'Model',
  path: '/models/model.safetensors',
  source: '/models/model.safetensors',
  source_type: 'path',
  type: 'main',
  ...overrides,
});

describe('MODEL_BASES', () => {
  it('matches known base labels, colors, and conversion support', () => {
    expect(getModelBaseInfo('sd-1')).toMatchObject({
      label: 'SD 1.x',
      colorPalette: 'green',
      supportsDiffusersConversion: true,
    });
    expect(getModelBaseInfo('sd-2')).toMatchObject({
      label: 'SD 2.x',
      colorPalette: 'teal',
      supportsDiffusersConversion: true,
    });
    expect(getModelBaseInfo('sdxl')).toMatchObject({
      label: 'SDXL',
      colorPalette: 'blue',
      supportsDiffusersConversion: true,
    });
    expect(getModelBaseInfo('flux2')).toMatchObject({ label: 'FLUX.2', colorPalette: 'cyan' });
    expect(getModelBaseInfo('qwen-image')).toMatchObject({ label: 'Qwen Image', colorPalette: 'cyan' });
    expect(isConvertibleToDiffusers(createModel({ base: 'sdxl', format: 'checkpoint', type: 'main' }))).toBe(true);
    expect(isConvertibleToDiffusers(createModel({ base: 'flux', format: 'checkpoint', type: 'main' }))).toBe(false);
  });

  it('falls back safely for unknown bases', () => {
    const base = 'made-up' as ModelBase;

    expect(getModelBaseInfo(base)).toEqual({ base, label: 'Made Up', colorPalette: 'gray' });
    expect(getModelBaseLabel(base)).toBe('Made Up');
    expect(getModelBaseColorPalette(base)).toBe('gray');
  });

  it('does not include generation fields', () => {
    for (const info of Object.values(MODEL_BASES)) {
      expect(info).not.toHaveProperty('defaults');
      expect(info).not.toHaveProperty('schedulerSet');
      expect(info).not.toHaveProperty('negativePrompt');
      expect(info).not.toHaveProperty('ui');
    }
  });

  it('covers all known ModelBase literals', () => {
    expect(KNOWN_MODEL_BASES).toEqual([
      'sd-1',
      'sd-2',
      'sdxl',
      'sdxl-refiner',
      'sd-3',
      'flux',
      'flux2',
      'cogview4',
      'qwen-image',
      'z-image',
      'anima',
      'any',
      'external',
      'unknown',
    ]);
  });
});
