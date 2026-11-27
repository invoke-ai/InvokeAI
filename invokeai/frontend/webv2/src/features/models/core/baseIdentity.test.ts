import { describe, expect, it } from 'vitest';

import type { ModelBase, ModelConfig } from './types';

import {
  getModelBaseColorPalette,
  getModelBaseInfo,
  getModelBaseLabel,
  isConvertibleToDiffusers,
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
  it('offers Diffusers conversion for SDXL checkpoints but not FLUX ones', () => {
    expect(isConvertibleToDiffusers(createModel({ base: 'sdxl', format: 'checkpoint', type: 'main' }))).toBe(true);
    expect(isConvertibleToDiffusers(createModel({ base: 'flux', format: 'checkpoint', type: 'main' }))).toBe(false);
  });

  it('falls back safely for unknown bases', () => {
    const base = 'made-up' as ModelBase;

    expect(getModelBaseInfo(base)).toEqual({ base, label: 'Made Up', colorPalette: 'gray' });
    expect(getModelBaseLabel(base)).toBe('Made Up');
    expect(getModelBaseColorPalette(base)).toBe('gray');
  });
});
