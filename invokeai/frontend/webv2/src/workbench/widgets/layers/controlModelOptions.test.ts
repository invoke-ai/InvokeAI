import type { ModelConfig } from '@features/models';

import { describe, expect, it } from 'vitest';

import {
  getCompatibleControlModels,
  resolveDefaultControlModel,
  resolveDefaultControlModelForBase,
} from './controlModelOptions';

const model = (key: string, base: string, type: string): ModelConfig => ({ base, key, name: key, type }) as ModelConfig;

describe('getCompatibleControlModels', () => {
  it('returns only Z-Image ControlNet models for z_image_control', () => {
    const models = [
      model('z-control', 'z-image', 'controlnet'),
      model('sd-control', 'sd-1', 'controlnet'),
      model('z-t2i', 'z-image', 't2i_adapter'),
    ];

    expect(getCompatibleControlModels(models, 'z-image', 'z_image_control').map((candidate) => candidate.key)).toEqual([
      'z-control',
    ]);
    expect(getCompatibleControlModels(models, 'sd-1', 'z_image_control')).toEqual([]);
  });

  it('keeps existing adapter filtering unchanged', () => {
    const models = [model('sd-control', 'sd-1', 'controlnet'), model('flux-lora', 'flux', 'control_lora')];
    expect(getCompatibleControlModels(models, 'sd-1', 'controlnet').map((candidate) => candidate.key)).toEqual([
      'sd-control',
    ]);
    expect(getCompatibleControlModels(models, null, 'control_lora').map((candidate) => candidate.key)).toEqual([
      'flux-lora',
    ]);
  });
});

describe('resolveDefaultControlModel', () => {
  it('prefers a union model, then tile, then the first compatible', () => {
    const canny = model('canny', 'sdxl', 'controlnet');
    const tile = { ...model('tile-model', 'sdxl', 'controlnet'), name: 'ControlNet Tile' } as ModelConfig;
    const union = { ...model('union-model', 'sdxl', 'controlnet'), name: 'Union Pro XL' } as ModelConfig;

    expect(resolveDefaultControlModel([canny, tile, union], 'sdxl', 'controlnet')).toBe('union-model');
    expect(resolveDefaultControlModel([canny, tile], 'sdxl', 'controlnet')).toBe('tile-model');
    expect(resolveDefaultControlModel([canny], 'sdxl', 'controlnet')).toBe('canny');
  });

  it('only considers models compatible with the base and kind', () => {
    const models = [
      { ...model('sd1-union', 'sd-1', 'controlnet'), name: 'Union' } as ModelConfig,
      model('sdxl-canny', 'sdxl', 'controlnet'),
    ];

    expect(resolveDefaultControlModel(models, 'sdxl', 'controlnet')).toBe('sdxl-canny');
    expect(resolveDefaultControlModel(models, 'sdxl', 't2i_adapter')).toBeNull();
    expect(resolveDefaultControlModel([], 'sdxl', 'controlnet')).toBeNull();
  });
});

describe('resolveDefaultControlModelForBase', () => {
  it('resolves the z_image_control kind for the z-image base and controlnet otherwise', () => {
    const models = [model('z-control', 'z-image', 'controlnet'), model('sd-control', 'sd-1', 'controlnet')];

    expect(resolveDefaultControlModelForBase(models, 'z-image')).toBe('z-control');
    expect(resolveDefaultControlModelForBase(models, 'sd-1')).toBe('sd-control');
    expect(resolveDefaultControlModelForBase(models, 'sd-2')).toBeNull();
  });
});
