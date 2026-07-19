import type { ModelConfig } from '@features/models';

import { describe, expect, it } from 'vitest';

import { getCompatibleControlModels } from './controlModelOptions';

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
