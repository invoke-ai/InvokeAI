import type { ModelConfig } from '@features/models';

import { describe, expect, it } from 'vitest';

import { getReidentifyPendingControlModels } from './controlModelOptions';

const model = (key: string, base: string, type: string, cond_in_channels: number | null): ModelConfig =>
  ({ base, cond_in_channels, key, name: key, type }) as ModelConfig;

describe('getReidentifyPendingControlModels', () => {
  const models = [
    model('sketch', 'anima', 'controlnet', 3),
    model('inpainting', 'anima', 'controlnet', 4),
    model('unidentified', 'anima', 'controlnet', null),
    model('sdxl-unknown', 'sdxl', 'controlnet', null),
    model('anima-lora', 'anima', 'lora', null),
  ];

  it('lists only Anima adapters of unknown type, and only for an Anima main model', () => {
    expect(getReidentifyPendingControlModels(models, 'anima').map((candidate) => candidate.key)).toEqual([
      'unidentified',
    ]);
    expect(getReidentifyPendingControlModels(models, 'sdxl')).toEqual([]);
    expect(getReidentifyPendingControlModels(models, null)).toEqual([]);
  });
});
