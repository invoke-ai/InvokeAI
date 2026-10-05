import type { ModelConfig } from '@features/models/core/types';

import { describe, expect, it } from 'vitest';

import { getFieldsForModel, validateDefaults } from './defaultSettingsFields';

const model = (base: string, type: string): Pick<ModelConfig, 'base' | 'type'> =>
  ({ base, type }) as Pick<ModelConfig, 'base' | 'type'>;

const fieldKeys = (candidate: Pick<ModelConfig, 'base' | 'type'>, fp8StorageSupported = true): string[] =>
  getFieldsForModel(candidate, fp8StorageSupported).map((field) => String(field.key));

describe('getFieldsForModel', () => {
  it('appends fp8_storage to the main-model fields without disturbing the existing ones', () => {
    const keys = fieldKeys(model('sdxl', 'main'));

    expect(keys).toContain('fp8_storage');
    expect(keys.slice(0, -1)).toEqual([
      'vae',
      'scheduler',
      'steps',
      'cfg_scale',
      'cfg_rescale_multiplier',
      'guidance',
      'width',
      'height',
      'vae_precision',
    ]);
  });

  it('gives LoRAs the weight and its slider bounds', () => {
    expect(fieldKeys(model('sdxl', 'lora'))).toEqual(['weight', 'weight_min', 'weight_max']);
  });

  it('adds fp8_storage alongside the preprocessor for a ControlNet', () => {
    expect(fieldKeys(model('sdxl', 'controlnet'))).toEqual(['preprocessor', 'fp8_storage']);
  });

  it('gives a ControlLoRA the preprocessor only', () => {
    // Not decided here any more: the backend's table has no `supported` row for a control LoRA, since
    // it is patched into a base model and the casting hooks would never fire.
    expect(fieldKeys(model('sdxl', 'control_lora'), false)).toEqual(['preprocessor']);
  });

  it('leaves the row out entirely when the backend says FP8 Storage does nothing for this model', () => {
    // The whole point of the change: a FLUX ControlNet, a GGUF main model or a model whose loader never
    // casts must not be offered a control that silently does nothing.
    expect(fieldKeys(model('flux', 'controlnet'), false)).toEqual(['preprocessor']);
    expect(fieldKeys(model('flux', 'main'), false)).not.toContain('fp8_storage');
  });
});

describe('validateDefaults', () => {
  const t = ((key: string) => key) as Parameters<typeof validateDefaults>[2];

  it('accepts the vae sentinel and a model key, rejects an empty string', () => {
    expect(validateDefaults(model('sdxl', 'main'), { vae: 'default' }, t)).toBeNull();
    expect(validateDefaults(model('sdxl', 'main'), { vae: 'some-model-key' }, t)).toBeNull();
    expect(validateDefaults(model('sdxl', 'main'), { vae: '' }, t)).not.toBeNull();
  });

  it('mirrors the backend LoRA bound validator: effective min must stay below max', () => {
    expect(validateDefaults(model('sdxl', 'lora'), { weight_max: 2, weight_min: -1 }, t)).toBeNull();
    expect(validateDefaults(model('sdxl', 'lora'), { weight_max: -1, weight_min: 1 }, t)).not.toBeNull();
    // Fallbacks are [-1, 2]: a min alone above 2 is invalid.
    expect(validateDefaults(model('sdxl', 'lora'), { weight_min: 3 }, t)).not.toBeNull();
  });

  it('requires an enabled weight to sit inside the effective range', () => {
    expect(validateDefaults(model('sdxl', 'lora'), { weight: 1.5 }, t)).toBeNull();
    expect(validateDefaults(model('sdxl', 'lora'), { weight: 2.5 }, t)).not.toBeNull();
    expect(validateDefaults(model('sdxl', 'lora'), { weight: 2.5, weight_max: 3 }, t)).toBeNull();
  });
});
