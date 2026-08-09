import type { ModelConfig } from '@features/models/core/types';

import { describe, expect, it } from 'vitest';

import { getFieldsForModel, supportsFp8Storage } from './defaultSettingsFields';

const model = (base: string, type: string): Pick<ModelConfig, 'base' | 'type'> =>
  ({ base, type }) as Pick<ModelConfig, 'base' | 'type'>;

const fieldKeys = (candidate: Pick<ModelConfig, 'base' | 'type'>): string[] =>
  getFieldsForModel(candidate).map((field) => String(field.key));

describe('supportsFp8Storage', () => {
  it('offers the toggle for main models', () => {
    for (const base of ['sdxl', 'flux', 'krea-2', 'qwen-image', 'wan', 'ideogram-4']) {
      expect(supportsFp8Storage(model(base, 'main'))).toBe(true);
    }
  });

  it('hides it for Z-Image, which the backend refuses to cast', () => {
    // _should_use_fp8 returns False for BaseModelType.ZImage: diffusers' layerwise casting hits
    // a dtype mismatch there (skipped modules bf16, hooked modules fp16).
    expect(supportsFp8Storage(model('z-image', 'main'))).toBe(false);
  });

  it('hides it for LoRA and ControlLoRA, which are patched rather than run', () => {
    // The casting hooks would never fire, so the toggle would be silently ignored.
    expect(supportsFp8Storage(model('sdxl', 'lora'))).toBe(false);
    expect(supportsFp8Storage(model('sdxl', 'control_lora'))).toBe(false);
  });

  it('offers it for the control adapters the backend does cast', () => {
    expect(supportsFp8Storage(model('sdxl', 'controlnet'))).toBe(true);
    expect(supportsFp8Storage(model('sdxl', 't2i_adapter'))).toBe(true);
  });

  it('hides it for VAEs', () => {
    // fp8 storage measurably degrades decode quality, so the backend excludes VAEs outright.
    expect(supportsFp8Storage(model('sdxl', 'vae'))).toBe(false);
  });
});

describe('getFieldsForModel', () => {
  it('appends fp8_storage to the main-model fields without disturbing the existing ones', () => {
    const keys = fieldKeys(model('sdxl', 'main'));

    expect(keys).toContain('fp8_storage');
    expect(keys.slice(0, -1)).toEqual([
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

  it('omits fp8_storage for a Z-Image main but keeps every other default', () => {
    const zImage = fieldKeys(model('z-image', 'main'));

    expect(zImage).not.toContain('fp8_storage');
    expect(zImage).toEqual(fieldKeys(model('sdxl', 'main')).filter((key) => key !== 'fp8_storage'));
  });

  it('leaves the LoRA field list untouched', () => {
    expect(fieldKeys(model('sdxl', 'lora'))).toEqual(['weight']);
  });

  it('adds fp8_storage alongside the preprocessor for a ControlNet', () => {
    expect(fieldKeys(model('sdxl', 'controlnet'))).toEqual(['preprocessor', 'fp8_storage']);
  });

  it('gives a ControlLoRA the preprocessor only', () => {
    expect(fieldKeys(model('sdxl', 'control_lora'))).toEqual(['preprocessor']);
  });
});
