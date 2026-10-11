import { describe, expect, it } from 'vitest';

import type { ExternalImageGeneratorModelConfig, GenerateModelConfig } from './types';

import {
  findExternalResolutionPreset,
  getExternalAspectRatioSizes,
  getExternalProviderOptionKeys,
  resolveExternalImageSize,
} from './externalProvider';

const external = (
  providerId: string,
  capabilities: ExternalImageGeneratorModelConfig['capabilities']
): GenerateModelConfig => ({
  base: 'external',
  capabilities,
  format: 'external_api',
  key: `${providerId}-model`,
  name: providerId,
  provider_id: providerId,
  type: 'external_image_generator',
});

const presets = [
  { aspect_ratio: '1:1', height: 1024, image_size: '1K', label: '1:1 (1K)', width: 1024 },
  { aspect_ratio: '16:9', height: 1152, image_size: '2K', label: '16:9 (2K)', width: 2048 },
];

describe('external provider options', () => {
  it('offers each provider only the options its node declares', () => {
    expect(getExternalProviderOptionKeys(external('gemini', { modes: ['txt2img'] }))).toEqual([
      'geminiTemperature',
      'geminiThinkingLevel',
    ]);
    expect(getExternalProviderOptionKeys(external('seedream', { modes: ['txt2img'] }))).toEqual([
      'seedreamWatermark',
      'seedreamOptimizePrompt',
    ]);
    // Alibaba Cloud and Atlas Cloud nodes add nothing beyond the shared fields.
    expect(getExternalProviderOptionKeys(external('alibabacloud', { modes: ['txt2img'] }))).toEqual([]);
    expect(getExternalProviderOptionKeys({ base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' })).toEqual([]);
    expect(getExternalProviderOptionKeys(undefined)).toEqual([]);
  });

  it('offers OpenAI input fidelity only to models that take image inputs', () => {
    expect(getExternalProviderOptionKeys(external('openai', { modes: ['txt2img'] }))).toEqual([
      'openaiQuality',
      'openaiBackground',
    ]);
    expect(getExternalProviderOptionKeys(external('openai', { modes: ['txt2img', 'img2img'] }))).toContain(
      'openaiInputFidelity'
    );
    expect(
      getExternalProviderOptionKeys(external('openai', { modes: ['txt2img'], supports_reference_images: true }))
    ).toContain('openaiInputFidelity');
  });

  it('resolves a named size only from the selected model’s own presets', () => {
    const gemini = external('gemini', { modes: ['txt2img'], resolution_presets: presets });

    expect(resolveExternalImageSize({ externalImageSize: '2K' }, gemini)).toBe('2K');
    expect(resolveExternalImageSize({ externalImageSize: '4K' }, gemini)).toBeNull();
    expect(
      resolveExternalImageSize({ externalImageSize: '2K' }, external('openai', { modes: ['txt2img'] }))
    ).toBeNull();
  });

  it('matches the preset select to the exact size and named size, so a hand-typed size reads as custom', () => {
    const gemini = external('gemini', { modes: ['txt2img'], resolution_presets: presets });

    expect(findExternalResolutionPreset({ externalImageSize: '2K', height: 1152, width: 2048 }, gemini)?.label).toBe(
      '16:9 (2K)'
    );
    expect(findExternalResolutionPreset({ externalImageSize: '2K', height: 1152, width: 2000 }, gemini)).toBeNull();
    expect(findExternalResolutionPreset({ externalImageSize: null, height: 1024, width: 1024 }, gemini)).toBeNull();
  });

  it('lists fixed ratio sizes in the provider’s order', () => {
    const openai = external('openai', {
      aspect_ratio_sizes: { '1:1': { height: 1024, width: 1024 }, '3:2': { height: 1024, width: 1536 } },
      modes: ['txt2img'],
    });

    expect(getExternalAspectRatioSizes(openai)).toEqual([
      ['1:1', { height: 1024, width: 1024 }],
      ['3:2', { height: 1024, width: 1536 }],
    ]);
    expect(getExternalAspectRatioSizes(external('gemini', { modes: ['txt2img'] }))).toEqual([]);
  });
});
