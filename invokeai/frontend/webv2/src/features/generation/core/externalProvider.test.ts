import { describe, expect, it } from 'vitest';

import type { ExternalImageGeneratorModelConfig, GenerateModelConfig } from './types';

import {
  findExternalResolutionPreset,
  getExternalAspectRatioSizes,
  getExternalProviderOptionKeys,
  getExternalReferenceImages,
  resolveExternalCanvasRequestSize,
  resolveExternalImageSize,
} from './externalProvider';

const external = (
  providerId: string,
  capabilities: ExternalImageGeneratorModelConfig['capabilities'],
  providerModelId = providerId === 'openai' ? 'gpt-image-1' : `${providerId}-model`
): GenerateModelConfig => ({
  base: 'external',
  capabilities,
  format: 'external_api',
  key: `${providerId}-model`,
  name: providerId,
  provider_id: providerId,
  provider_model_id: providerModelId,
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

  it('offers no OpenAI options to DALL·E, which the provider sends none of', () => {
    for (const modelId of ['dall-e-2', 'dall-e-3']) {
      expect(getExternalProviderOptionKeys(external('openai', { modes: ['txt2img', 'img2img'] }, modelId))).toEqual([]);
    }
    expect(getExternalProviderOptionKeys(external('openai', { modes: ['txt2img'] }, 'gpt-image-2'))).toEqual([
      'openaiQuality',
      'openaiBackground',
    ]);
  });

  it('sends enabled reference images with an image, and none to a model that takes no references', () => {
    const image = {
      height: 64,
      imageName: 'reference.png',
      original: { image: { height: 64, image_name: 'reference.png', width: 64 } },
      width: 64,
    };
    const settings = {
      referenceImages: [
        { config: { image, type: 'external_reference_image' as const }, id: 'on', isEnabled: true },
        { config: { image, type: 'external_reference_image' as const }, id: 'off', isEnabled: false },
        { config: { image: null, type: 'external_reference_image' as const }, id: 'empty', isEnabled: true },
      ],
    };

    expect(
      getExternalReferenceImages(
        settings,
        external('seedream', { modes: ['txt2img'], supports_reference_images: true })
      )
    ).toEqual([image]);
    expect(getExternalReferenceImages(settings, external('seedream', { modes: ['txt2img'] }))).toEqual([]);
  });

  it('resolves a named size only from the selected model’s own presets, at the preset’s exact size', () => {
    const gemini = external('gemini', { modes: ['txt2img'], resolution_presets: presets });

    expect(resolveExternalImageSize({ externalImageSize: '2K', height: 1152, width: 2048 }, gemini)).toBe('2K');
    expect(resolveExternalImageSize({ externalImageSize: '4K', height: 1152, width: 2048 }, gemini)).toBe('2K');
    expect(resolveExternalImageSize({ externalImageSize: null, height: 1152, width: 2048 }, gemini)).toBe('2K');
    // A size typed by hand after picking a preset is custom: the provider buckets the dimensions instead.
    expect(resolveExternalImageSize({ externalImageSize: '2K', height: 1152, width: 2000 }, gemini)).toBeNull();
    expect(
      resolveExternalImageSize(
        { externalImageSize: '2K', height: 1152, width: 2048 },
        external('openai', { modes: ['txt2img'] })
      )
    ).toBeNull();
  });

  it('recognizes typed preset dimensions while keeping unsupported dimensions custom', () => {
    const gemini = external('gemini', { modes: ['txt2img'], resolution_presets: presets });

    expect(findExternalResolutionPreset({ externalImageSize: '2K', height: 1152, width: 2048 }, gemini)?.label).toBe(
      '16:9 (2K)'
    );
    expect(findExternalResolutionPreset({ externalImageSize: '2K', height: 1152, width: 2000 }, gemini)).toBeNull();
    expect(
      findExternalResolutionPreset({ externalImageSize: null, height: 1024, width: 1024 }, gemini)?.image_size
    ).toBe('1K');
  });

  it('breaks a tie between presets of one size by the stored name, else the first', () => {
    const duplicate = external('gemini', {
      modes: ['txt2img'],
      resolution_presets: [
        ...presets,
        { aspect_ratio: '1:1', height: 1024, image_size: 'HD', label: '1:1 (HD)', width: 1024 },
      ],
    });

    expect(findExternalResolutionPreset({ externalImageSize: 'HD', height: 1024, width: 1024 }, duplicate)?.label).toBe(
      '1:1 (HD)'
    );
    expect(resolveExternalImageSize({ externalImageSize: 'HD', height: 1024, width: 1024 }, duplicate)).toBe('HD');
    expect(resolveExternalImageSize({ externalImageSize: null, height: 1024, width: 1024 }, duplicate)).toBe('1K');
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

describe('Canvas request size', () => {
  // The starter catalog's GPT Image and Seedream 2K size tables.
  const gptImage = external('openai', {
    aspect_ratio_sizes: {
      '1:1': { height: 1024, width: 1024 },
      '3:2': { height: 1024, width: 1536 },
      '2:3': { height: 1536, width: 1024 },
    },
    modes: ['txt2img', 'img2img'],
  });
  const seedream = external('seedream', {
    aspect_ratio_sizes: {
      '1:1': { height: 2048, width: 2048 },
      '4:3': { height: 1728, width: 2304 },
      '16:9': { height: 1600, width: 2848 },
      '21:9': { height: 1344, width: 3136 },
    },
    modes: ['txt2img', 'img2img'],
  });

  it('asks a fixed-size provider for its size at the ratio nearest the frame, whatever the frame’s size', () => {
    expect(resolveExternalCanvasRequestSize(gptImage, { height: 512, width: 640 })).toEqual({
      height: 1024,
      width: 1536,
    });
    expect(resolveExternalCanvasRequestSize(gptImage, { height: 500, width: 520 })).toEqual({
      height: 1024,
      width: 1024,
    });
    expect(resolveExternalCanvasRequestSize(gptImage, { height: 3000, width: 1800 })).toEqual({
      height: 1536,
      width: 1024,
    });
    expect(resolveExternalCanvasRequestSize(seedream, { height: 1080, width: 1920 })).toEqual({
      height: 1600,
      width: 2848,
    });
  });

  it('picks the smallest named size covering the frame at the nearest ratio, else the largest', () => {
    const gemini = external('gemini', {
      allowed_aspect_ratios: ['1:1', '16:9'],
      modes: ['txt2img'],
      resolution_presets: [
        { aspect_ratio: '1:1', height: 1024, image_size: '1K', label: '1:1 (1K)', width: 1024 },
        { aspect_ratio: '16:9', height: 576, image_size: '1K', label: '16:9 (1K)', width: 1024 },
        { aspect_ratio: '1:1', height: 2048, image_size: '2K', label: '1:1 (2K)', width: 2048 },
        { aspect_ratio: '1:1', height: 4096, image_size: '4K', label: '1:1 (4K)', width: 4096 },
      ],
    });

    expect(resolveExternalCanvasRequestSize(gemini, { height: 2000, width: 2000 })).toEqual({
      height: 2048,
      width: 2048,
    });
    expect(resolveExternalCanvasRequestSize(gemini, { height: 512, width: 512 })).toEqual({
      height: 1024,
      width: 1024,
    });
    expect(resolveExternalCanvasRequestSize(gemini, { height: 5000, width: 5000 })).toEqual({
      height: 4096,
      width: 4096,
    });
    // The only 16:9 size is too small for this frame, yet the shape wins over a larger size of another shape.
    expect(resolveExternalCanvasRequestSize(gemini, { height: 1080, width: 1920 })).toEqual({
      height: 576,
      width: 1024,
    });
  });

  it('asks a ratio-limited provider for an exact multiple of the nearest ratio covering the frame, within its maximum', () => {
    const ratios = external('custom', {
      allowed_aspect_ratios: ['1:1', '21:9', 'wide'],
      max_image_size: { height: 1024, width: 2048 },
      modes: ['txt2img'],
    });

    expect(resolveExternalCanvasRequestSize(ratios, { height: 270, width: 630 })).toEqual({ height: 270, width: 630 });
    expect(resolveExternalCanvasRequestSize(ratios, { height: 500, width: 1000 })).toEqual({
      height: 504,
      width: 1176,
    });
    expect(resolveExternalCanvasRequestSize(ratios, { height: 1500, width: 1500 })).toEqual({
      height: 1024,
      width: 1024,
    });
  });

  it('keeps an exact ratio multiple within the generation limit when the provider declares no maximum', () => {
    const ratios = external('custom', { allowed_aspect_ratios: ['21:9'], modes: ['txt2img'] });

    // The covering multiple would be 4116×1764; the largest one within 4096 px is 195 × (21, 9).
    expect(resolveExternalCanvasRequestSize(ratios, { height: 1755, width: 4096 })).toEqual({
      height: 1755,
      width: 4095,
    });
  });

  it('asks a free-form provider for the frame itself', () => {
    expect(
      resolveExternalCanvasRequestSize(external('atlascloud', { modes: ['txt2img'] }), { height: 333, width: 777 })
    ).toEqual({ height: 333, width: 777 });
  });
});
