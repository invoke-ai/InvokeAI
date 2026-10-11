import type { TFunction } from 'i18next';

import { describe, expect, it } from 'vitest';

import type { ExternalModelSettingsModel } from './externalModelFields';

import { toExternalModelChanges, toExternalModelValues, validateExternalModel } from './externalModelFields';

const t = ((key: string) => key) as unknown as TFunction;

const geminiModel: ExternalModelSettingsModel = {
  capabilities: {
    allowed_aspect_ratios: ['1:1', '16:9'],
    aspect_ratio_sizes: { '1:1': { height: 1024, width: 1024 } },
    mask_format: 'none',
    max_image_size: { height: 4096, width: 4096 },
    max_images_per_request: 1,
    max_reference_images: 14,
    modes: ['txt2img'],
    resolution_presets: [{ aspect_ratio: '1:1', height: 1024, image_size: '1K', label: '1:1 (1K)', width: 1024 }],
    supports_reference_images: true,
    supports_seed: true,
  },
  default_settings: { height: 1024, num_images: 1, width: 1024 },
  key: 'gemini-3-pro',
  name: 'Gemini 3 Pro',
  provider_id: 'gemini',
  provider_model_id: 'gemini-3-pro-image-preview',
  type: 'external_image_generator',
};

describe('external model record mapping', () => {
  it('reads the record into flat editor values, defaulting the fields the backend defaults', () => {
    expect(toExternalModelValues(geminiModel)).toEqual({
      defaultHeight: 1024,
      defaultNumImages: 1,
      defaultWidth: 1024,
      maxImageHeight: 4096,
      maxImageWidth: 4096,
      maxImagesPerRequest: 1,
      maxReferenceImages: 14,
      modeImg2img: false,
      modeInpaint: false,
      modeTxt2img: true,
      providerId: 'gemini',
      providerModelId: 'gemini-3-pro-image-preview',
      supportsNegativePrompt: true,
      supportsReferenceImages: true,
      supportsSeed: true,
    });
    // A record with no capability matrix reads as the backend's defaults: txt2img, negative prompt, nothing else.
    expect(
      toExternalModelValues({ ...geminiModel, capabilities: null, default_settings: null, provider_id: undefined })
    ).toMatchObject({
      defaultWidth: null,
      maxImagesPerRequest: null,
      modeTxt2img: true,
      providerId: '',
      supportsNegativePrompt: true,
      supportsReferenceImages: false,
      supportsSeed: false,
    });
  });

  it('writes a whole capability matrix, keeping the tables the editor does not show', () => {
    const changes = toExternalModelChanges(
      {
        ...toExternalModelValues(geminiModel),
        maxImageHeight: null,
        maxImageWidth: null,
        maxReferenceImages: null,
        modeImg2img: true,
        modeInpaint: true,
        providerId: ' gemini ',
        supportsSeed: false,
      },
      geminiModel
    );

    expect(changes.provider_id).toBe('gemini');
    expect(changes.capabilities).toEqual({
      allowed_aspect_ratios: ['1:1', '16:9'],
      aspect_ratio_sizes: { '1:1': { height: 1024, width: 1024 } },
      mask_format: 'none',
      max_image_size: null,
      max_images_per_request: 1,
      max_reference_images: null,
      modes: ['txt2img', 'img2img', 'inpaint'],
      resolution_presets: [{ aspect_ratio: '1:1', height: 1024, image_size: '1K', label: '1:1 (1K)', width: 1024 }],
      supports_negative_prompt: true,
      supports_reference_images: true,
      supports_seed: false,
    });
    expect(changes.default_settings).toEqual({ height: 1024, num_images: 1, width: 1024 });
  });

  it('clears the defaults record when every default is lifted', () => {
    const changes = toExternalModelChanges(
      { ...toExternalModelValues(geminiModel), defaultHeight: null, defaultNumImages: null, defaultWidth: null },
      geminiModel
    );

    expect(changes.default_settings).toBeNull();
  });

  it('rejects an empty route, no modes, half a max size, and non-positive limits', () => {
    const values = toExternalModelValues(geminiModel);

    expect(validateExternalModel(values, t)).toBeNull();
    expect(validateExternalModel({ ...values, providerModelId: '  ' }, t)).toBe('Provider model ID is required.');
    expect(validateExternalModel({ ...values, modeTxt2img: false }, t)).toBe('Choose at least one supported mode.');
    expect(validateExternalModel({ ...values, maxImageHeight: null }, t)).toBe(
      'Set both a max width and a max height, or neither.'
    );
    expect(validateExternalModel({ ...values, maxImagesPerRequest: 0 }, t)).toBe(
      'Images per request must be greater than 0.'
    );
    expect(validateExternalModel({ ...values, defaultWidth: 1024.5 }, t)).toBe('Default width must be a whole number.');
  });
});
