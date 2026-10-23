import { describe, expect, it } from 'vitest';

import type { ExternalModelSettingsModel } from './externalModelFields';

import {
  isStarterManagedExternalModel,
  toExternalModelChanges,
  toExternalModelValues,
  validateExternalModel,
} from './externalModelFields';

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
  source: 'external://gemini/gemini-3-pro-image-preview',
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

  it('rejects an empty route, no modes, half a max size, and non-positive limits, naming the field to mark', () => {
    const values = toExternalModelValues(geminiModel);

    expect(validateExternalModel(values)).toBeNull();
    expect(validateExternalModel({ ...values, providerModelId: '  ' })).toEqual({
      field: 'providerModelId',
      message: 'Provider model ID is required.',
    });
    expect(validateExternalModel({ ...values, modeTxt2img: false })).toEqual({
      field: 'modeTxt2img',
      message: 'Choose at least one supported mode.',
    });
    expect(validateExternalModel({ ...values, maxImageHeight: null })).toEqual({
      field: 'maxImageHeight',
      message: 'Set both a max width and a max height, or neither.',
    });
    expect(validateExternalModel({ ...values, maxImageWidth: null })?.field).toBe('maxImageWidth');
    expect(validateExternalModel({ ...values, maxImagesPerRequest: 0 })).toEqual({
      field: 'maxImagesPerRequest',
      message: 'Images per request must be greater than 0.',
    });
    expect(validateExternalModel({ ...values, defaultWidth: 1024.5 })).toEqual({
      field: 'defaultWidth',
      message: 'Default width must be a whole number.',
    });
  });

  it('recognises a record the starter catalog manages by its source, or by the source an install would record', () => {
    const starters = [
      { source: 'external://openai/gpt-image-1' },
      { source: 'external://gemini/gemini-2.5-flash-image' },
    ];

    expect(
      isStarterManagedExternalModel(
        { provider_id: 'openai', provider_model_id: 'renamed', source: 'external://openai/gpt-image-1' },
        starters
      )
    ).toBe(true);
    expect(isStarterManagedExternalModel({ provider_id: 'openai', provider_model_id: 'gpt-image-1' }, starters)).toBe(
      true
    );
    expect(
      isStarterManagedExternalModel(
        { provider_id: 'openai', provider_model_id: 'gpt-image-1', source: 'external://custom/gpt-image-1' },
        starters
      )
    ).toBe(false);
    expect(isStarterManagedExternalModel({ provider_id: 'custom', provider_model_id: 'mine' }, [])).toBe(false);
  });
});
