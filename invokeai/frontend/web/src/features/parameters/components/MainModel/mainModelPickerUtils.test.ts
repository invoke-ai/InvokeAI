import type {
  ExternalApiModelConfig,
  ExternalApiModelDefaultSettings,
  ExternalImageSize,
  ExternalModelCapabilities,
} from 'services/api/types';
import { describe, expect, it } from 'vitest';

import { isExternalModelUnsupportedForTab } from './mainModelPickerUtils';

const createExternalConfig = (modes: ExternalModelCapabilities['modes']): ExternalApiModelConfig => {
  const maxImageSize: ExternalImageSize = { width: 1024, height: 1024 };
  const defaultSettings: ExternalApiModelDefaultSettings = { width: 1024, height: 1024, steps: 30 };

  return {
    key: 'external-test',
    hash: 'external:openai:gpt-image-1',
    path: 'external://openai/gpt-image-1',
    file_size: 0,
    name: 'External Test',
    description: null,
    source: 'external://openai/gpt-image-1',
    source_type: 'url',
    source_api_response: null,
    cover_image: null,
    base: 'external',
    type: 'external_image_generator',
    format: 'external_api',
    provider_id: 'openai',
    provider_model_id: 'gpt-image-1',
    capabilities: {
      modes,
      supports_negative_prompt: true,
      supports_reference_images: false,
      max_image_size: maxImageSize,
    },
    default_settings: defaultSettings,
    tags: ['external'],
    is_default: false,
  };
};

describe('isExternalModelUnsupportedForTab', () => {
  it('disables external models without txt2img for generate', () => {
    const model = createExternalConfig(['img2img', 'inpaint']);

    expect(isExternalModelUnsupportedForTab(model, 'generate')).toBe(true);
  });

  it('allows external models with txt2img for generate', () => {
    const model = createExternalConfig(['txt2img']);

    expect(isExternalModelUnsupportedForTab(model, 'generate')).toBe(false);
  });

  it('allows external models on canvas', () => {
    const model = createExternalConfig(['inpaint']);

    expect(isExternalModelUnsupportedForTab(model, 'canvas')).toBe(false);
  });
});
