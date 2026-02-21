import { zModelIdentifierField } from 'features/nodes/types/common';
import type {
  ExternalApiModelConfig,
  ExternalApiModelDefaultSettings,
  ExternalImageSize,
  ExternalModelCapabilities,
} from 'services/api/types';
import { describe, expect, it } from 'vitest';

import { selectModelSupportsNegativePrompt, selectModelSupportsRefImages } from './paramsSlice';

const createExternalConfig = (capabilities: ExternalModelCapabilities): ExternalApiModelConfig => {
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
    capabilities: { ...capabilities, max_image_size: maxImageSize },
    default_settings: defaultSettings,
    tags: ['external'],
    is_default: false,
  };
};

describe('paramsSlice selectors for external models', () => {
  it('uses external capabilities for negative prompt support', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_negative_prompt: true,
      supports_reference_images: false,
    });
    const model = zModelIdentifierField.parse(config);

    expect(selectModelSupportsNegativePrompt.resultFunc(model, config)).toBe(true);
  });

  it('uses external capabilities for ref image support', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_negative_prompt: false,
      supports_reference_images: false,
    });
    const model = zModelIdentifierField.parse(config);

    expect(selectModelSupportsRefImages.resultFunc(model, config)).toBe(false);
  });
});
