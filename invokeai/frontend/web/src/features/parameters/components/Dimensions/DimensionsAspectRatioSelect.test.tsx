import type { ExternalApiModelConfig } from 'services/api/types';
import { describe, expect, test } from 'vitest';

const createExternalModel = (overrides: Partial<ExternalApiModelConfig> = {}): ExternalApiModelConfig => ({
  key: 'external-test',
  name: 'External Test',
  base: 'external',
  type: 'external_image_generator',
  format: 'external_api',
  provider_id: 'gemini',
  provider_model_id: 'gemini-2.5-flash-image',
  description: 'Test model',
  source: 'external://gemini/gemini-2.5-flash-image',
  source_type: 'url',
  source_api_response: null,
  path: '',
  file_size: 0,
  hash: 'external:gemini:gemini-2.5-flash-image',
  cover_image: null,
  is_default: false,
  tags: ['external'],
  capabilities: {
    modes: ['txt2img'],
    supports_reference_images: false,
    supports_negative_prompt: true,
    supports_seed: true,
    supports_guidance: true,
    max_images_per_request: 1,
    max_image_size: null,
    allowed_aspect_ratios: ['1:1', '16:9'],
    max_reference_images: null,
    mask_format: 'none',
    input_image_required_for: null,
  },
  default_settings: null,
  ...overrides,
});

describe('external model aspect ratios', () => {
  test('uses allowed aspect ratios for external models', () => {
    const model = createExternalModel();
    expect(model.capabilities.allowed_aspect_ratios).toEqual(['1:1', '16:9']);
  });
});
