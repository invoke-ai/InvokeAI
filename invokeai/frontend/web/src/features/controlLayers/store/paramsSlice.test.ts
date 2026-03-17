import type {
  ExternalApiModelConfig,
  ExternalApiModelDefaultSettings,
  ExternalImageSize,
  ExternalModelCapabilities,
  ExternalModelPanelSchema,
} from 'services/api/types';
import { describe, expect, it } from 'vitest';

import {
  selectModelSupportsDimensions,
  selectModelSupportsGuidance,
  selectModelSupportsNegativePrompt,
  selectModelSupportsRefImages,
  selectModelSupportsSeed,
  selectModelSupportsSteps,
} from './paramsSlice';

const buildExternalModelIdentifier = (config: ExternalApiModelConfig) =>
  ({
    key: config.key,
    hash: config.hash,
    name: config.name,
    base: config.base,
    type: config.type,
  }) as const;

const createExternalConfig = (
  capabilities: ExternalModelCapabilities,
  panelSchema?: ExternalModelPanelSchema
): ExternalApiModelConfig => {
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
    panel_schema: panelSchema,
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
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsNegativePrompt.resultFunc(model, config)).toBe(true);
  });

  it('uses external capabilities for ref image support', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_negative_prompt: false,
      supports_reference_images: false,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsRefImages.resultFunc(model, config)).toBe(false);
  });

  it('uses external capabilities for guidance support', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_negative_prompt: true,
      supports_reference_images: false,
      supports_guidance: true,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsGuidance.resultFunc(model, config)).toBe(true);
  });

  it('uses external capabilities for seed support', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_negative_prompt: true,
      supports_reference_images: false,
      supports_seed: false,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsSeed.resultFunc(model, config)).toBe(false);
  });

  it('uses external capabilities for steps support', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_negative_prompt: true,
      supports_reference_images: false,
      supports_steps: false,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsSteps.resultFunc(model, config)).toBe(false);
  });

  it('prefers panel schema over capabilities for control visibility', () => {
    const config = createExternalConfig(
      {
        modes: ['txt2img'],
        supports_negative_prompt: true,
        supports_reference_images: true,
        supports_guidance: true,
        supports_seed: true,
        supports_steps: true,
      },
      {
        prompts: [{ name: 'reference_images' }],
        image: [{ name: 'dimensions' }],
        generation: [],
      }
    );
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsNegativePrompt.resultFunc(model, config)).toBe(false);
    expect(selectModelSupportsRefImages.resultFunc(model, config)).toBe(true);
    expect(selectModelSupportsGuidance.resultFunc(model, config)).toBe(false);
    expect(selectModelSupportsSeed.resultFunc(model, config)).toBe(false);
    expect(selectModelSupportsSteps.resultFunc(model, config)).toBe(false);
    expect(selectModelSupportsDimensions.resultFunc(model, config)).toBe(true);
  });
});
