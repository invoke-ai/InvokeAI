import type {
  ExternalApiModelConfig,
  ExternalApiModelDefaultSettings,
  ExternalImageSize,
  ExternalModelCapabilities,
  ExternalModelPanelSchema,
} from 'services/api/types';
import { describe, expect, it } from 'vitest';

import {
  paramsSliceConfig,
  positivePromptAddedToHistory,
  promptRemovedFromHistory,
  selectModelSupportsDimensions,
  selectModelSupportsGuidance,
  selectModelSupportsNegativePrompt,
  selectModelSupportsRefImages,
  selectModelSupportsSeed,
  selectModelSupportsSteps,
} from './paramsSlice';
import { getInitialParamsState } from './types';

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
  const defaultSettings: ExternalApiModelDefaultSettings = { width: 1024, height: 1024 };

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
  it('returns false for negative prompt support on external models', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_reference_images: false,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsNegativePrompt.resultFunc(model)).toBe(false);
  });

  it('uses external capabilities for ref image support', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_reference_images: false,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsRefImages.resultFunc(model, config)).toBe(false);
  });

  it('returns false for guidance support on external models', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_reference_images: false,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsGuidance.resultFunc(model)).toBe(false);
  });

  it('uses external capabilities for seed support', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_reference_images: false,
      supports_seed: false,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsSeed.resultFunc(model, config)).toBe(false);
  });

  it('returns false for steps support on external models', () => {
    const config = createExternalConfig({
      modes: ['txt2img'],
      supports_reference_images: false,
    });
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsSteps.resultFunc(model)).toBe(false);
  });

  it('prefers panel schema over capabilities for control visibility', () => {
    const config = createExternalConfig(
      {
        modes: ['txt2img'],
        supports_reference_images: true,
        supports_seed: true,
      },
      {
        prompts: [{ name: 'reference_images' }],
        image: [{ name: 'dimensions' }],
        generation: [],
      }
    );
    const model = buildExternalModelIdentifier(config);

    expect(selectModelSupportsNegativePrompt.resultFunc(model)).toBe(false);
    expect(selectModelSupportsRefImages.resultFunc(model, config)).toBe(true);
    expect(selectModelSupportsGuidance.resultFunc(model)).toBe(false);
    expect(selectModelSupportsSeed.resultFunc(model, config)).toBe(false);
    expect(selectModelSupportsSteps.resultFunc(model)).toBe(false);
    expect(selectModelSupportsDimensions.resultFunc(model, config)).toBe(true);
  });
});

describe('paramsSliceConfig persisted state migration', () => {
  const migrate = paramsSliceConfig.persistConfig?.migrate;

  it('backfills new Qwen Image fields when migrating from v2 and preserves existing params', () => {
    expect(migrate).toBeDefined();

    // Build a valid pre-PR v2 persisted state by removing the fields that were added in v3
    const initial = getInitialParamsState();
    const v2State: Record<string, unknown> = {
      ...initial,
      _version: 2,
      positivePrompt: 'a fluffy cat',
      seed: 42,
      shouldRandomizeSeed: false,
      dimensions: { ...initial.dimensions, width: 768, height: 768 },
    };
    delete v2State.qwenImageVaeModel;
    delete v2State.qwenImageQwenVLEncoderModel;

    const result = migrate?.(v2State) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(4);
    expect(result.qwenImageVaeModel).toBeNull();
    expect(result.qwenImageQwenVLEncoderModel).toBeNull();
    // Existing params should be preserved
    expect(result.positivePrompt).toBe('a fluffy cat');
    expect(result.seed).toBe(42);
    expect(result.shouldRandomizeSeed).toBe(false);
    expect(result.dimensions.width).toBe(768);
    expect(result.dimensions.height).toBe(768);
  });

  it('backfills Krea-2 fields when migrating from v3 and preserves existing params', () => {
    expect(migrate).toBeDefined();

    const initial = getInitialParamsState();
    const v3State: Record<string, unknown> = {
      ...initial,
      _version: 3,
      positivePrompt: 'preserve this prompt',
      seed: 1234,
      dimensions: { ...initial.dimensions, width: 640, height: 896 },
    };
    delete v3State.krea2VaeModel;
    delete v3State.krea2Qwen3VlEncoderModel;
    delete v3State.krea2SeedVarianceEnabled;
    delete v3State.krea2SeedVarianceStrength;
    delete v3State.krea2SeedVarianceRandomizePercent;
    delete v3State.krea2RebalanceEnabled;
    delete v3State.krea2RebalanceMultiplier;
    delete v3State.krea2RebalanceWeights;

    const result = migrate?.(v3State) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(4);
    expect(result.krea2VaeModel).toBeNull();
    expect(result.krea2Qwen3VlEncoderModel).toBeNull();
    expect(result.krea2SeedVarianceEnabled).toBe(false);
    expect(result.krea2SeedVarianceStrength).toBe(20);
    expect(result.krea2SeedVarianceRandomizePercent).toBe(50);
    expect(result.krea2RebalanceEnabled).toBe(false);
    expect(result.krea2RebalanceMultiplier).toBe(4);
    expect(result.krea2RebalanceWeights).toBe('1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0');
    expect(result.positivePrompt).toBe('preserve this prompt');
    expect(result.seed).toBe(1234);
    expect(result.dimensions).toMatchObject({ width: 640, height: 896 });
  });

  it('migrates old positive prompt history entries to prompt pairs', () => {
    expect(migrate).toBeDefined();

    const initial = getInitialParamsState();
    const v3State: Record<string, unknown> = {
      ...initial,
      positivePromptHistory: ['a fluffy cat'],
    };

    const result = migrate?.(v3State) as ReturnType<typeof getInitialParamsState>;

    expect(result.positivePromptHistory).toEqual([{ positivePrompt: 'a fluffy cat', negativePrompt: null }]);
  });
});

describe('paramsSlice prompt history', () => {
  it('stores positive and negative prompts in the same history item', () => {
    const initial = getInitialParamsState();
    const state = paramsSliceConfig.slice.reducer(
      initial,
      positivePromptAddedToHistory({ positivePrompt: ' a fluffy cat ', negativePrompt: ' blurry ' })
    );

    expect(state.positivePromptHistory).toEqual([{ positivePrompt: 'a fluffy cat', negativePrompt: 'blurry' }]);
  });

  it('deduplicates and removes prompt history by positive and negative prompt pair', () => {
    const initial = getInitialParamsState();
    const withFirstPrompt = paramsSliceConfig.slice.reducer(
      initial,
      positivePromptAddedToHistory({ positivePrompt: 'a cat', negativePrompt: 'blurry' })
    );
    const withSecondPrompt = paramsSliceConfig.slice.reducer(
      withFirstPrompt,
      positivePromptAddedToHistory({ positivePrompt: 'a cat', negativePrompt: 'low quality' })
    );
    const removed = paramsSliceConfig.slice.reducer(
      withSecondPrompt,
      promptRemovedFromHistory({ positivePrompt: 'a cat', negativePrompt: 'blurry' })
    );

    expect(withSecondPrompt.positivePromptHistory).toEqual([
      { positivePrompt: 'a cat', negativePrompt: 'low quality' },
      { positivePrompt: 'a cat', negativePrompt: 'blurry' },
    ]);
    expect(removed.positivePromptHistory).toEqual([{ positivePrompt: 'a cat', negativePrompt: 'low quality' }]);
  });
});
