import type {
  ExternalApiModelConfig,
  ExternalApiModelDefaultSettings,
  ExternalImageSize,
  ExternalModelCapabilities,
  ExternalModelPanelSchema,
} from 'services/api/types';
import { describe, expect, it } from 'vitest';

import {
  isValidKrea2RebalanceWeights,
  KREA2_REBALANCE_WEIGHT_COUNT,
  modelChanged,
  paramsSliceConfig,
  parseKrea2RebalanceWeights,
  positivePromptAddedToHistory,
  promptRemovedFromHistory,
  selectModelSupportsDimensions,
  selectModelSupportsGuidance,
  selectModelSupportsNegativePrompt,
  selectModelSupportsRefImages,
  selectModelSupportsSeed,
  selectModelSupportsSteps,
  setIdeogram4Steps,
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

  it('backfills Qwen Image and HiDiffusion fields when migrating from v2 and preserves existing params', () => {
    expect(migrate).toBeDefined();

    // Build a valid pre-PR v2 persisted state by removing the fields that were added later.
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
    delete v2State.hiDiffusionEnabled;
    delete v2State.hiDiffusionRauNetEnabled;
    delete v2State.hiDiffusionWindowAttnEnabled;
    delete v2State.hiDiffusionT1Ratio;
    delete v2State.hiDiffusionT2Ratio;

    const result = migrate?.(v2State) as ReturnType<typeof getInitialParamsState>;

    // v2 migrates all the way through the current chain (v2 -> v3 adds Qwen fields,
    // v3 -> v4 adds Krea-2 and PiD fields).
    expect(result._version).toBe(4);
    expect(result.qwenImageVaeModel).toBeNull();
    expect(result.qwenImageQwenVLEncoderModel).toBeNull();
    expect(result.hiDiffusionEnabled).toBe(false);
    expect(result.hiDiffusionRauNetEnabled).toBe(true);
    expect(result.hiDiffusionWindowAttnEnabled).toBe(true);
    expect(result.hiDiffusionT1Ratio).toBe(0.4);
    expect(result.hiDiffusionT2Ratio).toBe(0.0);
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
    expect(result.krea2SeedVarianceStrength).toBe(0.1);
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

describe('paramsSlice PiD state on base change (modelChanged)', () => {
  const fluxModel = { key: 'flux', hash: 'h', name: 'FLUX', base: 'flux', type: 'main' };
  const modelWithBase = (base: string) => ({ key: base, hash: 'h', name: base, base, type: 'main' });

  const stateOnFluxWithNativePid = () =>
    ({
      ...getInitialParamsState(),
      model: fluxModel,
      pidMode: 'native',
      pidDecoderModel: { key: 'd', name: 'flux decoder', base: 'flux' },
    }) as ReturnType<typeof getInitialParamsState>;

  it('clears an incompatible PiD decoder when switching to a different PiD base (FLUX -> SDXL)', () => {
    const next = paramsSliceConfig.slice.reducer(
      stateOnFluxWithNativePid(),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      modelChanged({ model: modelWithBase('sdxl') as any, previousModel: fluxModel as any })
    );
    // The FLUX decoder is invalid for SDXL, so it is cleared; SDXL supports PiD so the mode is kept.
    expect(next.pidDecoderModel).toBeNull();
    expect(next.pidMode).toBe('native');
  });

  it('keeps the FLUX decoder when switching to Z-Image (which reuses the FLUX decoder)', () => {
    const next = paramsSliceConfig.slice.reducer(
      stateOnFluxWithNativePid(),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      modelChanged({ model: modelWithBase('z-image') as any, previousModel: fluxModel as any })
    );
    expect(next.pidDecoderModel).not.toBeNull();
    expect(next.pidMode).toBe('native');
  });

  it('turns PiD off (and clears the decoder) when switching to a non-PiD base (FLUX -> SD1)', () => {
    const next = paramsSliceConfig.slice.reducer(
      stateOnFluxWithNativePid(),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      modelChanged({ model: modelWithBase('sd-1') as any, previousModel: fluxModel as any })
    );
    expect(next.pidMode).toBe('off');
    expect(next.pidDecoderModel).toBeNull();
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

describe('paramsSlice ideogram4Steps normalization (backend requires >= 2)', () => {
  it('keeps a valid override step count', () => {
    const state = paramsSliceConfig.slice.reducer(getInitialParamsState(), setIdeogram4Steps(20));
    expect(state.ideogram4Steps).toBe(20);
  });

  it('accepts null (use the preset)', () => {
    const state = paramsSliceConfig.slice.reducer(getInitialParamsState(), setIdeogram4Steps(null));
    expect(state.ideogram4Steps).toBeNull();
  });

  it('normalizes a stale out-of-range value (1, below the backend min of 2) to null', () => {
    const state = paramsSliceConfig.slice.reducer(getInitialParamsState(), setIdeogram4Steps(1));
    expect(state.ideogram4Steps).toBeNull();
  });

  it('normalizes a stale rehydrated ideogram4Steps of 1 to null instead of failing the whole slice', () => {
    const migrate = paramsSliceConfig.persistConfig?.migrate;
    expect(migrate).toBeDefined();
    const rehydrated = migrate?.({ ...getInitialParamsState(), ideogram4Steps: 1 }) as ReturnType<
      typeof getInitialParamsState
    >;
    expect(rehydrated.ideogram4Steps).toBeNull();
  });
});

describe('isValidKrea2RebalanceWeights (backend rebalance node requires exactly 12 finite numbers)', () => {
  it('accepts exactly 12 finite comma-separated numbers', () => {
    const parsed = parseKrea2RebalanceWeights('1,1,1,1,1,1,1,2.5,5,1.1,4,1');
    expect(parsed).toEqual([1, 1, 1, 1, 1, 1, 1, 2.5, 5, 1.1, 4, 1]);
    expect(parsed).toHaveLength(KREA2_REBALANCE_WEIGHT_COUNT);
    // Tolerates surrounding whitespace and a trailing comma (empty segments are ignored, as in the backend).
    expect(isValidKrea2RebalanceWeights(' 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 ,')).toBe(true);
    expect(isValidKrea2RebalanceWeights('0,-1,1.5,-2.25,3,4,5,6,7,8,9,10')).toBe(true);
    // Scientific notation and leading-dot decimals are valid Python floats and must be accepted.
    expect(isValidKrea2RebalanceWeights('1e2,1.5e-3,.5,2.,+1,-1,1E3,3.14,0,10,11,12')).toBe(true);
  });

  it.each([
    ['too few', '1,2,3'],
    ['too many', '1,2,3,4,5,6,7,8,9,10,11,12,13'],
    ['nonnumeric', '1,2,3,4,5,6,7,8,9,10,11,x'],
    ['nan', '1,2,3,4,5,6,7,8,9,10,11,nan'],
    ['inf', '1,2,3,4,5,6,7,8,9,10,11,inf'],
    ['Infinity', '1,2,3,4,5,6,7,8,9,10,11,Infinity'],
    ['empty', ''],
    // JS Number() accepts these, but Python float() (the backend) rejects them, so we must too.
    ['hex', '0x10,2,3,4,5,6,7,8,9,10,11,12'],
    ['binary', '0b10,2,3,4,5,6,7,8,9,10,11,12'],
    ['octal', '0o10,2,3,4,5,6,7,8,9,10,11,12'],
  ])('rejects %s', (_label, value) => {
    expect(isValidKrea2RebalanceWeights(value)).toBe(false);
  });
});
