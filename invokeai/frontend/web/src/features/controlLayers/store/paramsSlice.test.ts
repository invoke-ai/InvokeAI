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
    delete v2State.minimaxH3DurationSeconds;
    delete v2State.minimaxH3OutputMode;
    delete v2State.minimaxH3TransformerModel;

    const result = migrate?.(v2State) as ReturnType<typeof getInitialParamsState>;

    // v2 migrates all the way through the current chain (v2 -> v3 adds Qwen fields,
    // v3 -> v4 adds Krea-2 and PiD fields, v5 -> v6 adds MiniMax H3 fields, v6 -> v7 adds the
    // MiniMax H3 single-file transformer override).
    expect(result._version).toBe(7);
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

  it('merges the separate Klein / dev VAE slots into flux2VaeModel when migrating from v3', () => {
    expect(migrate).toBeDefined();

    const initial = getInitialParamsState();
    const kleinVae = { key: 'klein-vae', hash: 'h', name: 'Klein VAE', base: 'flux2', type: 'vae' };
    // Pre-PR v3 state: separate Klein / dev VAE slots, no shared flux2VaeModel and none of the
    // other v4-only keys. Deleting both is what makes this a field-accurate v3 blob — without it
    // the fixture carries flux2DevMistralEncoderModel from getInitialParamsState() and masks the
    // migration's missing seed (which makes zParamsState.parse() throw on real upgrades).
    const v3State: Record<string, unknown> = {
      ...initial,
      _version: 3,
      positivePrompt: 'a fluffy cat',
      seed: 42,
      kleinVaeModel: kleinVae,
      flux2DevVaeModel: null,
    };
    delete v3State.flux2VaeModel;
    delete v3State.flux2DevMistralEncoderModel;

    const result = migrate?.(v3State) as ReturnType<typeof getInitialParamsState> & Record<string, unknown>;

    expect(result._version).toBe(7);
    expect((result.flux2VaeModel as { key: string } | null)?.key).toBe('klein-vae');
    // The new standalone dev Mistral encoder slot must be seeded, not left undefined.
    expect(result.flux2DevMistralEncoderModel).toBeNull();
    // Unrelated params must survive the migration (they'd be wiped if parse() threw).
    expect(result.positivePrompt).toBe('a fluffy cat');
    expect(result.seed).toBe(42);
    // The old slots must be gone.
    expect(result.kleinVaeModel).toBeUndefined();
    expect(result.flux2DevVaeModel).toBeUndefined();
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
    delete v3State.minimaxH3DurationSeconds;
    delete v3State.minimaxH3OutputMode;
    delete v3State.minimaxH3TransformerModel;

    const result = migrate?.(v3State) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(7);
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

  it('seeds the Wan fields for a released-build v3 blob that predates the Wan merge', () => {
    expect(migrate).toBeDefined();

    const initial = getInitialParamsState();
    // Released v6.13.x builds wrote v3 blobs before the Wan fields existed. They're nullable
    // with no default, so if the v3 -> v4 step didn't seed them, parse() would throw and the
    // whole slice would be wiped on upgrade.
    const v3State: Record<string, unknown> = {
      ...initial,
      _version: 3,
      positivePrompt: 'a fluffy cat',
    };
    delete v3State.wanTransformerLowNoise;
    delete v3State.wanComponentSource;
    delete v3State.wanVaeModel;
    delete v3State.wanT5EncoderModel;
    delete v3State.wanGuidanceScaleLowNoise;
    delete v3State.flux2VaeModel;
    delete v3State.flux2DevMistralEncoderModel;

    const result = migrate?.(v3State) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(7);
    expect(result.wanTransformerLowNoise).toBeNull();
    expect(result.wanComponentSource).toBeNull();
    expect(result.wanVaeModel).toBeNull();
    expect(result.wanT5EncoderModel).toBeNull();
    expect(result.wanGuidanceScaleLowNoise).toBeNull();
    expect(result.positivePrompt).toBe('a fluffy cat');
  });

  it('migrates a v4 blob written by main (PiD fields, no flux2 fields) without wiping it', () => {
    expect(migrate).toBeDefined();

    const initial = getInitialParamsState();
    const kleinVae = { key: 'klein-vae', hash: 'h', name: 'Klein VAE', base: 'flux2', type: 'vae' };
    // main and the FLUX.2 [dev] branch both shipped _version 4 with different keys. A blob from
    // main has the PiD fields and the old kleinVaeModel slot, but no flux2VaeModel /
    // flux2DevMistralEncoderModel.
    const mainV4State: Record<string, unknown> = {
      ...initial,
      _version: 4,
      positivePrompt: 'a fluffy cat',
      pidMode: 'fit',
      kleinVaeModel: kleinVae,
    };
    delete mainV4State.flux2VaeModel;
    delete mainV4State.flux2DevMistralEncoderModel;

    const result = migrate?.(mainV4State) as ReturnType<typeof getInitialParamsState> & Record<string, unknown>;

    expect(result._version).toBe(7);
    expect((result.flux2VaeModel as { key: string } | null)?.key).toBe('klein-vae');
    expect(result.flux2DevMistralEncoderModel).toBeNull();
    // main's own v4 values must survive untouched.
    expect(result.pidMode).toBe('fit');
    expect(result.positivePrompt).toBe('a fluffy cat');
    expect(result.kleinVaeModel).toBeUndefined();
  });

  it('migrates a v4 blob written by a pre-merge [dev] build (flux2 fields, no PiD fields) without wiping it', () => {
    expect(migrate).toBeDefined();

    const initial = getInitialParamsState();
    const flux2Vae = { key: 'flux2-vae', hash: 'h', name: 'FLUX.2 VAE', base: 'flux2', type: 'vae' };
    const devV4State: Record<string, unknown> = {
      ...initial,
      _version: 4,
      positivePrompt: 'a fluffy cat',
      flux2VaeModel: flux2Vae,
      flux2DevMistralEncoderModel: null,
    };
    delete devV4State.pidMode;
    delete devV4State.pidDecoderModel;
    delete devV4State.gemma2EncoderModel;
    delete devV4State.pidSteps;

    const result = migrate?.(devV4State) as ReturnType<typeof getInitialParamsState> & Record<string, unknown>;

    expect(result._version).toBe(7);
    // The branch's own v4 values must survive untouched.
    expect((result.flux2VaeModel as { key: string } | null)?.key).toBe('flux2-vae');
    expect(result.pidMode).toBe('off');
    expect(result.pidDecoderModel).toBeNull();
    expect(result.gemma2EncoderModel).toBeNull();
    expect(result.pidSteps).toBe(4);
    expect(result.positivePrompt).toBe('a fluffy cat');
  });

  it('backfills the MiniMax H3 fields when migrating from v5 and preserves existing params', () => {
    // main shipped its own v4 -> v5 (the FLUX.2 [dev] VAE/encoder merge) before this branch
    // landed, so the H3 step runs as v5 -> v6. A v5 blob written by a released build has no H3
    // keys, and they are required with no default -- if this step did not run, zParamsState.parse()
    // would throw and the whole slice would be wiped on upgrade.
    expect(migrate).toBeDefined();

    const initial = getInitialParamsState();
    const v5State: Record<string, unknown> = {
      ...initial,
      _version: 5,
      positivePrompt: 'preserve this prompt',
      seed: 4242,
      dimensions: { ...initial.dimensions, width: 1344, height: 768 },
    };
    delete v5State.minimaxH3DurationSeconds;
    delete v5State.minimaxH3OutputMode;

    const result = migrate?.(v5State) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(7);
    expect(result.minimaxH3DurationSeconds).toBe(5);
    expect(result.minimaxH3OutputMode).toBe('video');
    expect(result.positivePrompt).toBe('preserve this prompt');
    expect(result.seed).toBe(4242);
    expect(result.dimensions).toMatchObject({ width: 1344, height: 768 });
  });

  it('carries a v4 blob through both the flux2 and MiniMax H3 steps in one pass', () => {
    // The regression this guards: with both steps written as v4 -> v5, the flux2 block sets
    // _version = 5 and the H3 block never runs, so the parse below throws and the slice is wiped.
    expect(migrate).toBeDefined();

    const initial = getInitialParamsState();
    const v4State: Record<string, unknown> = {
      ...initial,
      _version: 4,
      positivePrompt: 'preserve this prompt',
    };
    delete v4State.minimaxH3DurationSeconds;
    delete v4State.minimaxH3OutputMode;
    delete v4State.flux2VaeModel;
    delete v4State.flux2DevMistralEncoderModel;
    delete v4State.minimaxH3TransformerModel;

    const result = migrate?.(v4State) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(7);
    expect(result.minimaxH3DurationSeconds).toBe(5);
    expect(result.minimaxH3OutputMode).toBe('video');
    expect(result.flux2DevMistralEncoderModel).toBeNull();
    expect(result.minimaxH3TransformerModel).toBeNull();
    expect(result.positivePrompt).toBe('preserve this prompt');
  });

  it('backfills the MiniMax H3 transformer override when migrating from v5 and preserves existing params', () => {
    expect(migrate).toBeDefined();

    const initial = getInitialParamsState();
    const v5State: Record<string, unknown> = {
      ...initial,
      _version: 5,
      positivePrompt: 'preserve this prompt',
      seed: 777,
      minimaxH3DurationSeconds: 10,
    };
    delete v5State.minimaxH3TransformerModel;

    const result = migrate?.(v5State) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(7);
    expect(result.minimaxH3TransformerModel).toBeNull();
    expect(result.minimaxH3DurationSeconds).toBe(10);
    expect(result.positivePrompt).toBe('preserve this prompt');
    expect(result.seed).toBe(777);
  });

  it('backfills the ERNIE-Image fields from their zod defaults without a version bump', () => {
    // The ERNIE-Image fields are additive with `.default()`, so there is no migration branch for
    // them. A persisted state written before they existed must still parse -- if it throws, the
    // caller's catch falls back to the initial state and the user loses every generation param.
    expect(migrate).toBeDefined();

    const initial = getInitialParamsState();
    const persisted: Record<string, unknown> = {
      ...initial,
      positivePrompt: 'preserve this prompt',
      seed: 99,
    };
    delete persisted.ernieImageScheduler;
    delete persisted.ernieImageUsePromptEnhancer;

    const result = migrate?.(persisted) as ReturnType<typeof getInitialParamsState>;

    expect(result.ernieImageScheduler).toBe('euler');
    expect(result.ernieImageUsePromptEnhancer).toBe(true);
    expect(result.positivePrompt).toBe('preserve this prompt');
    expect(result.seed).toBe(99);
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
