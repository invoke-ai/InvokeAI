import type {
  ExternalApiModelConfig,
  ExternalApiModelDefaultSettings,
  ExternalImageSize,
  ExternalModelCapabilities,
  ExternalModelPanelSchema,
} from 'services/api/types';
import { describe, expect, it } from 'vitest';

import {
  applyParamsVersionMigrations,
  backfillMissingParamsKeys,
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
import { getInitialParamsState, zParamsState } from './types';

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

/**
 * Top-level `zParamsState` key sets as actually shipped, taken from
 * `git show <tag>:...features/controlLayers/store/types.ts`.
 *
 * These are historical facts and must not be regenerated from `getInitialParamsState()` — a fixture
 * spread from the current initial state carries every current key and therefore cannot detect a
 * newly added key that the migration chain forgets to seed. That is precisely what masked the Wan
 * regression these tests exist to catch.
 *
 * One entry per persisted `_version` still in the wild, each picked as the *narrowest* key set among
 * the releases writing that version, so it is a subset of every real blob of that version:
 *   - v1: v6.6.0 only. Identical to the v6.7.0 set below minus `positivePromptHistory`, which the
 *     v1 -> v2 step seeds.
 *   - v2: v6.7.0 - v6.12.0. v6.7.0 is the narrowest and a strict subset of the rest; note v6.7.0 -
 *     v6.9.0 (46 keys) are considerably narrower than v6.10.0 (52) and v6.11.0 - v6.12.0 (60), so
 *     testing only the newest v2 release would miss six keys.
 *   - v3: v6.13.0 - v6.13.7. v6.13.7 is the narrowest (v6.13.0 additionally had `animaT5EncoderModel`,
 *     since removed from the schema; unknown keys are stripped by the non-strict object parse).
 * v4 blobs are written by v6.14.0-rc1 onward; the v4 -> v5 step is covered by its own tests below.
 *
 * Not covered here: v6.2.0a1 - v6.5.1 persist a blob with no `_version` at all (the v0 path). Those
 * also predate the current `dimensions` shape, so a faithful fixture cannot be built by filtering
 * `getInitialParamsState()` the way `buildReleaseBlob` does.
 */
const RELEASE_PARAMS_KEYS = {
  'v6.6.0': {
    version: 1,
    keys: [
      '_version',
      'canvasCoherenceEdgeSize',
      'canvasCoherenceMinDenoise',
      'canvasCoherenceMode',
      'cfgRescaleMultiplier',
      'cfgScale',
      'clipEmbedModel',
      'clipGEmbedModel',
      'clipLEmbedModel',
      'clipSkip',
      'controlLora',
      'dimensions',
      'fluxVAE',
      'guidance',
      'img2imgStrength',
      'infillColorValue',
      'infillMethod',
      'infillPatchmatchDownscaleSize',
      'infillTileSize',
      'iterations',
      'maskBlur',
      'maskBlurMethod',
      'model',
      'negativePrompt',
      'optimizedDenoisingEnabled',
      'positivePrompt',
      'refinerCFGScale',
      'refinerModel',
      'refinerNegativeAestheticScore',
      'refinerPositiveAestheticScore',
      'refinerScheduler',
      'refinerStart',
      'refinerSteps',
      'scheduler',
      'seamlessXAxis',
      'seamlessYAxis',
      'seed',
      'shouldRandomizeSeed',
      'shouldUseCpuNoise',
      'steps',
      't5EncoderModel',
      'upscaleCfgScale',
      'upscaleScheduler',
      'vae',
      'vaePrecision',
    ],
  },
  'v6.7.0': {
    version: 2,
    keys: [
      '_version',
      'canvasCoherenceEdgeSize',
      'canvasCoherenceMinDenoise',
      'canvasCoherenceMode',
      'cfgRescaleMultiplier',
      'cfgScale',
      'clipEmbedModel',
      'clipGEmbedModel',
      'clipLEmbedModel',
      'clipSkip',
      'controlLora',
      'dimensions',
      'fluxVAE',
      'guidance',
      'img2imgStrength',
      'infillColorValue',
      'infillMethod',
      'infillPatchmatchDownscaleSize',
      'infillTileSize',
      'iterations',
      'maskBlur',
      'maskBlurMethod',
      'model',
      'negativePrompt',
      'optimizedDenoisingEnabled',
      'positivePrompt',
      'positivePromptHistory',
      'refinerCFGScale',
      'refinerModel',
      'refinerNegativeAestheticScore',
      'refinerPositiveAestheticScore',
      'refinerScheduler',
      'refinerStart',
      'refinerSteps',
      'scheduler',
      'seamlessXAxis',
      'seamlessYAxis',
      'seed',
      'shouldRandomizeSeed',
      'shouldUseCpuNoise',
      'steps',
      't5EncoderModel',
      'upscaleCfgScale',
      'upscaleScheduler',
      'vae',
      'vaePrecision',
    ],
  },
  'v6.10.0': {
    version: 2,
    keys: [
      '_version',
      'canvasCoherenceEdgeSize',
      'canvasCoherenceMinDenoise',
      'canvasCoherenceMode',
      'cfgRescaleMultiplier',
      'cfgScale',
      'clipEmbedModel',
      'clipGEmbedModel',
      'clipLEmbedModel',
      'clipSkip',
      'colorCompensation',
      'controlLora',
      'dimensions',
      'fluxScheduler',
      'fluxVAE',
      'guidance',
      'img2imgStrength',
      'infillColorValue',
      'infillMethod',
      'infillPatchmatchDownscaleSize',
      'infillTileSize',
      'iterations',
      'maskBlur',
      'maskBlurMethod',
      'model',
      'negativePrompt',
      'optimizedDenoisingEnabled',
      'positivePrompt',
      'positivePromptHistory',
      'refinerCFGScale',
      'refinerModel',
      'refinerNegativeAestheticScore',
      'refinerPositiveAestheticScore',
      'refinerScheduler',
      'refinerStart',
      'refinerSteps',
      'scheduler',
      'seamlessXAxis',
      'seamlessYAxis',
      'seed',
      'shouldRandomizeSeed',
      'shouldUseCpuNoise',
      'steps',
      't5EncoderModel',
      'upscaleCfgScale',
      'upscaleScheduler',
      'vae',
      'vaePrecision',
      'zImageQwen3EncoderModel',
      'zImageQwen3SourceModel',
      'zImageScheduler',
      'zImageVaeModel',
    ],
  },
  'v6.13.7': {
    version: 3,
    keys: [
      '_version',
      'animaQwen3EncoderModel',
      'animaScheduler',
      'animaVaeModel',
      'canvasCoherenceEdgeSize',
      'canvasCoherenceMinDenoise',
      'canvasCoherenceMode',
      'cfgRescaleMultiplier',
      'cfgScale',
      'clipEmbedModel',
      'clipGEmbedModel',
      'clipLEmbedModel',
      'clipSkip',
      'colorCompensation',
      'controlLora',
      'dimensions',
      'fluxDypeExponent',
      'fluxDypePreset',
      'fluxDypeScale',
      'fluxScheduler',
      'fluxVAE',
      'geminiTemperature',
      'geminiThinkingLevel',
      'guidance',
      'imageSize',
      'img2imgStrength',
      'infillColorValue',
      'infillMethod',
      'infillPatchmatchDownscaleSize',
      'infillTileSize',
      'iterations',
      'kleinQwen3EncoderModel',
      'kleinVaeModel',
      'maskBlur',
      'maskBlurMethod',
      'model',
      'negativePrompt',
      'openaiBackground',
      'openaiInputFidelity',
      'openaiQuality',
      'optimizedDenoisingEnabled',
      'positivePrompt',
      'positivePromptHistory',
      'qwenImageComponentSource',
      'qwenImageQuantization',
      'qwenImageQwenVLEncoderModel',
      'qwenImageShift',
      'qwenImageVaeModel',
      'refinerCFGScale',
      'refinerModel',
      'refinerNegativeAestheticScore',
      'refinerPositiveAestheticScore',
      'refinerScheduler',
      'refinerStart',
      'refinerSteps',
      'scheduler',
      'seamlessXAxis',
      'seamlessYAxis',
      'seed',
      'seedreamOptimizePrompt',
      'seedreamWatermark',
      'shouldRandomizeSeed',
      'shouldUseCpuNoise',
      'steps',
      't5EncoderModel',
      'upscaleCfgScale',
      'upscaleScheduler',
      'vae',
      'vaePrecision',
      'zImageQwen3EncoderModel',
      'zImageQwen3SourceModel',
      'zImageScheduler',
      'zImageSeedVarianceEnabled',
      'zImageSeedVarianceRandomizePercent',
      'zImageSeedVarianceStrength',
      'zImageShift',
      'zImageVaeModel',
    ],
  },
} as const satisfies Record<string, { version: number; keys: readonly string[] }>;

/**
 * Build a blob shaped exactly like the one the given release persisted: current initial values, but
 * restricted to the keys that release's schema actually had.
 */
const buildReleaseBlob = (release: keyof typeof RELEASE_PARAMS_KEYS, overrides: Record<string, unknown> = {}) => {
  const { version, keys } = RELEASE_PARAMS_KEYS[release];
  const initial = getInitialParamsState() as unknown as Record<string, unknown>;
  const blob: Record<string, unknown> = {};
  for (const key of keys) {
    if (key in initial) {
      blob[key] = initial[key];
    }
  }
  blob._version = version;
  return { ...blob, ...overrides };
};

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
    expect(result._version).toBe(5);
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

    expect(result._version).toBe(5);
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

    const result = migrate?.(v3State) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(5);
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

    expect(result._version).toBe(5);
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

    expect(result._version).toBe(5);
    // The branch's own v4 values must survive untouched.
    expect((result.flux2VaeModel as { key: string } | null)?.key).toBe('flux2-vae');
    expect(result.pidMode).toBe('off');
    expect(result.pidDecoderModel).toBeNull();
    expect(result.gemma2EncoderModel).toBeNull();
    expect(result.pidSteps).toBe(4);
    expect(result.positivePrompt).toBe('a fluffy cat');
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

  it.each(['v6.6.0', 'v6.7.0', 'v6.10.0', 'v6.13.7'] as const)(
    'migrates a genuine %s blob without losing the user params',
    (release) => {
      expect(migrate).toBeDefined();

      // A real released-build blob only has the keys that release's schema declared. Any key added
      // since that has neither a zod default nor a migration seed fails the parse() at the end of
      // migrate(), and the caller in store.ts swallows the throw and falls back to the initial
      // state — silently wiping the user's prompts, model selection and dimensions on upgrade.
      const blob = buildReleaseBlob(release, {
        positivePrompt: 'a fluffy cat',
        seed: 42,
        shouldRandomizeSeed: false,
      });

      const result = migrate?.(blob) as ReturnType<typeof getInitialParamsState>;

      expect(result._version).toBe(5);
      expect(result.positivePrompt).toBe('a fluffy cat');
      expect(result.seed).toBe(42);
      expect(result.shouldRandomizeSeed).toBe(false);
    }
  );

  it('seeds every key of the current schema in the version steps themselves, for each released blob version', () => {
    // The general form of the defect this suite guards against: a key is added to zParamsState with
    // no `.default()`/`.optional()`/`.catch()` (so zod treats it as required) and no seed in the
    // migration chain. Every such key is a whole-slice wipe for anyone upgrading from a release
    // that predates it. Rather than enumerate keys by hand, assert the invariant over the whole
    // schema, so the next occurrence fails here instead of shipping.
    //
    // This deliberately runs the version steps *without* going through migrate(), because
    // backfillMissingParamsKeys() would otherwise repair the omission and hide it. The safety net
    // is there to protect users from a forgotten seed; this test is what stops one being merged.
    for (const release of Object.keys(RELEASE_PARAMS_KEYS) as (keyof typeof RELEASE_PARAMS_KEYS)[]) {
      const blob = buildReleaseBlob(release);

      applyParamsVersionMigrations(blob);
      const unseeded = backfillMissingParamsKeys(blob);

      expect(
        unseeded,
        `Keys missing from a genuine ${release} blob that neither carry a zod default nor get seeded by ` +
          `the migration chain. Upgrading from ${release} would throw in zParamsState.parse() and wipe the ` +
          `user's whole params slice. Give each key a zod default, or seed it in the _version ` +
          `${RELEASE_PARAMS_KEYS[release].version} migration step.`
      ).toEqual([]);
    }
  });

  it('backfills a key the version steps forget, instead of wiping the slice', () => {
    expect(migrate).toBeDefined();

    // Simulate the next occurrence of the defect: a required key that no migration step seeds. The
    // safety net must fill it and let everything else through, rather than throwing and handing the
    // caller in store.ts an excuse to reset the slice.
    const blob = buildReleaseBlob('v6.13.7', { positivePrompt: 'a fluffy cat', seed: 42 });
    applyParamsVersionMigrations(blob);
    delete blob.pidSteps;

    const backfilled = backfillMissingParamsKeys(blob);

    expect(backfilled).toEqual(['pidSteps']);
    expect(blob.pidSteps).toBe(4);
    expect(() => zParamsState.parse(blob)).not.toThrow();
    expect(blob.positivePrompt).toBe('a fluffy cat');
  });

  it('does not backfill over a key the persisted state already holds', () => {
    // The net must only fill omissions — never overwrite a real persisted value, and never mask a
    // present-but-invalid one (that still throws, same as before).
    const blob = buildReleaseBlob('v6.13.7');
    applyParamsVersionMigrations(blob);
    blob.pidSteps = 2;
    blob.positivePrompt = 'a fluffy cat';

    expect(backfillMissingParamsKeys(blob)).toEqual([]);
    expect(blob.pidSteps).toBe(2);
    expect(blob.positivePrompt).toBe('a fluffy cat');
  });

  it('leaves a defaulted key to zod rather than backfilling it from the initial state', () => {
    // The net must not pre-empt a field the schema can fill itself, or the schema's `.default()`
    // stops being authoritative the moment it diverges from getInitialParamsState(). `pidSteps`
    // (required) must be filled; `ernieImageScheduler` (`.default('euler')`) must not be.
    const blob = buildReleaseBlob('v6.13.7');
    applyParamsVersionMigrations(blob);
    delete blob.ernieImageScheduler;
    delete blob.pidSteps;

    expect(backfillMissingParamsKeys(blob)).toEqual(['pidSteps']);
    expect(blob.ernieImageScheduler).toBeUndefined();
    expect(zParamsState.parse(blob).ernieImageScheduler).toBe('euler');
  });

  it('never backfills _version, so version detection cannot be bypassed', () => {
    expect(migrate).toBeDefined();

    // The v0 branch keys off `!('_version' in state)` (presence) while the net keys off `undefined`
    // (value). If the net filled `_version`, a blob carrying an explicit undefined would be stamped
    // as current having run no migration step at all.
    const blob = buildReleaseBlob('v6.7.0', { _version: undefined, positivePrompt: 'a fluffy cat' });

    const result = migrate?.(blob) as ReturnType<typeof getInitialParamsState>;

    // It is treated as a v0 blob and walked through the whole chain, not stamped v5 in place.
    expect(result._version).toBe(5);
    expect(result.positivePromptHistory).toEqual([]);
    expect(result.qwenImageVaeModel).toBeNull();
    expect(result.wanVaeModel).toBeNull();
    expect(result.positivePrompt).toBe('a fluffy cat');
  });

  it('does not throw on a v0 blob whose dimensions are missing', () => {
    expect(migrate).toBeDefined();

    // A truncated or hand-edited pre-_version blob. The v0 step used to dereference
    // state.dimensions.rect unguarded, and the TypeError escaped migrate() — the one path that
    // could still cost the user the whole slice despite the safety net.
    const blob: Record<string, unknown> = { positivePrompt: 'a fluffy cat', seed: 7 };

    const result = migrate?.(blob) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(5);
    expect(result.positivePrompt).toBe('a fluffy cat');
    expect(result.seed).toBe(7);
    expect(result.dimensions).toBeDefined();
  });

  it('seeds the Wan fields for a released-build v3 blob that predates the Wan merge', () => {
    expect(migrate).toBeDefined();

    // Released v6.13.x builds wrote v3 blobs before the Wan fields existed. They're nullable with
    // no default, so if the v3 -> v4 step didn't seed them, parse() would throw and the whole
    // slice would be wiped on upgrade.
    const v3State = buildReleaseBlob('v6.13.7', { positivePrompt: 'a fluffy cat', seed: 42 });
    expect('wanVaeModel' in v3State).toBe(false);

    const result = migrate?.(v3State) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(5);
    expect(result.wanTransformerLowNoise).toBeNull();
    expect(result.wanComponentSource).toBeNull();
    expect(result.wanVaeModel).toBeNull();
    expect(result.wanT5EncoderModel).toBeNull();
    expect(result.wanGuidanceScaleLowNoise).toBeNull();
    // Unrelated params must survive the migration (they'd be wiped if parse() threw).
    expect(result.positivePrompt).toBe('a fluffy cat');
    expect(result.seed).toBe(42);
  });

  it('seeds the post-v3 fields for the oldest released v2 blob (v6.7.0 - v6.9.0)', () => {
    expect(migrate).toBeDefined();

    // Same class of defect one version earlier: these keys were added to the schema while releases
    // were still persisting v2 blobs, and the v2 -> v3 step seeded only the two Qwen Image fields.
    // v6.7.0 - v6.9.0 are the narrowest v2 blobs, missing the first six below in addition to
    // everything v6.10.0 is missing.
    const v2State = buildReleaseBlob('v6.7.0', { positivePrompt: 'a fluffy cat' });

    const result = migrate?.(v2State) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(5);
    expect(result.fluxScheduler).toBe('euler');
    expect(result.zImageScheduler).toBe('euler');
    expect(result.colorCompensation).toBe(false);
    expect(result.zImageVaeModel).toBeNull();
    expect(result.zImageQwen3EncoderModel).toBeNull();
    expect(result.zImageQwen3SourceModel).toBeNull();
    expect(result.fluxDypePreset).toBe('off');
    expect(result.fluxDypeScale).toBe(2.0);
    expect(result.fluxDypeExponent).toBe(2.0);
    expect(result.zImageShift).toBeNull();
    expect(result.zImageSeedVarianceEnabled).toBe(false);
    expect(result.zImageSeedVarianceStrength).toBe(0.1);
    expect(result.zImageSeedVarianceRandomizePercent).toBe(50);
    expect(result.animaVaeModel).toBeNull();
    expect(result.animaQwen3EncoderModel).toBeNull();
    expect(result.animaScheduler).toBe('euler');
    expect(result.kleinQwen3EncoderModel).toBeNull();
    expect(result.qwenImageComponentSource).toBeNull();
    expect(result.qwenImageQuantization).toBe('none');
    expect(result.qwenImageShift).toBeNull();
    expect(result.positivePrompt).toBe('a fluffy cat');
  });

  it('preserves post-v3 values already present in a dev-build v2 blob', () => {
    expect(migrate).toBeDefined();

    // v2 blobs written by dev builds after each field landed already carry the keys, possibly with
    // real values — the conditional seeds must not clobber them.
    const v2State = buildReleaseBlob('v6.7.0', {
      fluxScheduler: 'heun',
      colorCompensation: true,
      fluxDypePreset: 'auto',
      qwenImageQuantization: 'int8',
      qwenImageShift: 3.0,
      zImageSeedVarianceEnabled: true,
    });

    const result = migrate?.(v2State) as ReturnType<typeof getInitialParamsState>;

    expect(result.fluxScheduler).toBe('heun');
    expect(result.colorCompensation).toBe(true);
    expect(result.fluxDypePreset).toBe('auto');
    expect(result.qwenImageQuantization).toBe('int8');
    expect(result.qwenImageShift).toBe(3.0);
    expect(result.zImageSeedVarianceEnabled).toBe(true);
  });

  it('preserves Wan values already present in a dev-build v3 blob', () => {
    expect(migrate).toBeDefined();

    const initial = getInitialParamsState();
    const wanVae = { key: 'wan-vae', hash: 'h', name: 'Wan VAE', base: 'wan', type: 'vae' };
    // v3 blobs written by dev builds after the Wan merge already carry the keys, possibly with
    // real values — the conditional seeds must not clobber them.
    const v3State: Record<string, unknown> = {
      ...initial,
      _version: 3,
      wanVaeModel: wanVae,
      wanGuidanceScaleLowNoise: 3.5,
    };

    const result = migrate?.(v3State) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(5);
    expect((result.wanVaeModel as { key: string } | null)?.key).toBe('wan-vae');
    expect(result.wanGuidanceScaleLowNoise).toBe(3.5);
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
