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
  isValidKrea2RebalanceWeights,
  KREA2_REBALANCE_WEIGHT_COUNT,
  modelChanged,
  paramsSliceConfig,
  parseKrea2RebalanceWeights,
  positivePromptAddedToHistory,
  promptRemovedFromHistory,
  repairParamsState,
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

  it('supports reference images on Krea-2, which uses training-free style reference', () => {
    // Krea-2 has no ref-image adapter model, so the panel is gated purely on the base being listed in
    // SUPPORTS_REF_IMAGES_BASE_MODELS.
    const model = { key: 'krea2', hash: 'hash', name: 'Krea-2 Turbo', base: 'krea-2', type: 'main' } as never;

    expect(selectModelSupportsRefImages.resultFunc(model, null)).toBe(true);
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
 * What matters per version is the *intersection* of every build that wrote it, not the newest or even
 * the smallest single one: if any build omitted a key, the chain has to cope with that key's absence.
 * The ranges below come from reading `_version` and the key set out of all 66 `v6.*` tags. Do not
 * survey them with a glob like `v6.1*` — that matches v6.10 - v6.14 and silently drops v6.7 - v6.9.
 *   - v0: v6.0.0a1 - v6.6.0rc2, which persist a blob with no `_version` at all. Two entries, because
 *     no single v0 build has the intersecting key set (43 keys):
 *       - `v6.0.0a1` (46 keys) is the intersection once the three keys since removed from the schema
 *         are dropped — `positivePrompt2` / `negativePrompt2` / `shouldConcatPrompts`, which
 *         `buildReleaseBlob` filters out and the non-strict parse would strip anyway. Critically it
 *         has no `dimensions` key at all; that arrives in v6.0.0rc4.
 *       - `v6.5.1` (44 keys) is the widest-coverage v0 shape that *does* carry `dimensions`, in the
 *         pre-flattening form, which is the only way to exercise the v0 -> v1 lift.
 *   - v1: v6.6.0 and v6.7.0rc1, whose key sets are identical (45).
 *   - v2: v6.7.0 - v6.13.0.rc1. v6.7.0 is the intersection (46) and a strict subset of the rest; note
 *     v6.7.0 - v6.9.0 (46 keys) are considerably narrower than v6.10.0 (52), v6.11.0 - v6.12.0 (60)
 *     and v6.13.0.rc1 (73), so testing only the newest v2 build would miss six keys.
 *   - v3: v6.13.0 - v6.13.7. v6.13.7 is the intersection (77); v6.13.0 additionally had
 *     `animaT5EncoderModel`, since removed from the schema.
 *   - v4: the narrowest v4 blob is not a release at all — it is the one written by the build that did
 *     the bump, `1aeb05bbf0` (97 keys). Releases writing v4 start at v6.14.0-rc1.
 *   - v5: the current version, reached by the FLUX.2 [dev] merge `f10d2a4f5a`, which is also the
 *     build that wrote the narrowest v5 blob. Pinning the fixture at the bump commit is what keeps
 *     the invariant below meaningful for the current tier: the version steps can never cover it (a
 *     v5 blob matches no branch), so every key added since the bump has to carry a zod default, and
 *     this entry is what proves it does.
 */
const RELEASE_PARAMS_KEYS = {
  'v6.0.0a1': {
    version: 0,
    keys: [
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
      'negativePrompt2',
      'optimizedDenoisingEnabled',
      'positivePrompt',
      'positivePrompt2',
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
      'shouldConcatPrompts',
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
  'v6.5.1': {
    version: 0,
    keys: [
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
  '1aeb05bbf0': {
    version: 4,
    keys: [
      '_version',
      'animaLLLiteModel',
      'animaLLLiteWeight',
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
      'ideogram4ColorPalette',
      'ideogram4GuidanceScale',
      'ideogram4Mu',
      'ideogram4SamplerPreset',
      'ideogram4Steps',
      'imageSize',
      'img2imgStrength',
      'infillColorValue',
      'infillMethod',
      'infillPatchmatchDownscaleSize',
      'infillTileSize',
      'iterations',
      'kleinQwen3EncoderModel',
      'kleinVaeModel',
      'krea2Qwen3VlEncoderModel',
      'krea2RebalanceEnabled',
      'krea2RebalanceMultiplier',
      'krea2RebalanceWeights',
      'krea2SeedVarianceEnabled',
      'krea2SeedVarianceRandomizePercent',
      'krea2SeedVarianceStrength',
      'krea2VaeModel',
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
      'wanComponentSource',
      'wanGuidanceScaleLowNoise',
      'wanT5EncoderModel',
      'wanTransformerLowNoise',
      'wanVaeModel',
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
  f10d2a4f5a: {
    version: 5,
    keys: [
      '_version',
      'animaLLLiteModel',
      'animaLLLiteWeight',
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
      'ernieImageScheduler',
      'ernieImageUsePromptEnhancer',
      'flux2DevMistralEncoderModel',
      'flux2VaeModel',
      'fluxDypeExponent',
      'fluxDypePreset',
      'fluxDypeScale',
      'fluxScheduler',
      'fluxVAE',
      'geminiTemperature',
      'geminiThinkingLevel',
      'gemma2EncoderModel',
      'guidance',
      'hiDiffusionEnabled',
      'hiDiffusionRauNetEnabled',
      'hiDiffusionT1Ratio',
      'hiDiffusionT2Ratio',
      'hiDiffusionWindowAttnEnabled',
      'ideogram4ColorPalette',
      'ideogram4GuidanceScale',
      'ideogram4Mu',
      'ideogram4SamplerPreset',
      'ideogram4Steps',
      'imageSize',
      'img2imgStrength',
      'infillColorValue',
      'infillMethod',
      'infillPatchmatchDownscaleSize',
      'infillTileSize',
      'iterations',
      'kleinQwen3EncoderModel',
      'krea2Qwen3VlEncoderModel',
      'krea2RebalanceEnabled',
      'krea2RebalanceMultiplier',
      'krea2RebalanceWeights',
      'krea2SeedVarianceEnabled',
      'krea2SeedVarianceRandomizePercent',
      'krea2SeedVarianceStrength',
      'krea2VaeModel',
      'maskBlur',
      'maskBlurMethod',
      'model',
      'negativePrompt',
      'openaiBackground',
      'openaiInputFidelity',
      'openaiQuality',
      'optimizedDenoisingEnabled',
      'pidDecoderModel',
      'pidMode',
      'pidSteps',
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
      'wanComponentSource',
      'wanGuidanceScaleLowNoise',
      'wanT5EncoderModel',
      'wanTransformerLowNoise',
      'wanVaeModel',
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
 * Values for fixture keys that have since been removed from `zParamsState`, so a blob can still
 * reproduce them.
 *
 * `buildReleaseBlob` sources its values from `getInitialParamsState()`, which by definition only
 * knows today's keys. Without this table a fixture key that leaves the schema is dropped from the
 * blob *silently*, and the fixture quietly stops reproducing the shape the build actually persisted
 * — which in turn stops exercising whatever migration step reads that key. That is not theoretical:
 * `kleinVaeModel` left the schema when the v4 -> v5 step folded it into `flux2VaeModel`, and it is
 * the input to that fold. The guard test below fails on any fixture key that is in neither the
 * current schema nor this table, so the next removal has to be a decision rather than an accident.
 */
const REMOVED_SCHEMA_KEY_VALUES: Record<string, unknown> = {
  // Folded into `flux2VaeModel` and deleted by the v4 -> v5 step.
  kleinVaeModel: null,
  // The pre-merge FLUX.2 [dev] branch's VAE slot, folded into `flux2VaeModel` by the same step.
  flux2DevVaeModel: null,
  // The v0-era second prompt box, dropped when prompt concatenation was removed.
  positivePrompt2: '',
  negativePrompt2: '',
  shouldConcatPrompts: true,
  // Present in v6.13.0 only, removed before v6.13.7.
  animaT5EncoderModel: null,
};

/**
 * Build a blob shaped exactly like the one the given build persisted: current initial values, but
 * restricted to the keys that build's schema actually had.
 */
const buildReleaseBlob = (release: keyof typeof RELEASE_PARAMS_KEYS, overrides: Record<string, unknown> = {}) => {
  const { version, keys } = RELEASE_PARAMS_KEYS[release];
  const initial = getInitialParamsState() as unknown as Record<string, unknown>;
  const blob: Record<string, unknown> = {};
  for (const key of keys) {
    if (key in initial) {
      blob[key] = initial[key];
    } else if (key in REMOVED_SCHEMA_KEY_VALUES) {
      blob[key] = REMOVED_SCHEMA_KEY_VALUES[key];
    }
  }
  if (version === 0) {
    // v0 blobs carry no `_version` at all, and where they have `dimensions` it is the pre-flattening
    // shape — the one field `keys` alone cannot reproduce, since filtering the current initial state
    // would hand back the flat shape. `v6.0.0a1` has no `dimensions` key at all and must not gain
    // one here, or it stops testing the v0 seed.
    if ((keys as readonly string[]).includes('dimensions')) {
      blob.dimensions = buildReleaseDimensions(release, 512, 512);
    }
  } else {
    blob._version = version;
  }
  return { ...blob, ...overrides };
};

const probeModel = (key: string, base: string, type: string) => ({ key, hash: `${key}-hash`, name: key, base, type });

/**
 * Keys added to `zParamsState` after `1aeb05bbf0` bumped `_version` to 4, which reach users only via
 * a zod default. A tier that is current has no migration step by definition, so this is the only
 * route available to anything added after the most recent bump.
 */
const POST_V4_BUMP_DEFAULTED_KEYS = [
  'pidMode',
  'pidDecoderModel',
  'gemma2EncoderModel',
  'pidSteps',
  'hiDiffusionEnabled',
  'hiDiffusionRauNetEnabled',
  'hiDiffusionWindowAttnEnabled',
  'hiDiffusionT1Ratio',
  'hiDiffusionT2Ratio',
] as const satisfies readonly (keyof typeof zParamsState.shape)[];

/**
 * Every seed in the version steps written conditionally (`state.X = state.X ?? V`), paired with a
 * valid value that differs from what the seed would write.
 *
 * The `??` is the whole point of those lines: the field landed in the schema *before* the version
 * bump that seeds it, so a blob written by a dev build in that window already carries a real user
 * value, and an unconditional assignment would silently reset it. That is a one-field data loss per
 * key, invisible unless something probes the key specifically — so every conditional seed gets a
 * row here. `tier` is the fixture whose `_version` the seeding step consumes.
 */
const CONDITIONAL_SEED_PROBES: {
  tier: keyof typeof RELEASE_PARAMS_KEYS;
  key: keyof typeof zParamsState.shape;
  value: unknown;
}[] = [
  // v2 -> v3
  { tier: 'v6.7.0', key: 'fluxScheduler', value: 'lcm' },
  { tier: 'v6.7.0', key: 'zImageScheduler', value: 'lcm' },
  { tier: 'v6.7.0', key: 'colorCompensation', value: true },
  { tier: 'v6.7.0', key: 'zImageVaeModel', value: probeModel('z-vae', 'z-image', 'vae') },
  { tier: 'v6.7.0', key: 'zImageQwen3EncoderModel', value: probeModel('z-enc', 'z-image', 'main') },
  { tier: 'v6.7.0', key: 'zImageQwen3SourceModel', value: probeModel('z-src', 'z-image', 'main') },
  { tier: 'v6.7.0', key: 'fluxDypePreset', value: 'auto' },
  { tier: 'v6.7.0', key: 'fluxDypeScale', value: 3.5 },
  { tier: 'v6.7.0', key: 'fluxDypeExponent', value: 3.5 },
  { tier: 'v6.7.0', key: 'zImageShift', value: 3.0 },
  { tier: 'v6.7.0', key: 'zImageSeedVarianceEnabled', value: true },
  { tier: 'v6.7.0', key: 'zImageSeedVarianceStrength', value: 0.3 },
  { tier: 'v6.7.0', key: 'zImageSeedVarianceRandomizePercent', value: 75 },
  { tier: 'v6.7.0', key: 'animaVaeModel', value: probeModel('anima-vae', 'anima', 'vae') },
  { tier: 'v6.7.0', key: 'animaQwen3EncoderModel', value: probeModel('anima-enc', 'anima', 'main') },
  { tier: 'v6.7.0', key: 'animaScheduler', value: 'dpmpp_2m' },
  { tier: 'v6.7.0', key: 'kleinQwen3EncoderModel', value: probeModel('klein-enc', 'flux2', 'main') },
  { tier: 'v6.7.0', key: 'qwenImageComponentSource', value: probeModel('qwen-src', 'qwen-image', 'main') },
  { tier: 'v6.7.0', key: 'qwenImageQuantization', value: 'int8' },
  { tier: 'v6.7.0', key: 'qwenImageShift', value: 3.0 },
  // v3 -> v4
  { tier: 'v6.13.7', key: 'wanTransformerLowNoise', value: probeModel('wan-low', 'wan', 'main') },
  { tier: 'v6.13.7', key: 'wanComponentSource', value: probeModel('wan-src', 'wan', 'main') },
  { tier: 'v6.13.7', key: 'wanVaeModel', value: probeModel('wan-vae', 'wan', 'vae') },
  { tier: 'v6.13.7', key: 'wanT5EncoderModel', value: probeModel('wan-t5', 'wan', 'main') },
  { tier: 'v6.13.7', key: 'wanGuidanceScaleLowNoise', value: 3.5 },
  // v4 -> v5
  { tier: '1aeb05bbf0', key: 'flux2VaeModel', value: probeModel('flux2-vae', 'flux2', 'vae') },
  { tier: '1aeb05bbf0', key: 'flux2DevMistralEncoderModel', value: probeModel('mistral', 'flux2', 'main') },
  { tier: '1aeb05bbf0', key: 'pidMode', value: 'native' },
  { tier: '1aeb05bbf0', key: 'pidDecoderModel', value: probeModel('pid-dec', 'flux', 'main') },
  { tier: '1aeb05bbf0', key: 'gemma2EncoderModel', value: probeModel('gemma2', 'flux', 'main') },
  { tier: '1aeb05bbf0', key: 'pidSteps', value: 2 },
];

/**
 * `dimensions` in the shape the given build persisted it: `{ rect: { x, y, width, height },
 * aspectRatio }` before the v0 -> v1 flattening, `{ width, height, aspectRatio }` after.
 */
const buildReleaseDimensions = (release: keyof typeof RELEASE_PARAMS_KEYS, width: number, height: number) => {
  const { aspectRatio } = getInitialParamsState().dimensions;
  return RELEASE_PARAMS_KEYS[release].version === 0
    ? { rect: { x: 0, y: 0, width, height }, aspectRatio }
    : { width, height, aspectRatio };
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

  it.each(['v6.0.0a1', 'v6.5.1', 'v6.6.0', 'v6.7.0', 'v6.10.0', 'v6.13.7', '1aeb05bbf0', 'f10d2a4f5a'] as const)(
    'migrates a genuine %s blob without losing the user params',
    (release) => {
      expect(migrate).toBeDefined();

      // A real released-build blob only has the keys that release's schema declared. Any key added
      // since that has neither a zod default nor a migration seed fails the parse() at the end of
      // migrate(), and the caller in store.ts swallows the throw and falls back to the initial
      // state — silently wiping the user's prompts, model selection and dimensions on upgrade.
      const hasDimensions = (RELEASE_PARAMS_KEYS[release].keys as readonly string[]).includes('dimensions');
      const blob = buildReleaseBlob(release, {
        positivePrompt: 'a fluffy cat',
        seed: 42,
        shouldRandomizeSeed: false,
        // Every value here is non-default on purpose. The initial state is what a wiped slice looks
        // like, so a fixture carrying initial values cannot tell "migrated correctly" apart from
        // "silently reset" — `getInitialParamsState().model` is null, which is exactly why nothing
        // used to notice the repair pass clearing a model whose base has left the schema.
        model: { key: 'm1', hash: 'h1', name: 'Some SDXL Model', base: 'sdxl', type: 'main' },
        ...(hasDimensions ? { dimensions: buildReleaseDimensions(release, 768, 1024) } : {}),
      });

      const result = migrate?.(blob) as ReturnType<typeof getInitialParamsState>;

      expect(result._version).toBe(5);
      expect(result.positivePrompt).toBe('a fluffy cat');
      expect(result.seed).toBe(42);
      expect(result.shouldRandomizeSeed).toBe(false);
      expect(result.model).toMatchObject({ key: 'm1', base: 'sdxl' });
      if (hasDimensions) {
        expect(result.dimensions.width).toBe(768);
        expect(result.dimensions.height).toBe(1024);
      } else {
        // v6.0.0a1 - v6.0.0rc3 had no `dimensions` key; the v0 step seeds it.
        expect(result.dimensions).toEqual(getInitialParamsState().dimensions);
      }
    }
  );

  it('seeds every key of the current schema in the version steps themselves, for each persisted blob version', () => {
    // The general form of the defect this suite guards against: a key is added to zParamsState with
    // no `.default()`/`.optional()`/`.catch()` (so zod treats it as required) and no seed in the
    // migration chain. Every such key is a whole-slice wipe for anyone upgrading from a release
    // that predates it. Rather than enumerate keys by hand, assert the invariant over the whole
    // schema, so the next occurrence fails here instead of shipping.
    //
    // This deliberately runs the version steps *without* going through migrate(), because
    // repairParamsState() would otherwise repair the omission and hide it. The safety net is there
    // to protect users from a forgotten seed; this test is what stops one being merged.
    //
    // Scope limit worth knowing: fixture values all come from getInitialParamsState(), so this
    // covers *missing* keys, not keys whose persisted value a tightened schema would now reject.
    // Tightening a nested schema (an item in `positivePromptHistory`, say) still costs that whole
    // top-level key via `reset`, and no fixture here would notice. Realistic per-key persisted
    // values would be needed to close that, which is a bigger change than this suite.
    for (const release of Object.keys(RELEASE_PARAMS_KEYS) as (keyof typeof RELEASE_PARAMS_KEYS)[]) {
      const { version } = RELEASE_PARAMS_KEYS[release];
      const blob = buildReleaseBlob(release);

      applyParamsVersionMigrations(blob);
      const { backfilled, reset } = repairParamsState(blob);

      // The steps must not leave a value the schema rejects either — a half-applied migration that
      // writes a malformed value costs the user that field, silently.
      expect(
        reset,
        `The version steps left these keys holding a value that fails its own field schema, on a ` +
          `blob written at ${release}. Each costs the user that field on upgrade.`
      ).toEqual([]);

      expect(
        backfilled,
        version === getInitialParamsState()._version
          ? `Keys missing from a blob written at ${release}, the commit that bumped _version to ${version}. ` +
              `A blob already at the current version matches no branch in the migration chain, so no step can ` +
              `seed these — each needs a zod default, or upgrading throws in zParamsState.parse() and wipes ` +
              `the user's whole params slice.`
          : `Keys missing from a genuine ${release} blob that neither carry a zod default nor get seeded by ` +
              `the migration chain. Upgrading from ${release} would throw in zParamsState.parse() and wipe the ` +
              `user's whole params slice. Give each key a zod default, or seed it in the _version ` +
              `${version} migration step.`
      ).toEqual([]);
    }
  });

  it('reproduces every fixture key, including those since removed from the schema', () => {
    // `buildReleaseBlob` can only source values it knows about, so a fixture key that is neither in
    // the current schema nor in REMOVED_SCHEMA_KEY_VALUES is dropped from the blob without a word.
    // The fixture then still *looks* faithful — the key is right there in the table — while having
    // quietly stopped exercising whatever step consumes it. Fail instead, so removing a key from
    // zParamsState forces a decision about the fixtures that name it.
    const initial = getInitialParamsState() as unknown as Record<string, unknown>;

    for (const release of Object.keys(RELEASE_PARAMS_KEYS) as (keyof typeof RELEASE_PARAMS_KEYS)[]) {
      const { keys } = RELEASE_PARAMS_KEYS[release];
      const blob = buildReleaseBlob(release);
      const dropped = (keys as readonly string[]).filter(
        (key) => !(key in initial) && !(key in REMOVED_SCHEMA_KEY_VALUES) && !(key in blob)
      );

      expect(
        dropped,
        `The ${release} fixture lists these keys, but they are neither in the current zParamsState nor ` +
          `in REMOVED_SCHEMA_KEY_VALUES, so buildReleaseBlob silently omits them and the fixture no ` +
          `longer reproduces the blob that build persisted. Add each to REMOVED_SCHEMA_KEY_VALUES with ` +
          `the value that build wrote, or drop it from the fixture if it never existed.`
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
    delete blob.krea2VaeModel;

    const { backfilled } = repairParamsState(blob);

    expect(backfilled).toEqual(['krea2VaeModel']);
    expect(blob.krea2VaeModel).toBeNull();
    expect(() => zParamsState.parse(blob)).not.toThrow();
    expect(blob.positivePrompt).toBe('a fluffy cat');
  });

  it('does not repair over a key the persisted state already holds', () => {
    // The net must never overwrite a real persisted value — only fill omissions and replace values
    // the field's own schema rejects.
    const blob = buildReleaseBlob('v6.13.7');
    applyParamsVersionMigrations(blob);
    blob.pidSteps = 2;
    blob.positivePrompt = 'a fluffy cat';

    expect(repairParamsState(blob)).toEqual({ backfilled: [], reset: [] });
    expect(blob.pidSteps).toBe(2);
    expect(blob.positivePrompt).toBe('a fluffy cat');
  });

  it('leaves a defaulted key to zod rather than backfilling it from the initial state', () => {
    // The net must not pre-empt a field the schema can fill itself, or the schema's `.default()`
    // stops being authoritative the moment it diverges from getInitialParamsState().
    // `krea2VaeModel` (required) must be filled; `ernieImageScheduler` (`.default('euler')`) must not.
    const blob = buildReleaseBlob('v6.13.7');
    applyParamsVersionMigrations(blob);
    delete blob.ernieImageScheduler;
    delete blob.krea2VaeModel;

    expect(repairParamsState(blob).backfilled).toEqual(['krea2VaeModel']);
    expect(blob.ernieImageScheduler).toBeUndefined();
    expect(zParamsState.parse(blob).ernieImageScheduler).toBe('euler');
  });

  it('resets a present-but-invalid key instead of wiping the slice', () => {
    // The other half of the net: a value that no longer satisfies its field's schema costs the user
    // that one field, not every generation param they have. Without this, the parse at the end of
    // migrate() throws and store.ts falls back to the initial state wholesale.
    expect(migrate).toBeDefined();

    const blob = buildReleaseBlob('v6.13.7', { positivePrompt: 'a fluffy cat', seed: 42 });
    applyParamsVersionMigrations(blob);
    blob.pidSteps = 99; // out of the schema's 1-4 range
    blob.scheduler = 'not_a_scheduler';

    const { backfilled, reset } = repairParamsState(blob);

    expect(backfilled).toEqual([]);
    // Sorted: the net walks `zParamsState.shape`, so the raw order tracks field declaration order.
    expect([...reset].sort()).toEqual(['pidSteps', 'scheduler']);
    expect(blob.pidSteps).toBe(4);
    expect(blob.scheduler).toBe(getInitialParamsState().scheduler);
    expect(blob.positivePrompt).toBe('a fluffy cat');
    expect(blob.seed).toBe(42);
  });

  it.each([
    ['a dimensions object left without width/height', { rect: { x: 0, y: 0 }, aspectRatio: undefined }],
    ['a null dimensions', null],
  ])('repairs %s on a v0 blob rather than wiping the slice', (_label, dimensions) => {
    // The v0 step's `state.dimensions && state.dimensions.rect` guard only stops a TypeError; what
    // it leaves behind is a `dimensions` that the schema rejects. The net has to replace it, or the
    // guard buys nothing and the user still loses the slice — just via ZodError instead.
    expect(migrate).toBeDefined();

    const blob = buildReleaseBlob('v6.5.1', { dimensions, positivePrompt: 'a fluffy cat', seed: 7 });

    const result = migrate?.(blob) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(5);
    expect(result.dimensions).toEqual(getInitialParamsState().dimensions);
    expect(result.positivePrompt).toBe('a fluffy cat');
    expect(result.seed).toBe(7);
  });

  it('treats a blob whose _version has no value as v0 and walks the whole chain', () => {
    expect(migrate).toBeDefined();

    // The v0 branch keys off the *value*, not `'_version' in state`. With a presence check this blob
    // matches no branch at all, reaches the parse still at `_version: undefined` and takes the whole
    // slice down. (A real persisted blob comes from JSON.parse and so can never hold an explicit
    // `undefined` — this pins the invariant, not a reachable input.)
    const blob = buildReleaseBlob('v6.7.0', { _version: undefined, positivePrompt: 'a fluffy cat' });

    const result = migrate?.(blob) as ReturnType<typeof getInitialParamsState>;

    expect(result._version).toBe(5);
    expect(result.positivePromptHistory).toEqual([]);
    expect(result.qwenImageVaeModel).toBeNull();
    expect(result.wanVaeModel).toBeNull();
    expect(result.positivePrompt).toBe('a fluffy cat');
  });

  it('never repairs _version, so version detection cannot be bypassed', () => {
    // `_version` is the input to the version steps, so the net must leave it alone. If it repaired
    // it, any blob whose version is not the current literal — including one written by a *newer*
    // build — would be silently stamped v5 having run no step, and its stale field values would be
    // accepted as current. Deliberately not routed through migrate(): the version steps normalise
    // `_version` before the net ever sees it, so only calling the net directly tests the guard.
    // The blob is otherwise complete (the current tier's key set), so `_version` is the only thing
    // the parse below can object to.
    const blob = buildReleaseBlob('f10d2a4f5a', { _version: 6, positivePrompt: 'a fluffy cat' });

    const { backfilled, reset } = repairParamsState(blob);

    expect(backfilled).toEqual([]);
    expect(reset).toEqual([]);
    expect(blob._version).toBe(6);
    // Still fatal, which is the correct outcome for a downgrade: that slice really was written by a
    // schema this build does not know.
    expect(() => zParamsState.parse(blob)).toThrow();
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

  it.each(CONDITIONAL_SEED_PROBES)(
    'preserves a persisted $key rather than reseeding it (dev-build $tier blob)',
    ({ tier, key, value }) => {
      expect(migrate).toBeDefined();

      // Pin the probe itself, or this test goes quietly inert: the value has to satisfy the field's
      // own schema, and it has to differ from what the step would seed — otherwise "preserved" and
      // "clobbered" look identical and an unconditional assignment slips through.
      expect(zParamsState.shape[key].safeParse(value).success).toBe(true);
      const reseeded = migrate?.(buildReleaseBlob(tier)) as Record<string, unknown>;
      expect(reseeded[key]).not.toEqual(value);

      const blob = buildReleaseBlob(tier, { [key]: value, positivePrompt: 'a fluffy cat' });

      const result = migrate?.(blob) as Record<string, unknown>;

      expect(result[key]).toEqual(value);
      // The rest of the slice has to come through too — a reset here would be the whole-slice wipe.
      expect(result.positivePrompt).toBe('a fluffy cat');
    }
  );

  it('fills the fields added after the v4 bump from their zod defaults', () => {
    expect(migrate).toBeDefined();

    // Everything added since 1aeb05bbf0 bumped _version to 4 landed in a tier the chain could not
    // reach for as long as 4 was current: a v4 blob matched no branch, so no step could seed it and
    // a zod default was the only option. The PiD fields (3f5588f21f, one day after the bump) are the
    // case that dev builds actually hit; the ERNIE-Image and HiDiffusion fields followed the same
    // route. main's v4 -> v5 step now also seeds the PiD fields, but the defaults are what covers
    // the same gap on the current tier, which by definition still has no step.
    const blob = buildReleaseBlob('1aeb05bbf0', { positivePrompt: 'a fluffy cat', seed: 42 });
    expect('pidMode' in blob).toBe(false);
    expect('hiDiffusionEnabled' in blob).toBe(false);

    applyParamsVersionMigrations(blob);
    expect(blob._version).toBe(5);

    // The value assertions below cannot, on their own, prove the defaults exist: three mechanisms
    // produce the identical values, so any two can hide the third being reverted. Parsing directly
    // rather than through migrate() rules out the repair pass, but not the v4 -> v5 step, which
    // seeds all four PiD fields. Pin the property itself — carrying a default is exactly what lets a
    // field survive on a tier with no step, and it is the thing the current tier depends on.
    for (const key of POST_V4_BUMP_DEFAULTED_KEYS) {
      expect(
        zParamsState.shape[key].safeParse(undefined).success,
        `${key} was added after the _version 4 bump and must carry a zod default: the current tier ` +
          `has no migration step by definition, so a required key there fails the parse in migrate() ` +
          `and wipes the user's whole params slice.`
      ).toBe(true);
    }

    const result = zParamsState.parse(blob);

    expect(result.pidMode).toBe('off');
    expect(result.pidDecoderModel).toBeNull();
    expect(result.gemma2EncoderModel).toBeNull();
    expect(result.pidSteps).toBe(4);
    expect(result.hiDiffusionEnabled).toBe(false);
    expect(result.hiDiffusionRauNetEnabled).toBe(true);
    expect(result.hiDiffusionWindowAttnEnabled).toBe(true);
    expect(result.hiDiffusionT1Ratio).toBe(0.4);
    expect(result.hiDiffusionT2Ratio).toBe(0.0);
    expect(result.positivePrompt).toBe('a fluffy cat');
    expect(result.seed).toBe(42);
  });

  it('clears only the model when its base has since been removed from the schema', () => {
    expect(migrate).toBeDefined();

    // `zBaseModelType` dropped the external-API bases ('chatgpt-4o', 'imagen3', 'veo3', ...) in
    // v6.9.0rc1, so a genuine v6.5.1 - v6.8.1 blob can hold a `model` the current schema rejects.
    // Before the repair pass that failed zParamsState.parse() and cost the user every param; now it
    // costs them the model selection alone.
    const blob = buildReleaseBlob('v6.7.0', {
      model: { key: 'm1', hash: 'h1', name: 'GPT Image', base: 'chatgpt-4o', type: 'main' },
      positivePrompt: 'a fluffy cat',
      seed: 42,
    });

    const result = migrate?.(blob) as ReturnType<typeof getInitialParamsState>;

    expect(result.model).toBeNull();
    expect(result.positivePrompt).toBe('a fluffy cat');
    expect(result.seed).toBe(42);
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
