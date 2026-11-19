import type { GenerateLora, VaeModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import { getSchedulerOptions } from '@features/generation/settings';
import { resetArchitectureCapabilities } from '@features/models/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/models/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import {
  clearDeletedUpscaleInput,
  createDefaultUpscaleWidgetValues,
  getModelOnlyUpscalePlan,
  isUpscaleBatchCountFixed,
  resolveUpscaleSubmissionValues,
  getUpscaleOutputDimensions,
  getUpscaleValidationReasons,
  normalizeUpscaleWidgetValues,
  spandrelAutoscaleDimension,
  upscaleDenoiseDimension,
  isTileControlNetCandidate,
  selectUpscaleMainModel,
  syncUpscaleWidgetValuesWithModels,
  UPSCALE_ARCHITECTURES,
  UPSCALE_PRESETS,
  upscaleSchedulerOptions,
} from './settings';

seedArchitectureCapabilities();

const model = (key: string, type: string, base: string, name = key): ModelConfig => ({
  base,
  file_size: 1,
  format: 'checkpoint',
  hash: `${key}-hash`,
  key,
  name,
  path: key,
  source: key,
  source_type: 'path',
  type,
  ...(type === 'clip_embed' ? { variant: 'large' } : {}),
});

const MODELS = [
  model('sd15', 'main', 'sd-1'),
  model('spandrel', 'spandrel_image_to_image', 'any'),
  model('tile', 'controlnet', 'sd-1', 'ControlNet Tile'),
  model('lora', 'lora', 'sd-1'),
  model('vae', 'vae', 'sd-1'),
];

describe('upscale settings', () => {
  it('preserves the legacy defaults and named presets', () => {
    expect(createDefaultUpscaleWidgetValues()).toMatchObject({
      batchCount: 1,
      cfgScale: 2,
      creativity: 0,
      negativePromptHeightPx: 56,
      positivePromptHeightPx: 96,
      scale: 4,
      scheduler: 'kdpm_2',
      seedMode: 'random',
      steps: 30,
      structure: 0,
      tileOverlap: 128,
      tileSize: 1024,
    });
    expect(UPSCALE_PRESETS).toEqual({
      artistic: { creativity: 8, structure: -5 },
      balanced: { creativity: 0, structure: 0 },
      conservative: { creativity: -5, structure: 5 },
      creative: { creativity: 5, structure: -2 },
    });
  });

  it('reads the seed mode saved before modes existed from the random toggle', () => {
    expect(normalizeUpscaleWidgetValues({ shouldRandomizeSeed: false })?.seedMode).toBe('fixed');
    expect(normalizeUpscaleWidgetValues({ shouldRandomizeSeed: true })?.seedMode).toBe('random');
    expect(normalizeUpscaleWidgetValues({ seedMode: 'decrement', shouldRandomizeSeed: true })?.seedMode).toBe(
      'decrement'
    );
    expect(normalizeUpscaleWidgetValues({})?.seedMode).toBe('random');
  });

  it('reads values saved before FLUX.2 and Qwen Image with their encoders unset, and checks the type of saved ones', () => {
    const mistral = { ...model('m', 'mistral_encoder', 'any'), variant: 'mistral3_24b' };

    expect(normalizeUpscaleWidgetValues({ t5EncoderModel: null })).toMatchObject({
      mistralEncoderModel: null,
      qwenVlEncoderModel: null,
    });
    expect(normalizeUpscaleWidgetValues({ mistralEncoderModel: mistral, qwenVlEncoderModel: mistral })).toMatchObject({
      mistralEncoderModel: { key: 'm' },
      qwenVlEncoderModel: null,
    });
  });

  it('keeps a saved Text LLM that serves as the encoder through a role, and drops one without the role', () => {
    const textLlm = {
      ...model('llm', 'text_llm', 'any'),
      roles: [{ type: 'qwen3_encoder', variant: 'qwen3_4b' }],
    };

    expect(normalizeUpscaleWidgetValues({ qwen3EncoderModel: textLlm })).toMatchObject({
      qwen3EncoderModel: { key: 'llm', type: 'text_llm' },
    });
    expect(normalizeUpscaleWidgetValues({ qwen3EncoderModel: { ...textLlm, roles: [] } })).toMatchObject({
      qwen3EncoderModel: null,
    });
  });

  it('normalizes partial persisted values and calculates multiple-of-eight output dimensions', () => {
    expect(
      normalizeUpscaleWidgetValues({ negativePromptHeightPx: 4, positivePrompt: 'detail', positivePromptHeightPx: 999 })
    ).toMatchObject({
      negativePromptHeightPx: 56,
      positivePrompt: 'detail',
      positivePromptHeightPx: 360,
      scale: 4,
      tileSize: 1024,
    });
    expect(
      getUpscaleOutputDimensions(
        { height: 101, width: 203 },
        { mode: 'refine', model: MODELS[0] as never, scale: 2.5, upscaleModel: null }
      )
    ).toEqual({
      height: 248,
      width: 504,
    });
  });

  it('reconciles required models, refreshes configs, and removes incompatible selections', () => {
    const defaults = createDefaultUpscaleWidgetValues(MODELS);
    const stale = {
      ...defaults,
      loras: [
        { isEnabled: true, model: model('missing', 'lora', 'sd-1') as GenerateLora['model'], weight: 1 },
        { isEnabled: true, model: model('lora', 'lora', 'sd-1', 'Old name') as GenerateLora['model'], weight: 0.5 },
      ],
      vae: model('vae', 'vae', 'sd-1', 'Old VAE') as VaeModelConfig,
    };
    const synced = syncUpscaleWidgetValuesWithModels(stale, MODELS);

    expect(synced.model?.key).toBe('sd15');
    expect(synced.upscaleModel?.key).toBe('spandrel');
    expect(synced.tileControlnetModel?.key).toBe('tile');
    expect(synced.loras).toHaveLength(1);
    expect(synced.loras[0]?.model.name).toBe('lora');
    expect(synced.vae?.name).toBe('vae');
  });

  it('validates required fields and ranges and clears a deleted input image', () => {
    const defaults = createDefaultUpscaleWidgetValues();

    expect(getUpscaleValidationReasons({ ...defaults, cfgScale: 0.5, scale: 17 })).toEqual(
      expect.arrayContaining([
        'Upscale needs an input image. Upload one or send one from Gallery.',
        'Scale must be between 1 and 16.',
        'CFG scale must be between 1 and 100.',
      ])
    );

    const values = { ...defaults, inputImage: { height: 10, image_name: 'delete.png', width: 10 } };

    expect(clearDeletedUpscaleInput(values, new Set(['delete.png'])).inputImage).toBeNull();
  });
});

describe('architecture-driven validation', () => {
  const upscaleReady = (models: ModelConfig[]) => ({
    ...createDefaultUpscaleWidgetValues(models),
    inputImage: { height: 64, image_name: 'in.png', width: 64 },
  });

  it('does not demand a tile ControlNet for an architecture that has none', () => {
    const values = upscaleReady([
      model('main', 'main', 'flux'),
      model('spandrel', 'spandrel_image_to_image', 'any'),
      model('t5', 't5_encoder', 'any'),
      model('clip', 'clip_embed', 'any'),
      model('fluxvae', 'vae', 'flux'),
    ]);

    expect(getUpscaleValidationReasons(values)).toEqual([]);
    expect(values.tileControlnetModel).toBeNull();
  });

  it('blocks the run when a required component has been uninstalled', () => {
    const installed = [
      model('main', 'main', 'flux'),
      model('spandrel', 'spandrel_image_to_image', 'any'),
      model('t5', 't5_encoder', 'any'),
      model('clip', 'clip_embed', 'any'),
      model('fluxvae', 'vae', 'flux'),
    ];
    const values = upscaleReady(installed);
    const withoutT5 = installed.filter((candidate) => candidate.key !== 't5');

    expect(getUpscaleValidationReasons(values, withoutT5)).toContain('t5 is no longer installed.');
  });
});

describe('the architecture table', () => {
  it.each(Object.entries(UPSCALE_ARCHITECTURES))('agrees with the served %s capabilities', (base, architecture) => {
    // The fixture is the response the backend serves, pinned by `test_capabilities_fixture.py`, so a grid or
    // negative-prompt change on a denoise node fails here instead of at enqueue.
    const served = architectureCapabilitiesFixture.find((row) => row.base === base && row.variant === null);

    expect(served?.features.dimension_grid).toBe(architecture.denoiseGrid);
    expect(served?.features.negative_prompt.usage).toBe(architecture.negativePrompt);
  });
});

const withVariant = (config: ModelConfig, variant: string): ModelConfig => ({ ...config, variant }) as ModelConfig;

const QWEN3_4B = withVariant(model('qwen3-4b', 'qwen3_encoder', 'any'), 'qwen3_4b');
const QWEN3_8B = withVariant(model('qwen3-8b', 'qwen3_encoder', 'any'), 'qwen3_8b');
const QWEN3_06B = withVariant(model('qwen3-06b', 'qwen3_encoder', 'any'), 'qwen3_06b');
describe('text encoder and VAE components', () => {
  it('offers a Qwen3 4B Text LLM to Z-Image upscale through its role and refuses an 8B one', () => {
    const roleLlm = (variant: string): ModelConfig =>
      ({ ...model(`llm-${variant}`, 'text_llm', 'any'), roles: [{ type: 'qwen3_encoder', variant }] }) as ModelConfig;
    const values = (llm: ModelConfig) => ({
      ...createDefaultUpscaleWidgetValues([
        model('spandrel', 'spandrel_image_to_image', 'any'),
        model('z', 'main', 'z-image'),
        model('flux-vae', 'vae', 'flux'),
        llm,
      ]),
      inputImage: { height: 64, image_name: 'in.png', width: 64 },
    });

    expect(values(roleLlm('qwen3_4b')).qwen3EncoderModel?.key).toBe('llm-qwen3_4b');
    expect(values(roleLlm('qwen3_8b')).qwen3EncoderModel).toBeNull();
  });

  const ready = (models: ModelConfig[]) => ({
    ...createDefaultUpscaleWidgetValues(models),
    inputImage: { height: 64, image_name: 'in.png', width: 64 },
  });

  it('gives each architecture the Qwen3 its loader reads and the VAE it decodes with', () => {
    const shared = [model('spandrel', 'spandrel_image_to_image', 'any'), QWEN3_06B, QWEN3_4B];
    const zImage = ready([
      model('z', 'main', 'z-image'),
      ...shared,
      model('anima-vae', 'vae', 'qwen-image'),
      model('flux-vae', 'vae', 'flux'),
    ]);
    const anima = ready([
      withVariant(model('a', 'main', 'anima'), 'anima_qwen3'),
      ...shared,
      model('flux-vae', 'vae', 'flux'),
      model('anima-vae', 'vae', 'qwen-image'),
    ]);

    expect([zImage.qwen3EncoderModel?.key, zImage.vae?.key]).toEqual(['qwen3-4b', 'flux-vae']);
    expect([anima.qwen3EncoderModel?.key, anima.vae?.key]).toEqual(['qwen3-06b', 'anima-vae']);
    expect(getUpscaleValidationReasons(zImage)).toEqual([]);
    expect(getUpscaleValidationReasons(anima)).toEqual([]);
  });

  it('names a missing Qwen3 encoder rather than failing in the loader', () => {
    const values = ready([
      model('z', 'main', 'z-image'),
      model('spandrel', 'spandrel_image_to_image', 'any'),
      QWEN3_06B,
    ]);

    expect(getUpscaleValidationReasons(values)).toEqual([
      'Upscale needs a Qwen3 4B encoder for this main model.',
      'Upscale needs a VAE for this main model.',
    ]);
  });

  it('gives each FLUX.2 the encoder its loader takes: Klein the Qwen3 of its size, [dev] a Mistral', () => {
    const ministral = withVariant(model('ministral', 'mistral_encoder', 'any'), 'ministral3_3b');
    const mistral = withVariant(model('mistral', 'mistral_encoder', 'any'), 'mistral3_24b');
    const shared = [
      model('spandrel', 'spandrel_image_to_image', 'any'),
      model('flux2-vae', 'vae', 'flux2'),
      QWEN3_8B,
      QWEN3_06B,
      QWEN3_4B,
      ministral,
      mistral,
    ];
    const pick = (variant: string) => ready([withVariant(model('f', 'main', 'flux2'), variant), ...shared]);

    expect(
      ['klein_4b', 'klein_4b_base', 'klein_9b', 'klein_9b_base', 'dev'].map((variant) => [
        pick(variant).qwen3EncoderModel?.key ?? null,
        pick(variant).mistralEncoderModel?.key ?? null,
      ])
    ).toEqual([
      ['qwen3-4b', null],
      ['qwen3-4b', null],
      ['qwen3-8b', null],
      ['qwen3-8b', null],
      [null, 'mistral'],
    ]);
    expect(getUpscaleValidationReasons(pick('klein_9b'))).toEqual([]);
    expect(getUpscaleValidationReasons(pick('dev'))).toEqual([]);
  });
});

describe('picking a main model', () => {
  const MODELS_BY_BASE = [
    model('sd15', 'main', 'sd-1'),
    model('flux', 'main', 'flux'),
    model('z-turbo', 'main', 'z-image'),
    withVariant(model('z-base', 'main', 'z-image'), 'zbase'),
    withVariant(model('anima', 'main', 'anima'), 'anima_qwen3'),
    withVariant(model('anima-38', 'main', 'anima'), 'anima_qwen35'),
    withVariant(model('krea2-turbo', 'main', 'krea-2'), 'krea2_turbo'),
    withVariant(model('krea2-raw', 'main', 'krea-2'), 'krea2_base'),
    withVariant(model('klein', 'main', 'flux2'), 'klein_4b'),
    withVariant(model('klein-base', 'main', 'flux2'), 'klein_9b_base'),
    withVariant(model('flux2-dev', 'main', 'flux2'), 'dev'),
    withVariant(model('qwen', 'main', 'qwen-image'), 'generate'),
  ];
  const cfgAfterPicking = (key: string) =>
    selectUpscaleMainModel(
      { ...createDefaultUpscaleWidgetValues(MODELS_BY_BASE), cfgScale: 3 },
      MODELS_BY_BASE.find((candidate) => candidate.key === key) as never,
      MODELS_BY_BASE
    ).cfgScale;

  it('runs Z-Image and Anima at the CFG their model is served with', () => {
    expect(['z-turbo', 'z-base', 'anima', 'anima-38'].map(cfgAfterPicking)).toEqual([1, 4, 4.5, 6]);
  });

  it('refuses to run without the served table, rather than compile its fallbacks', () => {
    const turbo = selectUpscaleMainModel(
      {
        ...createDefaultUpscaleWidgetValues(MODELS_BY_BASE),
        inputImage: { height: 8, image_name: 'in.png', width: 8 },
      },
      MODELS_BY_BASE[2] as never,
      MODELS_BY_BASE
    );

    resetArchitectureCapabilities();

    expect(getUpscaleValidationReasons(turbo)).toEqual([
      'Model capabilities are not available. Upscale is blocked until they load; if this persists, retry from the Upscale panel.',
    ]);
  });
});

describe('schedulers per architecture', () => {
  it('offers only the schedulers that start where Creativity puts the pass', () => {
    const offered = (base: string) =>
      upscaleSchedulerOptions(model(base, 'main', base) as never).map((option) => option.value);

    expect(offered('flux')).toEqual(['euler', 'heun']);
    expect(offered('z-image')).toEqual(['euler', 'heun']);
    expect(offered('anima')).toEqual(['euler', 'dpmpp_2m', 'dpmpp_2m_sde', 'er_sde']);
    expect(offered('krea-2')).toEqual(['euler']);
    expect(offered('qwen-image')).toEqual(['euler']);
    expect(offered('flux2')).toEqual(['euler']);
    expect(offered('sdxl')).toEqual(
      getSchedulerOptions(model('sdxl', 'main', 'sdxl') as never).map((option) => option.value)
    );
  });
});

describe('tile ControlNet per architecture', () => {
  const zImage = { base: 'z-image' };
  const tile = model('zt', 'controlnet', 'z-image', 'Z-Image ControlNet Tile');
  const union = model('zu', 'controlnet', 'z-image', 'Z-Image ControlNet Union');

  it("accepts only Z-Image's tile model: its Union has no tile mode", () => {
    expect(isTileControlNetCandidate(tile, zImage)).toBe(true);
    expect(isTileControlNetCandidate(union, zImage)).toBe(false);
    expect(isTileControlNetCandidate(model('su', 'controlnet', 'sdxl', 'xinsir union'), { base: 'sdxl' })).toBe(true);
  });
});

describe('upscaleDenoiseDimension', () => {
  it('floors to the grid the architecture demands, not to the one Spandrel used', () => {
    // 1020 * 2 = 2040: a legal Spandrel size that flux_denoise rejects.
    expect(spandrelAutoscaleDimension(1020, 2)).toBe(2040);
    expect(upscaleDenoiseDimension(1020, 2, 8)).toBe(2040);
    expect(upscaleDenoiseDimension(1020, 2, 16)).toBe(2032);
  });
});

describe('upscale model-only mode', () => {
  const upscaler = (scale?: number | null) => ({
    ...model('ultrasharp', 'spandrel_image_to_image', 'any', '4x-UltraSharp'),
    type: 'spandrel_image_to_image' as const,
    ...(scale === undefined ? {} : { scale }),
  });
  const image = { height: 101, image_name: 'source.png', width: 203 };
  const modelOnly = (overrides = {}) => ({
    ...createDefaultUpscaleWidgetValues(),
    inputImage: image,
    mode: 'model-only' as const,
    upscaleModel: upscaler(4),
    ...overrides,
  });

  it('opens projects saved before modes, and anything unknown, in Refine', () => {
    expect(createDefaultUpscaleWidgetValues().mode).toBe('refine');
    expect(normalizeUpscaleWidgetValues({})?.mode).toBe('refine');
    expect(normalizeUpscaleWidgetValues({ mode: 'upscale-only' })?.mode).toBe('refine');
    expect(normalizeUpscaleWidgetValues({ mode: 'model-only' })?.mode).toBe('model-only');
  });

  it('is ready with an image and an upscaler alone, even before the served table loads', () => {
    resetArchitectureCapabilities();

    try {
      expect(getUpscaleValidationReasons(modelOnly({ model: null, tileControlnetModel: null, vae: null }))).toEqual([]);
    } finally {
      seedArchitectureCapabilities();
    }
  });

  it('plans the passes the autoscale node runs and whether it resizes after them', () => {
    expect(getModelOnlyUpscalePlan(4, 2)).toEqual({ kind: 'passes', passes: 1, resizes: true });
    expect(getModelOnlyUpscalePlan(4, 4)).toEqual({ kind: 'passes', passes: 1, resizes: false });
    expect(getModelOnlyUpscalePlan(4, 16)).toEqual({ kind: 'passes', passes: 2, resizes: false });
    expect(getModelOnlyUpscalePlan(2, 4)).toEqual({ kind: 'passes', passes: 2, resizes: false });
    expect(getModelOnlyUpscalePlan(2, 5)).toEqual({ kind: 'passes', passes: 3, resizes: true });
    expect(getModelOnlyUpscalePlan(1, 4)).toEqual({ kind: 'no-enlarge' });
    expect(getModelOnlyUpscalePlan(null, 4)).toBeNull();
  });

  it('runs a Model only submission once, with no prompt and a fixed seed, for the input its metadata was read from', () => {
    const draft = { negativePrompt: 'blur', negativePromptEnabled: true, positivePrompt: 'the Generate draft' };
    const values = modelOnly({ batchCount: 3, seedMode: 'increment' as const });

    expect(resolveUpscaleSubmissionValues(values, { promptDraft: draft, sourceImageName: 'source.png' })).toMatchObject(
      { batchCount: 1, negativePrompt: '', positivePrompt: '', seedMode: 'fixed' }
    );
    expect(resolveUpscaleSubmissionValues(values, { promptDraft: draft, sourceImageName: 'earlier.png' })).toBeNull();
    expect(resolveUpscaleSubmissionValues(values, { promptDraft: draft, sourceImageName: undefined })).toBeNull();
  });

  it('fixes the iteration count only for Model only, saved values included', () => {
    expect(isUpscaleBatchCountFixed({ batchCount: 3, mode: 'model-only' })).toBe(true);
    expect(isUpscaleBatchCountFixed({ batchCount: 3, mode: 'refine' })).toBe(false);
    expect(isUpscaleBatchCountFixed({ batchCount: 3 })).toBe(false);
  });
});
