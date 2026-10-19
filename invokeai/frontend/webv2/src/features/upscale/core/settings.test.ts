import type { GenerateLora, VaeModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import { resetArchitectureCapabilities } from '@features/generation/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities.testing';
import { getSchedulerOptions } from '@features/generation/settings';
import { describe, expect, it } from 'vitest';

import {
  clearDeletedUpscaleInput,
  createDefaultUpscaleWidgetValues,
  getUpscaleOutputDimensions,
  getUpscaleValidationReasons,
  normalizeUpscaleWidgetValues,
  spandrelAutoscaleDimension,
  upscaleDenoiseDimension,
  activeUpscaleComponents,
  coerceUpscaleScheduler,
  isTileControlNetCandidate,
  selectUpscaleMainModel,
  syncUpscaleWidgetValuesWithModels,
  UPSCALE_ARCHITECTURES,
  UPSCALE_PRESETS,
  upscaleSchedulerOptions,
} from './settings';

// VAE compatibility is served by the backend now, and the accessor fails closed without it.
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
  // Every installed CLIP Embed carries its variant; FLUX.1 takes only CLIP-L.
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
    expect(getUpscaleOutputDimensions({ height: 101, width: 203 }, 2.5, MODELS[0] as never)).toEqual({
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
    // The inversion this change is built on: compileUpscaleGraph used to throw without one.
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

  it('gives FLUX.1 a CLIP-L and never a CLIP-G, which its text encoder cannot run', () => {
    // A ComfyUI text_encoders folder installs both; whichever sorts first must not decide.
    const clipG = { ...model('clip-g', 'clip_embed', 'any'), variant: 'gigantic' } as ModelConfig;
    const installed = [
      clipG,
      model('main', 'main', 'flux'),
      model('spandrel', 'spandrel_image_to_image', 'any'),
      model('t5', 't5_encoder', 'any'),
      model('clip', 'clip_embed', 'any'),
      model('fluxvae', 'vae', 'flux'),
    ];

    const values = upscaleReady(installed);
    const restored = syncUpscaleWidgetValuesWithModels({ ...values, clipEmbedModel: clipG }, installed);

    expect(values.clipEmbedModel?.key).toBe('clip');
    expect(restored.clipEmbedModel?.key).toBe('clip');
  });

  it('still demands one where the architecture uses it', () => {
    const values = upscaleReady([model('main', 'main', 'sdxl'), model('spandrel', 'spandrel_image_to_image', 'any')]);

    expect(getUpscaleValidationReasons(values)).toContain(
      'Upscale needs a Tile or Union ControlNet compatible with the main model.'
    );
  });

  it.each([
    ['t5_encoder', 'Upscale needs a T5 encoder for this main model.'],
    ['clip_embed', 'Upscale needs a CLIP Embed model for this main model.'],
    ['vae', 'Upscale needs a VAE for this main model.'],
  ])('names the missing %s a bare FLUX checkpoint cannot supply', (missing, reason) => {
    const installed = [
      model('main', 'main', 'flux'),
      model('spandrel', 'spandrel_image_to_image', 'any'),
      model('t5', 't5_encoder', 'any'),
      model('clip', 'clip_embed', 'any'),
      model('fluxvae', 'vae', 'flux'),
    ].filter((candidate) => candidate.type !== missing);

    expect(getUpscaleValidationReasons(upscaleReady(installed))).toContain(reason);
  });

  it('exempts a self-contained SDNQ pipeline, which ships its own parts', () => {
    const sdnq = { ...model('main', 'main', 'flux'), format: 'sdnq_quantized' } as ModelConfig;
    const values = upscaleReady([sdnq, model('spandrel', 'spandrel_image_to_image', 'any')]);

    expect(getUpscaleValidationReasons(values)).toEqual([]);
  });

  it('refuses a FLUX Fill model, which the denoise node cannot run without fill conditioning', () => {
    const fill = { ...model('main', 'main', 'flux'), variant: 'dev_fill' } as ModelConfig;
    const values = upscaleReady([
      fill,
      model('spandrel', 'spandrel_image_to_image', 'any'),
      model('t5', 't5_encoder', 'any'),
      model('clip', 'clip_embed', 'any'),
      model('fluxvae', 'vae', 'flux'),
    ]);

    expect(getUpscaleValidationReasons(values)).toContain('Upscale cannot use a FLUX Fill model.');
  });

  it('blocks the run when a required component has been uninstalled', () => {
    // Without this the Invoke button stays enabled and the graph dies after Spandrel has run.
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
    // The fixture is the response the backend serves, pinned by `test_capabilities_fixture.py`, so
    // a grid or negative-prompt change on a denoise node fails here instead of at enqueue.
    const served = architectureCapabilitiesFixture.find((row) => row.base === base && row.variant === null);

    expect(served?.features.dimension_grid).toBe(architecture.denoiseGrid);
    expect(served?.features.negative_prompt.usage).toBe(architecture.negativePrompt);
  });
});

const withVariant = (config: ModelConfig, variant: string): ModelConfig => ({ ...config, variant }) as ModelConfig;

const QWEN3_4B = withVariant(model('qwen3-4b', 'qwen3_encoder', 'any'), 'qwen3_4b');
const QWEN3_8B = withVariant(model('qwen3-8b', 'qwen3_encoder', 'any'), 'qwen3_8b');
const QWEN3_06B = withVariant(model('qwen3-06b', 'qwen3_encoder', 'any'), 'qwen3_06b');
const QWEN35 = withVariant(model('qwen35', 'qwen3_5_encoder', 'any'), 'qwen3_5_4b');

describe('Z-Image and Anima components', () => {
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

  it('swaps the encoder when the main model moves to the other architecture', () => {
    const models = [
      model('z', 'main', 'z-image'),
      withVariant(model('a', 'main', 'anima'), 'anima_qwen3'),
      model('spandrel', 'spandrel_image_to_image', 'any'),
      QWEN3_4B,
      QWEN3_06B,
    ];
    const zImage = createDefaultUpscaleWidgetValues(models);
    const anima = syncUpscaleWidgetValuesWithModels({ ...zImage, model: models[1] as typeof zImage.model }, models);

    expect(zImage.qwen3EncoderModel?.key).toBe('qwen3-4b');
    expect(anima.qwen3EncoderModel?.key).toBe('qwen3-06b');
  });

  it('asks Anima-3.8B for the Qwen3.5 encoder and no other Anima for it', () => {
    const base = [model('spandrel', 'spandrel_image_to_image', 'any'), QWEN3_06B, model('vae', 'vae', 'qwen-image')];

    expect(
      getUpscaleValidationReasons(ready([withVariant(model('a', 'main', 'anima'), 'anima_qwen35'), ...base]))
    ).toEqual(['Upscale needs a Qwen3.5 encoder for this main model.']);
    expect(
      ready([withVariant(model('a', 'main', 'anima'), 'anima_qwen35'), ...base, QWEN35]).qwen35EncoderModel?.key
    ).toBe('qwen35');
    expect(
      ready([withVariant(model('a', 'main', 'anima'), 'anima_qwen3'), ...base, QWEN35]).qwen35EncoderModel
    ).toBeNull();
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

  it('never gives Z-Image the 8B, which its 2560-wide caption embedder cannot take', () => {
    // FLUX.2 Klein's 8B sorts first here, as it can by name in a real install.
    const values = ready([
      model('z', 'main', 'z-image'),
      model('spandrel', 'spandrel_image_to_image', 'any'),
      QWEN3_8B,
      QWEN3_4B,
    ]);

    expect(values.qwen3EncoderModel?.key).toBe('qwen3-4b');
    expect(getUpscaleValidationReasons({ ...values, qwen3EncoderModel: QWEN3_8B as never })).toContain(
      'qwen3-8b is not a Qwen3 4B encoder this main model can use.'
    );
  });

  it('gives a single-file Krea-2 the Qwen3-VL 4B, never the 8B its loader refuses', () => {
    // Ideogram 4's 8B is the same model type and sorts first here.
    const qwen3Vl8b = withVariant(model('qwen3vl-8b', 'qwen3_vl_encoder', 'any'), 'qwen3_vl_8b');
    const qwen3Vl4b = withVariant(model('qwen3vl-4b', 'qwen3_vl_encoder', 'any'), 'qwen3_vl_4b');
    const base = [model('spandrel', 'spandrel_image_to_image', 'any'), model('vae', 'vae', 'qwen-image')];
    const raw = { ...withVariant(model('k', 'main', 'krea-2'), 'krea2_base'), format: 'gguf_quantized' } as ModelConfig;
    const values = ready([raw, ...base, qwen3Vl8b, qwen3Vl4b]);

    expect([values.qwen3VlEncoderModel?.key, values.vae?.key]).toEqual(['qwen3vl-4b', 'vae']);
    expect(getUpscaleValidationReasons(values)).toEqual([]);
    expect(getUpscaleValidationReasons(ready([raw, ...base, qwen3Vl8b]))).toEqual([
      'Upscale needs a Qwen3-VL 4B encoder for this main model.',
    ]);
  });

  it('lets a Diffusers Krea-2 bring its own encoder and VAE', () => {
    const diffusers = {
      ...withVariant(model('k', 'main', 'krea-2'), 'krea2_turbo'),
      format: 'diffusers',
    } as ModelConfig;

    expect(
      getUpscaleValidationReasons(ready([diffusers, model('spandrel', 'spandrel_image_to_image', 'any')]))
    ).toEqual([]);
  });

  it('keeps a hand-picked encoder through a switch to an architecture that does not read it', () => {
    const models = [
      model('flux', 'main', 'flux'),
      model('z', 'main', 'z-image'),
      model('spandrel', 'spandrel_image_to_image', 'any'),
      model('t5-a', 't5_encoder', 'any'),
      model('t5-b', 't5_encoder', 'any'),
      model('clip', 'clip_embed', 'any'),
      QWEN3_4B,
    ];
    const flux = { ...createDefaultUpscaleWidgetValues(models), t5EncoderModel: models[4] as never };
    const zImage = syncUpscaleWidgetValuesWithModels({ ...flux, model: models[1] as typeof flux.model }, models);
    const backToFlux = syncUpscaleWidgetValuesWithModels({ ...zImage, model: models[0] as typeof flux.model }, models);

    expect(zImage.t5EncoderModel?.key).toBe('t5-b');
    expect(activeUpscaleComponents(zImage)).toMatchObject({
      qwen3EncoderModel: { key: 'qwen3-4b' },
      t5EncoderModel: null,
    });
    expect(backToFlux.t5EncoderModel?.key).toBe('t5-b');
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
  ];
  const cfgAfterPicking = (key: string) =>
    selectUpscaleMainModel(
      { ...createDefaultUpscaleWidgetValues(MODELS_BY_BASE), cfgScale: 3 },
      MODELS_BY_BASE.find((candidate) => candidate.key === key) as never,
      MODELS_BY_BASE
    ).cfgScale;

  it('runs Z-Image and Anima at the CFG their model is served with', () => {
    // Turbo is distilled for CFG 1; anything above doubles every step at upscale resolution.
    expect(['z-turbo', 'z-base', 'anima', 'anima-38'].map(cfgAfterPicking)).toEqual([1, 4, 4.5, 6]);
  });

  it('runs Krea-2 Turbo without CFG and Raw at the CFG it is served with', () => {
    expect(['krea2-turbo', 'krea2-raw'].map(cfgAfterPicking)).toEqual([1, 5.5]);
  });

  it('leaves SD and FLUX at the CFG the user set', () => {
    expect(['sd15', 'flux'].map(cfgAfterPicking)).toEqual([3, 3]);
  });

  it('gives SD and FLUX the widget CFG back after a model that set its own', () => {
    const fromTurbo = selectUpscaleMainModel(
      { ...createDefaultUpscaleWidgetValues(MODELS_BY_BASE), cfgScale: 3 },
      MODELS_BY_BASE[2] as never,
      MODELS_BY_BASE
    );
    const fromAnima38 = selectUpscaleMainModel(fromTurbo, MODELS_BY_BASE[5] as never, MODELS_BY_BASE);

    expect(fromTurbo.cfgScale).toBe(1);
    expect(selectUpscaleMainModel(fromTurbo, MODELS_BY_BASE[0] as never, MODELS_BY_BASE).cfgScale).toBe(2);
    expect(selectUpscaleMainModel(fromAnima38, MODELS_BY_BASE[1] as never, MODELS_BY_BASE).cfgScale).toBe(2);
  });

  it("gives a model the widget picks on its own that model's CFG, as a pick would", () => {
    // A fresh widget is created without models and gets its first one from the sync.
    const fresh = syncUpscaleWidgetValuesWithModels(createDefaultUpscaleWidgetValues(), [MODELS_BY_BASE[4]!]);
    // An uninstalled Z-Image Turbo falls back to SD, which gets the widget's CFG, not Turbo's 1.
    const turbo = selectUpscaleMainModel(fresh, MODELS_BY_BASE[2] as never, MODELS_BY_BASE);
    const fellBack = syncUpscaleWidgetValuesWithModels(turbo, [MODELS_BY_BASE[0]!]);

    expect(fresh.cfgScale).toBe(4.5);
    expect([turbo.cfgScale, fellBack.model?.key, fellBack.cfgScale]).toEqual([1, 'sd15', 2]);
  });

  it('refuses to run without the served table, rather than compile its fallbacks', () => {
    // A model picked without the table would keep a fallback CFG for good, every VAE would read as
    // incompatible and the schedulers would fall back; the widget is gated on the table for the same reason.
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

  it('keeps the CFG the user tuned when the same model is picked again', () => {
    const tuned = {
      ...selectUpscaleMainModel(
        createDefaultUpscaleWidgetValues(MODELS_BY_BASE),
        MODELS_BY_BASE[4] as never,
        MODELS_BY_BASE
      ),
      cfgScale: 5,
    };

    expect(selectUpscaleMainModel(tuned, MODELS_BY_BASE[4] as never, MODELS_BY_BASE).cfgScale).toBe(5);
  });

  it('asks index-clipping architectures for two steps, since one leaves no partial pass', () => {
    const zImage = { ...createDefaultUpscaleWidgetValues([model('z', 'main', 'z-image')]), steps: 1 };
    const sd = { ...createDefaultUpscaleWidgetValues([model('sd15', 'main', 'sd-1')]), steps: 1 };

    expect(getUpscaleValidationReasons(zImage)).toContain('Steps must be between 2 and 1000.');
    expect(getUpscaleValidationReasons(sd).some((reason) => reason.startsWith('Steps'))).toBe(false);
  });
});

describe('schedulers per architecture', () => {
  it('offers only the schedulers that start where Creativity puts the pass', () => {
    const offered = (base: string) =>
      upscaleSchedulerOptions(model(base, 'main', base) as never).map((option) => option.value);

    // A single scheduler hides the field.
    expect(offered('flux')).toEqual(['euler']);
    expect(offered('z-image')).toEqual(['euler']);
    expect(offered('anima')).toEqual(['euler', 'dpmpp_2m', 'dpmpp_2m_sde', 'er_sde']);
    // `krea2_denoise` has no scheduler input.
    expect(offered('krea-2')).toEqual(['euler']);
    // SD's pass adds noise with the scheduler itself, so every one of its schedulers starts there.
    expect(offered('sdxl')).toEqual(
      getSchedulerOptions(model('sdxl', 'main', 'sdxl') as never).map((option) => option.value)
    );
  });

  it('runs one of the schedulers it shows, whatever is stored', () => {
    const stored = ['euler', 'heun', 'lcm', 'dpmpp_2m', 'dpmpp_2m_sde', 'er_sde', 'kdpm_2', 'euler_a'];

    for (const base of Object.keys(UPSCALE_ARCHITECTURES)) {
      const main = model(base, 'main', base) as never;
      const shown = upscaleSchedulerOptions(main).map((option) => option.value);

      for (const scheduler of stored) {
        expect(shown, `${base} with ${scheduler} stored`).toContain(coerceUpscaleScheduler(main, scheduler));
      }
    }
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

  it('leaves the optional Z-Image ControlNet off until chosen, and keeps it once it is', () => {
    const models = [model('z', 'main', 'z-image'), model('spandrel', 'spandrel_image_to_image', 'any'), tile];
    const defaults = createDefaultUpscaleWidgetValues(models);
    const chosen = syncUpscaleWidgetValuesWithModels({ ...defaults, tileControlnetModel: tile as never }, models);

    expect(defaults.tileControlnetModel).toBeNull();
    expect(syncUpscaleWidgetValuesWithModels(defaults, models).tileControlnetModel).toBeNull();
    expect(chosen.tileControlnetModel?.key).toBe('zt');
  });

  it('refuses a Union model chosen for Z-Image', () => {
    const values = {
      ...createDefaultUpscaleWidgetValues([model('z', 'main', 'z-image')]),
      tileControlnetModel: union as never,
    };

    expect(getUpscaleValidationReasons(values)).toContain(
      'The Tile ControlNet must match the main model base and be a Tile model.'
    );
  });
});

describe('upscaleDenoiseDimension', () => {
  it('floors to the grid the architecture demands, not to the one Spandrel used', () => {
    // 1020 * 2 = 2040: a legal Spandrel size that flux_denoise rejects.
    expect(spandrelAutoscaleDimension(1020, 2)).toBe(2040);
    expect(upscaleDenoiseDimension(1020, 2, 8)).toBe(2040);
    expect(upscaleDenoiseDimension(1020, 2, 16)).toBe(2032);
  });

  it('never returns zero for a tiny frame', () => {
    expect(upscaleDenoiseDimension(4, 1, 16)).toBe(16);
  });

  it("shows the size the result comes out at, which a 16px pass makes smaller than Spandrel's", () => {
    // 1004 at 2x: Spandrel gives 2008, which FLUX.1 and Z-Image resize to 2000.
    const widths = ['sd-1', 'flux', 'z-image', 'anima'].map(
      (base) => getUpscaleOutputDimensions({ height: 1004, width: 1004 }, 2, model(base, 'main', base) as never).width
    );

    expect(widths).toEqual([2008, 2000, 2000, 2008]);
  });
});
