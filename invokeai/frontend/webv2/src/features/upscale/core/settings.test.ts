import type { GenerateLora, VaeModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { describe, expect, it } from 'vitest';

import {
  clearDeletedUpscaleInput,
  createDefaultUpscaleWidgetValues,
  getUpscaleOutputDimensions,
  getUpscaleValidationReasons,
  normalizeUpscaleWidgetValues,
  spandrelAutoscaleDimension,
  upscaleDenoiseDimension,
  syncUpscaleWidgetValuesWithModels,
  UPSCALE_PRESETS,
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
    expect(getUpscaleOutputDimensions({ height: 101, width: 203 }, 2.5)).toEqual({ height: 248, width: 504 });
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
});
