import type { GenerateLora, VaeModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import { describe, expect, it } from 'vitest';

import {
  clearDeletedUpscaleInput,
  createDefaultUpscaleWidgetValues,
  getUpscaleOutputDimensions,
  getUpscaleValidationReasons,
  normalizeUpscaleWidgetValues,
  syncUpscaleWidgetValuesWithModels,
  UPSCALE_PRESETS,
} from './settings';

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
      shouldRandomizeSeed: true,
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

    expect(getUpscaleValidationReasons({ ...defaults, scale: 17 })).toEqual(
      expect.arrayContaining([
        'Upscale needs an input image. Upload one or send one from Gallery.',
        'Scale must be between 1 and 16.',
      ])
    );

    const values = { ...defaults, inputImage: { height: 10, image_name: 'delete.png', width: 10 } };

    expect(clearDeletedUpscaleInput(values, new Set(['delete.png'])).inputImage).toBeNull();
  });
});
