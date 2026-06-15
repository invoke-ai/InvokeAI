import { describe, expect, it } from 'vitest';

import type { GenerateWidgetValues, MainModelConfig, VaeModelConfig } from '@workbench/generation/types';
import type { GalleryImage } from '@workbench/gallery/api';
import { buildImageRecallSettings, getImageRecallCapabilities } from './imageRecall';

const sdxlModel: MainModelConfig = { base: 'sdxl', key: 'sdxl-model', name: 'SDXL', type: 'main' };
const sd1Model: MainModelConfig = { base: 'sd-1', key: 'sd1-model', name: 'SD 1.5', type: 'main' };
const vaeModel: VaeModelConfig = { base: 'sd-1', key: 'vae-model', name: 'SD 1.5 VAE', type: 'vae' };

const createValues = (overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues => ({
  aspectRatioId: '1:1',
  aspectRatioIsLocked: false,
  aspectRatioValue: 1,
  batchCount: 1,
  cfgRescaleMultiplier: 0,
  cfgScale: 7,
  clipSkip: 0,
  height: 1024,
  model: sdxlModel,
  modelKey: sdxlModel.key,
  negativePrompt: '',
  positivePrompt: '',
  scheduler: 'euler_a',
  seamlessXAxis: false,
  seamlessYAxis: false,
  seed: 123,
  shouldRandomizeSeed: true,
  steps: 30,
  vae: null,
  vaePrecision: 'fp32',
  width: 1024,
  ...overrides,
});

const image: GalleryImage = {
  boardId: 'none',
  height: 770,
  imageCategory: 'general',
  imageName: 'image.png',
  imageUrl: '/image.png',
  queuedAt: '2026-06-12T00:00:00.000Z',
  sourceQueueItemId: 'backend-gallery',
  starred: false,
  thumbnailUrl: '/thumbnail.png',
  width: 513,
};

const metadata = {
  cfg_rescale_multiplier: 0.25,
  cfg_scale: 8,
  height: 770,
  model: { key: sd1Model.key },
  negative_prompt: null,
  positive_prompt: 'a recalled prompt',
  scheduler: 'euler',
  seamless_x: true,
  seed: 42,
  steps: 25,
  vae: { key: vaeModel.key },
  width: 513,
};

describe('image recall', () => {
  it('recalls supported image metadata into generate settings', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues(),
      image,
      kind: 'all',
      metadata,
      supportedModels: [sd1Model],
      vaeModels: [vaeModel],
    });

    expect(result?.values).toMatchObject({
      cfgRescaleMultiplier: 0.25,
      cfgScale: 8,
      clipSkip: 0,
      height: 768,
      modelKey: sd1Model.key,
      negativePrompt: '',
      positivePrompt: 'a recalled prompt',
      scheduler: 'euler',
      seamlessXAxis: true,
      seed: 42,
      shouldRandomizeSeed: false,
      steps: 25,
      width: 512,
    });
    expect(result?.values.vae).toEqual(vaeModel);
    expect(result?.fields).toContain('model');
    expect(result?.fields).toContain('vae');
    expect(result?.fields).toContain('prompts');
    expect(result?.fields).toContain('seed');
  });

  it('remixes without changing the current seed state', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues({ seed: 999, shouldRandomizeSeed: false }),
      image,
      kind: 'remix',
      metadata,
      supportedModels: [sd1Model],
      vaeModels: [],
    });

    expect(result?.values.seed).toBe(999);
    expect(result?.values.shouldRandomizeSeed).toBe(false);
    expect(result?.values.positivePrompt).toBe('a recalled prompt');
    expect(result?.fields).not.toContain('seed');
  });

  it('uses actual image dimensions for Use Size', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues(),
      image,
      kind: 'dimensions',
      metadata: null,
      supportedModels: [],
      vaeModels: [],
    });

    expect(result?.values.width).toBe(512);
    expect(result?.values.height).toBe(768);
    expect(result?.values.aspectRatioId).toBe('2:3');
  });

  it('only enables standalone CLIP skip when the current model supports it', () => {
    const sdxlCapabilities = getImageRecallCapabilities({
      currentValues: createValues({ model: sdxlModel, modelKey: sdxlModel.key }),
      image,
      metadata: { clip_skip: 3 },
      supportedModels: [],
      vaeModels: [],
    });
    const sd1Capabilities = getImageRecallCapabilities({
      currentValues: createValues({ model: sd1Model, modelKey: sd1Model.key }),
      image,
      metadata: { clip_skip: 3 },
      supportedModels: [],
      vaeModels: [],
    });

    expect(sdxlCapabilities.clipSkip).toBe(false);
    expect(sd1Capabilities.clipSkip).toBe(true);
  });
});
