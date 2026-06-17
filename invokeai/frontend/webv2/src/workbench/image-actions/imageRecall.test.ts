import type { GalleryImage } from '@workbench/gallery/api';
import type {
  ComponentModelConfig,
  GenerateWidgetValues,
  MainModelConfig,
  VaeModelConfig,
} from '@workbench/generation/types';

import { describe, expect, it } from 'vitest';

import { buildImageRecallSettings, getImageRecallCapabilities } from './imageRecall';

const sdxlModel: MainModelConfig = { base: 'sdxl', key: 'sdxl-model', name: 'SDXL', type: 'main' };
const sd1Model: MainModelConfig = { base: 'sd-1', key: 'sd1-model', name: 'SD 1.5', type: 'main' };
const animaModel: MainModelConfig = { base: 'anima', key: 'anima-model', name: 'Anima', type: 'main' };
const vaeModel: VaeModelConfig = { base: 'sd-1', key: 'vae-model', name: 'SD 1.5 VAE', type: 'vae' };
const qwenImageVae: VaeModelConfig = { base: 'qwen-image', key: 'qwen-vae', name: 'Qwen VAE', type: 'vae' };
const animaQwen3: ComponentModelConfig = {
  base: 'any',
  key: 'anima-qwen3',
  name: 'Anima Qwen3',
  type: 'qwen3_encoder',
  variant: 'qwen3_06b',
};
const qwenVLEncoder: ComponentModelConfig = {
  base: 'any',
  key: 'qwen-vl',
  name: 'Qwen VL',
  type: 'qwen_vl_encoder',
};

const createValues = (overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues => ({
  aspectRatioId: '1:1',
  aspectRatioIsLocked: false,
  aspectRatioValue: 1,
  batchCount: 1,
  cfgRescaleMultiplier: 0,
  cfgScale: 7,
  clipEmbedModel: null,
  clipGEmbedModel: null,
  clipLEmbedModel: null,
  clipSkip: 0,
  componentSourceModel: null,
  height: 1024,
  loras: [],
  model: sdxlModel,
  modelKey: sdxlModel.key,
  negativePrompt: '',
  positivePrompt: '',
  qwen3EncoderModel: null,
  qwenVLEncoderModel: null,
  scheduler: 'euler_a',
  seamlessXAxis: false,
  seamlessYAxis: false,
  seed: 123,
  shouldRandomizeSeed: true,
  steps: 30,
  t5EncoderModel: null,
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
      models: [sd1Model, vaeModel],
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
      models: [sd1Model],
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
      models: [],
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
      models: [],
      supportedModels: [],
      vaeModels: [],
    });
    const sd1Capabilities = getImageRecallCapabilities({
      currentValues: createValues({ model: sd1Model, modelKey: sd1Model.key }),
      image,
      metadata: { clip_skip: 3 },
      models: [],
      supportedModels: [],
      vaeModels: [],
    });

    expect(sdxlCapabilities.clipSkip).toBe(false);
    expect(sd1Capabilities.clipSkip).toBe(true);
  });

  it('enables recall actions for component-only metadata', () => {
    const capabilities = getImageRecallCapabilities({
      currentValues: createValues(),
      image,
      metadata: { qwen3_encoder: { key: animaQwen3.key } },
      models: [animaQwen3],
      supportedModels: [],
      vaeModels: [],
    });

    expect(capabilities.all).toBe(true);
    expect(capabilities.remix).toBe(true);
  });

  it('recalls required component models for component-based families', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues({ model: sd1Model, modelKey: sd1Model.key }),
      image,
      kind: 'all',
      metadata: {
        height: 1024,
        model: { key: animaModel.key },
        qwen3_encoder: { key: animaQwen3.key },
        vae: { key: qwenImageVae.key },
        width: 1024,
      },
      models: [animaModel, animaQwen3, qwenImageVae],
      supportedModels: [animaModel],
      vaeModels: [qwenImageVae],
    });

    expect(result?.values.model).toBe(animaModel);
    expect(result?.values.qwen3EncoderModel).toBe(animaQwen3);
    expect(result?.values.vae).toBe(qwenImageVae);
    expect(result?.fields).toContain('components');
  });

  it('clears component models when recalled metadata explicitly stores null', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues({
        componentSourceModel: sd1Model,
        qwen3EncoderModel: animaQwen3,
        qwenVLEncoderModel: qwenVLEncoder,
      }),
      image,
      kind: 'all',
      metadata: {
        qwen3_encoder: null,
        qwen3_source: null,
        qwen_image_qwen_vl_encoder: null,
      },
      models: [sd1Model, animaQwen3, qwenVLEncoder],
      supportedModels: [],
      vaeModels: [],
    });

    expect(result?.values.componentSourceModel).toBeNull();
    expect(result?.values.qwen3EncoderModel).toBeNull();
    expect(result?.values.qwenVLEncoderModel).toBeNull();
    expect(result?.fields).toContain('components');
  });
});
