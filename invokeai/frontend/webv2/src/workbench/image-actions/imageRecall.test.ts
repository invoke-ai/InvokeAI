import type { GalleryImage } from '@features/gallery';
import type {
  ComponentModelConfig,
  GenerateWidgetValues,
  MainModelConfig,
  VaeModelConfig,
} from '@features/generation/contracts';

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
  colorCompensation: false,
  componentSourceModel: null,
  height: 1024,
  loras: [],
  model: sdxlModel,
  modelKey: sdxlModel.key,
  negativePromptEnabled: true,
  negativePrompt: '',
  negativePromptHeightPx: 56,
  positivePrompt: '',
  positivePromptHeightPx: 96,
  qwen3EncoderModel: null,
  qwenVLEncoderModel: null,
  referenceImages: [],
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

  it('recalls reference images from metadata, re-targeting configs to the effective model', () => {
    // The metadata model is not installed, so the current SDXL model stays
    // selected and the Qwen reference config is rebuilt for SDXL.
    const result = buildImageRecallSettings({
      currentValues: createValues(),
      image,
      kind: 'all',
      metadata: {
        ref_images: [
          {
            id: 'ref-1',
            isEnabled: true,
            config: {
              image: {
                height: 770,
                imageName: 'reference.png',
                imageUrl: '/api/v1/images/i/reference.png/full',
                queuedAt: '2026-01-01T00:00:00.000Z',
                sourceQueueItemId: 'backend-gallery',
                thumbnailUrl: '/api/v1/images/i/reference.png/thumbnail',
                width: 513,
              },
              type: 'qwen_image_reference_image',
            },
          },
        ],
      },
      models: [],
      supportedModels: [sdxlModel],
      vaeModels: [],
    });

    expect(result?.values.referenceImages).toHaveLength(1);
    expect(result?.values.referenceImages[0]?.config).toMatchObject({
      image: { original: { image: { height: 770, image_name: 'reference.png', width: 513 } } },
      type: 'ip_adapter',
    });
    expect(result?.fields).toContain('referenceImages');
  });

  it('preserves canonical original and crop provenance from legacy metadata', () => {
    const crop = {
      box: { height: 256, width: 320, x: 12, y: 24 },
      image: { height: 256, image_name: 'crop.png', width: 320 },
      ratio: null,
    };
    const result = buildImageRecallSettings({
      currentValues: createValues(),
      image,
      kind: 'all',
      metadata: {
        ref_images: [
          {
            config: {
              image: {
                crop,
                original: { image: { height: 768, image_name: 'original.png', width: 512 } },
              },
              type: 'qwen_image_reference_image',
            },
            id: 'canonical-ref',
            isEnabled: false,
          },
        ],
      },
      models: [],
      supportedModels: [sdxlModel],
      vaeModels: [],
    });

    expect(result?.values.referenceImages[0]).toMatchObject({
      id: 'canonical-ref',
      isEnabled: false,
      config: {
        image: {
          crop,
          original: { image: { height: 768, image_name: 'original.png', width: 512 } },
        },
      },
    });
  });

  it('falls back to pre-v6 canvas ipAdapter metadata when ref_images has no valid entries', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues(),
      image,
      kind: 'all',
      metadata: {
        canvas_v2_metadata: {
          referenceImages: {
            entities: [
              {
                id: 'legacy-canvas-ref',
                ipAdapter: {
                  beginEndStepPct: [0.2, 0.9],
                  clipVisionModel: 'ViT-G',
                  image: { height: 512, image_name: 'legacy.png', width: 512 },
                  method: 'composition',
                  model: { base: 'sdxl', key: 'adapter', name: 'Adapter', type: 'ip_adapter' },
                  type: 'ip_adapter',
                  weight: 0.7,
                },
                isEnabled: false,
              },
            ],
          },
        },
        ref_images: [
          { id: 'invalid' },
          {
            config: { image: null, type: 'qwen_image_reference_image' },
            id: 'incomplete-direct-ref',
            isEnabled: true,
          },
        ],
      },
      models: [],
      supportedModels: [sdxlModel],
      vaeModels: [],
    });

    expect(result?.values.referenceImages).toHaveLength(1);
    expect(result?.values.referenceImages[0]).toMatchObject({
      id: 'legacy-canvas-ref',
      isEnabled: false,
      config: {
        image: { original: { image: { image_name: 'legacy.png' } } },
        type: 'ip_adapter',
      },
    });
  });

  it('prefers valid ref_images over canvas fallback entries', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues(),
      image,
      kind: 'all',
      metadata: {
        canvas_v2_metadata: {
          referenceImages: {
            entities: [
              {
                id: 'fallback',
                ipAdapter: { image: { height: 1, image_name: 'fallback.png', width: 1 }, type: 'flux_redux' },
              },
            ],
          },
        },
        ref_images: [
          {
            config: {
              image: { height: 64, image_name: 'direct.png', width: 64 },
              type: 'qwen_image_reference_image',
            },
            id: 'direct',
            isEnabled: true,
          },
        ],
      },
      models: [],
      supportedModels: [sdxlModel],
      vaeModels: [],
    });

    expect(result?.values.referenceImages.map((referenceImage) => referenceImage.id)).toEqual(['direct']);
  });

  it('drops recalled reference images when the effective model does not support them', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues({ model: animaModel, modelKey: animaModel.key }),
      image,
      kind: 'all',
      metadata: {
        positive_prompt: 'a recalled prompt',
        ref_images: [
          {
            id: 'ref-1',
            isEnabled: true,
            config: {
              image: {
                height: 770,
                imageName: 'reference.png',
                imageUrl: '/api/v1/images/i/reference.png/full',
                queuedAt: '2026-01-01T00:00:00.000Z',
                sourceQueueItemId: 'backend-gallery',
                thumbnailUrl: '/api/v1/images/i/reference.png/thumbnail',
                width: 513,
              },
              type: 'qwen_image_reference_image',
            },
          },
        ],
      },
      models: [],
      supportedModels: [animaModel],
      vaeModels: [],
    });

    expect(result?.fields).toContain('prompts');
    expect(result?.values.referenceImages).toEqual([]);
    expect(result?.fields).not.toContain('referenceImages');
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
