import type { GalleryImage } from '@features/gallery';
import type {
  ComponentModelConfig,
  GenerateWidgetValues,
  LoraModelConfig,
  MainModelConfig,
  VaeModelConfig,
} from '@features/generation/contracts';

import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities.testing';
import {
  fullyFilledSettingsFor,
  generateGraphCases,
  satisfiedSettingsFor,
} from '@features/generation/core/graphCoverage.testing';
import { compileGenerateGraph } from '@features/generation/graph';
import { normalizeReferenceImages } from '@features/generation/settings';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { buildImageRecallSettings, getImageRecallCapabilities, type ImageRecallKind } from './imageRecall';

const sdxlModel: MainModelConfig = { base: 'sdxl', key: 'sdxl-model', name: 'SDXL', type: 'main' };
const sd1Model: MainModelConfig = { base: 'sd-1', key: 'sd1-model', name: 'SD 1.5', type: 'main' };
const animaModel: MainModelConfig = { base: 'anima', key: 'anima-model', name: 'Anima', type: 'main' };
const krea2Model: MainModelConfig = { base: 'krea-2', key: 'krea2-model', name: 'Krea 2', type: 'main' };
const flux2DevModel: MainModelConfig = {
  base: 'flux2',
  key: 'flux2-dev',
  name: 'FLUX.2 dev',
  type: 'main',
  variant: 'dev',
};
const krea2Qwen3VL: ComponentModelConfig = {
  base: 'any',
  key: 'qwen3-vl-4b',
  name: 'Qwen3-VL 4B',
  type: 'qwen3_vl_encoder',
  variant: 'qwen3_vl_4b',
};
const ideogram4Qwen3VL: ComponentModelConfig = {
  base: 'any',
  key: 'qwen3-vl-8b',
  name: 'Qwen3-VL 8B',
  type: 'qwen3_vl_encoder',
  variant: 'qwen3_vl_8b',
};
const vaeModel: VaeModelConfig = { base: 'sd-1', key: 'vae-model', name: 'SD 1.5 VAE', type: 'vae' };
const qwenImageVae: VaeModelConfig = { base: 'qwen-image', key: 'qwen-vae', name: 'Qwen VAE', type: 'vae' };
const animaQwen3: ComponentModelConfig = {
  base: 'any',
  key: 'anima-qwen3',
  name: 'Anima Qwen3',
  type: 'qwen3_encoder',
  variant: 'qwen3_06b',
};
const mistralEncoder: ComponentModelConfig = {
  base: 'any',
  key: 'mistral',
  name: 'Mistral Encoder',
  type: 'mistral_encoder',
};
const qwenVLEncoder: ComponentModelConfig = {
  base: 'any',
  key: 'qwen-vl',
  name: 'Qwen VL',
  type: 'qwen_vl_encoder',
};

const inkLora: LoraModelConfig = { base: 'sdxl', hash: 'blake3:ink', key: 'ink-lora', name: 'Ink', type: 'lora' };
const grainLora: LoraModelConfig = {
  base: 'sdxl',
  hash: 'blake3:grain',
  key: 'grain-lora',
  name: 'Grain',
  type: 'lora',
};
const staleLora: LoraModelConfig = { base: 'sdxl', key: 'stale-lora', name: 'Stale', type: 'lora' };
const sd1Lora: LoraModelConfig = { base: 'sd-1', key: 'sd1-lora', name: 'Pixel', type: 'lora' };

/** A recorded `ModelIdentifierField`, as the generation graph writes it. */
const recordedLora = (lora: LoraModelConfig, weight: number) => ({
  model: { base: lora.base, hash: lora.hash, key: lora.key, name: lora.name, type: lora.type },
  weight,
});

const createValues = (overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues => ({
  aspectRatioId: '1:1',
  aspectRatioIsLocked: false,
  aspectRatioValue: 1,
  batchCount: 1,
  pidMode: 'off',
  pidDecoderModel: null,
  gemma2EncoderModel: null,
  pidSteps: 4,
  refinerModel: null,
  refinerSteps: 20,
  refinerCfgScale: 7.5,
  refinerScheduler: 'euler',
  refinerPositiveAestheticScore: 6,
  refinerNegativeAestheticScore: 2.5,
  refinerStart: 0.8,
  cfgRescaleMultiplier: 0,
  cfgScale: 7,
  clipEmbedModel: null,
  clipGEmbedModel: null,
  clipLEmbedModel: null,
  clipSkip: 0,
  colorCompensation: false,
  hiDiffusionEnabled: false,
  hiDiffusionRauNetEnabled: true,
  hiDiffusionT1Ratio: 0.4,
  hiDiffusionT2Ratio: 0,
  hiDiffusionWindowAttentionEnabled: true,
  dynamicPromptsCombinatorial: true,
  dynamicPromptsMaxPrompts: 100,
  dynamicPromptsSampleSeed: 0,
  dynamicPromptsResample: true,
  dynamicPromptsSeedBehaviour: 'per-iteration',
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
  expandPromptModelKey: null,
  imageToPromptModelKey: null,
  promptTemplate: null,
  promptTemplateViewMode: false,
  mistralEncoderModel: null,
  qwen3EncoderModel: null,
  qwenVLEncoderModel: null,
  qwen3VLEncoderModel: null,
  qwen35EncoderModel: null,
  animaInpaintAdapterModel: null,
  animaInpaintAdapterWeight: 1,
  wanT5EncoderModel: null,
  wanLowNoiseModel: null,
  ideogram4UnconditionalModel: null,
  wanGuidanceScaleLowNoise: null,
  ideogram4SamplerPreset: 'V4_QUALITY_48',
  ideogram4Steps: null,
  ideogram4GuidanceScale: null,
  ideogram4Mu: null,
  ideogram4ColorPalette: [],
  krea2RebalanceEnabled: false,
  krea2RebalanceMultiplier: 4,
  krea2RebalanceWeights: '1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0',
  krea2SeedVarianceEnabled: false,
  krea2SeedVarianceStrength: 0.1,
  krea2SeedVarianceRandomizePercent: 50,
  fluxDypePreset: 'off',
  fluxDypeScale: 2,
  fluxDypeExponent: 2,
  zImageShift: null,
  zImageSeedVarianceEnabled: false,
  zImageSeedVarianceStrength: 0.1,
  zImageSeedVarianceRandomizePercent: 50,
  qwenImageShift: null,
  qwenImageEncoderQuantization: 'none',
  referenceImages: [],
  scheduler: 'euler_a',
  seamlessXAxis: false,
  seamlessYAxis: false,
  seed: 123,
  seedMode: 'random',
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

seedArchitectureCapabilities();

describe('image recall', () => {
  it.each(['all', 'remix', 'prompts'] as const)(
    'clears the active template and view mode when %s recalls prompt metadata',
    (kind) => {
      const result = buildImageRecallSettings({
        currentValues: createValues({
          promptTemplate: {
            id: 'template-1',
            name: 'Cinematic',
            negativePrompt: 'lowres',
            positivePrompt: '{prompt}, cinematic',
          },
          promptTemplateViewMode: true,
        }),
        image,
        kind,
        metadata: { positive_prompt: 'a recalled prompt' },
        models: [],
        supportedModels: [],
        vaeModels: [],
      });

      expect(result?.values.promptTemplate).toBeNull();
      expect(result?.values.promptTemplateViewMode).toBe(false);
    }
  );

  it.each(['seed', 'dimensions', 'clipSkip'] as const)(
    'preserves the active template and view mode when %s has no prompt metadata',
    (kind) => {
      const currentValues = createValues({
        ...(kind === 'clipSkip' ? { model: sd1Model, modelKey: sd1Model.key } : {}),
        promptTemplate: {
          id: 'template-1',
          name: 'Cinematic',
          negativePrompt: 'lowres',
          positivePrompt: '{prompt}, cinematic',
        },
        promptTemplateViewMode: true,
      });
      const result = buildImageRecallSettings({
        currentValues,
        image,
        kind,
        metadata: kind === 'seed' ? { seed: 42 } : kind === 'clipSkip' ? { clip_skip: 1 } : null,
        models: [],
        supportedModels: [],
        vaeModels: [],
      });

      expect(result?.values.promptTemplate).toEqual(currentValues.promptTemplate);
      expect(result?.values.promptTemplateViewMode).toBe(true);
    }
  );

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
      seedMode: 'fixed',
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
      currentValues: createValues({ seed: 999, seedMode: 'fixed' }),
      image,
      kind: 'remix',
      metadata,
      models: [sd1Model],
      supportedModels: [sd1Model],
      vaeModels: [],
    });

    expect(result?.values.seed).toBe(999);
    expect(result?.values.seedMode).toBe('fixed');
    expect(result?.values.positivePrompt).toBe('a recalled prompt');
    expect(result?.fields).not.toContain('seed');
  });

  it.each(['all', 'remix'] as const)('recalls valid HiDiffusion metadata for supported models with %s', (kind) => {
    const result = buildImageRecallSettings({
      currentValues: createValues(),
      image,
      kind,
      metadata: {
        hidiffusion: true,
        hidiffusion_raunet: false,
        hidiffusion_t1_ratio: 0.35,
        hidiffusion_t2_ratio: 0.15,
        hidiffusion_window_attn: true,
        model: { key: sd1Model.key },
      },
      models: [sd1Model],
      supportedModels: [sd1Model],
      vaeModels: [],
    });

    expect(result?.values).toMatchObject({
      hiDiffusionEnabled: true,
      hiDiffusionRauNetEnabled: false,
      hiDiffusionT1Ratio: 0.35,
      hiDiffusionT2Ratio: 0.15,
      hiDiffusionWindowAttentionEnabled: true,
    });
    expect(result?.fields).toContain('hiDiffusion');
  });

  it('ignores HiDiffusion metadata for unsupported models', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues({ model: animaModel, modelKey: animaModel.key }),
      image,
      kind: 'all',
      metadata: {
        hidiffusion: true,
        hidiffusion_raunet: false,
        hidiffusion_t1_ratio: 0.35,
        hidiffusion_t2_ratio: 0.15,
        hidiffusion_window_attn: true,
        positive_prompt: 'keep the recall non-empty',
      },
      models: [],
      supportedModels: [animaModel],
      vaeModels: [],
    });

    expect(result?.values.hiDiffusionEnabled).toBe(false);
    expect(result?.fields).not.toContain('hiDiffusion');
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

  it('recalls the size for an external generator, which has no architecture row to wait for', () => {
    // External providers have no capability row and must remain eligible for dimension recall.
    const externalModel = {
      base: 'external',
      capabilities: { modes: ['txt2img'], supports_seed: true },
      format: 'external_api',
      key: 'external-model',
      name: 'OpenAI Image',
      provider_id: 'openai',
      type: 'external_image_generator',
    } as GenerateWidgetValues['model'];
    const recall = () =>
      buildImageRecallSettings({
        currentValues: createValues({ model: externalModel, modelKey: externalModel.key }),
        image,
        kind: 'dimensions',
        metadata: null,
        models: [],
        supportedModels: [],
        vaeModels: [],
      });

    expect(recall()?.values).toMatchObject({ height: 768, width: 512 });

    resetArchitectureCapabilities();
    try {
      expect(recall()?.values).toMatchObject({ height: 768, width: 512 });
    } finally {
      setArchitectureCapabilities(architectureCapabilitiesFixture);
    }
  });

  describe('before the capability table arrives', () => {
    // Require the model's grid before persisting recalled dimensions; fallback grid 8 can produce invalid sizes.
    beforeEach(() => {
      resetArchitectureCapabilities();
    });

    afterEach(() => {
      setArchitectureCapabilities(architectureCapabilitiesFixture);
    });

    it('recalls no size from the image rather than one snapped to the fallback grid', () => {
      const result = buildImageRecallSettings({
        currentValues: createValues(),
        image,
        kind: 'dimensions',
        metadata: null,
        models: [],
        supportedModels: [],
        vaeModels: [],
      });

      // Without the fix this is ['size'] with the width snapped to the fallback grid 8.
      expect(result?.fields ?? []).not.toContain('size');
    });

    it('withholds Recall All and Remix but keeps prompts and seed', () => {
      const metadata = { height: 768, model: { key: sdxlModel.key }, positive_prompt: 'a cat', seed: 7, width: 512 };
      const recall = (kind: ImageRecallKind) =>
        buildImageRecallSettings({
          currentValues: createValues(),
          image,
          kind,
          metadata,
          models: [],
          supportedModels: [sdxlModel],
          vaeModels: [],
        });

      // Recall All must not persist fallback defaults while capabilities are unavailable.
      expect(recall('all')).toBeNull();
      expect(recall('remix')).toBeNull();
      expect(recall('prompts')?.fields).toEqual(['prompts']);
      expect(recall('seed')?.fields).toEqual(['seed']);
      expect(
        getImageRecallCapabilities({
          currentValues: createValues(),
          image,
          metadata,
          models: [],
          supportedModels: [sdxlModel],
          vaeModels: [],
        })
      ).toMatchObject({ all: false, clipSkip: false, prompts: true, remix: false, seed: true });
    });
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

  it('recalls FLUX.2 [dev] Mistral encoder metadata', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues(),
      image,
      kind: 'all',
      metadata: { mistral_encoder: { key: mistralEncoder.key }, model: { key: flux2DevModel.key } },
      models: [mistralEncoder],
      supportedModels: [flux2DevModel],
      vaeModels: [],
    });

    expect(result?.values.mistralEncoderModel).toBe(mistralEncoder);
    expect(result?.fields).toContain('components');
  });

  it('remixes a Krea-2 image with its Qwen3-VL encoder', () => {
    const krea2Metadata = { model: { key: krea2Model.key }, qwen3_vl_encoder: { key: krea2Qwen3VL.key } };
    const recallInput = {
      currentValues: createValues(),
      image,
      metadata: krea2Metadata,
      models: [krea2Qwen3VL],
      supportedModels: [krea2Model],
      vaeModels: [],
    };

    expect(buildImageRecallSettings({ ...recallInput, kind: 'remix' })?.values.qwen3VLEncoderModel).toBe(krea2Qwen3VL);
    expect(
      getImageRecallCapabilities({ ...recallInput, metadata: { qwen3_vl_encoder: krea2Metadata.qwen3_vl_encoder } })
        .remix
    ).toBe(true);
  });

  it('keeps a fitting pick when the recorded component does not fit the current model', () => {
    // The image's Ideogram model is gone, so Remix stays on the current Krea-2 model and its 4B encoder.
    const result = buildImageRecallSettings({
      currentValues: createValues({ model: krea2Model, modelKey: krea2Model.key, qwen3VLEncoderModel: krea2Qwen3VL }),
      image,
      kind: 'remix',
      metadata: {
        model: { key: 'uninstalled-ideogram' },
        positive_prompt: 'a lighthouse',
        qwen3_vl_encoder: { key: ideogram4Qwen3VL.key },
      },
      models: [krea2Qwen3VL, ideogram4Qwen3VL],
      supportedModels: [krea2Model],
      vaeModels: [],
    });

    expect(result?.values.qwen3VLEncoderModel).toEqual(krea2Qwen3VL);
    expect(result?.fields).toEqual(['prompts']);
  });

  it('drops components that do not fit the remixed model, recorded or left over', () => {
    const recall = (metadata: Record<string, unknown>) =>
      buildImageRecallSettings({
        // Ideogram 4's 8B encoder does not fit Krea-2, which takes the 4B variant.
        currentValues: createValues({ qwen3VLEncoderModel: ideogram4Qwen3VL }),
        image,
        kind: 'remix',
        metadata: { model: { key: krea2Model.key }, ...metadata },
        models: [krea2Qwen3VL, ideogram4Qwen3VL],
        supportedModels: [krea2Model],
        vaeModels: [],
      })?.values.qwen3VLEncoderModel;

    expect(recall({})).toBeNull();
    expect(recall({ qwen3_vl_encoder: { key: ideogram4Qwen3VL.key } })).toBeNull();
  });

  it('clears component models when recalled metadata explicitly stores null', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues({
        componentSourceModel: sd1Model,
        mistralEncoderModel: mistralEncoder,
        qwen3EncoderModel: animaQwen3,
        qwenVLEncoderModel: qwenVLEncoder,
      }),
      image,
      kind: 'all',
      metadata: {
        mistral_encoder: null,
        qwen3_encoder: null,
        qwen3_source: null,
        qwen_image_qwen_vl_encoder: null,
      },
      models: [sd1Model, mistralEncoder, animaQwen3, qwenVLEncoder],
      supportedModels: [],
      vaeModels: [],
    });

    expect(result?.values.componentSourceModel).toBeNull();
    expect(result?.values.mistralEncoderModel).toBeNull();
    expect(result?.values.qwen3EncoderModel).toBeNull();
    expect(result?.values.qwenVLEncoderModel).toBeNull();
    expect(result?.fields).toContain('components');
  });

  describe('concepts', () => {
    const loraMetadata = {
      loras: [recordedLora(grainLora, 0.35), recordedLora(inkLora, 1.4)],
      model: { key: sdxlModel.key },
      positive_prompt: 'ink and grain',
      steps: 30,
    };
    const recallConcepts = (
      kind: ImageRecallKind,
      recordedMetadata: unknown,
      overrides: Partial<GenerateWidgetValues> = {},
      models: ComponentModelConfig[] = [sdxlModel, inkLora, grainLora, staleLora, sd1Lora]
    ) =>
      buildImageRecallSettings({
        currentValues: createValues({ loras: [{ isEnabled: true, model: staleLora, weight: 0.9 }], ...overrides }),
        image,
        kind,
        metadata: recordedMetadata,
        models,
        supportedModels: models.filter((model): model is MainModelConfig => model.type === 'main'),
        vaeModels: [],
      });

    it.each(['all', 'remix'] as const)('%s replaces the selected concepts with the recorded ones', (kind) => {
      const result = recallConcepts(kind, loraMetadata);

      expect(result?.values.loras).toEqual([
        { isEnabled: true, model: grainLora, weight: 0.35 },
        { isEnabled: true, model: inkLora, weight: 1.4 },
      ]);
      expect(result?.fields).toContain('loras');
      expect(result?.skipped).toEqual([]);
    });

    it('clears stale concepts when a generation record lists none', () => {
      const result = recallConcepts('all', {
        generation_mode: 'sdxl_txt2img',
        model: { key: sdxlModel.key },
        steps: 30,
      });

      expect(result?.values.loras).toEqual([]);
      expect(result?.fields).toContain('loras');
    });

    it('keeps the selected concepts for a canvas save, which names a model but never records concepts', () => {
      const result = recallConcepts('all', {
        height: 1024,
        model: { base: 'sdxl', key: sdxlModel.key, name: sdxlModel.name, type: 'main' },
        positive_prompt: 'saved from canvas',
        seed: 9,
        width: 1024,
      });

      expect(result?.values.loras).toEqual([{ isEnabled: true, model: staleLora, weight: 0.9 }]);
      expect(result?.fields).not.toContain('loras');
    });

    it('keeps the selected concepts and reports the recorded ones when the image model is not installed', () => {
      const result = recallConcepts('all', {
        ...loraMetadata,
        generation_mode: 'sdxl_txt2img',
        model: { base: 'sdxl', key: 'uninstalled-model', name: 'Gone', type: 'main' },
      });

      expect(result?.values.loras).toEqual([{ isEnabled: true, model: staleLora, weight: 0.9 }]);
      expect(result?.fields).not.toContain('loras');
      expect(result?.skipped).toEqual([
        { name: 'Grain', reason: 'modelUnavailable', subject: 'concept' },
        { name: 'Ink', reason: 'modelUnavailable', subject: 'concept' },
      ]);
    });

    it('keeps the selected concepts when the image model is the uninstalled one the panel still names', () => {
      const uninstalled: MainModelConfig = { base: 'sdxl', key: 'uninstalled-model', name: 'Gone', type: 'main' };
      const result = recallConcepts(
        'all',
        { generation_mode: 'sdxl_txt2img', model: { key: uninstalled.key }, positive_prompt: 'from the old model' },
        { model: uninstalled, modelKey: uninstalled.key }
      );

      expect(result?.values.loras).toEqual([{ isEnabled: true, model: staleLora, weight: 0.9 }]);
      expect(result?.fields).not.toContain('loras');
    });

    it('restores weights beyond the panel input range exactly', () => {
      const result = recallConcepts('all', {
        loras: [recordedLora(inkLora, 12), recordedLora(grainLora, -10.5)],
        model: { key: sdxlModel.key },
      });

      expect(result?.values.loras.map((lora) => lora.weight)).toEqual([12, -10.5]);
      expect(result?.skipped).toEqual([]);
    });

    it('restores concepts from records that name the LoRA under `lora`', () => {
      const result = recallConcepts('remix', {
        loras: [{ lora: { key: inkLora.key }, weight: 0.45 }],
        model: { key: sdxlModel.key },
      });

      expect(result?.values.loras).toEqual([{ isEnabled: true, model: inkLora, weight: 0.45 }]);
    });

    it('keeps the selected concepts when the metadata does not record a generation', () => {
      const result = recallConcepts('remix', { positive_prompt: 'prompt only' });

      expect(result?.values.loras).toEqual([{ isEnabled: true, model: staleLora, weight: 0.9 }]);
      expect(result?.fields).not.toContain('loras');
    });

    it.each(['prompts', 'seed', 'dimensions'] as const)('%s recall leaves the selected concepts alone', (kind) => {
      const result = recallConcepts(kind, { ...loraMetadata, seed: 7 });

      expect(result?.values.loras).toEqual([{ isEnabled: true, model: staleLora, weight: 0.9 }]);
      expect(result?.fields).not.toContain('loras');
      expect(result?.skipped).toEqual([]);
    });

    it('restores zero and negative weights as recorded, enabled', () => {
      const result = recallConcepts('all', {
        loras: [recordedLora(inkLora, 0), recordedLora(grainLora, -0.75)],
        model: { key: sdxlModel.key },
      });

      expect(result?.values.loras).toEqual([
        { isEnabled: true, model: inkLora, weight: 0 },
        { isEnabled: true, model: grainLora, weight: -0.75 },
      ]);
    });

    it('skips missing, incompatible, malformed, and duplicate entries and reports each', () => {
      const result = recallConcepts('all', {
        loras: [
          recordedLora(inkLora, 0.5),
          { model: { base: 'sdxl', key: 'deleted-lora', name: 'Deleted', type: 'lora' }, weight: 1 },
          recordedLora(sd1Lora, 0.8),
          { model: { key: grainLora.key } },
          'not an entry',
          recordedLora(inkLora, 0.9),
        ],
        model: { key: sdxlModel.key },
      });

      expect(result?.values.loras).toEqual([{ isEnabled: true, model: inkLora, weight: 0.5 }]);
      expect(result?.skipped).toEqual([
        { name: 'Deleted', reason: 'unresolved', subject: 'concept' },
        { name: 'Pixel', reason: 'incompatible', subject: 'concept' },
        { name: null, reason: 'invalid', subject: 'concept' },
        { name: null, reason: 'invalid', subject: 'concept' },
        { name: 'Ink', reason: 'duplicate', subject: 'concept' },
      ]);
    });

    it('keeps the selected concepts and reports a concept list it cannot read', () => {
      const result = recallConcepts('all', { loras: 'ink', model: { key: sdxlModel.key }, steps: 20 });

      expect(result?.values.loras).toEqual([{ isEnabled: true, model: staleLora, weight: 0.9 }]);
      expect(result?.skipped).toEqual([{ name: null, reason: 'invalid', subject: 'concept' }]);
      expect(result?.fields).toContain('steps');
    });

    it('judges compatibility against the recalled model, not the one it replaces', () => {
      const result = recallConcepts(
        'all',
        {
          loras: [recordedLora(sd1Lora, 0.6), recordedLora(inkLora, 0.4)],
          model: { key: sd1Model.key },
        },
        {},
        [sd1Model, sdxlModel, sd1Lora, inkLora, staleLora]
      );

      expect(result?.values.model).toEqual(sd1Model);
      expect(result?.values.loras).toEqual([{ isEnabled: true, model: sd1Lora, weight: 0.6 }]);
      expect(result?.skipped).toEqual([{ name: 'Ink', reason: 'incompatible', subject: 'concept' }]);
    });

    it('resolves a reinstalled LoRA by its content hash, or by name and base when no hash was recorded', () => {
      const reinstalledInk = { ...inkLora, key: 'reinstalled-ink' };
      const result = recallConcepts(
        'all',
        {
          loras: [
            { model: { ...recordedLora(inkLora, 1).model, key: 'old-ink-key' }, weight: 0.3 },
            { model: { base: 'sdxl', key: 'old-grain-key', name: 'Grain', type: 'lora' }, weight: 0.7 },
          ],
          model: { key: sdxlModel.key },
        },
        {},
        [sdxlModel, reinstalledInk, grainLora]
      );

      expect(result?.values.loras).toEqual([
        { isEnabled: true, model: reinstalledInk, weight: 0.3 },
        { isEnabled: true, model: grainLora, weight: 0.7 },
      ]);
    });

    it('does not pick between installed LoRAs a portable identifier matches equally', () => {
      const result = recallConcepts(
        'all',
        {
          loras: [{ model: { ...recordedLora(inkLora, 1).model, key: 'old-ink-key' }, weight: 1 }],
          model: { key: sdxlModel.key },
        },
        {},
        [sdxlModel, { ...inkLora, key: 'ink-copy-1' }, { ...inkLora, key: 'ink-copy-2' }]
      );

      expect(result?.values.loras).toEqual([]);
      expect(result?.skipped).toEqual([{ name: 'Ink', reason: 'ambiguous', subject: 'concept' }]);
    });

    it('does not substitute a same-named LoRA whose content differs from the recorded hash', () => {
      const result = recallConcepts(
        'all',
        {
          loras: [{ model: { ...recordedLora(inkLora, 1).model, key: 'old-ink-key' }, weight: 1 }],
          model: { key: sdxlModel.key },
        },
        {},
        [sdxlModel, { ...inkLora, hash: 'blake3:retrained', key: 'retrained-ink' }]
      );

      expect(result?.skipped).toEqual([{ name: 'Ink', reason: 'unresolved', subject: 'concept' }]);
    });

    it('offers Recall All and Remix for metadata that records only concepts it can restore', () => {
      const capabilitiesFor = (models: ComponentModelConfig[]) =>
        getImageRecallCapabilities({
          currentValues: createValues(),
          image,
          metadata: { loras: [recordedLora(inkLora, 0.5)] },
          models,
          supportedModels: [],
          vaeModels: [],
        });

      expect(capabilitiesFor([inkLora])).toMatchObject({ all: true, remix: true });
      expect(capabilitiesFor([sd1Lora])).toMatchObject({ all: false, remix: false });
    });

    it('submits the restored concepts at their recorded weights', () => {
      const { model, settings } = satisfiedSettingsFor('sdxl', {
        label: 'diffusers',
        overrides: { format: 'diffusers' },
      });
      const generatedWith = {
        ...settings,
        loras: [
          { isEnabled: true, model: grainLora, weight: -0.4 },
          { isEnabled: false, model: staleLora, weight: 1 },
          { isEnabled: true, model: inkLora, weight: 1.25 },
        ],
      };
      const loraWeights = (backendGraph: { nodes: Record<string, Record<string, unknown>> }) =>
        Object.values(backendGraph.nodes)
          .filter((node) => node.type === 'lora_selector')
          .map((node) => [(node.lora as { key: string }).key, node.weight]);
      const generated = compileGenerateGraph(generatedWith, model, 'gallery', {
        useCpuNoise: true,
        useNsfwChecker: false,
        useWatermark: false,
      }).backendGraph;
      const result = buildImageRecallSettings({
        currentValues: createValues({ loras: [{ isEnabled: true, model: staleLora, weight: 0.9 }], model: sd1Model }),
        image,
        kind: 'all',
        metadata: Object.values(generated.nodes).find((node) => node.type === 'core_metadata'),
        models: [model, inkLora, grainLora, staleLora],
        supportedModels: [model],
        vaeModels: [],
      });
      const recalled = compileGenerateGraph(result!.values, result!.values.model, 'gallery', {
        useCpuNoise: true,
        useNsfwChecker: false,
        useWatermark: false,
      });

      expect(loraWeights(recalled.backendGraph)).toEqual([
        [grainLora.key, -0.4],
        [inkLora.key, 1.25],
      ]);
      expect(loraWeights(recalled.backendGraph)).toEqual(loraWeights(generated));
    });
  });

  describe('krea-2 conditioning rebalance', () => {
    const krea2Metadata = {
      krea2_rebalance_enabled: true,
      krea2_rebalance_multiplier: 6.5,
      krea2_rebalance_weights: '1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0',
      model: { key: krea2Model.key },
    };

    it('restores the curve and gain a Krea-2 image was generated with', () => {
      const result = buildImageRecallSettings({
        currentValues: createValues(),
        image,
        kind: 'all',
        metadata: krea2Metadata,
        models: [],
        supportedModels: [krea2Model],
        vaeModels: [],
      });

      expect(result?.values.krea2RebalanceEnabled).toBe(true);
      expect(result?.values.krea2RebalanceMultiplier).toBe(6.5);
      expect(result?.values.krea2RebalanceWeights).toBe('1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0');
      expect(result?.fields).toContain('krea2Rebalance');
    });

    it('recalls the rebalance on a remix too', () => {
      const result = buildImageRecallSettings({
        currentValues: createValues(),
        image,
        kind: 'remix',
        metadata: krea2Metadata,
        models: [],
        supportedModels: [krea2Model],
        vaeModels: [],
      });

      expect(result?.fields).toContain('krea2Rebalance');
    });

    it('ignores the rebalance when the recalled model is not Krea-2', () => {
      const result = buildImageRecallSettings({
        currentValues: createValues(),
        image,
        kind: 'all',
        metadata: { ...krea2Metadata, model: { key: sdxlModel.key } },
        models: [],
        supportedModels: [sdxlModel],
        vaeModels: [],
      });

      expect(result?.values.krea2RebalanceEnabled).toBe(false);
      expect(result?.fields).not.toContain('krea2Rebalance');
    });

    it('drops a weights string the backend would reject', () => {
      // The string is forwarded to the node verbatim, so a malformed blob must not
      // become a queued item that fails mid-generation.
      const result = buildImageRecallSettings({
        currentValues: createValues(),
        image,
        kind: 'all',
        metadata: { ...krea2Metadata, krea2_rebalance_weights: '0x10,1,1,1,1,1,1,1,1,1,1,1' },
        models: [],
        supportedModels: [krea2Model],
        vaeModels: [],
      });

      expect(result?.values.krea2RebalanceWeights).toBe(createValues().krea2RebalanceWeights);
      // The enabled flag and gain are still sound on their own.
      expect(result?.values.krea2RebalanceMultiplier).toBe(6.5);
    });

    it('offers recall for an image whose only recallable delta is the rebalance', () => {
      const capabilities = getImageRecallCapabilities({
        currentValues: createValues({ model: krea2Model }),
        image,
        metadata: { krea2_rebalance_enabled: true },
        models: [],
        supportedModels: [],
        vaeModels: [],
      });

      expect(capabilities.all).toBe(true);
      expect(capabilities.remix).toBe(true);
    });
  });
});

describe('model-family tuning recall', () => {
  const fluxModel: MainModelConfig = { base: 'flux', key: 'flux-model', name: 'FLUX', type: 'main' };
  const zImageModel: MainModelConfig = { base: 'z-image', key: 'z-model', name: 'Z-Image', type: 'main' };
  const qwenModel: MainModelConfig = { base: 'qwen-image', key: 'qwen-model', name: 'Qwen', type: 'main' };
  const recall = (
    metadata: Record<string, unknown>,
    model: MainModelConfig,
    kind: ImageRecallKind = 'all',
    current: Partial<GenerateWidgetValues> = {}
  ) =>
    buildImageRecallSettings({
      currentValues: createValues(current),
      image,
      kind,
      metadata: { model: { key: model.key }, ...metadata },
      models: [],
      supportedModels: [model],
      vaeModels: [],
    });

  it('restores the DyPE recipe a FLUX image was generated with', () => {
    const result = recall({ dype_exponent: 0.5, dype_preset: 'manual', dype_scale: 3 }, fluxModel, 'remix');

    expect(result?.values).toMatchObject({ fluxDypeExponent: 0.5, fluxDypePreset: 'manual', fluxDypeScale: 3 });
    expect(result?.fields).toContain('modelTuning');
  });

  it('restores the Z-Image shift and seed variance, reading auto back as no override', () => {
    const fixed = recall(
      {
        z_image_seed_variance_enabled: true,
        z_image_seed_variance_randomize_percent: 25,
        z_image_seed_variance_strength: 0.3,
        z_image_shift: 2.5,
      },
      zImageModel
    );
    const auto = recall({ z_image_shift: 'auto' }, zImageModel, 'all', { zImageShift: 4 });

    expect(fixed?.values).toMatchObject({
      zImageSeedVarianceEnabled: true,
      zImageSeedVarianceRandomizePercent: 25,
      zImageSeedVarianceStrength: 0.3,
      zImageShift: 2.5,
    });
    expect(auto?.values.zImageShift).toBeNull();
    expect(auto?.fields).toContain('modelTuning');
  });

  it('does not count tuning the recall leaves as it was', () => {
    const unchanged = recall({ dype_preset: 'off', generation_mode: 'flux_txt2img', seed: 7 }, fluxModel);

    expect(unchanged?.values.fluxDypePreset).toBe('off');
    expect(unchanged?.fields).not.toContain('modelTuning');
  });

  it('restores the Qwen-Image shift and encoder quantization', () => {
    const result = recall({ qwen_image_quantization: 'nf4', qwen_image_shift: 3 }, qwenModel);

    expect(result?.values).toMatchObject({ qwenImageEncoderQuantization: 'nf4', qwenImageShift: 3 });
  });

  it('ignores tuning recorded for another family and values outside the node bounds', () => {
    const wrongFamily = recall({ dype_preset: 'manual', z_image_shift: 2 }, qwenModel);
    const outOfRange = recall({ dype_preset: 'turbo', dype_scale: 9 }, fluxModel);

    expect(wrongFamily?.values.fluxDypePreset).toBe('off');
    expect(wrongFamily?.values.zImageShift).toBeNull();
    expect(wrongFamily?.fields ?? []).not.toContain('modelTuning');
    expect(outOfRange?.values).toMatchObject({ fluxDypePreset: 'off', fluxDypeScale: 2 });
  });

  describe('over tuning the panel has since changed', () => {
    const changedTuning = {
      fluxDypePreset: '4k',
      qwenImageEncoderQuantization: 'nf4',
      qwenImageShift: 5,
      zImageSeedVarianceEnabled: true,
      zImageShift: 5,
    } as const;
    const recallOverChanged = (metadata: Record<string, unknown>, model: MainModelConfig) =>
      buildImageRecallSettings({
        currentValues: createValues(changedTuning),
        image,
        kind: 'all',
        metadata,
        models: [],
        supportedModels: [model],
        vaeModels: [],
      })?.values;
    const record = (model: MainModelConfig, extra: Record<string, unknown> = {}) => ({
      generation_mode: `${model.base}_txt2img`,
      model: { base: model.base, key: model.key, name: model.name, type: 'main' },
      ...extra,
    });

    it('restores the defaults a record of the same family leaves out', () => {
      expect(recallOverChanged(record(fluxModel), fluxModel)?.fluxDypePreset).toBe('off');
      expect(recallOverChanged(record(zImageModel), zImageModel)).toMatchObject({
        zImageSeedVarianceEnabled: false,
        zImageShift: null,
      });
      expect(recallOverChanged(record(qwenModel), qwenModel)).toMatchObject({
        qwenImageEncoderQuantization: 'none',
        qwenImageShift: null,
      });
    });

    it('reads the legacy null shift as auto', () => {
      expect(recallOverChanged(record(qwenModel, { qwen_image_shift: null }), qwenModel)?.qwenImageShift).toBeNull();
    });

    it('keeps the current tuning when the record is not a generation of that family', () => {
      const { generation_mode: _mode, ...withoutMode } = record(fluxModel);
      const otherFamily = record(fluxModel, {
        model: { base: 'sdxl', key: fluxModel.key, name: 'SDXL', type: 'main' },
      });

      expect(recallOverChanged(withoutMode, fluxModel)?.fluxDypePreset).toBe('4k');
      expect(recallOverChanged(otherFamily, fluxModel)?.fluxDypePreset).toBe('4k');
    });

    // What a graph records must bring the panel back to that recipe from either direction. A switch the graph stops
    // recording still passes here, since recall reads a missing key as the default; graph.test.ts pins that FLUX
    // records DyPE off.
    const tunedBases = generateGraphCases.filter(({ base }) => ['flux', 'qwen-image', 'z-image'].includes(base));
    const defaultTuning = {
      fluxDypePreset: 'off',
      qwenImageEncoderQuantization: 'none',
      qwenImageShift: null,
      zImageSeedVarianceEnabled: false,
      zImageShift: null,
    } as const;
    const roundTrip = ({ base, shape }: (typeof generateGraphCases)[number], recorded: object, current: object) => {
      const { model, settings } = fullyFilledSettingsFor(base, shape);
      const graph = compileGenerateGraph({ ...settings, ...recorded }, model, 'gallery', {
        useCpuNoise: true,
        useNsfwChecker: false,
        useWatermark: false,
      }).backendGraph;
      const values = buildImageRecallSettings({
        currentValues: createValues({ ...current, model }),
        image,
        kind: 'all',
        metadata: Object.values(graph.nodes).find((node) => node.type === 'core_metadata'),
        models: [],
        supportedModels: [model],
        vaeModels: [],
      })?.values;
      const keys = {
        flux: ['fluxDypePreset'],
        'qwen-image': ['qwenImageEncoderQuantization', 'qwenImageShift'],
        'z-image': ['zImageSeedVarianceEnabled', 'zImageShift'],
      }[base as 'flux' | 'qwen-image' | 'z-image'] as (keyof typeof defaultTuning)[];

      return {
        expected: Object.fromEntries(keys.map((key) => [key, (recorded as typeof defaultTuning)[key]])),
        restored: Object.fromEntries(keys.map((key) => [key, values?.[key]])),
      };
    };

    it.each(tunedBases)('$label restores its default tuning over changed settings', (graphCase) => {
      const { expected, restored } = roundTrip(graphCase, defaultTuning, changedTuning);

      expect(restored).toEqual(expected);
    });

    it.each(tunedBases)('$label restores its changed tuning over the defaults', (graphCase) => {
      const { expected, restored } = roundTrip(graphCase, changedTuning, defaultTuning);

      expect(restored).toEqual(expected);
    });
  });

  it('offers recall for an image whose only recallable delta is its tuning', () => {
    const capabilities = getImageRecallCapabilities({
      currentValues: createValues({ model: fluxModel }),
      image,
      metadata: { dype_preset: 'auto' },
      models: [],
      supportedModels: [],
      vaeModels: [],
    });

    expect(capabilities.all).toBe(true);
    expect(capabilities.remix).toBe(true);
  });

  it('restores a Krea-2 style reference with its strength through the recorded reference images', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues(),
      image,
      kind: 'all',
      metadata: {
        krea2_style_strength: 1.4,
        model: { key: krea2Model.key },
        ref_images: [
          {
            config: {
              image: { original: { image: { height: 512, image_name: 'style.png', width: 512 } } },
              styleStrength: 1.4,
              type: 'krea2_reference_image',
            },
            id: 'style',
            isEnabled: true,
          },
        ],
      },
      models: [],
      supportedModels: [krea2Model],
      vaeModels: [],
    });

    expect(result?.values.referenceImages).toEqual([
      {
        config: {
          image: { original: { image: { height: 512, image_name: 'style.png', width: 512 } } },
          styleStrength: 1.4,
          type: 'krea2_reference_image',
        },
        id: 'style',
        isEnabled: true,
      },
    ]);
    expect(result?.fields).toContain('referenceImages');
  });
});

describe('Krea-2 style reference without an image', () => {
  it('recalls the rest of the record and leaves reference images alone', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues(),
      image,
      kind: 'all',
      metadata: {
        krea2_style_strength: 1.4,
        model: { key: krea2Model.key },
        ref_images: [
          {
            config: { image: null, styleStrength: 1.4, type: 'krea2_reference_image' },
            id: 'style',
            isEnabled: true,
          },
        ],
        steps: 8,
      },
      models: [],
      supportedModels: [krea2Model],
      vaeModels: [],
    });

    expect(result?.fields).toEqual(expect.arrayContaining(['model', 'steps']));
    expect(result?.fields).not.toContain('referenceImages');
    expect(result?.values.referenceImages).toEqual([]);
  });
});

/** Remix from the metadata a fully filled graph records, starting from SDXL defaults. */
const remixRoundTrip = ({ base, shape }: (typeof generateGraphCases)[number]) => {
  const { filled, model, settings } = fullyFilledSettingsFor(base, shape);
  const graph = compileGenerateGraph(settings, model, 'gallery', {
    useCpuNoise: true,
    useNsfwChecker: false,
    useWatermark: false,
  }).backendGraph;
  const recordedMetadata = Object.values(graph.nodes).find((node) => node.type === 'core_metadata');
  const serializedMetadata = JSON.stringify(recordedMetadata);
  // A slot the graph leaves unused (e.g. a bundled model's) is not recorded, so there is nothing to restore.
  const recorded = filled.filter((key) => serializedMetadata.includes(`"${settings[key]?.key}"`));
  const components = recorded.map((key) => settings[key]).filter((component) => component !== null);
  const result = buildImageRecallSettings({
    currentValues: createValues(),
    image,
    kind: 'remix',
    metadata: recordedMetadata,
    models: components.filter((component) => component.type !== 'vae'),
    supportedModels: [model],
    vaeModels: components.filter((component): component is VaeModelConfig => component.type === 'vae'),
  });
  const keysOf = (values: GenerateWidgetValues | undefined) =>
    Object.fromEntries(recorded.map((key) => [key, values?.[key]?.key ?? null]));

  return { expected: keysOf(settings as GenerateWidgetValues), recorded, restored: keysOf(result?.values) };
};

describe('anima inpaint adapter', () => {
  const inpaintAdapter: ComponentModelConfig = {
    base: 'anima',
    cond_in_channels: 4,
    key: 'anima-inpaint',
    name: 'Anima LLLite Inpainting',
    type: 'controlnet',
  };
  const canvasInpaintMetadata = {
    anima_inpaint_adapter: { key: inpaintAdapter.key },
    anima_inpaint_adapter_weight: 0.7,
    model: { key: animaModel.key },
    qwen3_encoder: { key: animaQwen3.key },
    vae: { key: qwenImageVae.key },
  };
  const models = [animaModel, animaQwen3, qwenImageVae, inpaintAdapter];

  it.each(['all', 'remix'] as const)(
    'restores the adapter and its weight from a canvas inpaint image on %s',
    (kind) => {
      const result = buildImageRecallSettings({
        currentValues: createValues(),
        image,
        kind,
        metadata: canvasInpaintMetadata,
        models,
        supportedModels: [animaModel],
        vaeModels: [qwenImageVae],
      });

      expect(result?.values.model).toBe(animaModel);
      expect(result?.values.animaInpaintAdapterModel).toBe(inpaintAdapter);
      expect(result?.values.animaInpaintAdapterWeight).toBe(0.7);
      expect(result?.fields).toEqual(expect.arrayContaining(['components', 'animaInpaintAdapterWeight']));
    }
  );

  it('leaves the slot alone but still recalls the weight when the recorded adapter is not installed', () => {
    const result = buildImageRecallSettings({
      currentValues: createValues({ animaInpaintAdapterModel: inpaintAdapter, animaInpaintAdapterWeight: 1 }),
      image,
      kind: 'all',
      metadata: { ...canvasInpaintMetadata, anima_inpaint_adapter: { key: 'gone' } },
      models: [animaModel, animaQwen3, qwenImageVae],
      supportedModels: [animaModel],
      vaeModels: [qwenImageVae],
    });

    // Recall clones the current values, so the untouched slot keeps the same pick by value.
    expect(result?.values.animaInpaintAdapterModel).toEqual(inpaintAdapter);
    expect(result?.values.animaInpaintAdapterWeight).toBe(0.7);
    expect(result?.fields).toContain('animaInpaintAdapterWeight');
  });

  it('clears a stale adapter from an Anima inpaint record that lacks the key, which the backend persists for none', () => {
    const currentValues = createValues({ animaInpaintAdapterModel: inpaintAdapter, animaInpaintAdapterWeight: 0.5 });
    const {
      anima_inpaint_adapter: _adapter,
      anima_inpaint_adapter_weight: _weight,
      ...withoutAdapter
    } = canvasInpaintMetadata;
    const recall = (metadata: Record<string, unknown>, supportedModels = [animaModel]) =>
      buildImageRecallSettings({
        currentValues,
        image,
        kind: 'all',
        metadata,
        models: [...models, sd1Model],
        supportedModels,
        vaeModels: [qwenImageVae],
      });

    for (const generation_mode of ['anima_inpaint', 'anima_outpaint']) {
      const cleared = recall({ ...withoutAdapter, generation_mode });
      expect(cleared?.values.animaInpaintAdapterModel).toBeNull();
      expect(cleared?.values.animaInpaintAdapterWeight).toBe(0.5);
      expect(cleared?.fields).toContain('components');
      expect(cleared?.fields).not.toContain('animaInpaintAdapterWeight');
    }

    // A record from a mode that never writes the key, or none at all, says nothing about the adapter.
    expect(recall(withoutAdapter)?.values.animaInpaintAdapterModel).toEqual(inpaintAdapter);
    expect(recall({ ...withoutAdapter, generation_mode: 'anima_img2img' })?.values.animaInpaintAdapterModel).toEqual(
      inpaintAdapter
    );
    const notAnima = recall({ generation_mode: 'inpaint', model: { key: sd1Model.key } }, [sd1Model, animaModel]);
    expect(notAnima?.values.model).toBe(sd1Model);
    expect(notAnima?.fields).not.toContain('components');

    // The key present on such a record still sets the adapter.
    const restored = recall({ ...canvasInpaintMetadata, generation_mode: 'anima_inpaint' });
    expect(restored?.values.animaInpaintAdapterModel).toBe(inpaintAdapter);
    expect(restored?.values.animaInpaintAdapterWeight).toBe(0.7);
  });

  it.each([
    [25, 10],
    [-25, -10],
  ])(
    'clamps a recorded weight of %s to %s instead of applying a value canvas validation rejects',
    (recorded, applied) => {
      const result = buildImageRecallSettings({
        currentValues: createValues(),
        image,
        kind: 'all',
        metadata: { ...canvasInpaintMetadata, anima_inpaint_adapter_weight: recorded },
        models,
        supportedModels: [animaModel],
        vaeModels: [qwenImageVae],
      });

      expect(result?.values.animaInpaintAdapterWeight).toBe(applied);
      expect(result?.fields).toContain('animaInpaintAdapterWeight');
    }
  );

  it('skips the adapter and weight when the image is not an Anima image or the adapter is a control adapter', () => {
    const notAnima = buildImageRecallSettings({
      currentValues: createValues({ animaInpaintAdapterWeight: 1 }),
      image,
      kind: 'all',
      metadata: { ...canvasInpaintMetadata, model: { key: sd1Model.key } },
      models: [sd1Model, inpaintAdapter],
      supportedModels: [sd1Model],
      vaeModels: [],
    });
    expect(notAnima?.values.animaInpaintAdapterModel).toBeNull();
    expect(notAnima?.values.animaInpaintAdapterWeight).toBe(1);
    expect(notAnima?.fields).not.toContain('animaInpaintAdapterWeight');

    const controlAdapter = { ...inpaintAdapter, cond_in_channels: 3 };
    const wrongChannels = buildImageRecallSettings({
      currentValues: createValues(),
      image,
      kind: 'all',
      metadata: canvasInpaintMetadata,
      models: [animaModel, animaQwen3, qwenImageVae, controlAdapter],
      supportedModels: [animaModel],
      vaeModels: [qwenImageVae],
    });
    expect(wrongChannels?.values.animaInpaintAdapterModel).toBeNull();
  });
});

describe('remix round trip', () => {
  // Every component a graph records must come back; one recorded under a key Remix does not know fails here.
  it.each(generateGraphCases)('restores the components $label records', (graphCase) => {
    const { expected, restored } = remixRoundTrip(graphCase);

    expect(restored).toEqual(expected);
  });

  it('exercises every component setting Remix restores', () => {
    const recorded = new Set(generateGraphCases.flatMap((graphCase) => remixRoundTrip(graphCase).recorded));

    expect([...recorded].sort()).toEqual([
      'clipEmbedModel',
      'componentSourceModel',
      'gemma2EncoderModel',
      'ideogram4UnconditionalModel',
      'mistralEncoderModel',
      'pidDecoderModel',
      'qwen35EncoderModel',
      'qwen3EncoderModel',
      'qwen3VLEncoderModel',
      'qwenVLEncoderModel',
      't5EncoderModel',
      'vae',
      'wanLowNoiseModel',
      'wanT5EncoderModel',
    ]);
  });
});

describe('SDXL Refiner recall', () => {
  const refiner: MainModelConfig = {
    base: 'sdxl-refiner',
    hash: 'blake3:refiner',
    key: 'refiner-installed',
    name: 'SDXL Refiner',
    type: 'main',
  };
  const recordedRefiner = {
    refiner_cfg_scale: 5,
    refiner_model: { base: 'sdxl-refiner', hash: 'blake3:refiner', key: 'refiner-recorded', name: 'SDXL Refiner' },
    refiner_negative_aesthetic_score: 2,
    refiner_positive_aesthetic_score: 7,
    refiner_scheduler: 'dpmpp_2m',
    refiner_start: 0.7,
    refiner_steps: 12,
  };
  const currentRefinerSettings = {
    refinerCfgScale: 8,
    refinerModel: refiner,
    refinerNegativeAestheticScore: 3,
    refinerPositiveAestheticScore: 9,
    refinerScheduler: 'euler',
    refinerStart: 0.9,
    refinerSteps: 25,
  };
  const recall = (
    metadata: Record<string, unknown>,
    models: ComponentModelConfig[],
    kind: ImageRecallKind = 'all',
    currentValues = createValues()
  ) =>
    buildImageRecallSettings({
      currentValues,
      image,
      kind,
      metadata: { model: { key: sdxlModel.key }, ...metadata },
      models,
      supportedModels: [sdxlModel],
      vaeModels: [],
    });

  it.each(['all', 'remix'] as const)('restores the refiner and its tuning with %s, resolving by hash', (kind) => {
    const result = recall(recordedRefiner, [refiner], kind);

    expect(result?.values).toMatchObject({
      refinerCfgScale: 5,
      refinerModel: refiner,
      refinerNegativeAestheticScore: 2,
      refinerPositiveAestheticScore: 7,
      refinerScheduler: 'dpmpp_2m',
      refinerStart: 0.7,
      refinerSteps: 12,
    });
    expect(result?.fields).toContain('refiner');
  });

  it('resolves the refiner by key, then by name and base, and only when that is unambiguous', () => {
    const byKey = { ...refiner, hash: 'blake3:other', key: 'refiner-recorded' };

    expect(recall(recordedRefiner, [byKey, refiner])?.values.refinerModel).toEqual(byKey);

    const unhashed = { ...recordedRefiner, refiner_model: { base: 'sdxl-refiner', name: 'SDXL Refiner' } };

    expect(recall(unhashed, [refiner])?.values.refinerModel).toEqual(refiner);
    expect(recall(unhashed, [refiner])?.skipped).toEqual([]);
  });

  it.each([
    { metadata: recordedRefiner, models: [], name: 'SDXL Refiner', reason: 'unresolved' },
    {
      metadata: { ...recordedRefiner, refiner_model: { base: 'sdxl-refiner', name: 'SDXL Refiner' } },
      models: [refiner, { ...refiner, key: 'refiner-twin' }],
      name: 'SDXL Refiner',
      reason: 'ambiguous',
    },
    ...[{}, 'refiner', { key: 42 }, []].map((refiner_model) => ({
      metadata: { ...recordedRefiner, refiner_model },
      models: [refiner],
      name: null,
      reason: 'invalid',
    })),
  ])('keeps the model and all tuning together for a $reason refiner', ({ metadata, models, name, reason }) => {
    for (const refinerModel of [refiner, null]) {
      const current = { ...currentRefinerSettings, refinerModel };
      const result = recall(metadata, models, 'all', createValues(current));

      expect(result?.values).toMatchObject(current);
      expect(result?.fields).not.toContain('refiner');
      expect(result?.skipped).toEqual([{ name, reason, subject: 'refiner' }]);
    }
    expect(
      getImageRecallCapabilities({
        currentValues: createValues(),
        image,
        metadata,
        models,
        supportedModels: [sdxlModel],
        vaeModels: [],
      })
    ).toMatchObject({ all: false, remix: false });
  });

  it.each(['all', 'remix'] as const)(
    'keeps refiner settings during %s when the recorded main model is unavailable',
    (kind) => {
      // A persisted selection may still name the deleted main model's key.
      for (const key of ['deleted-main', sdxlModel.key]) {
        for (const refinerMetadata of [
          recordedRefiner,
          { refiner_model: {}, refiner_steps: 12 },
          {},
          { refiner_model: null },
        ]) {
          const metadata = { generation_mode: 'sdxl_txt2img', model: { key }, ...refinerMetadata };
          const currentValues = createValues(currentRefinerSettings);
          const result = buildImageRecallSettings({
            currentValues,
            image,
            kind,
            metadata,
            models: [refiner],
            supportedModels: [],
            vaeModels: [],
          });

          if ('refiner_model' in refinerMetadata && refinerMetadata.refiner_model) {
            expect(result?.values).toMatchObject(currentRefinerSettings);
            expect(result?.fields).not.toContain('refiner');
            expect(result?.skipped).toEqual([
              {
                name: refinerMetadata === recordedRefiner ? 'SDXL Refiner' : null,
                reason: 'modelUnavailable',
                subject: 'refiner',
              },
            ]);
          } else {
            // A record that ran without a refiner keeps the slot quietly: nothing to restore, nothing to report.
            expect(result).toBeNull();
          }
          expect(
            getImageRecallCapabilities({
              currentValues,
              image,
              metadata,
              models: [refiner],
              supportedModels: [],
              vaeModels: [],
            })
          ).toMatchObject({ all: false, remix: false });
        }
      }
    }
  );

  it('does not treat a non-refiner model named by the record as the refiner', () => {
    const result = recall({ ...recordedRefiner, refiner_model: { key: sdxlModel.key } }, [sdxlModel, refiner]);

    expect(result?.values.refinerModel).toBeNull();
    expect(result?.skipped).toEqual([{ name: sdxlModel.key, reason: 'unresolved', subject: 'refiner' }]);
  });

  it('clears the refiner for a generation record that ran without one and ignores refiner data on other families', () => {
    const cleared = buildImageRecallSettings({
      currentValues: createValues({ refinerModel: refiner }),
      image,
      kind: 'all',
      metadata: { generation_mode: 'sdxl_txt2img', model: { key: sdxlModel.key } },
      models: [refiner],
      supportedModels: [sdxlModel],
      vaeModels: [],
    });

    expect(cleared?.values.refinerModel).toBeNull();
    expect(cleared?.fields).toContain('refiner');

    // A canvas save names a model but is not a generation record, so it says nothing about the refiner.
    const canvasSave = buildImageRecallSettings({
      currentValues: createValues({ refinerModel: refiner }),
      image,
      kind: 'all',
      metadata: { model: { key: sdxlModel.key }, positive_prompt: 'kept' },
      models: [refiner],
      supportedModels: [sdxlModel],
      vaeModels: [],
    });

    expect(canvasSave?.values.refinerModel).toEqual(refiner);
    expect(canvasSave?.fields).not.toContain('refiner');

    const sd1 = buildImageRecallSettings({
      currentValues: createValues(),
      image,
      kind: 'all',
      metadata: { model: { key: sd1Model.key }, ...recordedRefiner },
      models: [sd1Model, refiner],
      supportedModels: [sd1Model],
      vaeModels: [],
    });

    expect(sd1?.values.refinerModel).toBeNull();
    expect(sd1?.fields).not.toContain('refiner');
  });

  it('drops out-of-range refiner tuning and unknown schedulers', () => {
    const result = recall({ ...recordedRefiner, refiner_scheduler: 'bogus', refiner_start: 3, refiner_steps: 0 }, [
      refiner,
    ]);

    expect(result?.values).toMatchObject({
      refinerModel: refiner,
      refinerScheduler: 'euler',
      refinerStart: 0.8,
      refinerSteps: 20,
    });
  });

  it('round-trips the refiner a compiled SDXL graph records', () => {
    const { model, settings } = satisfiedSettingsFor('sdxl', {
      label: 'diffusers',
      overrides: { format: 'diffusers' },
    });
    const graph = compileGenerateGraph({ ...settings, refinerModel: refiner, refinerStart: 0.65 }, model, 'gallery', {
      useCpuNoise: true,
      useNsfwChecker: false,
      useWatermark: false,
    }).backendGraph;
    const metadata = Object.values(graph.nodes).find((node) => node.type === 'core_metadata');
    const result = buildImageRecallSettings({
      currentValues: createValues({ model: sd1Model, modelKey: sd1Model.key }),
      image,
      kind: 'remix',
      metadata,
      models: [refiner],
      supportedModels: [model],
      vaeModels: [],
    });

    expect(result?.values.model).toEqual(model);
    expect(result?.values.refinerModel).toEqual(refiner);
    expect(result?.values.refinerStart).toBe(0.65);
    expect(
      getImageRecallCapabilities({
        currentValues: createValues(),
        image,
        metadata: { refiner_model: metadata?.refiner_model },
        models: [refiner],
        supportedModels: [model],
        vaeModels: [],
      })
    ).toMatchObject({ all: true, remix: true });
  });
});

describe('legacy records', () => {
  const fluxModel: MainModelConfig = { base: 'flux', key: 'flux-model', name: 'FLUX', type: 'main' };
  const recall = (
    metadata: unknown,
    overrides: Partial<GenerateWidgetValues>,
    supportedModels: MainModelConfig[],
    models: ComponentModelConfig[] = supportedModels,
    vaeModels: VaeModelConfig[] = []
  ) =>
    buildImageRecallSettings({
      currentValues: createValues(overrides),
      image,
      kind: 'all',
      metadata,
      models,
      supportedModels,
      vaeModels,
    });

  it('recalls FLUX guidance, which older records carry only under its own key', () => {
    const result = recall({ guidance: 7.25, model: { key: fluxModel.key } }, { cfgScale: 1.5, model: fluxModel }, [
      fluxModel,
    ]);

    expect(result?.values.cfgScale).toBe(7.25);
    expect(result?.fields).toContain('cfg');
  });

  it('reads cfg_scale for a CFG model even when a guidance key is present', () => {
    const result = recall({ cfg_scale: 4, guidance: 7.25, model: { key: sdxlModel.key } }, {}, [sdxlModel]);

    expect(result?.values.cfgScale).toBe(4);
  });

  it('recalls Krea-2 seed variance alongside the rebalance', () => {
    const result = recall(
      {
        krea2_seed_variance_enabled: true,
        krea2_seed_variance_randomize_percent: 73,
        krea2_seed_variance_strength: 0.42,
        model: { key: krea2Model.key },
      },
      { model: krea2Model, modelKey: krea2Model.key },
      [krea2Model]
    );

    expect(result?.values.krea2SeedVarianceEnabled).toBe(true);
    expect(result?.values.krea2SeedVarianceStrength).toBe(0.42);
    expect(result?.values.krea2SeedVarianceRandomizePercent).toBe(73);
    expect(result?.fields).toContain('krea2SeedVariance');
  });

  it('keeps the current seed variance for a value the enhancer cannot hold', () => {
    const result = recall(
      { krea2_seed_variance_enabled: true, krea2_seed_variance_strength: 5, model: { key: krea2Model.key } },
      { model: krea2Model, modelKey: krea2Model.key },
      [krea2Model]
    );

    expect(result?.values.krea2SeedVarianceEnabled).toBe(true);
    expect(result?.values.krea2SeedVarianceStrength).toBe(createValues().krea2SeedVarianceStrength);
  });

  it('ignores seed variance recorded for a model that is not Krea-2', () => {
    const result = recall({ krea2_seed_variance_enabled: true, model: { key: sdxlModel.key } }, {}, [sdxlModel]);

    expect(result?.values.krea2SeedVarianceEnabled).toBe(false);
    expect(result?.fields).not.toContain('krea2SeedVariance');
  });

  describe('explicitly empty references', () => {
    const currentReferenceImages = normalizeReferenceImages([
      {
        config: {
          image: { height: 512, image_name: 'current.png', width: 512 },
          model: { base: 'sdxl', key: 'adapter', name: 'Adapter', type: 'ip_adapter' },
          type: 'ip_adapter',
        },
        id: 'current-reference',
        isEnabled: true,
      },
    ]);
    const legacyCanvasReference = {
      referenceImages: {
        entities: [
          {
            id: 'legacy-canvas-ref',
            ipAdapter: {
              image: { height: 512, image_name: 'legacy.png', width: 512 },
              model: { base: 'sdxl', key: 'adapter', name: 'Adapter', type: 'ip_adapter' },
              type: 'ip_adapter',
            },
            isEnabled: true,
          },
        ],
      },
    };

    it('clears the current references when the record lists none', () => {
      const result = recall(
        { model: { key: sdxlModel.key }, positive_prompt: 'no references', ref_images: [] },
        { referenceImages: currentReferenceImages },
        [sdxlModel]
      );

      expect(result?.values.referenceImages).toEqual([]);
      expect(result?.fields).toContain('referenceImages');
    });

    it('does not fall back to pre-v6 canvas references for an empty modern list', () => {
      const result = recall(
        { canvas_v2_metadata: legacyCanvasReference, model: { key: sdxlModel.key }, ref_images: [] },
        { referenceImages: currentReferenceImages },
        [sdxlModel]
      );

      expect(result?.values.referenceImages).toEqual([]);
    });

    it('offers recall for an image whose only delta is clearing the references', () => {
      const capabilities = getImageRecallCapabilities({
        currentValues: createValues({ referenceImages: currentReferenceImages }),
        image,
        metadata: { ref_images: [] },
        models: [],
        supportedModels: [],
        vaeModels: [],
      });

      expect(capabilities.all).toBe(true);
    });

    it('leaves the current references when the record says nothing about them', () => {
      const result = recall(
        { model: { key: sdxlModel.key }, positive_prompt: 'silent' },
        { referenceImages: currentReferenceImages },
        [sdxlModel]
      );

      expect(result?.values.referenceImages).toEqual(currentReferenceImages);
      expect(result?.fields).not.toContain('referenceImages');
    });
  });

  describe('model identity', () => {
    const reinstalled: MainModelConfig = { ...sdxlModel, hash: 'blake3:sdxl', key: 'sdxl-reinstalled' };
    const recorded = { base: 'sdxl', hash: 'blake3:sdxl', key: 'sdxl-old-key', name: 'SDXL', type: 'main' };

    it('resolves a reinstalled main model by its recorded hash', () => {
      const result = recall({ model: recorded, positive_prompt: 'old image' }, { model: sd1Model }, [
        sd1Model,
        reinstalled,
      ]);

      expect(result?.values.model.key).toBe(reinstalled.key);
      expect(result?.fields).toContain('model');
    });

    it('does not let a recorded hash fall back to a same-named model', () => {
      const sameName: MainModelConfig = { ...sdxlModel, hash: 'blake3:other', key: 'sdxl-other' };
      const result = recall({ model: recorded, positive_prompt: 'old image' }, { model: sd1Model }, [
        sd1Model,
        sameName,
      ]);

      expect(result?.values.model.key).toBe(sd1Model.key);
      expect(result?.fields).not.toContain('model');
    });

    it('resolves a pre-identifier record by model name and base', () => {
      const result = recall(
        { model: { base_model: 'sdxl', model_name: 'SDXL' }, positive_prompt: 'very old image' },
        { model: sd1Model },
        [sd1Model, sdxlModel]
      );

      expect(result?.values.model.key).toBe(sdxlModel.key);
    });

    it('skips a name-and-base record that matches more than one installed model', () => {
      const twin: MainModelConfig = { ...sdxlModel, key: 'sdxl-twin' };
      const result = recall(
        { model: { base: 'sdxl', name: 'SDXL', type: 'main' }, positive_prompt: 'ambiguous' },
        { model: sd1Model },
        [sd1Model, sdxlModel, twin]
      );

      expect(result?.values.model.key).toBe(sd1Model.key);
      expect(result?.fields).not.toContain('model');
    });

    it('resolves components and the VAE by hash too', () => {
      const t5: ComponentModelConfig = {
        base: 'any',
        hash: 'blake3:t5',
        key: 't5-new',
        name: 'T5',
        type: 't5_encoder',
      };
      const vae: VaeModelConfig = { base: 'flux', hash: 'blake3:vae', key: 'vae-new', name: 'FLUX VAE', type: 'vae' };
      const result = recall(
        {
          model: { key: fluxModel.key },
          t5_encoder: { base: 'any', hash: 'blake3:t5', key: 't5-old', name: 'T5', type: 't5_encoder' },
          vae: { base: 'flux', hash: 'blake3:vae', key: 'vae-old', name: 'FLUX VAE', type: 'vae' },
        },
        { model: fluxModel, modelKey: fluxModel.key },
        [fluxModel],
        [fluxModel, t5],
        [vae]
      );

      expect(result?.values.t5EncoderModel?.key).toBe(t5.key);
      expect(result?.values.vae?.key).toBe(vae.key);
      expect(result?.fields).toEqual(expect.arrayContaining(['components', 'vae']));
    });

    describe('reference-image adapters', () => {
      const adapter: ComponentModelConfig = {
        base: 'sdxl',
        hash: 'blake3:adapter-new',
        key: 'adapter-new',
        name: 'Adapter',
        type: 'ip_adapter',
      };
      const recallAdapter = (installed: ComponentModelConfig[]) =>
        recall(
          {
            model: { key: sdxlModel.key },
            ref_images: [
              {
                config: {
                  image: { height: 512, image_name: 'reference.png', width: 512 },
                  model: {
                    base: 'sdxl',
                    hash: 'blake3:adapter-old',
                    key: 'adapter-old',
                    name: 'Adapter',
                    type: 'ip_adapter',
                  },
                  type: 'ip_adapter',
                },
                id: 'reference',
                isEnabled: true,
              },
            ],
          },
          {},
          [sdxlModel],
          [sdxlModel, ...installed]
        )?.values.referenceImages[0]?.config;

      it('resolves a reinstalled adapter by its recorded hash', () => {
        const rehashed = { ...adapter, hash: 'blake3:adapter-old', key: 'adapter-rehashed' };

        expect(recallAdapter([adapter, rehashed])).toMatchObject({ model: { key: rehashed.key } });
      });

      // Unlike the main model, legacy web falls back to the name when an adapter's hash matches nothing.
      it('falls back to name, base and type when the recorded hash matches nothing', () => {
        expect(recallAdapter([adapter, { ...adapter, key: 'lora-named-adapter', type: 'lora' }])).toMatchObject({
          model: { key: adapter.key },
        });
      });

      it('leaves the adapter unset when the name matches more than one installed adapter', () => {
        expect(recallAdapter([adapter, { ...adapter, key: 'adapter-twin' }])).toMatchObject({ model: null });
      });
    });
  });
});
