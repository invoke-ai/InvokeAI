import type { GalleryImage } from '@features/gallery';
import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateSettings,
  GenerateWidgetValues,
  LoraModelConfig,
  MainModelConfig,
  VaeModelConfig,
} from '@features/generation/contracts';

import {
  getAutoFlux2ComponentSourceModel,
  getComponentPolicyContext,
  getComponentSectionPolicy,
} from '@features/generation/core/baseGenerationPolicies';
import { isVaeCompatibleWithGenerateModel } from '@features/generation/core/componentCompatibility';
import {
  candidatesForSlot,
  flux2DevSourceRecallCase,
  fullyFilledSettingsFor,
  generateGraphCases,
  recallCatalogFor,
  satisfiedSettingsFor,
} from '@features/generation/core/graphCoverage.testing';
import { compileGenerateGraph, getRecordedComponents } from '@features/generation/graph';
import { getGenerationUiPolicy, normalizeReferenceImages } from '@features/generation/settings';
import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/models/core/architectureCapabilities';
import {
  architectureCapabilitiesFixture,
  seedArchitectureCapabilities,
} from '@features/models/core/architectureCapabilities.testing';
import { compileUpscaleGraph } from '@features/upscale/core/graph';
import { CASES as UPSCALE_CASES, valuesForCase } from '@features/upscale/core/upscaleGraphCoverage.testing';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { buildImageRecallSettings, getImageRecallCapabilities, type ImageRecallKind } from './imageRecall';
import { compileGenerateRecordCase, generateRecordCases, PROJECT_SETTINGS } from './imageRecordCases.testing';

const sdxlModel: MainModelConfig = { base: 'sdxl', key: 'sdxl-model', name: 'SDXL', type: 'main' };
const sd1Model: MainModelConfig = { base: 'sd-1', key: 'sd1-model', name: 'SD 1.5', type: 'main' };
const animaModel: MainModelConfig = { base: 'anima', key: 'anima-model', name: 'Anima', type: 'main' };
const krea2Model: MainModelConfig = { base: 'krea-2', key: 'krea2-model', name: 'Krea 2', type: 'main' };
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
  dynamicPromptsEnabled: true,
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
  qwen3VLVisionTowerModel: null,
  qwen35EncoderModel: null,
  animaInpaintAdapterModel: null,
  animaInpaintAdapterEnabled: true,
  animaInpaintAdapterWeight: 1,
  flux3PromptUpsamplerEnabled: false,
  flux3PromptUpsamplerModel: null,
  wanT5EncoderModel: null,
  wanLowNoiseModel: null,
  ideogram4UnconditionalModel: null,
  wanGuidanceScaleLowNoise: null,
  ideogram4SamplerPreset: 'V4_QUALITY_48',
  ideogram4Steps: null,
  ideogram4GuidanceScale: null,
  ideogram4Mu: null,
  ideogram4ColorPalette: [],
  externalImageSize: null,
  openaiQuality: 'auto',
  openaiBackground: 'auto',
  openaiInputFidelity: null,
  geminiTemperature: null,
  geminiThinkingLevel: null,
  seedreamWatermark: false,
  seedreamOptimizePrompt: false,
  krea2RebalanceEnabled: false,
  krea2RebalanceMultiplier: 4,
  krea2RebalanceWeights: '1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0',
  krea2SeedVarianceEnabled: false,
  krea2SeedVarianceStrength: 0.1,
  krea2SeedVarianceRandomizePercent: 50,
  krea2ReferenceKvCache: false,
  fluxDypePreset: 'off',
  fluxDypeScale: 2,
  fluxDypeExponent: 2,
  zImageShift: null,
  zImageSeedVarianceEnabled: false,
  zImageSeedVarianceStrength: 0.1,
  zImageSeedVarianceRandomizePercent: 50,
  qwenImageShift: null,
  qwenImageEncoderQuantization: 'none',
  mossImageUncondSkipLayers: [22],
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
  it.each(['positivePrompt', 'negativePrompt'] as const)(
    'recalls only the %s without replacing the other prompt',
    (kind) => {
      const currentValues = createValues({
        positivePrompt: 'keep positive',
        negativePrompt: 'keep negative',
        negativePromptEnabled: false,
      });
      const result = buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues,
        image,
        kind,
        metadata: { positive_prompt: 'saved positive', negative_prompt: 'saved negative' },
        models: [],
        supportedModels: [sdxlModel],
        vaeModels: [],
      });

      expect(result?.values).toMatchObject({
        positivePrompt: kind === 'positivePrompt' ? 'saved positive' : 'keep positive',
        negativePrompt: kind === 'negativePrompt' ? 'saved negative' : 'keep negative',
        negativePromptEnabled: kind === 'negativePrompt',
      });
    }
  );

  it.each(['positivePrompt', 'negativePrompt'] as const)(
    'preserves the other effective prompt when recalling %s clears a shared template',
    (kind) => {
      const result = buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues({
          positivePrompt: 'keep positive',
          negativePrompt: 'keep negative',
          negativePromptEnabled: false,
          promptTemplate: {
            id: 'template',
            name: 'Template',
            positivePrompt: 'cinematic {prompt}, 35mm',
            negativePrompt: 'avoid {prompt}, blurry',
          },
          promptTemplateViewMode: true,
        }),
        image,
        kind,
        metadata: { positive_prompt: 'saved positive', negative_prompt: 'saved negative' },
        models: [],
        supportedModels: [sdxlModel],
        vaeModels: [],
      });

      expect(result?.values).toMatchObject({
        positivePrompt: kind === 'positivePrompt' ? 'saved positive' : 'cinematic keep positive, 35mm',
        negativePrompt: kind === 'negativePrompt' ? 'saved negative' : 'avoid keep negative, blurry',
        negativePromptEnabled: kind === 'negativePrompt',
        promptTemplate: null,
        promptTemplateViewMode: false,
      });
    }
  );

  it.each(['prompts', 'all', 'remix'] as const)(
    'enables the recorded negative and preserves the unrecalled effective positive for %s',
    (kind) => {
      const result = buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues({
          positivePrompt: 'keep positive',
          negativePrompt: 'old negative',
          negativePromptEnabled: false,
          promptTemplate: {
            id: 'template',
            name: 'Template',
            positivePrompt: '{prompt}, detailed',
            negativePrompt: 'lowres',
          },
          promptTemplateViewMode: true,
        }),
        image,
        kind,
        metadata: { negative_prompt: 'recorded negative' },
        models: [],
        supportedModels: [sdxlModel],
        vaeModels: [],
      });

      expect(result?.values).toMatchObject({
        positivePrompt: 'keep positive, detailed',
        negativePrompt: 'recorded negative',
        negativePromptEnabled: true,
        promptTemplate: null,
        promptTemplateViewMode: false,
      });
    }
  );

  it.each([
    {
      kind: 'positivePrompt',
      metadata: { positive_prompt: '' },
      negativePromptEnabled: false,
      expected: { positivePrompt: '', negativePrompt: 'keep negative lowres', negativePromptEnabled: false },
    },
    {
      kind: 'negativePrompt',
      metadata: { negative_prompt: '' },
      negativePromptEnabled: true,
      expected: { positivePrompt: 'keep positive, detailed', negativePrompt: '', negativePromptEnabled: true },
    },
    {
      kind: 'negativePrompt',
      metadata: { negative_prompt: '' },
      negativePromptEnabled: false,
      expected: { positivePrompt: 'keep positive, detailed', negativePrompt: '', negativePromptEnabled: false },
    },
    {
      kind: 'negativePrompt',
      metadata: { negative_prompt: null },
      negativePromptEnabled: true,
      expected: { positivePrompt: 'keep positive, detailed', negativePrompt: '', negativePromptEnabled: false },
    },
    { kind: 'positivePrompt', metadata: { negative_prompt: 'recorded' }, negativePromptEnabled: false, expected: null },
    { kind: 'negativePrompt', metadata: { positive_prompt: 'recorded' }, negativePromptEnabled: false, expected: null },
    { kind: 'positivePrompt', metadata: { positive_prompt: null }, negativePromptEnabled: false, expected: null },
  ] as const)(
    'handles empty or absent $kind metadata without changing the other effective prompt',
    ({ kind, metadata, negativePromptEnabled, expected }) => {
      const result = buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues({
          positivePrompt: 'keep positive',
          negativePrompt: 'keep negative',
          negativePromptEnabled,
          promptTemplate: {
            id: 'template',
            name: 'Template',
            negativePrompt: 'lowres',
            positivePrompt: '{prompt}, detailed',
          },
          promptTemplateViewMode: true,
        }),
        image,
        kind,
        metadata,
        models: [],
        supportedModels: [sdxlModel],
        vaeModels: [],
      });

      if (expected === null) {
        expect(result).toBeNull();
      } else {
        expect(result?.values).toMatchObject({ ...expected, promptTemplate: null, promptTemplateViewMode: false });
      }
    }
  );

  it('recalls supported image metadata into generate settings', () => {
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
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

  describe("the recorded model's default VAE", () => {
    const sdxlVae: VaeModelConfig = { base: 'sdxl', key: 'sdxl-vae', name: 'SDXL VAE', type: 'vae' };
    const recall = (current: GenerateWidgetValues, model: MainModelConfig, vaeModels: VaeModelConfig[]) =>
      buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: current,
        image,
        kind: 'all',
        // No VAE recorded, so only the model's default could change it.
        metadata: { model: { key: model.key }, positive_prompt: 'a recalled prompt' },
        models: [model, ...vaeModels],
        supportedModels: [model],
        vaeModels,
      });

    // An image made with the bundled VAE records none; recalling it must not keep another VAE.
    it('uses the bundled VAE when the default names one that is not installed, as recall always has', () => {
      const model = { ...sdxlModel, default_settings: { vae: 'gone' } };
      const result = recall(createValues({ model: sdxlModel, modelKey: sdxlModel.key, vae: sdxlVae }), model, [
        sdxlVae,
      ]);

      expect(result?.values.vae).toBeNull();
    });
  });

  it('recalls reference images from metadata, re-targeting configs to the effective model', () => {
    // The metadata model is not installed: the SDXL model stays and the Qwen reference config is rebuilt for SDXL.
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues(),
      image,
      kind: 'all',
      metadata: {
        ref_images: [
          {
            id: 'ref-1',
            isEnabled: true,
            name: 'Hero pose',
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
    // The user's name stays with its image through the retarget to another family.
    expect(result?.values.referenceImages[0]?.name).toBe('Hero pose');
    expect(result?.fields).toContain('referenceImages');
  });

  it('preserves canonical original and crop provenance from legacy metadata', () => {
    const crop = {
      box: { height: 256, width: 320, x: 12, y: 24 },
      image: { height: 256, image_name: 'crop.png', width: 320 },
      ratio: null,
    };
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
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
      mayBeWorkflowRun: false,
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

  it.each(['all', 'remix'] as const)('recalls valid HiDiffusion metadata for supported models with %s', (kind) => {
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
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

  it('uses actual image dimensions for Use Size', () => {
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
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

  describe('before the capability table arrives', () => {
    // Require the model's grid before persisting recalled dimensions; fallback grid 8 can produce invalid sizes.
    beforeEach(() => {
      resetArchitectureCapabilities();
    });

    afterEach(() => {
      setArchitectureCapabilities(architectureCapabilitiesFixture);
    });

    it('withholds Recall All and Remix but keeps prompts and seed', () => {
      const metadata = { height: 768, model: { key: sdxlModel.key }, positive_prompt: 'a cat', seed: 7, width: 512 };
      const recall = (kind: ImageRecallKind) =>
        buildImageRecallSettings({
          mayBeWorkflowRun: false,
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

  it('recalls required component models for component-based families', () => {
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
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

  it('keeps a fitting pick when the recorded component does not fit the current model', () => {
    // The image's Ideogram model is gone, so Remix stays on the current Krea-2 model and its 4B encoder.
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
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
        mayBeWorkflowRun: false,
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

    it('restores concepts from records that name the LoRA under `lora`', () => {
      const result = recallConcepts('remix', {
        loras: [{ lora: { key: inkLora.key }, weight: 0.45 }],
        model: { key: sdxlModel.key },
      });

      expect(result?.values.loras).toEqual([{ isEnabled: true, model: inkLora, weight: 0.45 }]);
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
        mayBeWorkflowRun: false,
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

  describe('external provider options', () => {
    const openaiModel = {
      base: 'external',
      capabilities: { modes: ['txt2img', 'img2img'], supports_reference_images: true },
      format: 'external_api',
      key: 'gpt-image',
      name: 'GPT Image',
      provider_id: 'openai',
      provider_model_id: 'gpt-image-1',
      type: 'external_image_generator',
    } as GenerateWidgetValues['model'];
    // What the provider nodes record (`_build_output_provider_metadata`) plus the named size the base node records.
    const providerMetadata = {
      gemini_temperature: 0.4,
      gemini_thinking_level: 'high',
      image_size: '2K',
      openai_background: 'transparent',
      openai_input_fidelity: 'high',
      openai_quality: 'low',
      seedream_optimize_prompt: true,
      seedream_watermark: true,
    };
    const recall = (model: GenerateWidgetValues['model'], metadata: Record<string, unknown>, kind: 'all' | 'remix') =>
      buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues(),
        image,
        kind,
        metadata: { ...metadata, model: { key: model.key } },
        models: [],
        supportedModels: [model],
        vaeModels: [],
      });

    it('restores only the options the recalled OpenAI model accepts', () => {
      const result = recall(openaiModel, providerMetadata, 'all');

      expect(result?.values).toMatchObject({
        openaiBackground: 'transparent',
        openaiInputFidelity: 'high',
        openaiQuality: 'low',
      });
      // Another provider's keys and a named size the model does not list stay at the current values.
      expect(result?.values).toMatchObject({
        externalImageSize: null,
        geminiTemperature: null,
        geminiThinkingLevel: null,
        seedreamOptimizePrompt: false,
        seedreamWatermark: false,
      });
      expect(result?.fields).toContain('externalProvider');
    });

    it('skips provider metadata when the image was made with a local model', () => {
      const result = buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues(),
        image,
        kind: 'all',
        metadata: { ...providerMetadata, model: { key: sdxlModel.key } },
        models: [],
        supportedModels: [sdxlModel],
        vaeModels: [],
      });

      expect(result?.values).toMatchObject({
        externalImageSize: null,
        geminiTemperature: null,
        geminiThinkingLevel: null,
        openaiBackground: 'auto',
        openaiInputFidelity: null,
        openaiQuality: 'auto',
        seedreamOptimizePrompt: false,
        seedreamWatermark: false,
      });
      expect(result?.fields).not.toContain('externalProvider');
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
      mayBeWorkflowRun: false,
      currentValues: createValues(current),
      image,
      kind,
      metadata: { model: { key: model.key }, ...metadata },
      models: [],
      supportedModels: [model],
      vaeModels: [],
    });

  it('ignores tuning recorded for another family and values outside the node bounds', () => {
    const wrongFamily = recall({ dype_preset: 'manual', z_image_shift: 2 }, qwenModel);
    const outOfRange = recall({ dype_preset: 'turbo', dype_scale: 9 }, fluxModel);

    expect(wrongFamily?.values.fluxDypePreset).toBe('off');
    expect(wrongFamily?.values.zImageShift).toBeNull();
    expect(wrongFamily?.fields ?? []).not.toContain('modelTuning');
    expect(outOfRange?.values).toMatchObject({ fluxDypePreset: 'off', fluxDypeScale: 2 });
  });

  describe('PiD decoding', () => {
    const sdxlGeneration = (extra: Record<string, unknown> = {}) => ({
      generation_mode: 'sdxl_txt2img',
      model: { base: 'sdxl', key: sdxlModel.key, name: sdxlModel.name, type: 'main' },
      ...extra,
    });

    it.each(['fit', 'native'] as const)('restores %s decoding with its steps', (pidMode) => {
      const result = recall(sdxlGeneration({ pid_mode: pidMode, pid_steps: 8 }), sdxlModel);

      expect(result?.values).toMatchObject({ pidMode, pidSteps: 8 });
      expect(result?.fields).toContain('modelTuning');
    });
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
        mayBeWorkflowRun: false,
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
  });

  it('keeps the current Krea-2 style reference for a legacy record that did not list its references', () => {
    const styleReference = normalizeReferenceImages([
      {
        config: {
          image: { original: { image: { height: 512, image_name: 'current.png', width: 512 } } },
          styleStrength: 1,
          type: 'krea2_reference_image',
        },
        id: 'current-style',
        isEnabled: true,
      },
    ]);
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({ model: krea2Model, modelKey: krea2Model.key, referenceImages: styleReference }),
      image,
      kind: 'all',
      // The legacy UI recorded only the strength of the style reference it used.
      metadata: {
        app_version: '6.12.0',
        generation_mode: 'krea2_txt2img',
        krea2_style_strength: 0.6,
        model: { key: krea2Model.key },
      },
      models: [],
      supportedModels: [krea2Model],
      vaeModels: [],
    });

    expect(result?.values.referenceImages).toEqual(styleReference);
    expect(result?.fields).not.toContain('referenceImages');
  });
});

describe('recall offered for a single recallable delta', () => {
  const mossImageModel: MainModelConfig = {
    base: 'moss-image',
    format: 'diffusers',
    key: 'moss-image-model',
    name: 'MOSS-Image',
    type: 'main',
  };
  const seedreamModel = {
    base: 'external',
    capabilities: { modes: ['txt2img'] },
    format: 'external_api',
    key: 'seedream',
    name: 'Seedream',
    provider_id: 'seedream',
    provider_model_id: 'gpt-image-1',
    type: 'external_image_generator',
  } as GenerateWidgetValues['model'];
  const fluxModel: MainModelConfig = { base: 'flux', key: 'flux-model', name: 'FLUX', type: 'main' };

  it.each<{
    label: string;
    current: Partial<GenerateWidgetValues>;
    metadata: Record<string, unknown>;
    models: ComponentModelConfig[];
    offered: boolean;
  }>([
    {
      current: {},
      label: 'concepts it can restore',
      metadata: { loras: [recordedLora(inkLora, 0.5)] },
      models: [inkLora],
      offered: true,
    },
    {
      current: {},
      label: 'concepts of another base',
      metadata: { loras: [recordedLora(inkLora, 0.5)] },
      models: [sd1Lora],
      offered: false,
    },
    {
      current: { model: krea2Model },
      label: 'the Krea-2 rebalance',
      metadata: { krea2_rebalance_enabled: true },
      models: [],
      offered: true,
    },
    {
      current: { model: mossImageModel },
      label: 'the MOSS-Image skip layers',
      metadata: { moss_image_uncond_skip_layers: [] },
      models: [],
      offered: true,
    },
    {
      current: { model: seedreamModel, modelKey: seedreamModel.key },
      label: 'a provider option',
      metadata: { seedream_watermark: true },
      models: [],
      offered: true,
    },
    {
      current: { model: fluxModel },
      label: 'its tuning',
      metadata: { dype_preset: 'auto' },
      models: [],
      offered: true,
    },
  ])(
    'offers Recall All and Remix ($offered) for an image whose only delta is $label',
    ({ current, metadata, models, offered }) => {
      const capabilities = getImageRecallCapabilities({
        currentValues: createValues(current),
        image,
        metadata,
        models,
        supportedModels: [],
        vaeModels: [],
      });

      expect(capabilities).toMatchObject({ all: offered, remix: offered });
    }
  );
});

describe('Anima IP-Adapter recall', () => {
  it('restores the reference an Anima image was generated with and submits it unchanged', () => {
    const { model, settings } = satisfiedSettingsFor('anima', {
      label: 'qwen3',
      overrides: { format: 'checkpoint', variant: 'anima_qwen3' },
    });
    const animaIpAdapter: ComponentModelConfig = {
      base: 'anima',
      key: 'anima-ip-adapter',
      name: 'Anima IP-Adapter',
      type: 'ip_adapter',
    };
    const adapterNodes = (backendGraph: { nodes: Record<string, Record<string, unknown>> }) =>
      Object.values(backendGraph.nodes)
        .filter((node) => node.type === 'anima_ip_adapter')
        .map(({ id: _id, ...node }) => node);
    const generated = compileGenerateGraph(
      {
        ...settings,
        referenceImages: [
          {
            config: {
              beginEndStepPct: [0.1, 0.7],
              clipVisionModel: 'ViT-H',
              image: { original: { image: { height: 768, image_name: 'character.png', width: 512 } } },
              method: 'full',
              model: animaIpAdapter,
              type: 'ip_adapter',
              weight: 1.5,
            },
            id: 'anima-ref',
            isEnabled: true,
          },
        ],
      },
      model,
      'gallery',
      { useCpuNoise: true, useNsfwChecker: false, useWatermark: false }
    ).backendGraph;
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({ model }),
      image,
      kind: 'all',
      metadata: Object.values(generated.nodes).find((node) => node.type === 'core_metadata'),
      models: [model, animaIpAdapter],
      supportedModels: [model],
      vaeModels: [],
    });

    expect(result?.fields).toContain('referenceImages');
    const recalled = compileGenerateGraph(
      { ...settings, referenceImages: result!.values.referenceImages },
      model,
      'gallery',
      { useCpuNoise: true, useNsfwChecker: false, useWatermark: false }
    ).backendGraph;

    expect(adapterNodes(generated)).toHaveLength(1);
    expect(adapterNodes(recalled)).toEqual(adapterNodes(generated));
  });
});

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
        mayBeWorkflowRun: false,
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

  it('switches the adapter off but keeps the pick for an Anima inpaint record that lacks the key, which the backend persists for none', () => {
    const currentValues = createValues({ animaInpaintAdapterModel: inpaintAdapter, animaInpaintAdapterWeight: 0.5 });
    const {
      anima_inpaint_adapter: _adapter,
      anima_inpaint_adapter_weight: _weight,
      ...withoutAdapter
    } = canvasInpaintMetadata;
    const recall = (metadata: Record<string, unknown>, supportedModels = [animaModel]) =>
      buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues,
        image,
        kind: 'all',
        metadata,
        models: [...models, sd1Model],
        supportedModels,
        vaeModels: [qwenImageVae],
      });

    for (const generation_mode of ['anima_inpaint', 'anima_outpaint']) {
      const switchedOff = recall({ ...withoutAdapter, generation_mode });
      expect(switchedOff?.values.animaInpaintAdapterEnabled).toBe(false);
      expect(switchedOff?.values.animaInpaintAdapterModel).toEqual(inpaintAdapter);
      expect(switchedOff?.values.animaInpaintAdapterWeight).toBe(0.5);
      expect(switchedOff?.fields).toContain('components');
      expect(switchedOff?.fields).not.toContain('animaInpaintAdapterWeight');
    }

    // A record from a mode that never writes the key, or none at all, says nothing about the adapter.
    for (const metadata of [withoutAdapter, { ...withoutAdapter, generation_mode: 'anima_img2img' }]) {
      const untouched = recall(metadata);
      expect(untouched?.values.animaInpaintAdapterModel).toEqual(inpaintAdapter);
      expect(untouched?.values.animaInpaintAdapterEnabled).toBe(true);
    }
    const notAnima = recall({ generation_mode: 'inpaint', model: { key: sd1Model.key } }, [sd1Model, animaModel]);
    expect(notAnima?.values.model).toBe(sd1Model);
    expect(notAnima?.values.animaInpaintAdapterEnabled).toBe(true);
    expect(notAnima?.fields).not.toContain('components');
  });
});

describe('FLUX.3 prompt upsampler recall', () => {
  const flux3Model: MainModelConfig = {
    base: 'flux3',
    format: 'diffusers',
    key: 'flux3-klein',
    name: 'FLUX.3 Klein',
    type: 'main',
    variant: 'flux3_klein',
  };
  const upsampler: ComponentModelConfig = {
    base: 'any',
    key: 'pup',
    name: 'FLUX.3 Prompt Upsampler',
    supports_images: true,
    type: 'text_llm',
  };
  const models = [flux3Model, upsampler];
  const recall = (
    metadata: Record<string, unknown>,
    currentValues: GenerateWidgetValues,
    supportedModels = [flux3Model, sd1Model]
  ) =>
    buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues,
      image,
      kind: 'all',
      metadata,
      models: [...models, sd1Model],
      supportedModels,
      vaeModels: [],
    });
  const recordedBy = (settings: Partial<GenerateWidgetValues>) => {
    const generated = compileGenerateGraph(
      {
        ...createValues({ cfgScale: 1, model: flux3Model, modelKey: flux3Model.key }),
        positivePrompt: 'a beach at noon',
        ...settings,
      },
      flux3Model,
      'gallery',
      { useCpuNoise: true, useNsfwChecker: false, useWatermark: false }
    ).backendGraph;

    return Object.values(generated.nodes).find((node) => node.type === 'core_metadata')!;
  };

  it('turns the switch on with the recorded model from what a graph that upsampled records, and restores the typed prompt', () => {
    // At run time the edges add the typed prompt and the caption the upsampler wrote.
    const metadata = {
      ...recordedBy({ flux3PromptUpsamplerEnabled: true, flux3PromptUpsamplerModel: upsampler }),
      flux3_caption: 'A long caption. [{"id": "obj_1", "bbox": [0, 0, 500, 500], "desc": "a kite"}]',
      positive_prompt: 'a beach at noon',
    };
    const result = recall(metadata, createValues());

    expect(result?.values.flux3PromptUpsamplerEnabled).toBe(true);
    expect(result?.values.flux3PromptUpsamplerModel).toBe(upsampler);
    expect(result?.fields).toContain('components');
    // The caption the upsampler wrote is never the recalled prompt.
    expect(result?.values.positivePrompt).toBe('a beach at noon');
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
  const recall = (
    metadata: Record<string, unknown>,
    models: ComponentModelConfig[],
    kind: ImageRecallKind = 'all',
    currentValues = createValues()
  ) =>
    buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues,
      image,
      kind,
      metadata: { model: { key: sdxlModel.key }, ...metadata },
      models,
      supportedModels: [sdxlModel],
      vaeModels: [],
    });

  it('resolves the refiner by key, then by name and base, and only when that is unambiguous', () => {
    const byKey = { ...refiner, hash: 'blake3:other', key: 'refiner-recorded' };

    expect(recall(recordedRefiner, [byKey, refiner])?.values.refinerModel).toEqual(byKey);

    const unhashed = { ...recordedRefiner, refiner_model: { base: 'sdxl-refiner', name: 'SDXL Refiner' } };

    expect(recall(unhashed, [refiner])?.values.refinerModel).toEqual(refiner);
    expect(recall(unhashed, [refiner])?.skipped).toEqual([]);
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
      mayBeWorkflowRun: false,
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
      mayBeWorkflowRun: false,
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

    it('clears the current references when a generation record omits them', () => {
      const result = recall(
        { app_version: '6.0.0', generation_mode: 'txt2img', model: { key: sdxlModel.key }, positive_prompt: 'silent' },
        { referenceImages: currentReferenceImages },
        [sdxlModel]
      );

      expect(result?.values.referenceImages).toEqual([]);
      expect(result?.fields).toContain('referenceImages');
    });

    it('leaves the current references when the recorded list cannot be read', () => {
      // v6.3–6.10 recorded Kontext references as a list nested inside `ref_images`.
      const nested = [[{ config: { image: { image_name: 'kontext.png' }, type: 'flux_kontext' }, id: 'kontext' }]];
      const result = recall(
        {
          app_version: '7.0.0',
          generation_mode: 'txt2img',
          model: { key: sdxlModel.key },
          positive_prompt: 'nested',
          ref_images: nested,
        },
        { referenceImages: currentReferenceImages },
        [sdxlModel]
      );

      expect(result?.values.referenceImages).toEqual(currentReferenceImages);
      expect(result?.fields).not.toContain('referenceImages');
    });

    it('leaves the current references on a record that is not a generation record', () => {
      // An upscale or hand-built workflow record names a model without saying what references ran.
      const result = recall(
        { app_version: '7.0.0', model: { key: sdxlModel.key }, positive_prompt: 'upscaled' },
        { referenceImages: currentReferenceImages },
        [sdxlModel]
      );

      expect(result?.values.referenceImages).toEqual(currentReferenceImages);
      expect(result?.fields).not.toContain('referenceImages');
    });

    it('keeps the current references for a model that takes none', () => {
      const zImageModel: MainModelConfig = { base: 'z-image', key: 'z-model', name: 'Z-Image', type: 'main' };
      const result = recall(
        {
          app_version: '7.0.0',
          generation_mode: 'z_image_txt2img',
          model: { key: zImageModel.key },
          positive_prompt: 'no references',
        },
        { referenceImages: currentReferenceImages },
        [sdxlModel, zImageModel]
      );

      expect(result?.values.model.key).toBe(zImageModel.key);
      expect(result?.values.referenceImages.map((reference) => reference.id)).toEqual(['current-reference']);
      expect(result?.fields).not.toContain('referenceImages');
    });

    it('leaves the current references for a v4.2 record that kept its IP-Adapters in control layers', () => {
      const result = recall(
        {
          app_version: '6.0.0',
          control_layers: { layers: [{ type: 'ip_adapter_layer' }], version: 2 },
          generation_mode: 'txt2img',
          model: { key: sdxlModel.key },
          positive_prompt: 'v4.2',
        },
        { referenceImages: currentReferenceImages },
        [sdxlModel]
      );

      expect(result?.values.referenceImages).toEqual(currentReferenceImages);
      expect(result?.fields).not.toContain('referenceImages');
    });

    it('leaves the current references on a v5 record, which may have dropped the ones it used', () => {
      // v5 Canvas-staged images saved to the Gallery lost `canvas_v2_metadata`, and their references with it.
      const result = recall(
        {
          app_version: '5.15.0',
          generation_mode: 'sdxl_txt2img',
          model: { key: sdxlModel.key },
          positive_prompt: 'v5',
        },
        { referenceImages: currentReferenceImages },
        [sdxlModel]
      );

      expect(result?.values.referenceImages).toEqual(currentReferenceImages);
      expect(result?.fields).not.toContain('referenceImages');
    });

    it('leaves the current references when the record names no recallable model', () => {
      const result = recall(
        {
          app_version: '7.0.0',
          generation_mode: 'txt2img',
          model: { key: 'not-installed' },
          positive_prompt: 'silent',
        },
        { referenceImages: currentReferenceImages },
        [sdxlModel]
      );

      expect(result?.values.referenceImages).toEqual(currentReferenceImages);
      expect(result?.fields).not.toContain('referenceImages');
    });
  });

  describe('model identity', () => {
    const recorded = { base: 'sdxl', hash: 'blake3:sdxl', key: 'sdxl-old-key', name: 'SDXL', type: 'main' };

    it('does not let a recorded hash fall back to a same-named model', () => {
      const sameName: MainModelConfig = { ...sdxlModel, hash: 'blake3:other', key: 'sdxl-other' };
      const result = recall({ model: recorded, positive_prompt: 'old image' }, { model: sd1Model }, [
        sd1Model,
        sameName,
      ]);

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

    describe("the recalled model's default components", () => {
      const defaultT5: ComponentModelConfig = {
        base: 'any',
        key: 't5-default',
        name: 'Default T5',
        type: 't5_encoder',
      };
      const recordedT5: ComponentModelConfig = {
        base: 'any',
        hash: 'blake3:recorded',
        key: 't5-recorded',
        name: 'Recorded T5',
        type: 't5_encoder',
      };
      const defaultClip: ComponentModelConfig = {
        base: 'any',
        key: 'clip-default',
        name: 'Default CLIP',
        type: 'clip_embed',
        variant: 'large',
      };
      const withDefaults: MainModelConfig = {
        ...fluxModel,
        default_settings: { components: { clip_embed_model: defaultClip.key, t5_encoder: defaultT5.key } },
        key: 'flux-with-defaults',
      };
      const installed = [withDefaults, defaultT5, recordedT5, defaultClip];

      it('fills the slots the image recorded nothing for, and lets a recorded component win', () => {
        const result = recall(
          { model: { key: withDefaults.key }, t5_encoder: { ...recordedT5 } },
          { model: sdxlModel, modelKey: sdxlModel.key },
          [sdxlModel, withDefaults],
          installed
        );

        expect(result?.values.t5EncoderModel?.key).toBe(recordedT5.key);
        expect(result?.values.clipEmbedModel?.key).toBe(defaultClip.key);
      });

      it('replaces a current selection that fits when the image recorded nothing for the slot, like Reset', () => {
        const result = recall(
          { model: { key: withDefaults.key } },
          { model: withDefaults, modelKey: withDefaults.key, t5EncoderModel: recordedT5 },
          [sdxlModel, withDefaults],
          installed
        );

        expect(result?.values.t5EncoderModel?.key).toBe(defaultT5.key);
      });

      it('lets a component the image recorded as empty clear the default', () => {
        const result = recall(
          { model: { key: withDefaults.key }, t5_encoder: null },
          { model: sdxlModel, modelKey: sdxlModel.key },
          [sdxlModel, withDefaults],
          installed
        );

        expect(result?.values.t5EncoderModel).toBeNull();
        expect(result?.values.clipEmbedModel?.key).toBe(defaultClip.key);
      });

      describe('on a generation record, which records what ran', () => {
        const q21: MainModelConfig = {
          base: 'qwen-image-2-1',
          default_settings: { components: { qwen3_vl_encoder: 'q21-gguf-encoder' }, vae: 'q21-vae' },
          format: 'diffusers',
          key: 'q21-main',
          name: 'Qwen-Image-2.1',
          type: 'main',
          variant: 'qwen_image_21_base',
        };
        const ggufEncoder: ComponentModelConfig = {
          base: 'any',
          format: 'gguf_quantized',
          key: 'q21-gguf-encoder',
          name: 'GGUF encoder',
          type: 'qwen3_vl_encoder',
          variant: 'qwen3_vl_8b',
        };
        const otherEncoder: ComponentModelConfig = {
          ...ggufEncoder,
          format: 'checkpoint',
          key: 'q21-other-encoder',
          name: 'Other encoder',
        };
        const vae: VaeModelConfig = { base: 'qwen-image-2-1', key: 'q21-vae', name: 'Q21 VAE', type: 'vae' };
        const recallQ21 = (metadata: Record<string, unknown>, overrides: Partial<GenerateWidgetValues> = {}) =>
          recall(
            { model: { key: q21.key }, ...metadata },
            { model: sdxlModel, modelKey: sdxlModel.key, ...overrides },
            [sdxlModel, q21],
            [q21, ggufEncoder, otherEncoder],
            [vae]
          );

        it('applies no saved default component, and an unnamed slot keeps the current pick', () => {
          const result = recallQ21({ generation_mode: 'txt2img' }, { qwen3VLEncoderModel: otherEncoder });
          const empty = recallQ21({ generation_mode: 'txt2img' });

          expect(result?.values.qwen3VLEncoderModel?.key).toBe(otherEncoder.key);
          expect(empty?.values.qwen3VLEncoderModel).toBeNull();
        });

        it('still restores what the record names', () => {
          const result = recallQ21({ generation_mode: 'txt2img', qwen3_vl_encoder: { ...ggufEncoder } });

          expect(result?.values.qwen3VLEncoderModel?.key).toBe(ggufEncoder.key);
        });

        it('fills the defaults for a record that is no generation record, which says nothing of components', () => {
          const result = recallQ21({});

          expect(result?.values.qwen3VLEncoderModel?.key).toBe(ggufEncoder.key);
          expect(result?.values.vae?.key).toBe(vae.key);
        });
      });

      it('keeps the current VAE when the image clears the component source whose bundled VAE the default took', () => {
        const pipeline: MainModelConfig = {
          base: 'z-image',
          format: 'diffusers',
          key: 'z-pipeline',
          name: 'Z-Image pipeline',
          type: 'main',
        };
        const zGguf: MainModelConfig = {
          base: 'z-image',
          default_settings: { components: { component_source: pipeline.key }, vae: 'default' },
          format: 'gguf_quantized',
          key: 'z-gguf',
          name: 'Z-Image GGUF',
          type: 'main',
        };
        const vae: VaeModelConfig = { base: 'flux', key: 'z-vae', name: 'FLUX VAE', type: 'vae' };
        const result = recall(
          { model: { key: zGguf.key }, qwen3_source: null },
          { model: sdxlModel, modelKey: sdxlModel.key, vae },
          [sdxlModel, zGguf],
          [zGguf, pipeline],
          [vae]
        );

        expect(result?.values.componentSourceModel).toBeNull();
        expect(result?.values.vae).toEqual(vae);
      });

      it('fills every slot of an image that recorded none', () => {
        const result = recall(
          { model: { key: withDefaults.key } },
          { model: sdxlModel, modelKey: sdxlModel.key },
          [sdxlModel, withDefaults],
          installed
        );

        expect(result?.values.t5EncoderModel?.key).toBe(defaultT5.key);
        expect(result?.values.clipEmbedModel?.key).toBe(defaultClip.key);
      });
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
    });
  });
});

describe('legacy family metadata parity', () => {
  it('restores recorded Ideogram sampler overrides and explicit preset defaults', () => {
    const model: MainModelConfig = {
      base: 'ideogram-4',
      key: 'ideogram',
      name: 'Ideogram',
      type: 'main',
      format: 'diffusers',
    };
    const recall = (metadata: unknown) =>
      buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues({
          model,
          modelKey: model.key,
          ideogram4Steps: 70,
          ideogram4GuidanceScale: 6,
          ideogram4Mu: 3,
        }),
        image,
        kind: 'all',
        metadata,
        models: [model],
        supportedModels: [model],
        vaeModels: [],
      });
    const metadata = {
      model,
      ideogram4_sampler_preset: 'V4_TURBO_12',
      ideogram4_steps: 26,
      ideogram4_guidance_scale: 4.5,
      ideogram4_mu: -1.5,
      ideogram4_color_palette: ['green', 'gold'],
    };
    expect(recall(metadata)?.values).toMatchObject({
      ideogram4SamplerPreset: 'V4_TURBO_12',
      ideogram4Steps: 26,
      ideogram4GuidanceScale: 4.5,
      ideogram4Mu: -1.5,
      ideogram4ColorPalette: ['green', 'gold'],
    });
    expect(
      recall({ model, ideogram4_steps: null, ideogram4_guidance_scale: null, ideogram4_mu: null })?.values
    ).toMatchObject({ ideogram4Steps: null, ideogram4GuidanceScale: null, ideogram4Mu: null });
    expect(recall({ model, generation_mode: 'ideogram4_txt2img', steps: 49, cfg_scale: 1.5 })?.values).toMatchObject({
      steps: 49,
      cfgScale: 1.5,
      ideogram4Steps: null,
      ideogram4GuidanceScale: null,
      ideogram4Mu: null,
    });
    expect(recall({ model, ideogram4_steps: 2.5, ideogram4_mu: 100 })?.values).toMatchObject({
      ideogram4Steps: 70,
      ideogram4Mu: 3,
    });
  });
});

it.each(['array', 'entities'] as const)(
  'restores historical Canvas referenceImages %s through Generate recall',
  (shape) => {
    const ip = { base: 'sdxl', type: 'ip_adapter', key: 'ip', name: 'IP' } as ComponentModelConfig;
    const entry = {
      id: 'old',
      isEnabled: false,
      ipAdapter: {
        type: 'ip_adapter',
        model: ip,
        image: { image_name: 'portrait.png', width: 320, height: 200 },
        weight: 0.3,
        method: 'composition',
        clipVisionModel: 'ViT-L',
        beginEndStepPct: [0.1, 0.8],
      },
    };
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues(),
      image,
      kind: 'all',
      metadata: { canvas_v2_metadata: { referenceImages: shape === 'array' ? [entry] : { entities: [entry] } } },
      models: [sdxlModel, ip],
      supportedModels: [sdxlModel],
      vaeModels: [],
    });
    expect(result?.values.referenceImages).toEqual([
      {
        id: 'old',
        isEnabled: false,
        config: { ...entry.ipAdapter, image: { original: { image: entry.ipAdapter.image } } },
      },
    ]);
  }
);

it.each([false, true])('keeps modern references authoritative over v4 aliases (empty=%s)', (empty) => {
  const ip = { base: 'sdxl', type: 'ip_adapter', key: 'ip', name: 'IP' } as ComponentModelConfig;
  const legacy = {
    ip_adapter_model: ip,
    image: [{ image_name: 'legacy.png', width: 200, height: 100 }],
    weight: 0.5,
    clip_vision_model: 'ViT-G',
    method: 'style',
    begin_step_percent: 0.2,
    end_step_percent: 0.8,
  };
  const modern = {
    id: 'modern',
    isEnabled: true,
    config: {
      type: 'ip_adapter',
      model: ip,
      image: { image_name: 'modern.png', width: 400, height: 200 },
      weight: 0.4,
      clipVisionModel: 'ViT-H',
      method: 'full',
      beginEndStepPct: [0, 1],
    },
  };
  const input = {
    currentValues: createValues({ referenceImages: normalizeReferenceImages([modern]) }),
    image,
    kind: 'all' as const,
    models: [sdxlModel, ip],
    supportedModels: [sdxlModel],
    vaeModels: [],
  };
  const result = buildImageRecallSettings({
    mayBeWorkflowRun: false,
    ...input,
    metadata: { ref_images: empty ? [] : [modern], ipAdapters: [legacy] },
  });
  expect(result?.values.referenceImages.map(({ id }) => id)).toEqual(empty ? [] : ['modern']);
  const legacyResult = buildImageRecallSettings({
    mayBeWorkflowRun: false,
    ...input,
    metadata: { ipAdapters: [legacy] },
  });
  expect(legacyResult?.values.referenceImages).toEqual([
    expect.objectContaining({
      config: expect.objectContaining({
        model: ip,
        image: { original: { image: { image_name: 'legacy.png', width: 200, height: 100 } } },
        clipVisionModel: 'ViT-G',
        method: 'style',
        weight: 0.5,
        beginEndStepPct: [0.2, 0.8],
      }),
    }),
  ]);
});

describe('CFG scale schedule recall', () => {
  it('recalls the initial step value from a valid CFG schedule array', () => {
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({ cfgScale: 7 }),
      image,
      kind: 'all',
      metadata: { cfg_scale: [6.5, 4.5, 2.5], model: sdxlModel },
      models: [sdxlModel],
      supportedModels: [sdxlModel],
      vaeModels: [],
    });

    expect(result?.fields).toContain('cfg');
    expect(result?.values.cfgScale).toBe(6.5);
  });

  it('recalls the initial step value from a valid guidance schedule array for distilled-guidance models', () => {
    const fluxModel: MainModelConfig = { base: 'flux', key: 'flux-model', name: 'FLUX', type: 'main' };
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({ cfgScale: 3.5, model: fluxModel, modelKey: fluxModel.key }),
      image,
      kind: 'all',
      metadata: { guidance: [4.2, 3.0], model: fluxModel },
      models: [fluxModel],
      supportedModels: [fluxModel],
      vaeModels: [],
    });

    expect(result?.fields).toContain('cfg');
    expect(result?.values.cfgScale).toBe(4.2);
  });

  it.each<[unknown[]]>([[[]], [[6.5, 'bad']], [[6.5, Number.NaN]], [[0.5, 6]]])(
    'ignores invalid or below-minimum CFG schedule array %j without overwriting cfgScale',
    (cfgScale) => {
      const result = buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues({ cfgScale: 7 }),
        image,
        kind: 'all',
        metadata: { cfg_scale: cfgScale, model: sdxlModel, positive_prompt: 'keep recall active' },
        models: [sdxlModel],
        supportedModels: [sdxlModel],
        vaeModels: [],
      });

      expect(result?.fields).not.toContain('cfg');
      expect(result?.values.cfgScale).toBe(7);
    }
  );
});

describe('Wan LoRA expert target recall', () => {
  const wanMain: MainModelConfig = {
    base: 'wan',
    format: 'diffusers',
    key: 'wan-main',
    name: 'Wan A14B',
    type: 'main',
    variant: 't2v_a14b',
  };
  const wanLoraHigh: LoraModelConfig = {
    base: 'wan',
    key: 'wan-high',
    name: 'Wan High',
    type: 'lora',
    variant: 'a14b',
  };
  const wanLoraLow: LoraModelConfig = { base: 'wan', key: 'wan-low', name: 'Wan Low', type: 'lora', variant: 'a14b' };
  const wanLoraDual: LoraModelConfig = {
    base: 'wan',
    key: 'wan-dual',
    name: 'Wan Dual',
    type: 'lora',
    variant: 'a14b',
  };

  it('preserves target and low_noise_weight and folds dual high/low entries for the same LoRA', () => {
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({ model: wanMain, modelKey: wanMain.key }),
      image,
      kind: 'all',
      metadata: {
        generation_mode: 'wan_txt2img',
        loras: [
          { model: wanLoraHigh, target: 'high', weight: 0.9 },
          { model: wanLoraLow, target: 'low', weight: 0.6 },
          { low_noise_weight: 0.35, model: wanLoraDual, target: 'both', weight: 0.85 },
        ],
        model: wanMain,
      },
      models: [wanMain, wanLoraHigh, wanLoraLow, wanLoraDual],
      supportedModels: [wanMain],
      vaeModels: [],
    });

    expect(result?.fields).toContain('loras');
    expect(result?.values.loras).toEqual([
      { isEnabled: true, model: wanLoraHigh, target: 'high', weight: 0.9 },
      { isEnabled: true, model: wanLoraLow, target: 'low', weight: 0.6 },
      { isEnabled: true, lowNoiseWeight: 0.35, model: wanLoraDual, target: 'both', weight: 0.85 },
    ]);

    const folded = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({ model: wanMain, modelKey: wanMain.key }),
      image,
      kind: 'all',
      metadata: {
        generation_mode: 'wan_txt2img',
        loras: [
          { model: wanLoraDual, target: 'low', weight: 0.4 },
          { model: wanLoraDual, target: 'high', weight: 0.8 },
        ],
        model: wanMain,
      },
      models: [wanMain, wanLoraDual],
      supportedModels: [wanMain],
      vaeModels: [],
    });

    expect(folded?.skipped).toEqual([]);
    expect(folded?.values.loras).toEqual([
      { isEnabled: true, lowNoiseWeight: 0.4, model: wanLoraDual, target: 'both', weight: 0.8 },
    ]);
  });
});

describe('a record of VAE settings only', () => {
  it.each([
    [{ vae_precision: 'fp16' }, { vaePrecision: 'fp16' }],
    [{ color_compensation: true }, { colorCompensation: true }],
  ] as const)('enables recall actions for the VAE-only record %j, which recall applies', (metadata, restored) => {
    const input = {
      currentValues: createValues({ model: sdxlModel, modelKey: sdxlModel.key }),
      image,
      metadata,
      models: [],
      supportedModels: [],
      vaeModels: [],
    };
    const capabilities = getImageRecallCapabilities(input);

    expect(capabilities).toMatchObject({ all: true, remix: true });
    expect(buildImageRecallSettings({ mayBeWorkflowRun: false, ...input, kind: 'all' })?.values).toMatchObject(
      restored
    );
  });
});

describe('Diffusers main that records itself as its component source', () => {
  const projectSettings = { useCpuNoise: true, useNsfwChecker: false, useWatermark: false };
  const sourceOf = (graph: { nodes: Record<string, Record<string, unknown>> }, field: string) => {
    const metadata = Object.values(graph.nodes).find((node) => node.type === 'core_metadata');

    return (metadata![field] as { key: string }).key;
  };

  it.each([['z-image', 'qwen3_source']] as const)(
    'recalls a %s record onto a panel holding another source without keeping it',
    (base, key) => {
      const { model, settings } = satisfiedSettingsFor(base, {
        label: 'diffusers',
        overrides: { format: 'diffusers' },
      });
      const original = compileGenerateGraph(
        { ...settings, componentSourceModel: null },
        model,
        'gallery',
        projectSettings
      );
      const other = { ...model, key: `other-${base}-source`, name: 'Other source' } as MainModelConfig;
      const result = buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues({ ...settings, componentSourceModel: other }),
        image,
        kind: 'all',
        metadata: Object.values(original.backendGraph.nodes).find((node) => node.type === 'core_metadata'),
        models: [model, other],
        supportedModels: [model],
        vaeModels: [],
      });
      const recalled = compileGenerateGraph(result!.values, result!.values.model, 'gallery', projectSettings);

      expect(sourceOf(original.backendGraph, key)).toBe(model.key);
      expect(result?.values.componentSourceModel).toBeNull();
      expect(sourceOf(recalled.backendGraph, key)).toBe(model.key);
    }
  );
});

// Their loaders read a source only beside a main that bundles no components, so a source left in the panel must
// reach neither the loader nor the record, or Image Details would name a model that did not run.
describe('Diffusers main beside a component source its loader never reads', () => {
  const projectSettings = { useCpuNoise: true, useNsfwChecker: false, useWatermark: false };
  const SOURCE_FIELDS = ['component_source', 'mistral_source_model', 'qwen3_source_model'];
  const RECORD_KEYS = ['qwen_image_component_source', 'wan_component_source', 'mistral_source', 'qwen3_source'];

  it.each([
    ['qwen-image', {}],
    ['wan', {}],
    ['flux2', { variant: 'klein_4b' }],
    ['flux2', { variant: 'dev' }],
  ] as const)('sends and records no source for a %s %j main', (base, variant) => {
    const shape = { label: 'diffusers', overrides: { format: 'diffusers', ...variant } };
    const { model, settings } = satisfiedSettingsFor(base, shape);
    const source = { ...model, key: `${base}-panel-source`, name: 'Panel source' } as MainModelConfig;
    const { backendGraph } = compileGenerateGraph(
      { ...settings, componentSourceModel: source },
      model,
      'gallery',
      projectSettings
    );
    const loader = backendGraph.nodes.model_loader!;
    const metadata = Object.values(backendGraph.nodes).find((node) => node.type === 'core_metadata')!;

    expect(SOURCE_FIELDS.filter((field) => loader[field] !== undefined)).toEqual([]);
    expect(RECORD_KEYS.filter((key) => key in metadata)).toEqual([]);
  });

  it('still sends and records the source of a single-file main, whose loader reads it', () => {
    const { model, settings } = satisfiedSettingsFor('wan', { label: 'gguf', overrides: { format: 'gguf_quantized' } });
    const source = { ...model, format: 'diffusers', key: 'wan-source', name: 'Source' } as MainModelConfig;
    const { backendGraph } = compileGenerateGraph(
      { ...settings, componentSourceModel: source },
      model,
      'gallery',
      projectSettings
    );
    const metadata = Object.values(backendGraph.nodes).find((node) => node.type === 'core_metadata')!;

    expect(backendGraph.nodes.model_loader!.component_source).toMatchObject({ key: source.key });
    expect(metadata.wan_component_source).toMatchObject({ key: source.key });
  });
});

describe('Diffusers main that supplied its own VAE', () => {
  const projectSettings = { useCpuNoise: true, useNsfwChecker: false, useWatermark: false };
  const coreMetadata = (graph: { nodes: Record<string, Record<string, unknown>> }) =>
    Object.values(graph.nodes).find((node) => node.type === 'core_metadata')!;

  // What each family's loader takes from a bundled Diffusers main when no standalone model is picked. Written out by
  // hand from the loaders; the record leaves these slots out, so a recall must not let a panel pick take them over.
  const BUNDLED_CASES = [
    ['z-image', {}, ['qwen3EncoderModel', 'vae']],
    ['qwen-image', {}, ['qwenVLEncoderModel', 'vae']],
    ['wan', {}, ['wanT5EncoderModel', 'vae']],
    ['flux2', { variant: 'klein_4b' }, ['qwen3EncoderModel', 'vae']],
    ['flux2', { variant: 'dev' }, ['mistralEncoderModel', 'vae']],
    ['krea-2', {}, ['qwen3VLEncoderModel', 'vae']],
    ['flux3', {}, ['qwen3VLEncoderModel', 'vae']],
    ['moss-image', {}, ['qwen35EncoderModel', 'vae']],
    ['qwen-image-2-1', {}, ['qwen3VLEncoderModel', 'vae']],
    ['ernie-image', {}, ['mistralEncoderModel', 'vae']],
    ['ideogram-4', {}, ['ideogram4UnconditionalModel', 'qwen3VLEncoderModel', 'vae']],
  ] as const;

  it.each(BUNDLED_CASES)(
    'clears the standalone components a panel holds when a %s record left out what its bundled main supplied',
    (base, variant, cleared) => {
      const shape = { label: 'diffusers', overrides: { format: 'diffusers', ...variant } };
      const { model, settings } = satisfiedSettingsFor(base, shape);
      const { settings: filled } = fullyFilledSettingsFor(base, shape);
      const recorded = {
        ...settings,
        componentSourceModel: null,
        ...Object.fromEntries(cleared.map((key) => [key, null])),
      };
      const original = compileGenerateGraph(recorded, model, 'gallery', projectSettings);
      const panel = { ...filled, componentSourceModel: null };
      const picks = cleared.filter((key) => panel[key]);
      const result = buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues(panel),
        image,
        kind: 'all',
        metadata: coreMetadata(original.backendGraph),
        ...recallCatalogFor(model, panel),
      });

      expect(picks).toEqual(cleared);
      for (const key of cleared) {
        expect(result?.values[key], key).toBeNull();
      }
    }
  );

  it.each([
    ['z-image', 'flux', null],
    ['qwen-image', 'qwen-image', null],
    ['wan', 'wan', 16],
  ] as const)('drops the panel VAE when a %s record names no VAE', (base, vaeBase, latentChannels) => {
    const { model, settings } = satisfiedSettingsFor(base, { label: 'diffusers', overrides: { format: 'diffusers' } });
    const original = compileGenerateGraph(
      { ...settings, componentSourceModel: null, vae: null },
      model,
      'gallery',
      projectSettings
    );
    const panelVae = {
      base: vaeBase,
      key: `panel-${base}-vae`,
      name: 'Panel VAE',
      type: 'vae',
      latent_channels: latentChannels,
    } as VaeModelConfig;
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({ ...settings, componentSourceModel: null, vae: panelVae }),
      image,
      kind: 'all',
      metadata: coreMetadata(original.backendGraph),
      models: [model],
      supportedModels: [model],
      vaeModels: [panelVae],
    });
    const recalled = compileGenerateGraph(result!.values, result!.values.model, 'gallery', projectSettings);

    expect(coreMetadata(original.backendGraph)).not.toHaveProperty('vae');
    expect(result?.values.vae).toBeNull();
    expect(coreMetadata(recalled.backendGraph)).not.toHaveProperty('vae');
  });
});

describe('FLUX.2 dev record whose standalone encoder is no longer installed', () => {
  it('keeps the recorded VAE source beside the panel encoder that replaces it, and drops the panel VAE', () => {
    const { model, otherModels, settings } = flux2DevSourceRecallCase();
    const vaeSource = { ...settings.componentSourceModel!, key: 'flux2-klein-vae-source', variant: 'klein_9b' };
    const recorded = { ...settings, componentSourceModel: vaeSource };
    const metadata = Object.values(
      compileGenerateGraph(recorded, model, 'gallery', {
        useCpuNoise: true,
        useNsfwChecker: false,
        useWatermark: false,
      }).backendGraph.nodes
    ).find((node) => node.type === 'core_metadata');
    const panelEncoder = { ...settings.mistralEncoderModel!, key: 'panel-mistral', name: 'Panel Mistral' };
    const panelVae: VaeModelConfig = { base: 'flux2', key: 'panel-vae', name: 'Panel VAE', type: 'vae' };
    const installed = [...otherModels, ...recallCatalogFor(model, recorded).models].filter(
      (candidate) => candidate.key !== settings.mistralEncoderModel!.key
    );
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({
        ...recorded,
        componentSourceModel: null,
        mistralEncoderModel: panelEncoder,
        vae: panelVae,
      }),
      image,
      kind: 'all',
      metadata,
      models: [...installed, panelEncoder],
      supportedModels: [model],
      vaeModels: [panelVae],
    });

    expect(metadata).not.toHaveProperty('vae');
    expect(result?.values.mistralEncoderModel?.key).toBe('panel-mistral');
    expect(result?.values.componentSourceModel?.key).toBe(vaeSource.key);
    expect(result?.values.vae).toBeNull();
  });
});

describe('FLUX.2 dev record recalled over a panel encoder of another family', () => {
  it('does not let an encoder the model rejects decide that the recorded source supplies the VAE', () => {
    const { model, otherModels, settings } = flux2DevSourceRecallCase();
    const vaeSource = { ...settings.componentSourceModel!, key: 'flux2-klein-vae-source', variant: 'klein_9b' };
    const recorded = { ...settings, componentSourceModel: vaeSource };
    const metadata = Object.values(
      compileGenerateGraph(recorded, model, 'gallery', {
        useCpuNoise: true,
        useNsfwChecker: false,
        useWatermark: false,
      }).backendGraph.nodes
    ).find((node) => node.type === 'core_metadata');
    const foreignEncoder = { ...settings.mistralEncoderModel!, key: 'sdxl-clip', base: 'sdxl', type: 'clip_embed' };
    const panelVae: VaeModelConfig = { base: 'flux2', key: 'panel-vae', name: 'Panel VAE', type: 'vae' };
    const installed = [...otherModels, ...recallCatalogFor(model, recorded).models].filter(
      (candidate) => candidate.key !== settings.mistralEncoderModel!.key
    );
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({
        ...recorded,
        componentSourceModel: null,
        mistralEncoderModel: foreignEncoder as typeof settings.mistralEncoderModel,
        vae: panelVae,
      }),
      image,
      kind: 'all',
      metadata,
      models: installed,
      supportedModels: [model],
      vaeModels: [panelVae],
    });

    expect(result?.values.mistralEncoderModel).toBeNull();
    expect(result?.values.vae?.key).toBe('panel-vae');
  });
});

describe('FLUX.2 dev record whose source supplies only the VAE', () => {
  it('restores the recorded source beside the recorded standalone encoder, ahead of another pipeline', () => {
    const { model, otherModels, settings } = flux2DevSourceRecallCase();
    // A Klein pipeline gives a dev model its VAE but not its encoder, so the Mistral encoder stays standalone.
    const vaeSource = { ...settings.componentSourceModel!, key: 'flux2-klein-vae-source', variant: 'klein_9b' };
    const recorded = { ...settings, componentSourceModel: vaeSource };
    const record = compileGenerateGraph(recorded, model, 'gallery', {
      useCpuNoise: true,
      useNsfwChecker: false,
      useWatermark: false,
    }).backendGraph.nodes;
    const catalog = recallCatalogFor(model, recorded);
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({ ...recorded, componentSourceModel: null, mistralEncoderModel: null }),
      image,
      kind: 'all',
      metadata: Object.values(record).find((node) => node.type === 'core_metadata'),
      models: [...otherModels, ...catalog.models],
      supportedModels: [model],
      vaeModels: [],
    });

    expect(result?.values.mistralEncoderModel?.key).toBe(settings.mistralEncoderModel?.key);
    expect(result?.values.componentSourceModel?.key).toBe(vaeSource.key);
  });
});

describe('FLUX.2 dev record whose source supplies the encoder and the VAE', () => {
  it('clears the standalone encoder and VAE a panel holds, so the recorded source supplies them again', () => {
    const { model, otherModels, settings } = flux2DevSourceRecallCase();
    const standaloneVae: VaeModelConfig = {
      base: 'flux2',
      key: 'flux2-standalone-vae',
      name: 'FLUX.2 VAE',
      type: 'vae',
    };
    const recorded = { ...settings, mistralEncoderModel: null, vae: null };
    const record = compileGenerateGraph(recorded, model, 'gallery', {
      useCpuNoise: true,
      useNsfwChecker: false,
      useWatermark: false,
    }).backendGraph.nodes;
    const metadata = Object.values(record).find((node) => node.type === 'core_metadata');
    const catalog = recallCatalogFor(model, recorded);
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({
        ...recorded,
        mistralEncoderModel: settings.mistralEncoderModel,
        vae: standaloneVae,
      }),
      image,
      kind: 'all',
      metadata,
      models: [...otherModels, ...catalog.models, settings.mistralEncoderModel!],
      supportedModels: [model],
      vaeModels: [standaloneVae],
    });

    expect(metadata).toMatchObject({ mistral_source: { key: settings.componentSourceModel!.key } });
    expect(metadata).not.toHaveProperty('mistral_encoder');
    expect(metadata).not.toHaveProperty('vae');
    expect(result?.values.componentSourceModel?.key).toBe(settings.componentSourceModel!.key);
    expect(result?.values.mistralEncoderModel).toBeNull();
    expect(result?.values.vae).toBeNull();
  });
});

describe('FLUX.2 dev record naming a source and a standalone component that is not installed', () => {
  const setup = (recordedVae: VaeModelConfig | null) => {
    const { model, otherModels, settings } = flux2DevSourceRecallCase();
    const recorded = { ...settings, vae: recordedVae };
    const record = compileGenerateGraph(recorded, model, 'gallery', {
      useCpuNoise: true,
      useNsfwChecker: false,
      useWatermark: false,
    }).backendGraph.nodes;
    const metadata = Object.values(record).find((node) => node.type === 'core_metadata');
    const uninstalled = new Set([settings.mistralEncoderModel!.key, recordedVae?.key]);
    const installed = [...otherModels, ...recallCatalogFor(model, recorded).models].filter(
      (candidate) => !uninstalled.has(candidate.key)
    );

    return { installed, metadata, model, recorded, settings };
  };

  it('keeps the panel encoder when the recorded standalone encoder is not installed', () => {
    const { installed, metadata, model, recorded, settings } = setup(null);
    const panelEncoder = { ...settings.mistralEncoderModel!, key: 'panel-mistral', name: 'Panel Mistral' };
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({ ...recorded, mistralEncoderModel: panelEncoder }),
      image,
      kind: 'all',
      metadata,
      models: [...installed, panelEncoder],
      supportedModels: [model],
      vaeModels: [],
    });

    expect(metadata).toHaveProperty('mistral_encoder');
    expect(metadata).toHaveProperty('mistral_source');
    expect(result?.values.mistralEncoderModel?.key).toBe('panel-mistral');
  });

  it('keeps the panel VAE when the recorded VAE is not installed', () => {
    const recordedVae: VaeModelConfig = { base: 'flux2', key: 'recorded-vae', name: 'Recorded VAE', type: 'vae' };
    const panelVae: VaeModelConfig = { base: 'flux2', key: 'panel-vae', name: 'Panel VAE', type: 'vae' };
    const { installed, metadata, model, recorded } = setup(recordedVae);
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({ ...recorded, vae: panelVae }),
      image,
      kind: 'all',
      metadata,
      models: installed,
      supportedModels: [model],
      vaeModels: [panelVae],
    });

    expect(metadata).toHaveProperty('vae');
    expect(result?.values.vae?.key).toBe('panel-vae');
  });
});

describe('a record that names no model', () => {
  const recall = (panel: Partial<GenerateWidgetValues>, models: ComponentModelConfig[], vaeModels: VaeModelConfig[]) =>
    buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues(panel),
      image,
      kind: 'all',
      metadata: { positive_prompt: 'a lighthouse at dusk' },
      models,
      supportedModels: [panel.model as MainModelConfig],
      vaeModels,
    })?.values;

  it('restores the prompt and keeps the standalone encoder of a panel model that bundles its own', () => {
    const { model, settings } = satisfiedSettingsFor('qwen-image', {
      label: 'diffusers',
      overrides: { format: 'diffusers' },
    });
    const { settings: filled } = fullyFilledSettingsFor('qwen-image', {
      label: 'diffusers',
      overrides: { format: 'diffusers' },
    });
    const encoder = filled.qwenVLEncoderModel!;
    const values = recall(
      { ...settings, model, modelKey: model.key, qwenVLEncoderModel: encoder },
      [model, encoder],
      []
    );

    expect(values?.positivePrompt).toBe('a lighthouse at dusk');
    expect(values?.qwenVLEncoderModel?.key).toBe(encoder.key);
  });

  it('keeps the VAE override of an SDXL panel', () => {
    const vae: VaeModelConfig = { base: 'sdxl', key: 'sdxl-vae-override', name: 'SDXL VAE', type: 'vae' };
    const values = recall({ model: sdxlModel, modelKey: sdxlModel.key, vae }, [sdxlModel], [vae]);

    expect(values?.positivePrompt).toBe('a lighthouse at dusk');
    expect(values?.vae?.key).toBe(vae.key);
  });
});

/**
 * Recall All of a compiled image onto a panel holding standalone components the image never used must rebuild a graph
 * that loads what the original loaded: the record leaves out what the main or its source supplied.
 */
describe('recall onto a panel holding unused standalone components', () => {
  const SD_VAE_SLOT = { key: 'vae', label: 'VAE', modelTypes: ['vae'], valueKind: 'vae' } as const;
  const componentKeys = (graph: Parameters<typeof getRecordedComponents>[0]) =>
    Object.fromEntries(
      Object.entries(getRecordedComponents(graph)).map(([key, value]) => [key, (value as { key: string }).key])
    );

  const roundTrip = (
    model: GenerateModelConfig,
    recorded: GenerateSettings,
    panel: GenerateSettings,
    otherModels: readonly ComponentModelConfig[] = []
  ) => {
    const original = compileGenerateGraph(recorded, model, 'gallery', PROJECT_SETTINGS).backendGraph;
    const catalogs = [recallCatalogFor(model, recorded), recallCatalogFor(model, panel)];
    const result = buildImageRecallSettings({
      mayBeWorkflowRun: false,
      currentValues: createValues({ ...panel, model, modelKey: model.key }),
      image,
      kind: 'all',
      // The server stamps the version of every record it stores.
      metadata: {
        ...Object.values(original.nodes).find((node) => node.type === 'core_metadata'),
        app_version: '7.0.0',
      },
      models: [...otherModels, ...catalogs.flatMap((catalog) => catalog.models)],
      supportedModels: [model],
      vaeModels: catalogs.flatMap((catalog) => catalog.vaeModels),
    })!;
    const rebuilt = compileGenerateGraph(result.values, result.values.model, 'gallery', PROJECT_SETTINGS).backendGraph;

    return { original: componentKeys(original), rebuilt: componentKeys(rebuilt) };
  };

  /**
   * `settings` with every empty component slot, the SD VAE override and a FLUX.2 pipeline (a source the policy lists no
   * slot for) filled by a compatible standalone model.
   */
  const withStandaloneComponents = (model: GenerateModelConfig, settings: GenerateSettings): GenerateSettings => {
    let panel = settings;

    for (let pass = 0; pass < 3; pass += 1) {
      for (const slot of getComponentSectionPolicy(model, panel).slots) {
        const context = getComponentPolicyContext(model, panel);
        const candidate = panel[slot.key]
          ? undefined
          : candidatesForSlot(slot).find((c) => c.key !== model.key && (!slot.filter || slot.filter(c, context)));

        if (candidate) {
          panel = { ...panel, [slot.key]: candidate };
        }
      }
    }

    const sdVae =
      getGenerationUiPolicy(model).sdVaeVisible && !panel.vae
        ? candidatesForSlot(SD_VAE_SLOT).find((c) => isVaeCompatibleWithGenerateModel(model, c as VaeModelConfig))
        : undefined;
    const flux2Source =
      model.base === 'flux2' && !panel.componentSourceModel
        ? ({ ...model, format: 'diffusers', key: `${model.key}-panel-source`, submodels: undefined } as MainModelConfig)
        : undefined;

    return {
      ...panel,
      ...(sdVae ? { vae: sdVae as VaeModelConfig } : {}),
      ...(flux2Source ? { componentSourceModel: flux2Source } : {}),
    };
  };

  it('loads the VAE a Qwen-Image Diffusers main supplied, not the panel pick, though a source sat in the panel', () => {
    const shape = { label: 'diffusers', overrides: { format: 'diffusers' } };
    const { model, settings } = satisfiedSettingsFor('qwen-image', shape);
    const source = { ...model, key: 'qwen-image-fallback-source', name: 'Qwen-Image source' } as MainModelConfig;
    const recorded = { ...settings, componentSourceModel: source, qwenVLEncoderModel: null, vae: null };
    const panelVae: VaeModelConfig = { base: 'qwen-image', key: 'panel-qwen-vae', name: 'Panel VAE', type: 'vae' };
    const { original, rebuilt } = roundTrip(model, recorded, {
      ...recorded,
      componentSourceModel: null,
      vae: panelVae,
    });

    expect(original).toEqual({});
    expect(rebuilt).toEqual(original);
  });

  it('loads the VAE an SDXL main bundled, not the panel override', () => {
    const { model, settings } = satisfiedSettingsFor('sdxl', {
      label: 'diffusers',
      overrides: { format: 'diffusers' },
    });
    const panelVae: VaeModelConfig = { base: 'sdxl', key: 'panel-sdxl-vae', name: 'Panel VAE', type: 'vae' };
    const { original, rebuilt } = roundTrip(model, { ...settings, vae: null }, { ...settings, vae: panelVae });

    expect(original).toEqual({});
    expect(rebuilt).toEqual(original);
  });

  /**
   * The shape with only what it needs, then with a Diffusers source in place of the standalone models it can skip,
   * where the graph uses it and the panel keeps it (the model selection drops a FLUX.2 source a bundled main ignores).
   */
  const minimalCases = () =>
    generateGraphCases.flatMap(({ base, label, shape }) => {
      const { model, settings } = satisfiedSettingsFor(base, shape);
      const source = {
        ...model,
        format: 'diffusers',
        key: `${model.key}-source`,
        submodels: undefined,
      } as MainModelConfig;
      let sourced: GenerateSettings = { ...settings, componentSourceModel: source };

      for (const slot of getComponentSectionPolicy(model, sourced).slots) {
        if (slot.key !== 'componentSourceModel' && !slot.required?.(getComponentPolicyContext(model, sourced))) {
          sourced = { ...sourced, [slot.key]: null };
        }
      }

      const recorded = componentKeys(compileGenerateRecordCase({ label, model, settings: sourced }).backendGraph);
      const keepsSource =
        Object.values(recorded).includes(source.key) &&
        (base !== 'flux2' || getAutoFlux2ComponentSourceModel(model, sourced, [source])?.key === source.key);

      return [
        { label: `${label} (minimal)`, model, settings },
        ...(keepsSource ? [{ label: `${label} (minimal, component source)`, model, settings: sourced }] : []),
      ];
    });

  it('rebuilds the components of every Generate record case and minimal shape', () => {
    const cases = [...generateRecordCases(), ...minimalCases()];
    const failures = cases.flatMap(({ label, model, settings, ...testCase }) => {
      const otherModels = 'otherModels' in testCase ? (testCase.otherModels ?? []) : [];
      const { original, rebuilt } = roundTrip(model, settings, withStandaloneComponents(model, settings), otherModels);

      return [...new Set([...Object.keys(original), ...Object.keys(rebuilt)])]
        .filter((key) => original[key] !== rebuilt[key])
        .map((key) => `${label}: \`${key}\` was ${original[key] ?? 'absent'}, rebuilt ${rebuilt[key] ?? 'absent'}`);
    });

    expect(failures).toEqual([]);
  });

  describe('records of writers that left out what they used', () => {
    const recallStripped = (
      model: GenerateModelConfig,
      recorded: GenerateSettings,
      panel: GenerateSettings,
      stripped: readonly string[],
      appVersion: string,
      options: { kind?: 'all' | 'remix'; mayBeWorkflowRun?: boolean } = {}
    ) => {
      const original = compileGenerateGraph(recorded, model, 'gallery', PROJECT_SETTINGS).backendGraph;
      const metadata: Record<string, unknown> = {
        ...Object.values(original.nodes).find((node) => node.type === 'core_metadata'),
        app_version: appVersion,
      };

      for (const key of stripped) {
        delete metadata[key];
      }

      const catalogs = [recallCatalogFor(model, recorded), recallCatalogFor(model, panel)];

      return buildImageRecallSettings({
        currentValues: createValues({ ...panel, model, modelKey: model.key }),
        image,
        kind: options.kind ?? 'all',
        mayBeWorkflowRun: options.mayBeWorkflowRun ?? false,
        metadata,
        models: catalogs.flatMap((catalog) => catalog.models),
        supportedModels: [model],
        vaeModels: catalogs.flatMap((catalog) => catalog.vaeModels),
      })!.values;
    };

    const splitWithSource = (base: 'wan' | 'z-image') => {
      const { model, settings } = satisfiedSettingsFor(base, {
        label: 'gguf',
        overrides: { format: 'gguf_quantized' },
      });
      const source = { ...model, format: 'diffusers', key: `${base}-source`, name: 'Source' } as MainModelConfig;
      const encoder = base === 'wan' ? 'wanT5EncoderModel' : 'qwen3EncoderModel';

      return { model, panel: { ...settings, componentSourceModel: source, [encoder]: null, vae: null }, source };
    };

    // A workflow can wire a denoise node's `metadata` output straight to its decoder; the node records no components.
    it('keeps the source a split Wan panel selected for a record a Wan denoise node wrote', () => {
      const { model, panel, source } = splitWithSource('wan');
      const compileComponents = (settings: GenerateSettings) =>
        componentKeys(compileGenerateGraph(settings, model, 'gallery', PROJECT_SETTINGS).backendGraph);
      // As `WanDenoiseInvocation._build_metadata` and `build_denoise_metadata` write it without a metadata input.
      const metadata = {
        add_noise: true,
        app_version: '7.0.0-beta.2-post1',
        cfg_scale: panel.cfgScale,
        denoising_end: 1,
        denoising_start: 0,
        generation_mode: 'wan_txt2img',
        guidance: panel.cfgScale,
        height: 512,
        loras: [],
        metadata_version: '1.4.0',
        model: { base: model.base, hash: 'hash', key: model.key, name: model.name, type: model.type },
        seed: 7,
        steps: panel.steps,
        width: 512,
      };
      const values = buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues({ ...panel, model, modelKey: model.key }),
        image,
        kind: 'all',
        metadata,
        ...recallCatalogFor(model, panel),
      })!.values;

      expect(compileComponents(panel)).toEqual({ wan_component_source: source.key });
      expect(values.componentSourceModel?.key).toBe(source.key);
      expect(compileComponents(values)).toEqual(compileComponents(panel));
    });

    // The legacy Z-Image builder writes no `rand_device`, and v6.10.0rc1 sent a source it never recorded.
    it('keeps the source a split Z-Image panel selected for a legacy frontend record that omits it', () => {
      const { model, panel, source } = splitWithSource('z-image');
      const values = recallStripped(model, panel, panel, ['qwen3_source', 'rand_device'], '6.10.0rc1');

      expect(values.componentSourceModel?.key).toBe(source.key);
    });

    // The legacy SD 3 builder passed standalone encoders to the loader and recorded none of them.
    it.each([
      ['', true],
      ['unreleased', true],
      ['6.14.4', true],
      ['7.0.0-alpha.4', true],
      ['7.0.0-beta.1', true],
      ['7.0.0-beta.2', false],
      ['7.0.0-beta.2-post1', false],
      ['7.0.0', false],
    ] as const)('keeps the standalone SD 3 encoders for a %s record', (appVersion, keeps) => {
      const shape = { label: 'diffusers', overrides: { format: 'diffusers' } };
      const { model } = satisfiedSettingsFor('sd-3', shape);
      const { settings: filled } = fullyFilledSettingsFor('sd-3', shape);
      const encoders = ['clipGEmbedModel', 'clipLEmbedModel', 't5EncoderModel'] as const;
      const recorded = { ...filled, ...Object.fromEntries(encoders.map((key) => [key, null])) } as GenerateSettings;
      const values = recallStripped(model, recorded, filled, [], appVersion);

      expect(encoders.every((key) => filled[key] !== null)).toBe(true);
      expect(encoders.map((key) => values[key]?.key ?? null)).toEqual(
        encoders.map((key) => (keeps ? filled[key]!.key : null))
      );
    });

    // A 7.0 server stamps the version on records of the legacy frontend too; only webv2 writes `rand_device`.
    it('keeps the standalone SD 3 encoders for a legacy frontend record stamped by a 7.0 server', () => {
      const shape = { label: 'diffusers', overrides: { format: 'diffusers' } };
      const { model } = satisfiedSettingsFor('sd-3', shape);
      const { settings: filled } = fullyFilledSettingsFor('sd-3', shape);
      const encoders = ['clipGEmbedModel', 'clipLEmbedModel', 't5EncoderModel'] as const;
      const recorded = { ...filled, ...Object.fromEntries(encoders.map((key) => [key, null])) } as GenerateSettings;
      const values = recallStripped(model, recorded, filled, ['rand_device'], '7.0.0-beta.2-post1');

      expect(encoders.map((key) => values[key]?.key ?? null)).toEqual(encoders.map((key) => filled[key]!.key));
    });

    // A workflow opened from a Generate graph keeps its metadata node (`rand_device` and all) while the user picks
    // components on its loader, so the record omits an encoder the run used.
    it.each(['all', 'remix'] as const)(
      'keeps the standalone SD 3 encoders for a %s of a run that embeds a workflow',
      (kind) => {
        const shape = { label: 'diffusers', overrides: { format: 'diffusers' } };
        const { model } = satisfiedSettingsFor('sd-3', shape);
        const { settings: filled } = fullyFilledSettingsFor('sd-3', shape);
        const encoders = ['clipGEmbedModel', 'clipLEmbedModel', 't5EncoderModel'] as const;
        const recorded = { ...filled, ...Object.fromEntries(encoders.map((key) => [key, null])) } as GenerateSettings;
        const graphRun = recallStripped(model, recorded, filled, [], '7.0.0', { kind });
        const workflowRun = recallStripped(model, recorded, filled, [], '7.0.0', { kind, mayBeWorkflowRun: true });

        expect(encoders.map((key) => graphRun[key]?.key ?? null)).toEqual(encoders.map(() => null));
        expect(encoders.map((key) => workflowRun[key]?.key ?? null)).toEqual(encoders.map((key) => filled[key]!.key));
      }
    );

    it.each(['krea-2', 'qwen-image-2-1'] as const)(
      'keeps the GGUF encoder and Vision Tower a %s panel holds when the recorded encoder is not installed',
      (base) => {
        const { model, settings } = satisfiedSettingsFor(base, {
          ggufEncoder: true,
          label: 'standalone-gguf-encoder',
          overrides: { format: 'gguf_quantized' },
        });
        const ggufEncoder = settings.qwen3VLEncoderModel!;
        // The run used a regular encoder, which takes no tower; the panel's catalog does not hold it.
        const regular = { ...ggufEncoder, format: 'checkpoint', key: 'regular-encoder', name: 'Regular encoder' };
        const recorded = { ...settings, qwen3VLEncoderModel: regular, qwen3VLVisionTowerModel: null };
        const metadata = Object.values(
          compileGenerateGraph(recorded, model, 'gallery', PROJECT_SETTINGS).backendGraph.nodes
        ).find((node) => node.type === 'core_metadata');
        const panel = recallCatalogFor(model, settings);
        const values = buildImageRecallSettings({
          mayBeWorkflowRun: false,
          currentValues: createValues({ ...settings, model, modelKey: model.key }),
          image,
          kind: 'all',
          metadata,
          models: panel.models,
          supportedModels: [model],
          vaeModels: panel.vaeModels,
        })!.values;

        expect(metadata).toHaveProperty('qwen3_vl_encoder');
        expect(values.qwen3VLEncoderModel?.key).toBe(ggufEncoder.key);
        expect(values.qwen3VLVisionTowerModel?.key).toBe(settings.qwen3VLVisionTowerModel!.key);
      }
    );

    it.each(['krea-2', 'qwen-image-2-1'] as const)(
      'clears the Vision Tower a %s panel holds when the recorded run used none',
      (base) => {
        const { model, settings } = satisfiedSettingsFor(base, {
          ggufEncoder: true,
          label: 'standalone-gguf-encoder',
          overrides: { format: 'gguf_quantized' },
        });
        const { original, rebuilt } = roundTrip(model, { ...settings, qwen3VLVisionTowerModel: null }, settings);

        expect(settings.qwen3VLVisionTowerModel).toBeTruthy();
        expect(original).not.toHaveProperty('qwen3_vl_vision_tower');
        expect(rebuilt).toEqual(original);
      }
    );
  });

  describe('a recorded source that is not installed', () => {
    const recallWithoutSource = (base: 'qwen-image' | 'z-image') => {
      const shape = { label: 'diffusers', overrides: { format: 'diffusers' } };
      const { model, settings } = satisfiedSettingsFor(base, shape);
      const { settings: filled } = fullyFilledSettingsFor(base, shape);
      const source = { ...model, key: `${base}-uninstalled-source`, name: 'Uninstalled source' } as MainModelConfig;
      const recorded = { ...settings, componentSourceModel: source };
      const written = Object.values(
        compileGenerateGraph(recorded, model, 'gallery', PROJECT_SETTINGS).backendGraph.nodes
      ).find((node) => node.type === 'core_metadata');
      // Records of releases whose Qwen-Image builder sent a source to a Diffusers main's loader still name it.
      const metadata =
        base === 'qwen-image'
          ? {
              ...written,
              qwen_image_component_source: { base: model.base, key: source.key, name: source.name, type: 'main' },
            }
          : written;
      const panel = { ...filled, componentSourceModel: null, model, modelKey: model.key };
      const encoderKey = base === 'z-image' ? 'qwen3EncoderModel' : 'qwenVLEncoderModel';
      const values = buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues(panel),
        image,
        kind: 'all',
        metadata,
        models: [model, panel[encoderKey]!],
        supportedModels: [model],
        vaeModels: [panel.vae!],
      })!.values;

      return { encoder: values[encoderKey], metadata, panel, panelEncoder: panel[encoderKey]!, source, values };
    };

    it('keeps the panel encoder and VAE of a Z-Image main, whose loader may have taken both from the source', () => {
      const { encoder, metadata, panel, panelEncoder, source, values } = recallWithoutSource('z-image');

      expect(metadata).toMatchObject({ qwen3_source: { key: source.key } });
      expect(encoder?.key).toBe(panelEncoder.key);
      expect(values.vae?.key).toBe(panel.vae!.key);
    });

    it('clears the panel encoder and VAE of a Qwen-Image main, which its loader fills ahead of any source', () => {
      const { encoder, metadata, source, values } = recallWithoutSource('qwen-image');

      expect(metadata).toMatchObject({ qwen_image_component_source: { key: source.key } });
      expect(encoder).toBeNull();
      expect(values.vae).toBeNull();
    });
  });

  describe('a record naming a standalone encoder and VAE beside a panel running on a component source', () => {
    const recall = (base: 'qwen-image' | 'wan' | 'z-image', installed: boolean) => {
      const { model, settings } = satisfiedSettingsFor(base, {
        label: 'gguf',
        overrides: { format: 'gguf_quantized' },
      });
      const source = { ...model, format: 'diffusers', key: `${base}-source`, name: 'Source' } as MainModelConfig;
      const encoder = (
        { 'qwen-image': 'qwenVLEncoderModel', wan: 'wanT5EncoderModel', 'z-image': 'qwen3EncoderModel' } as const
      )[base];
      const panel = { ...settings, componentSourceModel: source, [encoder]: null, vae: null };
      const metadata = Object.values(
        compileGenerateGraph(settings, model, 'gallery', PROJECT_SETTINGS).backendGraph.nodes
      ).find((node) => node.type === 'core_metadata');
      const catalog = recallCatalogFor(model, settings);
      const values = buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues({ ...panel, model, modelKey: model.key }),
        image,
        kind: 'all',
        metadata,
        models: installed ? [...catalog.models, source] : [model, source],
        supportedModels: [model],
        vaeModels: installed ? catalog.vaeModels : [],
      })!.values;

      return { metadata, model, panel, recorded: settings, source, values, encoder };
    };

    it.each(['qwen-image', 'wan', 'z-image'] as const)(
      'keeps the source of a %s panel while the recorded encoder and VAE are not installed',
      (base) => {
        const { encoder, metadata, model, panel, recorded, source, values } = recall(base, false);

        expect(recorded[encoder]).toBeTruthy();
        expect(recorded.vae).toBeTruthy();
        expect(metadata).toMatchObject({ vae: { key: recorded.vae!.key } });
        // The panel compiled on the source before recall, and still does.
        expect(() => compileGenerateGraph(panel, model, 'gallery', PROJECT_SETTINGS)).not.toThrow();
        expect(values.componentSourceModel?.key).toBe(source.key);
        expect(() => compileGenerateGraph(values, values.model, 'gallery', PROJECT_SETTINGS)).not.toThrow();
      }
    );

    it.each(['qwen-image', 'wan', 'z-image'] as const)(
      'still clears the source of a %s panel once the recorded encoder and VAE are restored',
      (base) => {
        const { encoder, recorded, values } = recall(base, true);

        expect(values[encoder]?.key).toBe(recorded[encoder]!.key);
        expect(values.vae?.key).toBe(recorded.vae!.key);
        expect(values.componentSourceModel).toBeNull();
      }
    );
  });

  describe('a record that names no components', () => {
    const wan = satisfiedSettingsFor('wan', { label: 'gguf', overrides: { format: 'gguf_quantized' } });
    const qwenImage = satisfiedSettingsFor('qwen-image', { label: 'diffusers', overrides: { format: 'diffusers' } });
    const sdxl = satisfiedSettingsFor('sdxl', { label: 'diffusers', overrides: { format: 'diffusers' } });

    const recallOnto = (model: GenerateModelConfig, panel: GenerateSettings, metadata: Record<string, unknown>) =>
      buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues({ ...panel, model, modelKey: model.key }),
        image,
        kind: 'all',
        metadata,
        ...recallCatalogFor(model, panel),
      })!.values;
    const componentsOf = (values: GenerateSettings) =>
      Object.fromEntries(
        (['componentSourceModel', 'qwenVLEncoderModel', 'vae', 'wanLowNoiseModel', 'wanT5EncoderModel'] as const).map(
          (key) => [key, values[key]?.key ?? null]
        )
      );
    // As `buildCanvasSaveMetadata` in src/workbench/canvas-operations/saveCanvasToGallery.ts writes it.
    const canvasSave = (model: GenerateModelConfig, panel: GenerateSettings) => ({
      cfg_scale: panel.cfgScale,
      height: 512,
      metadata_source: 'canvas_save',
      model: { base: model.base, key: model.key, name: model.name, type: model.type },
      positive_prompt: 'a canvas save',
      steps: panel.steps,
      vae: panel.vae ? { base: panel.vae.base, key: panel.vae.key, name: panel.vae.name, type: 'vae' } : null,
      width: 512,
    });

    it.each([
      ['wan', wan],
      ['qwen-image', qwenImage],
    ] as const)('keeps the %s components of the panel a Canvas save recalls onto', (_base, { model, settings }) => {
      const panel = withStandaloneComponents(model, settings);
      const values = recallOnto(model, panel, canvasSave(model, panel));

      expect(componentsOf(values)).toEqual(componentsOf(panel));
    });

    it('keeps the panel VAE of an SDXL Canvas save written by an older UI, which records no VAE', () => {
      const panel = withStandaloneComponents(sdxl.model, sdxl.settings);
      const values = recallOnto(sdxl.model, panel, {
        canvas_v2_metadata: {},
        model: { key: sdxl.model.key },
        positive_prompt: 'a canvas save',
        seed: 7,
      });

      expect(values.vae?.key).toBe(panel.vae!.key);
    });

    it('clears the panel components of an Upscale record, which names those its loader used', () => {
      const testCase = UPSCALE_CASES.find(({ label }) => label === 'qwen-image-diffusers')!;
      const { installed, values: upscale } = valuesForCase(testCase);
      const metadata = Object.values(
        compileUpscaleGraph({ ...upscale, vae: null }, 'gallery', { useCpuNoise: true }).backendGraph.nodes
      ).find((node) => node.type === 'core_metadata')!;
      const model = testCase.main as MainModelConfig;
      const panel = withStandaloneComponents(model, qwenImage.settings);
      const values = buildImageRecallSettings({
        mayBeWorkflowRun: false,
        currentValues: createValues({ ...panel, model, modelKey: model.key }),
        image,
        kind: 'all',
        metadata,
        models: [...installed, ...recallCatalogFor(model, panel).models],
        supportedModels: [model],
        vaeModels: [panel.vae!],
      })!.values;

      expect(metadata).not.toHaveProperty('qwen_image_qwen_vl_encoder');
      expect(metadata).not.toHaveProperty('vae');
      expect(panel.qwenVLEncoderModel && panel.vae).toBeTruthy();
      expect(values.qwenVLEncoderModel).toBeNull();
      expect(values.vae).toBeNull();
    });
  });
});
