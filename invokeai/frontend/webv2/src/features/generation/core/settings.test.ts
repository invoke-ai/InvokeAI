import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { getSeedSequenceLength, getSeedStep, planSeedSubmission, SEED_MAX, wrapSeed } from '@platform/core/seed';
import { describe, expect, it } from 'vitest';

import type {
  ComponentModelConfig,
  GenerateWidgetValues,
  LoraModelConfig,
  MainModelConfig,
  VaeModelConfig,
} from './types';

import {
  calculateNewSize,
  cloneGenerateWidgetValues,
  isLoraCompatibleWithModel,
  isGenerateSettings,
  MAX_REFERENCE_IMAGE_NAME_LENGTH,
  normalizeGenerateSettings,
  normalizeReferenceImages,
  parseMossImageSkipLayers,
  syncGenerateWidgetValuesWithModels,
} from './settings';

const legacyStoredValues = {
  batchCount: 2,
  cfgRescaleMultiplier: 0,
  cfgScale: 7,
  height: 768,
  modelKey: 'legacy-model',
  negativePrompt: 'blurry',
  positivePrompt: 'a castle',
  scheduler: 'euler_a',
  seed: 123,
  shouldRandomizeSeed: false,
  steps: 30,
  width: 512,
};

seedArchitectureCapabilities();

describe('normalizeGenerateSettings', () => {
  it.each([null, 0, 0.4])('preserves saved HiDiffusion ratio values (%s)', (ratio) => {
    expect(
      normalizeGenerateSettings({ ...legacyStoredValues, hiDiffusionT1Ratio: ratio, hiDiffusionT2Ratio: ratio })
    ).toMatchObject({ hiDiffusionT1Ratio: ratio, hiDiffusionT2Ratio: ratio });
  });

  it('reads the seed mode saved before modes existed from the random toggle', () => {
    expect(normalizeGenerateSettings(legacyStoredValues)?.seedMode).toBe('fixed');
    expect(normalizeGenerateSettings({ ...legacyStoredValues, shouldRandomizeSeed: true })?.seedMode).toBe('random');
    expect(
      normalizeGenerateSettings({ ...legacyStoredValues, seedMode: 'decrement', shouldRandomizeSeed: true })?.seedMode
    ).toBe('decrement');
    expect(
      normalizeGenerateSettings({ ...legacyStoredValues, seedMode: 'bogus', shouldRandomizeSeed: true })?.seedMode
    ).toBe('random');
  });

  it('reads prompt tool model picks, defaulting projects saved before them to none', () => {
    expect(normalizeGenerateSettings(legacyStoredValues)).toMatchObject({
      expandPromptModelKey: null,
      imageToPromptModelKey: null,
    });
    expect(
      normalizeGenerateSettings({ ...legacyStoredValues, expandPromptModelKey: 'llm', imageToPromptModelKey: 7 })
    ).toMatchObject({ expandPromptModelKey: 'llm', imageToPromptModelKey: null });
  });

  it('normalizes persisted reference images and drops invalid entries', () => {
    const normalized = normalizeGenerateSettings({
      ...legacyStoredValues,
      referenceImages: [
        {
          id: 'ref-1',
          isEnabled: true,
          config: {
            type: 'qwen_image_reference_image',
            image: {
              height: 768,
              imageName: 'reference.png',
              imageUrl: '/api/v1/images/i/reference.png/full',
              queuedAt: '2026-01-01T00:00:00.000Z',
              sourceQueueItemId: 'backend-gallery',
              thumbnailUrl: '/api/v1/images/i/reference.png/thumbnail',
              width: 512,
            },
          },
        },
        { id: 'broken', isEnabled: true, config: { type: 'qwen_image_reference_image', image: { imageName: 3 } } },
      ],
    });

    expect(normalized?.referenceImages).toHaveLength(1);
    expect(normalized?.referenceImages[0]).toMatchObject({
      id: 'ref-1',
      isEnabled: true,
      config: {
        type: 'qwen_image_reference_image',
        image: { original: { image: { height: 768, image_name: 'reference.png', width: 512 } } },
      },
    });
    expect(normalized && isGenerateSettings(normalized)).toBe(true);
    expect(
      isGenerateSettings({
        ...normalized,
        referenceImages: [
          {
            ...normalized?.referenceImages[0],
            config: {
              ...normalized?.referenceImages[0]?.config,
              image: {
                height: 768,
                imageName: 'reference.png',
                imageUrl: '/full',
                queuedAt: 'now',
                sourceQueueItemId: 'queue',
                thumbnailUrl: '/thumbnail',
                width: 512,
              },
            },
          },
        ],
      })
    ).toBe(false);
  });

  it('upgrades legacy persisted values without losing the core fields', () => {
    const normalized = normalizeGenerateSettings(legacyStoredValues);

    expect(normalized).not.toBeNull();
    expect(normalized?.positivePrompt).toBe('a castle');
    expect(normalized?.width).toBe(512);
    expect(normalized?.height).toBe(768);
    expect(normalized?.aspectRatioId).toBe('2:3');
    expect(normalized?.aspectRatioIsLocked).toBe(false);
    expect(normalized?.clipSkip).toBe(0);
    expect(normalized?.colorCompensation).toBe(false);
    expect(normalized?.hiDiffusionEnabled).toBe(false);
    expect(normalized?.hiDiffusionRauNetEnabled).toBe(true);
    expect(normalized?.hiDiffusionWindowAttentionEnabled).toBe(true);
    expect(normalized?.hiDiffusionT1Ratio).toBeNull();
    expect(normalized?.hiDiffusionT2Ratio).toBeNull();
    expect(normalized?.loras).toEqual([]);
    expect(normalized?.negativePromptEnabled).toBe(true);
    expect(normalized?.negativePromptHeightPx).toBe(56);
    expect(normalized?.positivePromptHeightPx).toBe(96);
    expect(normalized?.seamlessXAxis).toBe(false);
    expect(normalized?.vae).toBeNull();
    expect(normalized?.vaePrecision).toBe('fp32');
    expect(normalized && isGenerateSettings(normalized)).toBe(true);
  });

  it('rejects values missing core fields', () => {
    expect(normalizeGenerateSettings({})).toBeNull();
    expect(normalizeGenerateSettings({ ...legacyStoredValues, seed: Number.NaN })).toBeNull();
    expect(normalizeGenerateSettings({ ...legacyStoredValues, positivePrompt: undefined })).toBeNull();
    expect(normalizeGenerateSettings(null)).toBeNull();
  });

  it('drops malformed values for the newer fields back to defaults', () => {
    const normalized = normalizeGenerateSettings({
      ...legacyStoredValues,
      aspectRatioId: 'bogus',
      hiDiffusionEnabled: 'yes',
      hiDiffusionRauNetEnabled: null,
      hiDiffusionT1Ratio: Number.NaN,
      hiDiffusionT2Ratio: 'late',
      hiDiffusionWindowAttentionEnabled: 1,
      loras: [{ isEnabled: true, model: { key: 'k', name: 'n', type: 'main' }, weight: 1 }],
      vae: { key: 'k', name: 'n', type: 'main' },
      vaePrecision: 'fp64',
    });

    expect(normalized?.aspectRatioId).toBe('2:3');
    expect(normalized?.hiDiffusionEnabled).toBe(false);
    expect(normalized?.hiDiffusionRauNetEnabled).toBe(true);
    expect(normalized?.hiDiffusionWindowAttentionEnabled).toBe(true);
    expect(normalized?.hiDiffusionT1Ratio).toBeNull();
    expect(normalized?.hiDiffusionT2Ratio).toBeNull();
    expect(normalized?.loras).toEqual([]);
    expect(normalized?.vae).toBeNull();
    expect(normalized?.vaePrecision).toBe('fp32');
  });

  it('keeps valid hosted-provider options and drops the rest back to provider defaults', () => {
    const kept = normalizeGenerateSettings({
      ...legacyStoredValues,
      externalImageSize: '2K',
      geminiTemperature: 0.5,
      geminiThinkingLevel: 'minimal',
      openaiBackground: 'opaque',
      openaiInputFidelity: 'high',
      openaiQuality: 'medium',
      seedreamOptimizePrompt: true,
      seedreamWatermark: true,
    });

    expect(kept).toMatchObject({
      externalImageSize: '2K',
      geminiTemperature: 0.5,
      geminiThinkingLevel: 'minimal',
      openaiBackground: 'opaque',
      openaiInputFidelity: 'high',
      openaiQuality: 'medium',
      seedreamOptimizePrompt: true,
      seedreamWatermark: true,
    });

    const dropped = normalizeGenerateSettings({
      ...legacyStoredValues,
      externalImageSize: '',
      geminiTemperature: 2.5,
      geminiThinkingLevel: 'max',
      openaiBackground: 'clear',
      openaiInputFidelity: 'medium',
      openaiQuality: 'ultra',
      seedreamOptimizePrompt: 'yes',
      seedreamWatermark: 1,
    });

    expect(dropped).toMatchObject({
      externalImageSize: null,
      geminiTemperature: null,
      geminiThinkingLevel: null,
      openaiBackground: 'auto',
      openaiInputFidelity: null,
      openaiQuality: 'auto',
      seedreamOptimizePrompt: false,
      seedreamWatermark: false,
    });
  });

  it('bounds the model-family tuning fields and defaults the rest', () => {
    const normalized = normalizeGenerateSettings({
      ...legacyStoredValues,
      fluxDypeExponent: 5000,
      fluxDypePreset: 'turbo',
      fluxDypeScale: -1,
      qwenImageEncoderQuantization: 'fp8',
      qwenImageShift: 3,
      zImageSeedVarianceEnabled: 'yes',
      zImageSeedVarianceRandomizePercent: 0,
      zImageSeedVarianceStrength: 9,
      zImageShift: 42,
    });

    expect(normalized).toMatchObject({
      fluxDypeExponent: 1000,
      fluxDypePreset: 'off',
      fluxDypeScale: 0,
      qwenImageEncoderQuantization: 'none',
      qwenImageShift: 3,
      zImageSeedVarianceEnabled: false,
      zImageSeedVarianceRandomizePercent: 1,
      zImageSeedVarianceStrength: 2,
      zImageShift: null,
    });
    expect(normalizeGenerateSettings(legacyStoredValues)).toMatchObject({
      fluxDypeExponent: 2,
      fluxDypePreset: 'off',
      fluxDypeScale: 2,
      qwenImageEncoderQuantization: 'none',
      qwenImageShift: null,
      zImageSeedVarianceEnabled: false,
      zImageSeedVarianceRandomizePercent: 50,
      zImageSeedVarianceStrength: 0.1,
      zImageShift: null,
    });
  });

  it('drops malformed persisted model identifiers missing a base', () => {
    const normalized = normalizeGenerateSettings({
      ...legacyStoredValues,
      loras: [{ isEnabled: true, model: { key: 'lora', name: 'LoRA', type: 'lora' }, weight: 1 }],
      mistralEncoderModel: { key: 'mistral', name: 'Mistral', type: 'mistral_encoder' },
      qwen3EncoderModel: { key: 'qwen3', name: 'Qwen3', type: 'qwen3_encoder' },
      vae: { key: 'vae', name: 'VAE', type: 'vae' },
    });

    expect(normalized?.loras).toEqual([]);
    expect(normalized?.mistralEncoderModel).toBeNull();
    expect(normalized?.qwen3EncoderModel).toBeNull();
    expect(normalized?.vae).toBeNull();
  });

  it('is idempotent across persisted-value variations', () => {
    for (let index = 0; index < 128; index += 1) {
      const first = normalizeGenerateSettings({
        ...legacyStoredValues,
        aspectRatioId: index % 3 === 0 ? 'bogus' : index % 2 === 0 ? '1:1' : 'Free',
        batchCount: (index % 20) + 1,
        cfgScale: (index % 30) / 2,
        height: 64 + ((index * 8) % 1984),
        negativePromptEnabled: index % 2 === 0,
        seed: index * 7919,
        shouldRandomizeSeed: index % 5 === 0,
        width: 64 + ((index * 16) % 1984),
      });

      expect(first).not.toBeNull();
      expect(normalizeGenerateSettings(first)).toEqual(first);
    }
  });
});

describe('the FLUX.3 prompt upsampler settings', () => {
  const upsampler = {
    base: 'any',
    key: 'pup',
    name: 'FLUX.3 Prompt Upsampler',
    supports_images: true,
    type: 'text_llm',
  };

  it('defaults a project saved without them to off with no model, and keeps what a later one stored', () => {
    expect(normalizeGenerateSettings(legacyStoredValues)).toMatchObject({
      flux3PromptUpsamplerEnabled: false,
      flux3PromptUpsamplerModel: null,
    });
    expect(
      normalizeGenerateSettings({
        ...legacyStoredValues,
        flux3PromptUpsamplerEnabled: true,
        flux3PromptUpsamplerModel: upsampler,
      })
    ).toMatchObject({ flux3PromptUpsamplerEnabled: true, flux3PromptUpsamplerModel: upsampler });
    expect(
      normalizeGenerateSettings({
        ...legacyStoredValues,
        flux3PromptUpsamplerEnabled: 'yes',
        flux3PromptUpsamplerModel: { key: 'no-name' },
      })
    ).toMatchObject({ flux3PromptUpsamplerEnabled: false, flux3PromptUpsamplerModel: null });
  });
});

describe('the Anima inpaint adapter settings', () => {
  it('defaults the weight, clamps it to the node bounds, and drops a malformed adapter', () => {
    expect(normalizeGenerateSettings(legacyStoredValues)).toMatchObject({
      animaInpaintAdapterModel: null,
      animaInpaintAdapterWeight: 1,
    });
    const adapter = { base: 'anima', cond_in_channels: 4, key: 'inpaint', name: 'Inpainting', type: 'controlnet' };
    expect(
      normalizeGenerateSettings({
        ...legacyStoredValues,
        animaInpaintAdapterModel: adapter,
        animaInpaintAdapterWeight: 12,
      })
    ).toMatchObject({ animaInpaintAdapterModel: adapter, animaInpaintAdapterWeight: 10 });
    expect(
      normalizeGenerateSettings({
        ...legacyStoredValues,
        animaInpaintAdapterModel: { key: 'no-name' },
        animaInpaintAdapterWeight: 'heavy',
      })
    ).toMatchObject({ animaInpaintAdapterModel: null, animaInpaintAdapterWeight: 1 });
  });
});

describe('the MOSS-Image skip layers', () => {
  it('parses comma-separated block indices into unique ascending ones, an empty field into none', () => {
    expect(parseMossImageSkipLayers(' 26, 22,24 ,22')).toEqual([22, 24, 26]);
    expect(parseMossImageSkipLayers('22, ')).toEqual([22]);
    expect(parseMossImageSkipLayers('')).toEqual([]);
    expect(parseMossImageSkipLayers(' , ')).toEqual([]);
  });
});

describe('Generate widget value snapshots', () => {
  const model: MainModelConfig = { base: 'flux2', key: 'model', name: 'Old Model', type: 'main' };
  const component: ComponentModelConfig = { base: 'any', key: 'qwen3', name: 'Old Qwen3', type: 'qwen3_encoder' };
  const mistral: ComponentModelConfig = {
    base: 'any',
    key: 'mistral',
    name: 'Old Mistral',
    type: 'mistral_encoder',
  };
  const vae: VaeModelConfig = { base: 'flux2', key: 'vae', name: 'Old VAE', type: 'vae' };
  const lora: LoraModelConfig = { base: 'flux2', key: 'lora', name: 'Old LoRA', type: 'lora' };
  const values: GenerateWidgetValues = {
    ...(normalizeGenerateSettings(legacyStoredValues) as NonNullable<ReturnType<typeof normalizeGenerateSettings>>),
    clipEmbedModel: { base: 'any', key: 'clip', name: 'CLIP', type: 'clip_embed', variant: 'large' },
    componentSourceModel: model,
    loras: [{ isEnabled: true, model: lora, weight: 0.5 }],
    mistralEncoderModel: mistral,
    model,
    modelKey: model.key,
    qwen3EncoderModel: component,
    vae,
  };

  it('deep-clones every nested model selection', () => {
    const clone = cloneGenerateWidgetValues(values);

    expect(clone).toEqual(values);
    expect(clone.model).not.toBe(values.model);
    expect(clone.componentSourceModel).not.toBe(values.componentSourceModel);
    expect(clone.mistralEncoderModel).not.toBe(values.mistralEncoderModel);
    expect(clone.qwen3EncoderModel).not.toBe(values.qwen3EncoderModel);
    expect(clone.vae).not.toBe(values.vae);
    expect(clone.loras[0]).not.toBe(values.loras[0]);
    expect(clone.loras[0]?.model).not.toBe(values.loras[0]?.model);
  });

  it('uses current backend model records for same-key stored selections', () => {
    const currentModel = { ...model, format: 'diffusers' as const, name: 'Current Model' };
    const currentComponent = { ...component, name: 'Current Qwen3' };
    const currentMistral = { ...mistral, name: 'Current Mistral' };
    const currentVae = { ...vae, name: 'Current VAE' };
    const currentLora = { ...lora, name: 'Current LoRA' };
    const synced = syncGenerateWidgetValuesWithModels(values, [
      currentModel,
      currentComponent,
      currentMistral,
      currentVae,
      currentLora,
    ]);

    expect(synced.model).toBe(currentModel);
    expect(synced.componentSourceModel).toBe(currentModel);
    expect(synced.mistralEncoderModel).toBe(currentMistral);
    expect(synced.qwen3EncoderModel).toBe(currentComponent);
    expect(synced.vae).toBe(currentVae);
    expect(synced.loras[0]?.model).toBe(currentLora);
  });

  it('does not resync cloned current model snapshots by reference alone', () => {
    const currentModel = { ...model, format: 'diffusers' as const, name: 'Current Model' };
    const currentComponent = { ...component, name: 'Current Qwen3' };
    const currentMistral = { ...mistral, name: 'Current Mistral' };
    const currentVae = { ...vae, name: 'Current VAE' };
    const currentLora = { ...lora, name: 'Current LoRA' };
    const currentValues = {
      ...values,
      componentSourceModel: currentModel,
      loras: [{ isEnabled: true, model: currentLora, weight: 0.5 }],
      model: currentModel,
      mistralEncoderModel: currentMistral,
      qwen3EncoderModel: currentComponent,
      vae: currentVae,
    };
    const clonedValues = cloneGenerateWidgetValues(currentValues);
    const synced = syncGenerateWidgetValuesWithModels(clonedValues, [
      currentModel,
      currentComponent,
      currentMistral,
      currentVae,
      currentLora,
    ]);

    expect(synced).toBe(clonedValues);
  });
});

describe('LoRA settings helpers', () => {
  it('restricts FLUX.2 LoRAs by base and variant', () => {
    const flux2Model: MainModelConfig = { base: 'flux2', key: 'flux2', name: 'FLUX.2', type: 'main' };

    expect(isLoraCompatibleWithModel({ base: 'flux' }, flux2Model)).toBe(false);
    expect(
      isLoraCompatibleWithModel({ base: 'flux2', variant: 'klein_4b' }, { ...flux2Model, variant: 'klein_9b' })
    ).toBe(false);
  });
});

describe('dimension helpers', () => {
  it('fits a ratio into a pixel area on the dimension grid', () => {
    const { height, width } = calculateNewSize(1, 1024 * 1024);

    expect(width).toBe(1024);
    expect(height).toBe(1024);

    const wide = calculateNewSize(16 / 9, 1024 * 1024);

    expect(wide.width % 8).toBe(0);
    expect(wide.height % 8).toBe(0);
    expect(wide.width / wide.height).toBeCloseTo(16 / 9, 1);
  });

  it('keeps the ratio at the size limits, scaling both sides instead of clamping one', () => {
    // 6144² at 16:9 is 8192×4608; halved to fit 4096.
    expect(calculateNewSize(16 / 9, 6144 * 6144)).toEqual({ height: 2304, width: 4096 });
    // 2048² at 8:1 is 5792.6×724.1; scaled to fit 4096 wide, the height is exactly 512.
    expect(calculateNewSize(8, 2048 * 2048)).toEqual({ height: 512, width: 4096 });
    // 32² is below the 64px minimum on both sides.
    expect(calculateNewSize(1, 32 * 32)).toEqual({ height: 64, width: 64 });
    // 48² at 4:1 is 96×24; scaled so the short side is 64, then 256×64.
    expect(calculateNewSize(4, 48 * 48)).toEqual({ height: 64, width: 256 });
    // 64:1 is the widest ratio both limits allow: 4096×64 exactly.
    expect(calculateNewSize(64, 1024 * 1024)).toEqual({ height: 64, width: 4096 });
  });

  it('keeps a reference name through normalizing, trimmed and capped, and drops one that is empty or not text', () => {
    const config = { image: null, type: 'flux2_reference_image' as const };
    const long = `${'é'.repeat(MAX_REFERENCE_IMAGE_NAME_LENGTH)}tail`;
    const [named, untouched, blank, wrongType, capped] = normalizeReferenceImages([
      { config, id: 'named', isEnabled: true, name: '  Hero pose  ' },
      { config, id: 'untouched', isEnabled: true },
      { config, id: 'blank', isEnabled: true, name: '   ' },
      { config, id: 'wrong-type', isEnabled: true, name: 7 },
      { config, id: 'capped', isEnabled: true, name: long },
    ]);

    expect(named?.name).toBe('Hero pose');
    expect(untouched).toEqual({ config, id: 'untouched', isEnabled: true });
    expect(blank && 'name' in blank).toBe(false);
    expect(wrongType && 'name' in wrongType).toBe(false);
    expect(capped?.name).toBe('é'.repeat(80));
  });
});

describe('seed modes', () => {
  it('steps forward for random, since a random start still runs consecutive seeds', () => {
    expect(getSeedStep('random')).toBe(1);
    expect(getSeedStep('increment')).toBe(1);
    expect(getSeedStep('fixed')).toBe(0);
    expect(getSeedStep('decrement')).toBe(-1);
  });

  it('wraps over the inclusive seed range in both directions', () => {
    expect(wrapSeed(SEED_MAX)).toBe(SEED_MAX);
    expect(wrapSeed(SEED_MAX + 1)).toBe(0);
    expect(wrapSeed(-1)).toBe(SEED_MAX);
    expect(wrapSeed(-2)).toBe(SEED_MAX - 1);
  });
});

describe('getSeedSequenceLength', () => {
  it('takes one entry per iteration unless every image of a prompt set gets its own', () => {
    expect(
      getSeedSequenceLength({ batchCount: 3, promptCount: 1, seedBehaviour: 'per-image', seedMode: 'increment' })
    ).toBe(3);
    expect(
      getSeedSequenceLength({ batchCount: 3, promptCount: 2, seedBehaviour: 'per-iteration', seedMode: 'increment' })
    ).toBe(3);
    expect(
      getSeedSequenceLength({ batchCount: 3, promptCount: 2, seedBehaviour: 'per-image', seedMode: 'random' })
    ).toBe(6);
  });
});

describe('planSeedSubmission', () => {
  const plan = (seedMode: Parameters<typeof planSeedSubmission>[0]['seedMode'], startSeed = 42, batchCount = 3) =>
    planSeedSubmission({ batchCount, promptCount: 1, seedBehaviour: 'per-iteration', seedMode, startSeed });

  it('advances the editable seed past the batch only in the stepping modes', () => {
    expect(plan('increment')).toEqual({
      lastSeed: 44,
      nextSeed: 45,
      seedMode: 'increment',
      sequenceLength: 3,
      startSeed: 42,
      step: 1,
    });
    expect(plan('decrement')).toEqual({
      lastSeed: 40,
      nextSeed: 39,
      seedMode: 'decrement',
      sequenceLength: 3,
      startSeed: 42,
      step: -1,
    });
    expect(plan('fixed')).toEqual({
      lastSeed: 42,
      nextSeed: null,
      seedMode: 'fixed',
      sequenceLength: 1,
      startSeed: 42,
      step: 0,
    });
    expect(plan('random')).toEqual({
      lastSeed: 44,
      nextSeed: null,
      seedMode: 'random',
      sequenceLength: 3,
      startSeed: 42,
      step: 1,
    });
  });
});

describe('normalizeGenerateSettings refiner fields', () => {
  it('defaults values saved before the refiner existed', () => {
    expect(normalizeGenerateSettings(legacyStoredValues)).toMatchObject({
      refinerCfgScale: 7.5,
      refinerModel: null,
      refinerNegativeAestheticScore: 2.5,
      refinerPositiveAestheticScore: 6,
      refinerScheduler: 'euler',
      refinerStart: 0.8,
      refinerSteps: 20,
    });
  });

  it('preserves valid Wan LoRA targets and lowNoiseWeight and strips invalid ones on normalizeGenerateSettings', () => {
    const loraModel = { base: 'wan', key: 'wan-lora', name: 'Wan LoRA', type: 'lora' };
    const normalized = normalizeGenerateSettings({
      ...legacyStoredValues,
      loras: [
        { isEnabled: true, lowNoiseWeight: 0.35, model: loraModel, target: 'both', weight: 0.85 },
        { isEnabled: true, lowNoiseWeight: 0.5, model: loraModel, target: 'high', weight: 0.9 },
        { isEnabled: true, lowNoiseWeight: Number.NaN, model: loraModel, target: 'invalid', weight: 0.7 },
      ],
    });

    expect(normalized?.loras).toEqual([
      { isEnabled: true, lowNoiseWeight: 0.35, model: loraModel, target: 'both', weight: 0.85 },
      { isEnabled: true, model: loraModel, target: 'high', weight: 0.9 },
      { isEnabled: true, model: loraModel, weight: 0.7 },
    ]);
    expect(normalized && isGenerateSettings(normalized)).toBe(true);
    expect(
      isGenerateSettings({
        ...normalized,
        loras: [{ isEnabled: true, lowNoiseWeight: 0.5, model: loraModel, target: 'high', weight: 0.9 }],
      })
    ).toBe(false);
  });
});
