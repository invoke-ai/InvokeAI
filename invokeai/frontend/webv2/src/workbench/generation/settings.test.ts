import { describe, expect, it } from 'vitest';

import type {
  ComponentModelConfig,
  GenerateLora,
  GenerateWidgetValues,
  LoraModelConfig,
  MainModelConfig,
  VaeModelConfig,
} from './types';

import { getGenerationDimensions } from './baseGenerationPolicies';
import {
  calculateNewSize,
  clampDimension,
  cloneGenerateWidgetValues,
  deriveAspectRatioId,
  getDefaultLoraWeight,
  getModelDefaultVae,
  hasModelDefaultVae,
  isLoraCompatibleWithModel,
  isGenerateSettings,
  normalizeGenerateSettings,
  syncGenerateWidgetValuesWithModels,
  syncGenerateLorasWithModels,
} from './settings';

/** The persisted widget-value shape from before aspect ratio / VAE / seamless / CLIP skip landed. */
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

describe('normalizeGenerateSettings', () => {
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
    expect(normalized?.loras).toEqual([]);
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
      loras: [{ isEnabled: true, model: { key: 'k', name: 'n', type: 'main' }, weight: 1 }],
      vae: { key: 'k', name: 'n', type: 'main' },
      vaePrecision: 'fp64',
    });

    expect(normalized?.aspectRatioId).toBe('2:3');
    expect(normalized?.loras).toEqual([]);
    expect(normalized?.vae).toBeNull();
    expect(normalized?.vaePrecision).toBe('fp32');
  });

  it('drops malformed persisted model identifiers missing a base', () => {
    const normalized = normalizeGenerateSettings({
      ...legacyStoredValues,
      loras: [{ isEnabled: true, model: { key: 'lora', name: 'LoRA', type: 'lora' }, weight: 1 }],
      qwen3EncoderModel: { key: 'qwen3', name: 'Qwen3', type: 'qwen3_encoder' },
      vae: { key: 'vae', name: 'VAE', type: 'vae' },
    });

    expect(normalized?.loras).toEqual([]);
    expect(normalized?.qwen3EncoderModel).toBeNull();
    expect(normalized?.vae).toBeNull();
  });
});

describe('Generate widget value snapshots', () => {
  const model: MainModelConfig = { base: 'flux2', key: 'model', name: 'Old Model', type: 'main' };
  const component: ComponentModelConfig = { base: 'any', key: 'qwen3', name: 'Old Qwen3', type: 'qwen3_encoder' };
  const vae: VaeModelConfig = { base: 'flux2', key: 'vae', name: 'Old VAE', type: 'vae' };
  const lora: LoraModelConfig = { base: 'flux2', key: 'lora', name: 'Old LoRA', type: 'lora' };
  const values: GenerateWidgetValues = {
    ...(normalizeGenerateSettings(legacyStoredValues) as NonNullable<ReturnType<typeof normalizeGenerateSettings>>),
    clipEmbedModel: { base: 'any', key: 'clip', name: 'CLIP', type: 'clip_embed' },
    componentSourceModel: model,
    loras: [{ isEnabled: true, model: lora, weight: 0.5 }],
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
    expect(clone.qwen3EncoderModel).not.toBe(values.qwen3EncoderModel);
    expect(clone.vae).not.toBe(values.vae);
    expect(clone.loras[0]).not.toBe(values.loras[0]);
    expect(clone.loras[0]?.model).not.toBe(values.loras[0]?.model);
  });

  it('uses current backend model records for same-key stored selections', () => {
    const currentModel = { ...model, format: 'diffusers' as const, name: 'Current Model' };
    const currentComponent = { ...component, name: 'Current Qwen3' };
    const currentVae = { ...vae, name: 'Current VAE' };
    const currentLora = { ...lora, name: 'Current LoRA' };
    const synced = syncGenerateWidgetValuesWithModels(values, [
      currentModel,
      currentComponent,
      currentVae,
      currentLora,
    ]);

    expect(synced.model).toBe(currentModel);
    expect(synced.componentSourceModel).toBe(currentModel);
    expect(synced.qwen3EncoderModel).toBe(currentComponent);
    expect(synced.vae).toBe(currentVae);
    expect(synced.loras[0]?.model).toBe(currentLora);
  });

  it('does not resync cloned current model snapshots by reference alone', () => {
    const currentModel = { ...model, format: 'diffusers' as const, name: 'Current Model' };
    const currentComponent = { ...component, name: 'Current Qwen3' };
    const currentVae = { ...vae, name: 'Current VAE' };
    const currentLora = { ...lora, name: 'Current LoRA' };
    const currentValues = {
      ...values,
      componentSourceModel: currentModel,
      loras: [{ isEnabled: true, model: currentLora, weight: 0.5 }],
      model: currentModel,
      qwen3EncoderModel: currentComponent,
      vae: currentVae,
    };
    const clonedValues = cloneGenerateWidgetValues(currentValues);
    const synced = syncGenerateWidgetValuesWithModels(clonedValues, [
      currentModel,
      currentComponent,
      currentVae,
      currentLora,
    ]);

    expect(synced).toBe(clonedValues);
  });
});

describe('LoRA settings helpers', () => {
  it('uses current model records when reading selected LoRA defaults', () => {
    const staleModel: LoraModelConfig = {
      base: 'sdxl',
      default_settings: { weight: 0.75 },
      key: 'lora-key',
      name: 'Old LoRA Name',
      type: 'lora',
    };
    const currentModel: LoraModelConfig = {
      ...staleModel,
      default_settings: { weight: 1.25 },
      name: 'Updated LoRA Name',
      trigger_phrases: ['updated trigger'],
    };
    const selectedLora: GenerateLora = { isEnabled: true, model: staleModel, weight: 0.5 };

    const [syncedLora] = syncGenerateLorasWithModels([selectedLora], [currentModel]);

    expect(syncedLora?.model).toBe(currentModel);
    expect(syncedLora?.weight).toBe(0.5);
    expect(getDefaultLoraWeight(syncedLora!.model)).toBe(1.25);
  });

  it('allows FLUX LoRAs for FLUX.2 while preserving FLUX.2 variant checks', () => {
    const flux2Model: MainModelConfig = { base: 'flux2', key: 'flux2', name: 'FLUX.2', type: 'main' };

    expect(isLoraCompatibleWithModel({ base: 'flux' }, flux2Model)).toBe(true);
    expect(
      isLoraCompatibleWithModel({ base: 'flux2', variant: 'klein_4b' }, { ...flux2Model, variant: 'klein_9b' })
    ).toBe(false);
  });
});

describe('model defaults', () => {
  it('uses shared VAE compatibility for cross-base default VAEs', () => {
    const zImageModel: MainModelConfig = {
      base: 'z-image',
      default_settings: { vae: 'flux-vae' },
      key: 'z-image',
      name: 'Z-Image',
      type: 'main',
    };
    const fluxVae: VaeModelConfig = { base: 'flux', key: 'flux-vae', name: 'FLUX VAE', type: 'vae' };

    expect(getModelDefaultVae(zImageModel, [fluxVae])).toBe(fluxVae);
  });

  it('distinguishes absent VAE defaults from explicit VAE defaults', () => {
    expect(hasModelDefaultVae({ base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' })).toBe(false);
    expect(
      hasModelDefaultVae({ base: 'sdxl', default_settings: { vae: null }, key: 'sdxl', name: 'SDXL', type: 'main' })
    ).toBe(false);
    expect(
      hasModelDefaultVae({ base: 'sdxl', default_settings: { vae: 'vae' }, key: 'sdxl', name: 'SDXL', type: 'main' })
    ).toBe(true);
  });
});

describe('dimension helpers', () => {
  it('derives the closest preset aspect ratio', () => {
    expect(deriveAspectRatioId(1024, 1024)).toBe('1:1');
    expect(deriveAspectRatioId(1536, 1024)).toBe('3:2');
    expect(deriveAspectRatioId(1000, 770)).toBe('Free');
  });

  it('fits a ratio into a pixel area on the dimension grid', () => {
    const { height, width } = calculateNewSize(1, 1024 * 1024);

    expect(width).toBe(1024);
    expect(height).toBe(1024);

    const wide = calculateNewSize(16 / 9, 1024 * 1024);

    expect(wide.width % 8).toBe(0);
    expect(wide.height % 8).toBe(0);
    expect(wide.width / wide.height).toBeCloseTo(16 / 9, 1);
  });

  it('uses larger dimension grids for model families that require them', () => {
    expect(getGenerationDimensions({ base: 'sdxl', type: 'main' }).grid).toBe(8);
    expect(getGenerationDimensions({ base: 'anima', type: 'main' }).grid).toBe(8);
    expect(getGenerationDimensions({ base: 'flux2', type: 'main' }).grid).toBe(16);
    expect(getGenerationDimensions({ base: 'qwen-image', type: 'main' }).grid).toBe(16);
    expect(getGenerationDimensions({ base: 'cogview4', type: 'main' }).grid).toBe(32);
    expect(clampDimension(888, getGenerationDimensions({ base: 'flux2', type: 'main' }).grid)).toBe(896);
  });
});
