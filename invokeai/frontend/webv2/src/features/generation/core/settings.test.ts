import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import {
  getSeedSequenceLength,
  getSeedStep,
  isSeedMode,
  planSeedSubmission,
  SEED_MAX,
  wrapSeed,
} from '@platform/core/seed';
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
  moveReferenceImage,
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

seedArchitectureCapabilities();

describe('normalizeGenerateSettings', () => {
  it('reads the seed mode saved before modes existed from the random toggle', () => {
    expect(normalizeGenerateSettings(legacyStoredValues)?.seedMode).toBe('fixed');
    expect(normalizeGenerateSettings({ ...legacyStoredValues, shouldRandomizeSeed: true })?.seedMode).toBe('random');
    // A saved mode wins over a stale toggle left behind by a partial patch.
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

  it('rejects values that name neither a seed mode nor the random toggle', () => {
    const { shouldRandomizeSeed: _, ...withoutSeedPolicy } = legacyStoredValues;

    expect(normalizeGenerateSettings(withoutSeedPolicy)).toBeNull();
  });

  it('never lets the strict guard pass a record whose seed mode still needs inventing', () => {
    // The normalized guard must cover every field because accepted records bypass normalization.
    const normalized = normalizeGenerateSettings(legacyStoredValues);

    expect(normalized && isGenerateSettings(normalized)).toBe(true);
    const { seedMode: _, ...withoutSeedMode } = normalized as NonNullable<typeof normalized>;

    expect(isGenerateSettings({ ...withoutSeedMode, shouldRandomizeSeed: true })).toBe(false);
    expect(isGenerateSettings({ ...withoutSeedMode, seedMode: 'bogus', shouldRandomizeSeed: true })).toBe(false);
  });

  it('treats any key normalize would fill in or repair as non-canonical, not only the seed mode', () => {
    // Saved before PiD existed, or with a ratio outside the range normalize clamps to.
    const normalized = normalizeGenerateSettings(legacyStoredValues) as NonNullable<
      ReturnType<typeof normalizeGenerateSettings>
    >;
    const { pidMode: _, ...withoutPidMode } = normalized;

    expect(isGenerateSettings(withoutPidMode)).toBe(false);
    expect(isGenerateSettings({ ...normalized, hiDiffusionT1Ratio: 99 })).toBe(false);
    expect(normalizeGenerateSettings(withoutPidMode)?.pidMode).toBe(normalized.pidMode);
  });

  it('enforces template view mode only when a valid template remains', () => {
    const validTemplate = {
      id: 'template-1',
      name: 'Cinematic',
      negativePrompt: '',
      positivePrompt: '{prompt}, cinematic',
    };

    expect(
      normalizeGenerateSettings({ ...legacyStoredValues, promptTemplate: null, promptTemplateViewMode: true })
        ?.promptTemplateViewMode
    ).toBe(false);
    expect(
      normalizeGenerateSettings({
        ...legacyStoredValues,
        promptTemplate: { id: '', name: 'Broken' },
        promptTemplateViewMode: true,
      })?.promptTemplateViewMode
    ).toBe(false);
    expect(
      normalizeGenerateSettings({ ...legacyStoredValues, promptTemplate: validTemplate, promptTemplateViewMode: true })
        ?.promptTemplateViewMode
    ).toBe(true);
    expect(isGenerateSettings({ ...legacyStoredValues, promptTemplate: null, promptTemplateViewMode: true })).toBe(
      false
    );
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

  it('keeps more than five valid persisted reference images for provider-specific caps', () => {
    const referenceImages = Array.from({ length: 6 }, (_, index) => ({
      id: `ref-${index}`,
      isEnabled: true,
      config: {
        type: 'external_reference_image',
        image: {
          height: 768,
          imageName: `reference-${index}.png`,
          imageUrl: `/api/v1/images/i/reference-${index}.png/full`,
          queuedAt: '2026-01-01T00:00:00.000Z',
          sourceQueueItemId: 'backend-gallery',
          thumbnailUrl: `/api/v1/images/i/reference-${index}.png/thumbnail`,
          width: 512,
        },
      },
    }));

    const normalized = normalizeGenerateSettings({ ...legacyStoredValues, referenceImages });

    expect(normalized?.referenceImages).toHaveLength(6);
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
    expect(normalized?.hiDiffusionT1Ratio).toBe(0.4);
    expect(normalized?.hiDiffusionT2Ratio).toBe(0);
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

  it('defaults an omitted batch count to one', () => {
    const { batchCount, ...storedValues } = legacyStoredValues;

    expect(batchCount).toBe(2);
    expect(normalizeGenerateSettings(storedValues)?.batchCount).toBe(1);
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
    expect(normalized?.hiDiffusionT1Ratio).toBe(0.4);
    expect(normalized?.hiDiffusionT2Ratio).toBe(0);
    expect(normalized?.loras).toEqual([]);
    expect(normalized?.vae).toBeNull();
    expect(normalized?.vaePrecision).toBe('fp32');
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

  it('restricts FLUX.2 LoRAs by base and variant', () => {
    const flux2Model: MainModelConfig = { base: 'flux2', key: 'flux2', name: 'FLUX.2', type: 'main' };

    expect(isLoraCompatibleWithModel({ base: 'flux' }, flux2Model)).toBe(false);
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

  it('moves a reference image one step through the stack, and no-ops at the ends', () => {
    const referenceImages = ['a', 'b', 'c'].map((id) => ({
      config: { image: null, model: null, type: 'flux_kontext_reference_image' as const },
      id,
      isEnabled: true,
    }));
    const idsOf = (list: readonly { id: string }[]) => list.map(({ id }) => id);

    expect(idsOf(moveReferenceImage(referenceImages, 'c', -1))).toEqual(['a', 'c', 'b']);
    expect(idsOf(moveReferenceImage(referenceImages, 'a', 1))).toEqual(['b', 'a', 'c']);

    // Reordering preserves item identities and changes conditioning order.
    const moved = moveReferenceImage(referenceImages, 'a', 1);

    expect(moved[1]).toBe(referenceImages[0]);

    // Return the same array for no-ops so callers can skip persistence.
    expect(moveReferenceImage(referenceImages, 'a', -1)).toBe(referenceImages);
    expect(moveReferenceImage(referenceImages, 'c', 1)).toBe(referenceImages);
    expect(moveReferenceImage(referenceImages, 'missing', 1)).toBe(referenceImages);
    expect(moveReferenceImage([], 'a', 1)).toEqual([]);
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

describe('seed modes', () => {
  it('recognizes only the four modes', () => {
    expect(isSeedMode('random')).toBe(true);
    expect(isSeedMode('decrement')).toBe(true);
    expect(isSeedMode('shuffle')).toBe(false);
    expect(isSeedMode(true)).toBe(false);
  });

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
  it('uses one seed for the whole fixed submission, whatever its shape', () => {
    expect(
      getSeedSequenceLength({ batchCount: 3, promptCount: 2, seedBehaviour: 'per-image', seedMode: 'fixed' })
    ).toBe(1);
  });

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

  it('wraps the next seed at either end of the range', () => {
    expect(plan('increment', SEED_MAX - 1, 2)).toMatchObject({ lastSeed: SEED_MAX, nextSeed: 0 });
    expect(plan('decrement', 1, 2)).toMatchObject({ lastSeed: 0, nextSeed: SEED_MAX });
  });

  it('advances by every seed a prompt set consumes', () => {
    expect(
      planSeedSubmission({
        batchCount: 2,
        promptCount: 2,
        seedBehaviour: 'per-image',
        seedMode: 'increment',
        startSeed: 42,
      })
    ).toMatchObject({ lastSeed: 45, nextSeed: 46, sequenceLength: 4 });
  });
});
