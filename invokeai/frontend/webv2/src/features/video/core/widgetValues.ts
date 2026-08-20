import type {
  GenerationModelCatalogItem as ModelConfig,
  MainModelConfig,
  ModelIdentifierConfig,
} from '@features/generation/contracts';

import { isLoraCompatibleWithModel, isLoraModelConfig, SEED_MAX } from '@features/generation/settings';

import type { VideoWidgetValues } from './types';

import {
  getDefaultVideoSettings,
  getVideoComponentSectionPolicy,
  getVideoModelAvailabilityReasons,
  getVideoModelSelectionResult,
  getVideoValidationReasons,
  isSupportedVideoModel,
  type VideoComponentPolicyContext,
  type VideoComponentValueKey,
} from './videoPolicies';

/**
 * The panel-facing layer over the pure settings: default values seeded from
 * the installed catalog, catalog reconciliation (upscale-style: render-time,
 * mount write-back, and compile all reuse this one function), the aggregate
 * validation the invoke route consumes, and seed resolution at submit.
 */

export const createDefaultVideoWidgetValues = (models: readonly ModelConfig[] = []): VideoWidgetValues => {
  const model = (models.find((candidate) => isSupportedVideoModel(candidate)) as MainModelConfig | undefined) ?? null;

  return { ...getDefaultVideoSettings(model ?? undefined, models), model };
};

/**
 * Re-resolves every model reference against the live catalog. MUST return the
 * same object when nothing changes — the mount-time reconciler and render
 * memoization rely on that identity guarantee to avoid write-back loops.
 */
export const syncVideoWidgetValuesWithModels = (
  values: VideoWidgetValues,
  models: readonly ModelConfig[]
): VideoWidgetValues => {
  const modelsByKey = new Map(models.map((model) => [model.key, model]));
  const storedMain = values.model ? modelsByKey.get(values.model.key) : undefined;
  const model: MainModelConfig | null =
    storedMain && isSupportedVideoModel(storedMain)
      ? storedMain
      : ((models.find((candidate) => isSupportedVideoModel(candidate)) as MainModelConfig | undefined) ?? null);

  // The main changed identity under us (nothing stored, or the stored model was
  // uninstalled and another family got auto-picked): run the canonical
  // selection transition first, or the new family would inherit the old one's
  // frames/fps/resolution — an H3 main stuck at Wan's fps 16 has no working
  // FPS control to fix it with.
  const base: VideoWidgetValues =
    model && model.key !== values.model?.key
      ? { ...getVideoModelSelectionResult({ currentSettings: values, model, models }).settings, model }
      : values;

  const componentPolicy = model ? getVideoComponentSectionPolicy(model, base) : null;
  const slotsByKey = new Map(componentPolicy?.slots.map((slot) => [slot.key, slot]) ?? []);
  const componentContext: VideoComponentPolicyContext | null = model
    ? { model, selectedComponents: base, settings: base }
    : null;

  const syncComponent = <T extends ModelIdentifierConfig | MainModelConfig>(
    key: VideoComponentValueKey,
    value: T | null
  ): T | null => {
    if (!value) {
      return null;
    }

    const installed = modelsByKey.get(value.key);
    const slot = slotsByKey.get(key);

    if (!installed || !slot || !componentContext) {
      return null;
    }

    return !slot.filter || slot.filter(installed, componentContext) ? (installed as T) : null;
  };

  const loras = model
    ? base.loras.flatMap((lora) => {
        const installed = modelsByKey.get(lora.model.key);

        return installed && isLoraModelConfig(installed) && isLoraCompatibleWithModel(installed, model)
          ? [{ ...lora, model: installed }]
          : [];
      })
    : [];
  // The accelerator flag cannot outlive its recorded LoRAs.
  const acceleratorAlive =
    base.acceleratorEnabled &&
    base.acceleratorLoraKeys.length > 0 &&
    base.acceleratorLoraKeys.every((key) => loras.some((lora) => lora.model.key === key && lora.isEnabled));
  // Reuse the stored array whenever the content is unchanged — the identity
  // guarantee below depends on it.
  const acceleratorLoraKeys = acceleratorAlive || base.acceleratorLoraKeys.length === 0 ? base.acceleratorLoraKeys : [];

  const next: VideoWidgetValues = {
    ...base,
    acceleratorEnabled: acceleratorAlive,
    acceleratorLoraKeys,
    componentSourceModel: syncComponent('componentSourceModel', base.componentSourceModel),
    h3TextEncoderModel: syncComponent('h3TextEncoderModel', base.h3TextEncoderModel),
    h3TransformerModel: syncComponent('h3TransformerModel', base.h3TransformerModel),
    loras,
    model,
    modelKey: model?.key ?? base.modelKey,
    vae: syncComponent('vae', base.vae),
    wanLowNoiseModel: syncComponent('wanLowNoiseModel', base.wanLowNoiseModel),
    wanT5EncoderModel: syncComponent('wanT5EncoderModel', base.wanT5EncoderModel),
  };

  const isUnchanged =
    base === values &&
    next.model === values.model &&
    next.modelKey === values.modelKey &&
    next.acceleratorEnabled === values.acceleratorEnabled &&
    next.acceleratorLoraKeys === values.acceleratorLoraKeys &&
    next.vae === values.vae &&
    next.wanT5EncoderModel === values.wanT5EncoderModel &&
    next.wanLowNoiseModel === values.wanLowNoiseModel &&
    next.componentSourceModel === values.componentSourceModel &&
    next.h3TransformerModel === values.h3TransformerModel &&
    next.h3TextEncoderModel === values.h3TextEncoderModel &&
    next.loras.length === values.loras.length &&
    next.loras.every((lora, index) => lora.model === values.loras[index]?.model);

  return isUnchanged ? values : next;
};

/** The aggregate readiness check the invoke route and the compiler share. */
export const getVideoWidgetValidationReasons = (
  values: VideoWidgetValues,
  models?: readonly ModelConfig[]
): string[] => {
  if (!values.model) {
    return ['Video needs a Wan 2.2 or MiniMax H3 main model.'];
  }

  const reasons = getVideoValidationReasons(values.model, values);

  if (models) {
    reasons.push(...getVideoModelAvailabilityReasons(values.model, values, models));
  }

  return reasons;
};

export const resolveVideoSeed = (values: Pick<VideoWidgetValues, 'seed' | 'shouldRandomizeSeed'>): number =>
  values.shouldRandomizeSeed ? Math.floor(Math.random() * SEED_MAX) : values.seed;
