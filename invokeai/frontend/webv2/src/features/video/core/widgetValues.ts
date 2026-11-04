import type {
  GenerationModelCatalogItem as ModelConfig,
  MainModelConfig,
  ModelIdentifierConfig,
} from '@features/generation/contracts';

import { isLoraCompatibleWithModel, isLoraModelConfig } from '@features/generation/settings';
import { SEED_MAX } from '@platform/core/seed';

import type { VideoWidgetValues } from './types';

import {
  getAcceleratorLoraChangeResult,
  getDefaultVideoSettings,
  getVideoComponentSectionPolicy,
  getVideoModelAvailabilityReasons,
  getVideoModelSelectionResult,
  getVideoModes,
  getVideoValidationReasons,
  isSupportedVideoModel,
  isVideoModelSelectable,
  type VideoComponentPolicyContext,
  type VideoComponentValueKey,
} from './videoPolicies';

// Shared empty array, so clearing already-clear keys keeps its identity.
const EMPTY_ACCELERATOR_LORA_KEYS: string[] = [];

export const createDefaultVideoWidgetValues = (models: readonly ModelConfig[] = []): VideoWidgetValues => {
  const model = (models.find((candidate) => isVideoModelSelectable(candidate)) as MainModelConfig | undefined) ?? null;

  return { ...getDefaultVideoSettings(model ?? undefined, models), model };
};

/**
 * Resolve catalog references while preserving object identity when unchanged to avoid reconciliation write-back
 * loops.
 */
export const syncVideoWidgetValuesWithModels = (
  values: VideoWidgetValues,
  models: readonly ModelConfig[]
): VideoWidgetValues => {
  // Promote only installed H3 checkpoint overrides before other reconciliation; retain the old Diffusers main as
  // components and drop invalid overrides without evicting it.
  if (values.model?.base === 'minimax-h3' && values.model.format === 'diffusers' && values.h3TransformerModel) {
    const installedTransformer = models.find((candidate) => candidate.key === values.h3TransformerModel?.key);

    if (
      installedTransformer &&
      installedTransformer.type === 'main' &&
      installedTransformer.base === 'minimax-h3' &&
      installedTransformer.format === 'checkpoint'
    ) {
      values = {
        ...values,
        componentSourceModel: values.model,
        h3TransformerModel: null,
        model: installedTransformer as MainModelConfig,
        modelKey: installedTransformer.key,
      };
    }
  }

  const modelsByKey = new Map(models.map((model) => [model.key, model]));
  const storedMain = values.model ? modelsByKey.get(values.model.key) : undefined;
  // Preserve supported legacy nonselectable mains for repair guidance; automatic choices use the stricter
  // selectable filter.
  const registered: MainModelConfig | null =
    storedMain && isSupportedVideoModel(storedMain)
      ? storedMain
      : ((models.find((candidate) => isVideoModelSelectable(candidate)) as MainModelConfig | undefined) ?? null);
  const sameH3 =
    registered?.base === 'minimax-h3' && registered.format === 'checkpoint' && registered.key === values.modelKey;
  const detected = registered?.variant === 'ref2va' ? 'ref2va' : 'fl2va';
  // Legacy records may contain either the classified model or the already-effective synthetic model.
  const storedTask = values.model?.variant === 'ref2va' ? 'ref2va' : 'fl2va';
  const effectiveVariant = sameH3
    ? (values.h3SelectedTask ??
      (values.h3ModeOverride
        ? storedTask !== detected
          ? storedTask
          : detected === 'ref2va'
            ? 'fl2va'
            : 'ref2va'
        : detected))
    : detected;
  const override = Boolean(sameH3 && effectiveVariant !== detected);
  // Reuse identity only while every catalog field still agrees: same-key reinstalls or edits must
  // refresh the synthetic model instead of retaining stale hashes, paths, names, or capabilities.
  const candidate =
    registered?.base === 'minimax-h3' && registered.format === 'checkpoint'
      ? { ...registered, variant: effectiveVariant }
      : registered;
  const sameCandidate =
    candidate &&
    values.model &&
    Object.keys(candidate).length === Object.keys(values.model).length &&
    Object.entries(candidate).every(
      ([key, value]) =>
        JSON.stringify(value) === JSON.stringify((values.model as unknown as Record<string, unknown>)[key])
    );
  const model: MainModelConfig | null = sameCandidate ? values.model : candidate;

  // Run canonical selection transitions after model replacement so family-specific fps, frames, and resolution
  // cannot remain stale.
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
  // Repair active acceleration after catalog changes, restoring normal defaults if no complete set remains. Never
  // enable it implicitly; preserve unchanged array identities.
  const unchangedAccelerator = {
    acceleratorEnabled: base.acceleratorEnabled,
    acceleratorLoraKeys: base.acceleratorLoraKeys,
    audioCfgScale: base.audioCfgScale,
    cfgScale: base.cfgScale,
    cfgScaleLowNoise: base.cfgScaleLowNoise,
    modalityScale: base.modalityScale,
    steps: base.steps,
    stgScale: base.stgScale,
  };
  const acceleratorSync = model ? getAcceleratorLoraChangeResult(base, model, models, loras) : null;
  let accelerator = unchangedAccelerator;

  if (!model) {
    // Without a main, clear acceleration keys alongside empty LoRAs to keep persisted settings valid.
    if (base.acceleratorEnabled || base.acceleratorLoraKeys.length > 0) {
      accelerator = {
        ...unchangedAccelerator,
        acceleratorEnabled: false,
        acceleratorLoraKeys: EMPTY_ACCELERATOR_LORA_KEYS,
      };
    }
  } else if (acceleratorSync && acceleratorSync.outcome !== 'unchanged') {
    // Every field the change result rewrites, not a hand-picked five. An accelerator that removes
    // guidance also restores it when it goes away, and naming fields here meant the Concepts-list
    // route honoured that while this one -- the LoRA leaving the catalog entirely -- dropped it,
    // leaving a guided run with its audio, STG and modality guidance silently pinned at identity.
    accelerator = {
      acceleratorEnabled: acceleratorSync.settings.acceleratorEnabled,
      acceleratorLoraKeys: acceleratorSync.settings.acceleratorLoraKeys,
      audioCfgScale: acceleratorSync.settings.audioCfgScale,
      cfgScale: acceleratorSync.settings.cfgScale,
      cfgScaleLowNoise: acceleratorSync.settings.cfgScaleLowNoise,
      modalityScale: acceleratorSync.settings.modalityScale,
      steps: acceleratorSync.settings.steps,
      stgScale: acceleratorSync.settings.stgScale,
    };
  }

  const next: VideoWidgetValues = {
    ...base,
    ...accelerator,
    componentSourceModel: syncComponent('componentSourceModel', base.componentSourceModel),
    h3HybridBaseModel: syncComponent('h3HybridBaseModel', base.h3HybridBaseModel),
    h3TextEncoderModel: syncComponent('h3TextEncoderModel', base.h3TextEncoderModel),
    h3TransformerModel: syncComponent('h3TransformerModel', base.h3TransformerModel),
    loras,
    ltx2DurationHeadModel: syncComponent('ltx2DurationHeadModel', base.ltx2DurationHeadModel),
    ltx2TextEncoderModel: syncComponent('ltx2TextEncoderModel', base.ltx2TextEncoderModel),
    model,
    modelKey: model?.key ?? base.modelKey,
    h3ModeOverride: override,
    h3SelectedTask:
      model?.base === 'minimax-h3' && model.format === 'checkpoint'
        ? model.variant === 'ref2va'
          ? 'ref2va'
          : 'fl2va'
        : undefined,
    vae: syncComponent('vae', base.vae),
    wanLowNoiseModel: syncComponent('wanLowNoiseModel', base.wanLowNoiseModel),
    wanT5EncoderModel: syncComponent('wanT5EncoderModel', base.wanT5EncoderModel),
  };

  // Auto duration belongs to the head it was turned on with: an uninstalled head takes it along, so
  // reinstalling the head later does not silently switch it back on.
  if (!next.ltx2DurationHeadModel) {
    next.autoDuration = false;
  }

  // Drop orphaned references when the replacement model lacks reference mode, preserving identity otherwise.
  if (next.references.length > 0 && model && !getVideoModes(model).includes('reference')) {
    next.references = [];
  }

  const isUnchanged =
    base === values &&
    next.model === values.model &&
    next.modelKey === values.modelKey &&
    next.h3SelectedTask === values.h3SelectedTask &&
    next.h3ModeOverride === values.h3ModeOverride &&
    next.acceleratorEnabled === values.acceleratorEnabled &&
    next.acceleratorLoraKeys === values.acceleratorLoraKeys &&
    next.vae === values.vae &&
    next.wanT5EncoderModel === values.wanT5EncoderModel &&
    next.wanLowNoiseModel === values.wanLowNoiseModel &&
    next.componentSourceModel === values.componentSourceModel &&
    next.h3TransformerModel === values.h3TransformerModel &&
    next.h3TextEncoderModel === values.h3TextEncoderModel &&
    next.h3HybridBaseModel === values.h3HybridBaseModel &&
    next.ltx2TextEncoderModel === values.ltx2TextEncoderModel &&
    next.ltx2DurationHeadModel === values.ltx2DurationHeadModel &&
    next.autoDuration === values.autoDuration &&
    next.references === values.references &&
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
    return ['Video needs a Wan 2.2, MiniMax H3 or LTX-2 main model.'];
  }

  // Readiness may receive the persisted (classified) record before the view's catalog sync.
  // Resolve the same effective H3 task the graph compiler will use, without changing any media.
  const registered = models?.find((item) => item.key === values.model?.key);
  const source = registered && isSupportedVideoModel(registered) ? registered : values.model;
  const effectiveModel =
    source.base === 'minimax-h3' && source.format === 'checkpoint'
      ? {
          ...source,
          variant:
            values.h3SelectedTask ??
            (registered && values.h3ModeOverride
              ? source.variant === 'ref2va'
                ? 'fl2va'
                : 'ref2va'
              : source.variant === 'ref2va'
                ? 'ref2va'
                : 'fl2va'),
        }
      : source;
  const reasons = getVideoValidationReasons(effectiveModel, values);

  if (models) {
    reasons.push(...getVideoModelAvailabilityReasons(effectiveModel, values, models));
  }

  return reasons;
};

export const resolveVideoSeed = (values: Pick<VideoWidgetValues, 'seed' | 'seedMode'>): number =>
  values.seedMode === 'random' ? Math.floor(Math.random() * SEED_MAX) : values.seed;
