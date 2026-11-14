import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import {
  getModelDefaultComponents,
  getSettingsWithModelDefaults,
} from '@features/generation/core/baseGenerationPolicies';
import { DEFAULT_COMPONENT_SLOTS } from '@features/generation/core/componentDefaults';
import { getModelDefaultVae, hasModelDefaultVae, isVaeModelConfig } from '@features/generation/core/settings';

const MODEL_DEFAULT_VALUE_KEYS = [
  'aspectRatioId',
  'aspectRatioIsLocked',
  'aspectRatioValue',
  'cfgRescaleMultiplier',
  'cfgScale',
  'height',
  'modelKey',
  'scheduler',
  'steps',
  'vaePrecision',
  'width',
] as const satisfies readonly (keyof GenerateSettings)[];

/**
 * Settings with the model's defaults applied, including its default VAE and default components. A component slot the
 * model stores no default for keeps its selection.
 */
export const getModelDefaultSettings = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  models: readonly ModelConfig[]
): GenerateSettings => {
  const nextSettings = {
    ...getSettingsWithModelDefaults(settings, model),
    ...getModelDefaultComponents(model, settings, models, { replace: true }),
  };

  return hasModelDefaultVae(model)
    ? { ...nextSettings, vae: getModelDefaultVae(model, models.filter(isVaeModelConfig)) }
    : nextSettings;
};

/**
 * Patch limited to the model-governed keys, so prompts and other fields stay untouched. Concepts are left as they are:
 * the model already skips the ones it cannot load, and keeping them lets a compatible model run them again.
 */
export const getModelDefaultsPatch = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  models: readonly ModelConfig[]
): Partial<GenerateSettings> => {
  const defaults = getModelDefaultSettings(settings, model, models);
  const patch: Partial<GenerateSettings> = { vae: defaults.vae };

  for (const key of MODEL_DEFAULT_VALUE_KEYS) {
    (patch as Record<string, unknown>)[key] = defaults[key];
  }

  for (const { setting } of DEFAULT_COMPONENT_SLOTS) {
    if (defaults[setting] !== settings[setting]) {
      (patch as Record<string, unknown>)[setting] = defaults[setting];
    }
  }

  return patch;
};

/** Multi-key decisions such as size count as one override. */
const MODEL_DEFAULT_DECISION_GROUPS: readonly (readonly (typeof MODEL_DEFAULT_VALUE_KEYS)[number][])[] = [
  ['aspectRatioId', 'aspectRatioIsLocked', 'aspectRatioValue', 'height', 'width'],
  ['steps'],
  ['cfgScale'],
  ['cfgRescaleMultiplier'],
  ['scheduler'],
  ['vaePrecision'],
];

/** Count a VAE override once, and each default component swapped for another model. */
export const countModelDefaultOverrides = (
  settings: GenerateSettings,
  modelDefaultSettings: GenerateSettings
): number => {
  let count = MODEL_DEFAULT_DECISION_GROUPS.filter((group) =>
    group.some((key) => !Object.is(settings[key], modelDefaultSettings[key]))
  ).length;

  if (settings.vae?.key !== modelDefaultSettings.vae?.key) {
    count += 1;
  }

  count += DEFAULT_COMPONENT_SLOTS.filter(
    ({ setting }) => settings[setting]?.key !== modelDefaultSettings[setting]?.key
  ).length;

  return count;
};

export const settingsMatchModelDefaults = (settings: GenerateSettings, modelDefaultSettings: GenerateSettings) =>
  countModelDefaultOverrides(settings, modelDefaultSettings) === 0;
