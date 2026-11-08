import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import {
  GENERATE_COMPONENT_VALUE_KEYS,
  getSettingsWithModelDefaults,
} from '@features/generation/core/baseGenerationPolicies';

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
 * Patch limited to the model-governed keys, so prompts and other fields stay untouched. Concepts are left as they are:
 * the model already skips the ones it cannot load, and keeping them lets a compatible model run them again.
 */
export const getModelDefaultsPatch = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  models: readonly ModelConfig[]
): Partial<GenerateSettings> => {
  const defaults = getSettingsWithModelDefaults(settings, model, models);
  const patch: Partial<GenerateSettings> = {};

  for (const key of MODEL_DEFAULT_VALUE_KEYS) {
    (patch as Record<string, unknown>)[key] = defaults[key];
  }

  for (const key of GENERATE_COMPONENT_VALUE_KEYS) {
    if (defaults[key] !== settings[key]) {
      (patch as Record<string, unknown>)[key] = defaults[key];
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

/** Count each component (the VAE, an encoder) once. */
export const countModelDefaultOverrides = (
  settings: GenerateSettings,
  modelDefaultSettings: GenerateSettings
): number =>
  MODEL_DEFAULT_DECISION_GROUPS.filter((group) =>
    group.some((key) => !Object.is(settings[key], modelDefaultSettings[key]))
  ).length +
  GENERATE_COMPONENT_VALUE_KEYS.filter((key) => settings[key]?.key !== modelDefaultSettings[key]?.key).length;

export const settingsMatchModelDefaults = (settings: GenerateSettings, modelDefaultSettings: GenerateSettings) =>
  countModelDefaultOverrides(settings, modelDefaultSettings) === 0;
