import type { GenerateModelConfig, GenerateSettings, VaeModelConfig } from '@features/generation/core/types';

import { getSettingsWithModelDefaults } from '@features/generation/core/baseGenerationPolicies';
import { getModelDefaultVae, hasModelDefaultVae } from '@features/generation/core/settings';

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

/** Settings with the model's defaults applied, including its bundled default VAE when it has one. */
export const getModelDefaultSettings = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  vaeModels: VaeModelConfig[]
): GenerateSettings => {
  const nextSettings = getSettingsWithModelDefaults(settings, model);

  return hasModelDefaultVae(model) ? { ...nextSettings, vae: getModelDefaultVae(model, vaeModels) } : nextSettings;
};

/** Patch limited to the model-governed keys, so prompts and other fields stay untouched. */
export const getModelDefaultsPatch = (
  settings: GenerateSettings,
  model: GenerateModelConfig,
  vaeModels: VaeModelConfig[]
): Partial<GenerateSettings> => {
  const defaults = getModelDefaultSettings(settings, model, vaeModels);
  const patch: Partial<GenerateSettings> = { loras: defaults.loras, vae: defaults.vae };

  for (const key of MODEL_DEFAULT_VALUE_KEYS) {
    (patch as Record<string, unknown>)[key] = defaults[key];
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

/** Count each VAE or LoRA set once. */
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

  const lorasMatch =
    settings.loras.length === modelDefaultSettings.loras.length &&
    settings.loras.every((lora, index) => lora.isEnabled === modelDefaultSettings.loras[index]?.isEnabled);

  if (!lorasMatch) {
    count += 1;
  }

  return count;
};

export const settingsMatchModelDefaults = (settings: GenerateSettings, modelDefaultSettings: GenerateSettings) =>
  countModelDefaultOverrides(settings, modelDefaultSettings) === 0;
