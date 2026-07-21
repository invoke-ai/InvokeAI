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

export const settingsMatchModelDefaults = (settings: GenerateSettings, modelDefaultSettings: GenerateSettings) =>
  MODEL_DEFAULT_VALUE_KEYS.every((key) => Object.is(settings[key], modelDefaultSettings[key])) &&
  settings.vae?.key === modelDefaultSettings.vae?.key &&
  settings.loras.length === modelDefaultSettings.loras.length &&
  settings.loras.every((lora, index) => lora.isEnabled === modelDefaultSettings.loras[index]?.isEnabled);
