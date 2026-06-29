import type { GenerateSettings } from '@workbench/generation/types';

export type GenerateSettingsUpdate = Partial<GenerateSettings> | ((settings: GenerateSettings) => GenerateSettings);

export type PendingGenerateSettingsUpdate = ((settings: GenerateSettings) => GenerateSettings) | null;

export const applyGenerateSettingsPatch = (
  settings: GenerateSettings,
  patch: Partial<GenerateSettings> | null
): GenerateSettings => (patch ? { ...settings, ...patch } : settings);

const getGenerateSettingsUpdater = (
  update: GenerateSettingsUpdate
): ((settings: GenerateSettings) => GenerateSettings) => {
  if (typeof update === 'function') {
    return update;
  }

  return (settings) => applyGenerateSettingsPatch(settings, update);
};

export const mergeGenerateSettingsUpdate = (
  pendingUpdate: PendingGenerateSettingsUpdate,
  update: GenerateSettingsUpdate
): PendingGenerateSettingsUpdate => {
  const updater = getGenerateSettingsUpdater(update);

  if (!pendingUpdate) {
    return updater;
  }

  return (settings) => updater(pendingUpdate(settings));
};

export const applyGenerateSettingsUpdate = (
  settings: GenerateSettings,
  update: PendingGenerateSettingsUpdate
): GenerateSettings => (update ? update(settings) : settings);

export const getChangedGenerateSettingsPatch = (
  current: GenerateSettings,
  next: GenerateSettings
): Partial<GenerateSettings> => {
  const patch: Partial<GenerateSettings> = {};

  for (const key of Object.keys(next) as Array<keyof GenerateSettings>) {
    if (!Object.is(current[key], next[key])) {
      patch[key] = next[key] as never;
    }
  }

  return patch;
};
