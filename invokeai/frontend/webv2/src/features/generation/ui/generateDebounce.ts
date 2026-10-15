import type { GenerateSettings } from '@features/generation/core/types';
import type { ReadableExternalStore } from '@platform/state/projectedExternalStore';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';

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

/** Keeps `current`'s values where `next` only re-created equal ones, so identity-keyed subscribers see no change. */
export const reuseEqualGenerateSettingsValues = (
  current: GenerateSettings,
  next: GenerateSettings
): GenerateSettings => {
  let reconciled: GenerateSettings | null = null;

  for (const key of Object.keys(next) as Array<keyof GenerateSettings>) {
    if (
      !Object.is(current[key], next[key]) &&
      Object.hasOwn(current, key) &&
      areJsonValuesStructurallyEqual(current[key], next[key])
    ) {
      reconciled ??= { ...next };
      reconciled[key] = current[key] as never;
    }
  }

  return reconciled ?? next;
};

/**
 * The form's draft settings as a read-only store: each snapshot is plain immutable `GenerateSettings`. Sections
 * subscribe through a selector of the fields they render and read `getSnapshot()` in handlers that need more.
 */
export type GenerateDraft = ReadableExternalStore<GenerateSettings>;

/**
 * Selects the listed settings. Under the external-store selector hooks' shallow equality the selection keeps its
 * identity until a listed value changes, so a subscriber re-renders only for its own fields.
 */
export const pickGenerateSettings =
  <Key extends keyof GenerateSettings>(keys: readonly Key[]) =>
  (settings: GenerateSettings): Pick<GenerateSettings, Key> => {
    const picked = {} as Pick<GenerateSettings, Key>;

    for (const key of keys) {
      picked[key] = settings[key];
    }

    return picked;
  };
