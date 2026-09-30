import type { GenerateSettings } from '@features/generation/core/types';
import type { ExternalStoreCore } from '@platform/state/externalStoreCore';

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

export type GenerateDraftStore = ExternalStoreCore<GenerateSettings>;

const draftViews = new WeakSet<object>();

export const isDraftView = (value: object): boolean => draftViews.has(value);

/**
 * Returns a `getSnapshot` for one subscriber that yields a live view of the draft. The view is replaced (so the
 * subscriber re-renders) only when a key it has read changes; enumerating it counts as reading every key.
 *
 * Contract for holders of a view:
 * - Every read returns the latest draft, including reads through a view retained from an earlier render, so a
 *   retained view cannot be diffed against a newer one; compare read values instead.
 * - A view must not escape into stores, query keys, structuredClone, or persistence. Copy the fields you need.
 *   Only a whole view handed back to the form's settings commit is unwrapped (see `isDraftView`).
 */
export const createDraftTracker = (draft: GenerateDraftStore) => {
  let baseline = draft.getSnapshot();
  let readKeys = new Set<PropertyKey>();
  let readsAll = false;
  const read = (key: PropertyKey) => {
    readKeys.add(key);
    return Reflect.get(draft.getSnapshot(), key);
  };
  const createView = (): GenerateSettings => {
    readKeys = new Set();
    readsAll = false;
    const view = new Proxy({} as GenerateSettings, {
      defineProperty: () => false,
      deleteProperty: () => false,
      get: (_target, key) => read(key),
      getOwnPropertyDescriptor: (_target, key) => {
        read(key);
        const descriptor = Reflect.getOwnPropertyDescriptor(draft.getSnapshot(), key);
        return descriptor ? { ...descriptor, configurable: true } : undefined;
      },
      has: (_target, key) => {
        readKeys.add(key);
        return Reflect.has(draft.getSnapshot(), key);
      },
      ownKeys: () => {
        readsAll = true;
        return Reflect.ownKeys(draft.getSnapshot());
      },
      set: () => false,
    });
    draftViews.add(view);
    return view;
  };
  let view = createView();

  return () => {
    const snapshot = draft.getSnapshot();

    if (snapshot !== baseline) {
      const previous = baseline;
      const changed =
        readsAll || [...readKeys].some((key) => !Object.is(Reflect.get(previous, key), Reflect.get(snapshot, key)));
      baseline = snapshot;

      if (changed) {
        view = createView();
      }
    }

    return view;
  };
};
