import type { ModelConfig } from '@features/models/core/types';

import {
  type AccountScope,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { createTrailingSingleFlight } from '@platform/state/singleFlight';
import { getApiErrorMessage } from '@platform/transport/http';

import { getModelsDir, listMissingModels, listModelKeysByDateAdded, listModels } from './api';

/**
 * Share backend-owned model state outside the workbench reducer; mutations patch snapshots or refresh to keep all
 * surfaces consistent.
 */

export interface ModelsSnapshot {
  models: ModelConfig[];
  /** Same models keyed for point lookups; always derived from `models`. */
  modelsByKey: ReadonlyMap<string, ModelConfig>;
  missingModelKeys: ReadonlySet<string>;
  /** Each model's position in install order, oldest first; null until the Date added sort asks for it. */
  dateAddedRanks: ReadonlyMap<string, number> | null;
  /** Bumped when a model's cover image changes; cache-busts thumbnail URLs. */
  coverImageVersions: Readonly<Record<string, number>>;
  /** Absolute server path of the models directory (resolves relative model paths). */
  modelsDir: string | null;
  status: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
}

const EMPTY_MISSING_KEYS: ReadonlySet<string> = new Set<string>();
const EMPTY_MODELS_BY_KEY: ReadonlyMap<string, ModelConfig> = new Map<string, ModelConfig>();

const EMPTY_MODELS_SNAPSHOT: ModelsSnapshot = {
  coverImageVersions: {},
  dateAddedRanks: null,
  error: null,
  missingModelKeys: EMPTY_MISSING_KEYS,
  models: [],
  modelsByKey: EMPTY_MODELS_BY_KEY,
  modelsDir: null,
  status: 'idle',
};

/** Every `models` write goes through here so the by-key index never drifts. */
const withModels = (models: ModelConfig[]): Pick<ModelsSnapshot, 'models' | 'modelsByKey'> => ({
  models,
  modelsByKey: new Map(models.map((model) => [model.key, model])),
});
const store = createExternalStore<ModelsSnapshot>(EMPTY_MODELS_SNAPSHOT);

const refreshFlight = createTrailingSingleFlight();
let mutationGeneration = 0;
let dateAddedOrderWanted = false;

registerAccountOwnedResource({
  clear: () => {
    mutationGeneration = 0;
    dateAddedOrderWanted = false;
    refreshFlight.reset();
    store.setSnapshot(EMPTY_MODELS_SNAPSHOT);
  },
  name: 'models-library',
});

/** Re-fetch the library; concurrent calls share one request, and a call made mid-flight queues one trailing rerun. */
// Flagged on the entry itself, so every consumer of the catalog sees it, not only the library views.
const withMissingFlags = (models: ModelConfig[], missingKeys: ReadonlySet<string>): ModelConfig[] =>
  missingKeys.size > 0
    ? models.map((model) => (missingKeys.has(model.key) ? { ...model, is_missing: true } : model))
    : models;

/**
 * A fresh read of the library, not this tab's snapshot, with current file availability on each entry as the snapshot
 * has it. Detection is best-effort: a failed missing-file lookup lists every model as present.
 */
export const listModelsWithAvailability = async (signal?: AbortSignal): Promise<ModelConfig[]> => {
  const [models, missing] = await Promise.all([listModels(signal), listMissingModels(signal).catch(() => [])]);

  return withMissingFlags(models, new Set(missing.map((model) => model.key)));
};

export const refreshModels = (owner: AccountScope = captureAccountScope()): Promise<void> =>
  refreshFlight.run(() => {
    const generation = mutationGeneration;
    store.patchSnapshot({ status: store.getSnapshot().status === 'loaded' ? 'loaded' : 'loading' });

    return Promise.all([
      listModels(owner.signal),
      // Missing-file detection is best-effort; never fail the whole library.
      listMissingModels(owner.signal).catch(() => [] as ModelConfig[]),
      // Static server config: fetched once, best-effort.
      store.getSnapshot().modelsDir ?? getModelsDir(owner.signal).catch(() => null),
      // Read with the catalog so the ranks describe the same refresh as `models`; best-effort like the two above.
      dateAddedOrderWanted ? listModelKeysByDateAdded(owner.signal).catch(() => null) : null,
    ])
      .then(([models, missingModels, modelsDir, dateAddedKeys]) => {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        if (generation !== mutationGeneration) {
          void refreshModels(owner);
          return;
        }

        const missingKeys =
          missingModels.length > 0 ? new Set(missingModels.map((model) => model.key)) : EMPTY_MISSING_KEYS;

        store.patchSnapshot({
          ...withModels(withMissingFlags(models, missingKeys)),
          // A failed order read keeps the previous ranks.
          ...(dateAddedKeys ? { dateAddedRanks: new Map(dateAddedKeys.map((key, index) => [key, index])) } : null),
          error: null,
          missingModelKeys: missingKeys,
          modelsDir,
          status: 'loaded',
        });
      })
      .catch((error: unknown) => {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        if (generation !== mutationGeneration) {
          void refreshModels(owner);
          return;
        }

        store.patchSnapshot({
          error: getApiErrorMessage(error, 'Failed to load models.'),
          status: store.getSnapshot().models.length > 0 ? 'loaded' : 'error',
        });
      });
  });

/** Fetch on first use or retry after an error; callers share and can await the request. */
export const ensureModelsLoaded = (): Promise<void> => {
  const { status } = store.getSnapshot();

  if (status === 'idle' || status === 'error') {
    return refreshModels();
  }

  return refreshFlight.inflight() ?? Promise.resolve();
};

/**
 * Start reading install order with the catalog, from now until the account changes. Resolves whether the order is
 * loaded once the read this call asked for has settled; after a failed read, calling again retries.
 */
export const ensureDateAddedOrderLoaded = async (): Promise<boolean> => {
  if (store.getSnapshot().dateAddedRanks) {
    return true;
  }

  dateAddedOrderWanted = true;
  await refreshModels();

  // A refresh already in flight when this was called did not read the order, and one a mutation fenced discarded
  // it; each queues a rerun that reads it, so wait those out too.
  while (!store.getSnapshot().dateAddedRanks && refreshFlight.inflight()) {
    await refreshFlight.inflight();
  }

  return store.getSnapshot().dateAddedRanks !== null;
};

export const getModelsSnapshot = (): ModelsSnapshot => store.getSnapshot();

/** Read-only subscription for App-owned cross-feature runtimes. */
export const subscribeModels = (listener: () => void): (() => void) => store.subscribe(listener);

/**
 * Patch one model in place after a successful update/convert. A server record carries no file availability, so the
 * model keeps the missing flag the last refresh gave it; whoever moves its files refreshes the library afterwards.
 */
export const replaceModelInStore = (model: ModelConfig): void => {
  mutationGeneration += 1;
  const { missingModelKeys, models } = store.getSnapshot();
  const flagged = missingModelKeys.has(model.key) ? { ...model, is_missing: true } : model;

  store.patchSnapshot(withModels(models.map((existing) => (existing.key === model.key ? flagged : existing))));
};

/** Apply a narrow optimistic model patch without replacing unrelated server fields. */
export const patchModelInStore = (key: string, changes: Partial<ModelConfig>): void => {
  mutationGeneration += 1;
  store.patchSnapshot(
    withModels(store.getSnapshot().models.map((model) => (model.key === key ? { ...model, ...changes } : model)))
  );
};

export const removeModelsFromStore = (keys: string[]): void => {
  mutationGeneration += 1;
  const removed = new Set(keys);

  store.patchSnapshot(withModels(store.getSnapshot().models.filter((model) => !removed.has(model.key))));
};

/** Update cover presence and URL version without refetching; backend cover URLs remain stable. */
export const markCoverImageChanged = (key: string, hasImage: boolean): void => {
  mutationGeneration += 1;
  const { coverImageVersions, models } = store.getSnapshot();

  store.patchSnapshot({
    ...withModels(
      models.map((model) =>
        model.key === key ? { ...model, cover_image: hasImage ? (model.cover_image ?? 'present') : null } : model
      )
    ),
    coverImageVersions: { ...coverImageVersions, [key]: (coverImageVersions[key] ?? 0) + 1 },
  });
};

export const setModelsSnapshotForTests = (next: Partial<ModelsSnapshot>): void => {
  store.patchSnapshot(next.models ? { ...next, ...withModels(next.models) } : next);
};

export const useModelsSelector = store.useSelector;
