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

import { getModelsDir, listMissingModels, listModels } from './api';

/**
 * Shared library store for installed model configs. Lives outside the
 * workbench reducer because the list is backend-owned server state shared by
 * every model surface (manager, detail views, pickers); a single
 * module store keeps them consistent and avoids re-fetch storms. Mutations go
 * through the API layer and then either patch the snapshot in place (fast
 * path) or trigger a refresh.
 */

export interface ModelsSnapshot {
  models: ModelConfig[];
  /** Same models keyed for point lookups; always derived from `models`. */
  modelsByKey: ReadonlyMap<string, ModelConfig>;
  /** Keys of models whose files are missing on disk. */
  missingModelKeys: ReadonlySet<string>;
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

registerAccountOwnedResource({
  clear: () => {
    refreshFlight.reset();
    store.setSnapshot(EMPTY_MODELS_SNAPSHOT);
  },
  name: 'models-library',
});

/** Re-fetch the library; concurrent calls share one request, and a call made mid-flight queues one trailing rerun. */
export const refreshModels = (owner: AccountScope = captureAccountScope()): Promise<void> =>
  refreshFlight.run(() => {
    store.patchSnapshot({ status: store.getSnapshot().status === 'loaded' ? 'loaded' : 'loading' });

    return Promise.all([
      listModels(owner.signal),
      // Missing-file detection is best-effort; never fail the whole library.
      listMissingModels(owner.signal).catch(() => [] as ModelConfig[]),
      // Static server config: fetched once, best-effort.
      store.getSnapshot().modelsDir ?? getModelsDir(owner.signal).catch(() => null),
    ])
      .then(([models, missingModels, modelsDir]) => {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        store.patchSnapshot({
          ...withModels(models),
          error: null,
          missingModelKeys:
            missingModels.length > 0 ? new Set(missingModels.map((model) => model.key)) : EMPTY_MISSING_KEYS,
          modelsDir,
          status: 'loaded',
        });
      })
      .catch((error: unknown) => {
        if (!isAccountScopeCurrent(owner)) {
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

export const getModelsSnapshot = (): ModelsSnapshot => store.getSnapshot();

/** Read-only subscription for App-owned cross-feature runtimes. */
export const subscribeModels = (listener: () => void): (() => void) => store.subscribe(listener);

/** Patch one model in place after a successful update/convert. */
export const replaceModelInStore = (model: ModelConfig): void => {
  store.patchSnapshot(
    withModels(store.getSnapshot().models.map((existing) => (existing.key === model.key ? model : existing)))
  );
};

/** Apply a narrow optimistic model patch without replacing unrelated server fields. */
export const patchModelInStore = (key: string, changes: Partial<ModelConfig>): void => {
  store.patchSnapshot(
    withModels(store.getSnapshot().models.map((model) => (model.key === key ? { ...model, ...changes } : model)))
  );
};

export const removeModelsFromStore = (keys: string[]): void => {
  const removed = new Set(keys);

  store.patchSnapshot(withModels(store.getSnapshot().models.filter((model) => !removed.has(model.key))));
};

/**
 * Record a cover image upload/removal without refetching: keeps the truthy
 * `cover_image` marker in sync for thumbnails and bumps the version that
 * cache-busts their URLs (the backend serves the image at a stable URL).
 */
export const markCoverImageChanged = (key: string, hasImage: boolean): void => {
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
