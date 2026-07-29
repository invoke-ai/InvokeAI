import type { ModelConfig } from '@features/models/core/types';

import {
  type AccountScope,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

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

const EMPTY_MODELS_SNAPSHOT: ModelsSnapshot = {
  coverImageVersions: {},
  error: null,
  missingModelKeys: EMPTY_MISSING_KEYS,
  models: [],
  modelsDir: null,
  status: 'idle',
};
const store = createExternalStore<ModelsSnapshot>(EMPTY_MODELS_SNAPSHOT);

let inflightRefresh: Promise<void> | null = null;

registerAccountOwnedResource({
  clear: () => {
    inflightRefresh = null;
    store.setSnapshot(EMPTY_MODELS_SNAPSHOT);
  },
  name: 'models-library',
});

/** Re-fetch the library; concurrent calls share one request. */
export const refreshModels = (owner: AccountScope = captureAccountScope()): Promise<void> => {
  if (inflightRefresh) {
    return inflightRefresh;
  }

  store.patchSnapshot({ status: store.getSnapshot().status === 'loaded' ? 'loaded' : 'loading' });

  const refresh = Promise.all([
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
        error: null,
        missingModelKeys:
          missingModels.length > 0 ? new Set(missingModels.map((model) => model.key)) : EMPTY_MISSING_KEYS,
        models,
        modelsDir,
        status: 'loaded',
      });
    })
    .catch((error: unknown) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      store.patchSnapshot({
        error: error instanceof Error ? error.message : 'Failed to load models.',
        status: store.getSnapshot().models.length > 0 ? 'loaded' : 'error',
      });
    })
    .finally(() => {
      if (inflightRefresh === refresh) {
        inflightRefresh = null;
      }
    });

  inflightRefresh = refresh;
  return inflightRefresh;
};

/** Fetch on first use or retry after an error; callers share and can await the request. */
export const ensureModelsLoaded = (): Promise<void> => {
  const { status } = store.getSnapshot();

  if (status === 'idle' || status === 'error') {
    return refreshModels();
  }

  return inflightRefresh ?? Promise.resolve();
};

export const getModelsSnapshot = (): ModelsSnapshot => store.getSnapshot();

/** Read-only subscription for App-owned cross-feature runtimes. */
export const subscribeModels = (listener: () => void): (() => void) => store.subscribe(listener);

/** Patch one model in place after a successful update/convert. */
export const replaceModelInStore = (model: ModelConfig): void => {
  store.patchSnapshot({
    models: store.getSnapshot().models.map((existing) => (existing.key === model.key ? model : existing)),
  });
};

/** Apply a narrow optimistic model patch without replacing unrelated server fields. */
export const patchModelInStore = (key: string, changes: Partial<ModelConfig>): void => {
  store.patchSnapshot({
    models: store.getSnapshot().models.map((model) => (model.key === key ? { ...model, ...changes } : model)),
  });
};

export const removeModelsFromStore = (keys: string[]): void => {
  const removed = new Set(keys);

  store.patchSnapshot({ models: store.getSnapshot().models.filter((model) => !removed.has(model.key)) });
};

/**
 * Record a cover image upload/removal without refetching: keeps the truthy
 * `cover_image` marker in sync for thumbnails and bumps the version that
 * cache-busts their URLs (the backend serves the image at a stable URL).
 */
export const markCoverImageChanged = (key: string, hasImage: boolean): void => {
  const { coverImageVersions, models } = store.getSnapshot();

  store.patchSnapshot({
    coverImageVersions: { ...coverImageVersions, [key]: (coverImageVersions[key] ?? 0) + 1 },
    models: models.map((model) =>
      model.key === key ? { ...model, cover_image: hasImage ? (model.cover_image ?? 'present') : null } : model
    ),
  });
};

export const setModelsSnapshotForTests = (next: Partial<ModelsSnapshot>): void => {
  store.patchSnapshot(next);
};

export const useModelsSelector = store.useSelector;

export const useModelsSnapshot = (): ModelsSnapshot => store.useSnapshot();
