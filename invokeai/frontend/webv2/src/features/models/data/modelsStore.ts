import type { ModelConfig } from '@features/models/core/types';

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

const store = createExternalStore<ModelsSnapshot>({
  coverImageVersions: {},
  error: null,
  missingModelKeys: EMPTY_MISSING_KEYS,
  models: [],
  modelsDir: null,
  status: 'idle',
});

let inflightRefresh: Promise<void> | null = null;

/** Re-fetch the library; concurrent calls share one request. */
export const refreshModels = (): Promise<void> => {
  if (inflightRefresh) {
    return inflightRefresh;
  }

  store.patchSnapshot({ status: store.getSnapshot().status === 'loaded' ? 'loaded' : 'loading' });

  inflightRefresh = Promise.all([
    listModels(),
    // Missing-file detection is best-effort; never fail the whole library.
    listMissingModels().catch(() => [] as ModelConfig[]),
    // Static server config: fetched once, best-effort.
    store.getSnapshot().modelsDir ?? getModelsDir().catch(() => null),
  ])
    .then(([models, missingModels, modelsDir]) => {
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
      store.patchSnapshot({
        error: error instanceof Error ? error.message : 'Failed to load models.',
        status: store.getSnapshot().models.length > 0 ? 'loaded' : 'error',
      });
    })
    .finally(() => {
      inflightRefresh = null;
    });

  return inflightRefresh;
};

/** Fetch once on first use; later callers get the cached snapshot. */
export const ensureModelsLoaded = (): void => {
  if (store.getSnapshot().status === 'idle') {
    void refreshModels();
  }
};

export const getModelsSnapshot = (): ModelsSnapshot => store.getSnapshot();

/** Patch one model in place after a successful update/convert. */
export const replaceModelInStore = (model: ModelConfig): void => {
  store.patchSnapshot({
    models: store.getSnapshot().models.map((existing) => (existing.key === model.key ? model : existing)),
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
