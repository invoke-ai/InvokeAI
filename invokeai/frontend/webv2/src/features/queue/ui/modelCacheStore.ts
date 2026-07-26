import type { TFunction } from 'i18next';

import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { apiFetchJson, getApiErrorMessage } from '@platform/transport/http';

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'] as const;

export const formatModelCacheBytes = (bytes: number | null | undefined): string => {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes) || bytes < 0) {
    return '—';
  }

  let value = bytes;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < BYTE_UNITS.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${unitIndex === 0 ? value : value.toFixed(1)} ${BYTE_UNITS[unitIndex]}`;
};

/**
 * RAM/VRAM model-cache statistics for the Queue widget's footer. Pull-based:
 * the backend exposes `GET /api/v2/models/stats` (no socket events for cache
 * stats), so we refresh on mount, after a model loads, and after a clear.
 */

export interface ModelCacheStats {
  hits: number;
  misses: number;
  /** Peak bytes resident in the cache. */
  high_watermark: number;
  /** Current bytes resident in the cache. */
  cache_used?: number;
  /** Number of models currently held in the cache. */
  in_cache: number;
  cleared: number;
  /** Configured cache budget, in bytes. */
  cache_size: number;
  loaded_model_sizes?: Record<string, number>;
}

export interface ModelCacheClearResult {
  models_cleared: number;
  bytes_freed: number;
}

const EMPTY_MODEL_CACHE_CLEAR_RESULT: ModelCacheClearResult = { bytes_freed: 0, models_cleared: 0 };

export const normalizeModelCacheClearResult = (result: ModelCacheClearResult | null): ModelCacheClearResult =>
  result ?? EMPTY_MODEL_CACHE_CLEAR_RESULT;

export const getModelCacheUsage = (stats: ModelCacheStats | null): { used: number; total: number } => ({
  total: stats?.cache_size ?? 0,
  used: stats?.cache_used ?? stats?.high_watermark ?? 0,
});

export const getModelCacheClearToast = (
  { bytes_freed, models_cleared }: ModelCacheClearResult,
  t: TFunction
): { title: string; description: string; status: 'info' | 'success' } => {
  if (models_cleared === 0) {
    return {
      description: t('widgets.queue.modelCache.noModelsClearedDescription'),
      status: 'info',
      title: t('widgets.queue.modelCache.noModelsCleared'),
    };
  }

  return {
    description: t('widgets.queue.modelCache.clearedDescription', {
      bytes: formatModelCacheBytes(bytes_freed),
      count: models_cleared,
    }),
    status: 'success',
    title: t('widgets.queue.modelCache.cleared'),
  };
};

export interface ModelCacheSnapshot {
  stats: ModelCacheStats | null;
  loadState: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
}

const EMPTY_MODEL_CACHE_SNAPSHOT: ModelCacheSnapshot = { error: null, loadState: 'idle', stats: null };
const store = createExternalStore<ModelCacheSnapshot>(EMPTY_MODEL_CACHE_SNAPSHOT);

let inflight: Promise<void> | null = null;

registerAccountOwnedResource({
  clear: () => {
    inflight = null;
    store.setSnapshot(EMPTY_MODEL_CACHE_SNAPSHOT);
  },
  name: 'model-cache-stats',
});

export const refreshModelCacheStats = (): Promise<void> => {
  if (inflight) {
    return inflight;
  }

  const owner = captureAccountScope();

  if (store.getSnapshot().loadState === 'idle') {
    store.patchSnapshot({ loadState: 'loading' });
  }

  const refresh = apiFetchJson<ModelCacheStats | null>('/api/v2/models/stats', { signal: owner.signal })
    .then((stats) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      store.patchSnapshot({ error: null, loadState: 'loaded', stats });
    })
    .catch((error: unknown) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      store.patchSnapshot({ error: getApiErrorMessage(error, 'Failed to load model cache stats'), loadState: 'error' });
    })
    .finally(() => {
      if (inflight === refresh) {
        inflight = null;
      }
    });

  inflight = refresh;
  return inflight;
};

export const clearModelCache = async (): Promise<ModelCacheClearResult> => {
  const owner = captureAccountScope();
  const result = await apiFetchJson<ModelCacheClearResult | null>('/api/v2/models/empty_model_cache', {
    method: 'POST',
    signal: owner.signal,
  });

  assertAccountScopeCurrent(owner);
  await refreshModelCacheStats();
  assertAccountScopeCurrent(owner);

  return normalizeModelCacheClearResult(result);
};

export const getModelCacheSnapshot = (): ModelCacheSnapshot => store.getSnapshot();

export const useModelCacheStats = (): ModelCacheSnapshot => store.useSnapshot();
