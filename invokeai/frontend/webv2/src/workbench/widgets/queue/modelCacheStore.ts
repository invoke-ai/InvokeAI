import { apiFetchJson, getApiErrorMessage } from '@workbench/backend/http';
import { createExternalStore } from '@workbench/externalStore';
import { formatBytes } from '@workbench/models/taxonomy';

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

export const getModelCacheClearToast = ({
  bytes_freed,
  models_cleared,
}: ModelCacheClearResult): { title: string; description: string; status: 'info' | 'success' } => {
  if (models_cleared === 0) {
    return {
      description:
        'No unlocked cached models were available to clear. Active models stay loaded until they are no longer in use.',
      status: 'info',
      title: 'No cached models cleared',
    };
  }

  return {
    description: `Cleared ${models_cleared} cached model${models_cleared === 1 ? '' : 's'} and freed ${formatBytes(bytes_freed)}.`,
    status: 'success',
    title: 'Model cache cleared',
  };
};

interface ModelCacheSnapshot {
  stats: ModelCacheStats | null;
  loadState: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
}

const store = createExternalStore<ModelCacheSnapshot>({ error: null, loadState: 'idle', stats: null });

let inflight: Promise<void> | null = null;

export const refreshModelCacheStats = (): Promise<void> => {
  if (inflight) {
    return inflight;
  }

  if (store.getSnapshot().loadState === 'idle') {
    store.patchSnapshot({ loadState: 'loading' });
  }

  inflight = apiFetchJson<ModelCacheStats | null>('/api/v2/models/stats')
    .then((stats) => {
      store.patchSnapshot({ error: null, loadState: 'loaded', stats });
    })
    .catch((error: unknown) => {
      store.patchSnapshot({ error: getApiErrorMessage(error, 'Failed to load cache stats'), loadState: 'error' });
    })
    .finally(() => {
      inflight = null;
    });

  return inflight;
};

export const clearModelCache = async (): Promise<ModelCacheClearResult> => {
  const result = await apiFetchJson<ModelCacheClearResult | null>('/api/v2/models/empty_model_cache', {
    method: 'POST',
  });
  await refreshModelCacheStats();
  return normalizeModelCacheClearResult(result);
};

export const useModelCacheStats = (): ModelCacheSnapshot => store.useSnapshot();
