import type { TFunction } from 'i18next';

import { formatBytes } from '@platform/i18n/languages';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { apiFetchJson, getApiErrorMessage } from '@platform/transport/http';

/** The backend sizes the model cache in binary gigabytes (GiB). */
export const formatModelCacheBytes = (bytes: number | null | undefined): string => formatBytes(bytes, { binary: true });

/** `used/total` sharing one unit when both sizes read in the same one ("6.2/12.0 GiB"), each with its own otherwise. */
export const formatModelCacheRange = (used: number, total: number): string => {
  const usedText = formatModelCacheBytes(used);
  const totalText = formatModelCacheBytes(total);
  const unitStart = usedText.lastIndexOf('\u00a0');

  return unitStart >= 0 && totalText.endsWith(usedText.slice(unitStart))
    ? `${usedText.slice(0, unitStart)}/${totalText}`
    : `${usedText}/${totalText}`;
};

/** Cache statistics have no socket stream; they refresh while watched (on watch and model load) and after a clear. */

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

let watchers = 0;

/** Marks the stats as on screen and reads them; returns the release. */
export const watchModelCacheStats = (): (() => void) => {
  watchers += 1;
  void refreshModelCacheStats();

  return () => {
    watchers -= 1;
  };
};

/** The realtime runtime's model-load hook: a reading nobody shows is not worth a request. */
export const refreshWatchedModelCacheStats = (): Promise<void> =>
  watchers > 0 ? refreshModelCacheStats() : Promise.resolve();

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
