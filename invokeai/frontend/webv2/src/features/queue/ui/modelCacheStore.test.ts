import { accountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  getModelCacheClearToast,
  getModelCacheSnapshot,
  getModelCacheUsage,
  normalizeModelCacheClearResult,
  refreshModelCacheStats,
} from './modelCacheStore';

const dependencies = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', () => ({
  apiFetchJson: dependencies.apiFetchJson,
  getApiErrorMessage: (error: unknown, fallback: string) => (error instanceof Error ? error.message : fallback),
}));

const deferred = <T>(): { promise: Promise<T>; resolve: (value: T) => void } => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });

  return { promise, resolve };
};

const t = ((key: string, values?: Record<string, unknown>) => {
  const translations: Record<string, string> = {
    'widgets.queue.modelCache.cleared': 'Model cache cleared',
    'widgets.queue.modelCache.clearedDescription': `Cleared ${values?.count} cached models and freed ${values?.bytes}.`,
    'widgets.queue.modelCache.noModelsCleared': 'No cached models cleared',
    'widgets.queue.modelCache.noModelsClearedDescription':
      'No unlocked cached models were available to clear. Active models stay loaded until they are no longer in use.',
  };

  return translations[key] ?? key;
}) as never;

beforeEach(() => {
  accountLifecycle.invalidate();
  dependencies.apiFetchJson.mockReset();
});

describe('model cache stats helpers', () => {
  it('uses current cache usage instead of the historical high watermark', () => {
    expect(
      getModelCacheUsage({
        cache_size: 100,
        cache_used: 12,
        cleared: 0,
        high_watermark: 90,
        hits: 0,
        in_cache: 1,
        misses: 0,
      })
    ).toEqual({ total: 100, used: 12 });
  });

  it('reports whether clear cache actually cleared models', () => {
    expect(getModelCacheClearToast({ bytes_freed: 1024, models_cleared: 2 }, t)).toEqual({
      description: 'Cleared 2 cached models and freed 1.0 KB.',
      status: 'success',
      title: 'Model cache cleared',
    });

    expect(getModelCacheClearToast({ bytes_freed: 0, models_cleared: 0 }, t)).toEqual({
      description:
        'No unlocked cached models were available to clear. Active models stay loaded until they are no longer in use.',
      status: 'info',
      title: 'No cached models cleared',
    });
  });

  it('normalizes legacy null clear-cache responses', () => {
    expect(normalizeModelCacheClearResult(null)).toEqual({ bytes_freed: 0, models_cleared: 0 });
  });

  it('drops a stale stats response without releasing the replacement account refresh', async () => {
    const first = deferred<null>();
    const second = deferred<{
      cache_size: number;
      cleared: number;
      high_watermark: number;
      hits: number;
      in_cache: number;
      misses: number;
    }>();

    dependencies.apiFetchJson.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);

    const staleRefresh = refreshModelCacheStats();

    accountLifecycle.invalidate();

    expect(getModelCacheSnapshot()).toEqual({ error: null, loadState: 'idle', stats: null });

    const currentRefresh = refreshModelCacheStats();

    first.resolve(null);
    await staleRefresh;

    expect(getModelCacheSnapshot()).toEqual({ error: null, loadState: 'loading', stats: null });
    expect(refreshModelCacheStats()).toBe(currentRefresh);
    expect(dependencies.apiFetchJson).toHaveBeenCalledTimes(2);

    second.resolve({
      cache_size: 100,
      cleared: 0,
      high_watermark: 20,
      hits: 1,
      in_cache: 1,
      misses: 2,
    });
    await currentRefresh;

    expect(getModelCacheSnapshot()).toMatchObject({
      error: null,
      loadState: 'loaded',
      stats: { cache_size: 100, high_watermark: 20, in_cache: 1 },
    });
  });
});
