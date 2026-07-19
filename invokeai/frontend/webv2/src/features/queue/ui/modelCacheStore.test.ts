import { describe, expect, it } from 'vitest';

import { getModelCacheClearToast, getModelCacheUsage, normalizeModelCacheClearResult } from './modelCacheStore';

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
});
