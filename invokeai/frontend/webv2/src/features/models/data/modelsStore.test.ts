import type { ModelConfig } from '@features/models/core/types';

import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
  getModelsDir: vi.fn(),
  listMissingModels: vi.fn(),
  listModels: vi.fn(),
}));

vi.mock('./api', () => api);

describe('models store loading', () => {
  beforeEach(() => {
    vi.resetModules();
    api.getModelsDir.mockReset().mockResolvedValue('/models');
    api.listMissingModels.mockReset().mockResolvedValue([]);
    api.listModels.mockReset();
  });

  it('returns its shared request and retries after an initial failure', async () => {
    api.listModels.mockRejectedValueOnce(new Error('temporary outage')).mockResolvedValueOnce([
      {
        base: 'sdxl',
        description: null,
        file_size: 1,
        format: 'checkpoint',
        hash: 'hash',
        key: 'model-key',
        name: 'Model',
        source: 'model.safetensors',
        type: 'main',
      },
    ]);
    const { ensureModelsLoaded, getModelsSnapshot } = await import('./modelsStore');

    const first = ensureModelsLoaded();
    expect(ensureModelsLoaded()).toBe(first);
    await first;
    expect(getModelsSnapshot()).toMatchObject({ error: 'temporary outage', status: 'error' });

    await ensureModelsLoaded();
    expect(getModelsSnapshot()).toMatchObject({ error: null, status: 'loaded' });
    expect(getModelsSnapshot().models.map((model) => model.key)).toEqual(['model-key']);
    expect(api.listModels).toHaveBeenCalledTimes(2);
  });

  it('keeps the by-key index in lockstep with every models write', async () => {
    api.listModels.mockResolvedValueOnce([
      { base: 'sdxl', key: 'a', name: 'A', type: 'main' },
      { base: 'sdxl', key: 'b', name: 'B', type: 'main' },
    ]);
    const store = await import('./modelsStore');

    await store.refreshModels();
    expect(store.getModelsSnapshot().modelsByKey.get('b')?.name).toBe('B');

    store.patchModelInStore('a', { name: 'A2' });
    expect(store.getModelsSnapshot().modelsByKey.get('a')?.name).toBe('A2');

    store.removeModelsFromStore(['b']);
    expect(store.getModelsSnapshot().modelsByKey.has('b')).toBe(false);
    expect(store.getModelsSnapshot().models.map((model) => model.key)).toEqual(['a']);
  });

  it('unwraps FastAPI detail bodies into the snapshot error', async () => {
    const { ApiError } = await import('@platform/transport/http');
    api.listModels.mockRejectedValueOnce(new ApiError('{"detail":"Model records unavailable"}', 500));
    const { getModelsSnapshot, refreshModels } = await import('./modelsStore');

    await refreshModels();

    expect(getModelsSnapshot()).toMatchObject({ error: 'Model records unavailable', status: 'error' });
  });

  it('keeps the new account request authoritative when the old request resolves last', async () => {
    const account = await import('@platform/state/accountLifecycle');
    account.accountLifecycle.activate('user-a');
    const modelsStore = await import('./modelsStore');
    const modelB: ModelConfig = {
      base: 'sdxl',
      description: null,
      file_size: 1,
      format: 'checkpoint',
      hash: 'hash-b',
      key: 'model-b',
      name: 'Model B',
      path: 'model-b.safetensors',
      source: 'model-b.safetensors',
      source_type: 'path',
      type: 'main',
    } as ModelConfig;
    let resolveA: ((value: ModelConfig[]) => void) | undefined;
    let resolveB: ((value: ModelConfig[]) => void) | undefined;
    api.listModels
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveA = resolve;
        })
      )
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveB = resolve;
        })
      )
      // The join during user B's flight queues a trailing rerun.
      .mockResolvedValueOnce([modelB]);
    const userARefresh = modelsStore.refreshModels();

    account.accountLifecycle.invalidate();
    account.accountLifecycle.activate('user-b');
    const userBRefresh = modelsStore.refreshModels();

    resolveA?.([]);
    await userARefresh;
    expect(modelsStore.refreshModels()).toBe(userBRefresh);

    resolveB?.([modelB]);
    await userBRefresh;

    expect(modelsStore.getModelsSnapshot().models.map((model) => model.key)).toEqual(['model-b']);

    // The mid-flight join re-fetches once more after the shared flight settles.
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    expect(api.listModels).toHaveBeenCalledTimes(3);
    expect(modelsStore.getModelsSnapshot().models.map((model) => model.key)).toEqual(['model-b']);
  });
});
