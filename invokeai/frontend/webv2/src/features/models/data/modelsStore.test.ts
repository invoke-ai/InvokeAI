import type { ModelConfig } from '@features/models/core/types';

import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
  getModelsDir: vi.fn(),
  listMissingModels: vi.fn(),
  listModelKeysByDateAdded: vi.fn(),
  listModels: vi.fn(),
}));

vi.mock('./api', () => api);

describe('models store loading', () => {
  beforeEach(() => {
    vi.resetModules();
    api.getModelsDir.mockReset().mockResolvedValue('/models');
    api.listMissingModels.mockReset().mockResolvedValue([]);
    api.listModelKeysByDateAdded.mockReset();
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

  it('reads the library fresh with file availability, and lists every model present when detection fails', async () => {
    const a = { base: 'sdxl', key: 'a', name: 'A', type: 'main' };
    const b = { base: 'sdxl', key: 'b', name: 'B', type: 'main' };
    const { listModelsWithAvailability } = await import('./modelsStore');

    api.listModels.mockResolvedValueOnce([a, b]);
    api.listMissingModels.mockResolvedValueOnce([b]);
    await expect(listModelsWithAvailability()).resolves.toEqual([a, { ...b, is_missing: true }]);

    api.listModels.mockResolvedValueOnce([a, b]);
    api.listMissingModels.mockRejectedValueOnce(new Error('down'));
    await expect(listModelsWithAvailability()).resolves.toEqual([a, b]);
  });

  it('flags a registered model whose files are gone on its catalog entry', async () => {
    const a = { base: 'sdxl', key: 'a', name: 'A', type: 'main' };
    const b = { base: 'sdxl', key: 'b', name: 'B', type: 'main' };
    api.listModels.mockResolvedValueOnce([a, b]);
    api.listMissingModels.mockResolvedValueOnce([b]);
    const store = await import('./modelsStore');

    await store.refreshModels();

    expect(store.getModelsSnapshot().models).toEqual([a, { ...b, is_missing: true }]);
    expect(store.getModelsSnapshot().modelsByKey.get('b')).toMatchObject({ is_missing: true });
  });

  it('keeps a missing model flagged when a server response replaces its record', async () => {
    const b = { base: 'sdxl', key: 'b', name: 'B', type: 'main' };
    api.listModels.mockResolvedValueOnce([b]);
    api.listMissingModels.mockResolvedValueOnce([b]);
    const store = await import('./modelsStore');

    await store.refreshModels();
    store.replaceModelInStore({ ...b, name: 'Renamed' } as ModelConfig);

    expect(store.getModelsSnapshot().modelsByKey.get('b')).toMatchObject({ is_missing: true, name: 'Renamed' });
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

  it('reads install order only once it is asked for, even mid-refresh, then with every refresh', async () => {
    const a = { base: 'sdxl', key: 'a', name: 'A', type: 'main' };
    const b = { base: 'sdxl', key: 'b', name: 'B', type: 'main' };
    api.listModels.mockResolvedValue([a, b]);
    api.listModelKeysByDateAdded
      // Outlasts the refresh already in flight, so only waiting for the trailing read can answer true.
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            setTimeout(() => resolve(['b', 'a']));
          })
      )
      .mockResolvedValueOnce(['a', 'b']);
    const store = await import('./modelsStore');

    await store.refreshModels();
    expect(api.listModelKeysByDateAdded).not.toHaveBeenCalled();
    expect(store.getModelsSnapshot().dateAddedRanks).toBeNull();

    void store.refreshModels();
    await expect(store.ensureDateAddedOrderLoaded()).resolves.toBe(true);
    expect([...store.getModelsSnapshot().dateAddedRanks!]).toEqual([
      ['b', 0],
      ['a', 1],
    ]);
    expect(store.getModelsSnapshot().models.map((model) => model.key)).toEqual(['a', 'b']);

    await store.refreshModels();
    expect([...store.getModelsSnapshot().dateAddedRanks!]).toEqual([
      ['a', 0],
      ['b', 1],
    ]);
  });

  it('reports a failed install-order read without failing the library, and retries when asked again', async () => {
    api.listModels.mockResolvedValue([{ base: 'sdxl', key: 'a', name: 'A', type: 'main' }]);
    api.listModelKeysByDateAdded.mockRejectedValueOnce(new Error('down')).mockResolvedValueOnce(['a']);
    const store = await import('./modelsStore');

    await expect(store.ensureDateAddedOrderLoaded()).resolves.toBe(false);
    expect(store.getModelsSnapshot()).toMatchObject({ dateAddedRanks: null, error: null, status: 'loaded' });
    expect(store.getModelsSnapshot().models.map((model) => model.key)).toEqual(['a']);

    await expect(store.ensureDateAddedOrderLoaded()).resolves.toBe(true);
    expect([...store.getModelsSnapshot().dateAddedRanks!]).toEqual([['a', 0]]);
  });

  it.each(['replace', 'patch', 'delete', 'cover'] as const)(
    'fences an older list after a confirmed %s and reads unrelated changes on the trailing pass',
    async (mutation) => {
      const original = { key: 'a', name: 'Before', cover_image: null } as ModelConfig;
      let finishOld!: (models: ModelConfig[]) => void;
      let finishFresh!: (models: ModelConfig[]) => void;
      api.listModels
        .mockReturnValueOnce(
          new Promise((resolve) => {
            finishOld = resolve;
          })
        )
        .mockReturnValueOnce(
          new Promise((resolve) => {
            finishFresh = resolve;
          })
        );
      const store = await import('./modelsStore');
      store.setModelsSnapshotForTests({ models: [original], status: 'loaded' });
      const refresh = store.refreshModels();
      if (mutation === 'replace') {
        store.replaceModelInStore({ ...original, name: 'After' });
      }
      if (mutation === 'patch') {
        store.patchModelInStore('a', { name: 'After' });
      }
      if (mutation === 'delete') {
        store.removeModelsFromStore(['a']);
      }
      if (mutation === 'cover') {
        store.markCoverImageChanged('a', true);
      }
      const confirmed = store.getModelsSnapshot().models;
      finishOld([original]);
      await refresh;
      expect(store.getModelsSnapshot().models).toEqual(confirmed);
      expect(api.listModels).toHaveBeenCalledTimes(2);
      finishFresh([...confirmed, { key: 'b', name: 'Unrelated' } as ModelConfig]);
      await store.ensureModelsLoaded();
      expect(store.getModelsSnapshot().modelsByKey.get('b')?.name).toBe('Unrelated');
    }
  );

  it('discards an obsolete list error after a mutation and still reads the authoritative trailing result', async () => {
    const original = { key: 'a', name: 'Before' } as ModelConfig;
    let rejectOld!: (error: Error) => void;
    let finishFresh!: (models: ModelConfig[]) => void;
    api.listModels
      .mockReturnValueOnce(
        new Promise((_resolve, reject) => {
          rejectOld = reject;
        })
      )
      .mockReturnValueOnce(
        new Promise((resolve) => {
          finishFresh = resolve;
        })
      );
    const store = await import('./modelsStore');
    store.setModelsSnapshotForTests({ models: [original], status: 'loaded' });
    const refresh = store.refreshModels();
    store.replaceModelInStore({ ...original, name: 'Confirmed' });
    rejectOld(new Error('Obsolete failure'));
    await refresh;
    expect(store.getModelsSnapshot()).toMatchObject({ error: null, status: 'loaded' });
    expect(store.getModelsSnapshot().modelsByKey.get('a')?.name).toBe('Confirmed');
    expect(api.listModels).toHaveBeenCalledTimes(2);
    finishFresh([{ ...original, name: 'Confirmed' }, { key: 'b', name: 'Unrelated' } as ModelConfig]);
    await store.ensureModelsLoaded();
    expect(store.getModelsSnapshot().modelsByKey.get('b')?.name).toBe('Unrelated');
  });

  it('accepts authoritative data after an optimistic change rolls back', async () => {
    const model = { key: 'a', name: 'Before', cpu_only: null } as ModelConfig;
    let finishOld!: (models: ModelConfig[]) => void;
    api.listModels
      .mockReturnValueOnce(
        new Promise((resolve) => {
          finishOld = resolve;
        })
      )
      .mockResolvedValueOnce([{ ...model, name: 'Server rename' }]);
    const store = await import('./modelsStore');
    store.setModelsSnapshotForTests({ models: [model], status: 'loaded' });
    const refresh = store.refreshModels();
    store.patchModelInStore('a', { cpu_only: true });
    store.patchModelInStore('a', { cpu_only: null });
    finishOld([model]);
    await refresh;
    await store.ensureModelsLoaded();
    expect(store.getModelsSnapshot().modelsByKey.get('a')).toMatchObject({ name: 'Server rename', cpu_only: null });
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

    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
    expect(api.listModels).toHaveBeenCalledTimes(3);
    expect(modelsStore.getModelsSnapshot().models.map((model) => model.key)).toEqual(['model-b']);
  });
});
