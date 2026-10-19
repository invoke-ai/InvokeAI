import type { ModelConfig } from '@features/models/core/types';

import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
  addModelRelationship: vi.fn(),
  getRelatedModelKeys: vi.fn(),
  removeModelRelationship: vi.fn(),
}));

vi.mock('./relationshipsApi', () => api);

describe('relationships store', () => {
  beforeEach(() => {
    vi.resetModules();
    api.addModelRelationship.mockReset().mockResolvedValue(undefined);
    api.getRelatedModelKeys.mockReset();
    api.removeModelRelationship.mockReset().mockResolvedValue(undefined);
  });

  it('caches per key and dedupes concurrent ensures', async () => {
    api.getRelatedModelKeys.mockResolvedValue(['b']);
    const store = await import('./relationshipsStore');

    const first = store.ensureRelatedModelKeysLoaded('a');
    expect(store.ensureRelatedModelKeysLoaded('a')).toBe(first);
    await first;

    await store.ensureRelatedModelKeysLoaded('a');
    expect(api.getRelatedModelKeys).toHaveBeenCalledTimes(1);
    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: ['b'] });
  });

  it('refresh revalidates a cached entry and rejects with an empty fallback on first failure', async () => {
    api.getRelatedModelKeys.mockRejectedValueOnce(new Error('outage')).mockResolvedValueOnce(['c']);
    const store = await import('./relationshipsStore');

    await expect(store.refreshRelatedModelKeys('a')).rejects.toThrow('outage');
    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: [] });

    await store.refreshRelatedModelKeys('a');
    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: ['c'] });
  });

  it('link patches both cached directions and leaves uncached entries absent', async () => {
    const store = await import('./relationshipsStore');
    store.setRelationshipsSnapshotForTests({ relatedKeysByModelKey: { a: [], b: [] } });

    await store.linkModels('a', 'b');
    expect(api.addModelRelationship).toHaveBeenCalledWith('a', 'b', expect.anything());
    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: ['b'], b: ['a'] });

    await store.linkModels('a', 'uncached');
    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: ['b', 'uncached'], b: ['a'] });
  });

  it('unlink removes both cached directions', async () => {
    const store = await import('./relationshipsStore');
    store.setRelationshipsSnapshotForTests({ relatedKeysByModelKey: { a: ['b'], b: ['a', 'c'] } });

    await store.unlinkModels('a', 'b');
    expect(api.removeModelRelationship).toHaveBeenCalledWith('a', 'b', expect.anything());
    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: [], b: ['c'] });
  });

  it('scrubs deleted models from every entry', async () => {
    const store = await import('./relationshipsStore');
    store.setRelationshipsSnapshotForTests({ relatedKeysByModelKey: { a: ['b', 'c'], b: ['a'], c: ['a'] } });

    store.removeModelsFromRelationships(['b']);
    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: ['c'], c: ['a'] });
  });

  it('prunes a library delete exactly once via the subscription', async () => {
    const libraryModel = (key: string): ModelConfig => ({ base: 'sdxl', key, name: key, type: 'main' }) as ModelConfig;
    const models = await import('./modelsStore');

    // Loaded before this module evaluates: the seeded baseline must cover it.
    models.setModelsSnapshotForTests({ models: [libraryModel('a'), libraryModel('b')], status: 'loaded' });

    const store = await import('./relationshipsStore');

    store.setRelationshipsSnapshotForTests({ relatedKeysByModelKey: { a: ['b'], b: ['a'] } });

    let notifications = 0;
    const unsubscribe = store.subscribeToRelationships(() => {
      notifications += 1;
    });

    models.removeModelsFromStore(['b']);

    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: [] });
    expect(notifications).toBe(1);
    unsubscribe();
  });

  it('records tombstones without notifying when nothing referenced the removed keys', async () => {
    const store = await import('./relationshipsStore');

    store.setRelationshipsSnapshotForTests({ relatedKeysByModelKey: { a: ['b'] } });

    let notifications = 0;
    const unsubscribe = store.subscribeToRelationships(() => {
      notifications += 1;
    });

    store.removeModelsFromRelationships(['unrelated']);

    expect(notifications).toBe(0);
    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: ['b'] });
    unsubscribe();
  });

  it('discards a stale refresh that resolves after a link and keeps the patched entry', async () => {
    let resolveStale: ((keys: string[]) => void) | undefined;
    api.getRelatedModelKeys
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveStale = resolve;
        })
      )
      .mockResolvedValueOnce(['b']);
    const store = await import('./relationshipsStore');
    store.setRelationshipsSnapshotForTests({ relatedKeysByModelKey: { a: [], b: [] } });

    const staleRefresh = store.refreshRelatedModelKeys('a').catch(() => {});
    await store.linkModels('a', 'b');

    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: ['b'], b: ['a'] });
    // The mutation evicted the stale GET, which unblocked the dedupe for a fresh one.
    expect(api.getRelatedModelKeys.mock.calls.filter(([key]) => key === 'a')).toHaveLength(2);

    await Promise.resolve();
    resolveStale?.(['stale']);
    await staleRefresh;

    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey['a']).toEqual(['b']);
  });

  it('link during the initial load ends with the entry present and containing the linked key', async () => {
    let resolveStale: ((keys: string[]) => void) | undefined;
    api.getRelatedModelKeys
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveStale = resolve;
        })
      )
      .mockResolvedValueOnce(['b']);
    const store = await import('./relationshipsStore');

    const initialLoad = store.ensureRelatedModelKeysLoaded('a').catch(() => {});
    await store.linkModels('a', 'b');

    resolveStale?.([]);
    await initialLoad;

    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey['a']).toEqual(['b']);
  });

  it('refetches a key whose fetch failed on the next ensure and clears the flag on success', async () => {
    api.getRelatedModelKeys.mockRejectedValueOnce(new Error('outage')).mockResolvedValueOnce(['b']);
    const store = await import('./relationshipsStore');

    await expect(store.ensureRelatedModelKeysLoaded('a')).rejects.toThrow('outage');
    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: [] });

    await store.ensureRelatedModelKeysLoaded('a');
    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: ['b'] });

    await store.ensureRelatedModelKeysLoaded('a');
    expect(api.getRelatedModelKeys).toHaveBeenCalledTimes(2);
  });

  it("a removed model's late fetch cannot resurrect its entry", async () => {
    let resolveLate: ((keys: string[]) => void) | undefined;
    api.getRelatedModelKeys.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveLate = resolve;
      })
    );
    const store = await import('./relationshipsStore');

    const lateFetch = store.ensureRelatedModelKeysLoaded('a').catch(() => {});
    store.removeModelsFromRelationships(['a']);

    resolveLate?.(['b']);
    await lateFetch;

    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({});
    expect(api.getRelatedModelKeys).toHaveBeenCalledTimes(1);
  });

  it('a link that settles after its model was deleted does not resurrect it', async () => {
    let resolveLink: (() => void) | undefined;
    api.addModelRelationship.mockReturnValueOnce(
      new Promise<void>((resolve) => {
        resolveLink = resolve;
      })
    );
    const store = await import('./relationshipsStore');
    store.setRelationshipsSnapshotForTests({ relatedKeysByModelKey: { a: [], b: [] } });

    const link = store.linkModels('a', 'b');
    store.removeModelsFromRelationships(['b']);

    resolveLink?.();
    await link;

    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: [] });
  });

  it('a link that settles after a newer unlink of the same pair does not re-add it', async () => {
    let resolveLink: (() => void) | undefined;
    api.addModelRelationship.mockReturnValueOnce(
      new Promise<void>((resolve) => {
        resolveLink = resolve;
      })
    );
    const store = await import('./relationshipsStore');
    store.setRelationshipsSnapshotForTests({ relatedKeysByModelKey: { a: [], b: [] } });

    const link = store.linkModels('a', 'b');
    // The unlink is issued from the other side of the pair and settles first.
    await store.unlinkModels('b', 'a');

    resolveLink?.();
    await link;

    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: [], b: [] });
  });

  it('an unlink that settles after a newer link does not clobber it', async () => {
    let resolveUnlink: (() => void) | undefined;
    api.removeModelRelationship.mockReturnValueOnce(
      new Promise<void>((resolve) => {
        resolveUnlink = resolve;
      })
    );
    const store = await import('./relationshipsStore');
    store.setRelationshipsSnapshotForTests({ relatedKeysByModelKey: { a: ['b'], b: ['a'] } });

    const unlink = store.unlinkModels('a', 'b');
    await store.linkModels('a', 'b');

    resolveUnlink?.();
    await unlink;

    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: ['b'], b: ['a'] });
  });

  it('clears the cache and inflight requests on account switch', async () => {
    const account = await import('@platform/state/accountLifecycle');
    account.accountLifecycle.activate('user-a');
    const store = await import('./relationshipsStore');
    let resolveA: ((keys: string[]) => void) | undefined;
    api.getRelatedModelKeys.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveA = resolve;
      })
    );

    const userAFetch = store.ensureRelatedModelKeysLoaded('a');

    account.accountLifecycle.invalidate();
    account.accountLifecycle.activate('user-b');

    resolveA?.(['stale']);
    await userAFetch;

    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({});
  });
});

describe('relationships pruning from library snapshots', () => {
  beforeEach(() => {
    vi.resetModules();
    api.getRelatedModelKeys.mockReset();
  });

  const model = (key: string) => ({ key, name: key }) as ModelConfig;

  it('drops entries for models missing from the next loaded snapshot and scrubs siblings', async () => {
    const store = await import('./relationshipsStore');
    const models = await import('./modelsStore');

    models.setModelsSnapshotForTests({ models: [model('a'), model('b'), model('c')], status: 'loaded' });
    store.setRelationshipsSnapshotForTests({ relatedKeysByModelKey: { a: ['b'], b: ['a', 'c'], c: ['b'] } });

    models.setModelsSnapshotForTests({ models: [model('b'), model('c')], status: 'loaded' });

    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ b: ['c'], c: ['b'] });
  });

  it('does not prune from non-loaded snapshots or before a first loaded baseline', async () => {
    const store = await import('./relationshipsStore');
    const models = await import('./modelsStore');

    store.setRelationshipsSnapshotForTests({ relatedKeysByModelKey: { a: ['b'], b: ['a'] } });

    // First loaded snapshot only establishes the baseline.
    models.setModelsSnapshotForTests({ models: [model('b')], status: 'loaded' });
    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: ['b'], b: ['a'] });

    // A loading patch must not prune either.
    models.setModelsSnapshotForTests({ models: [], status: 'loading' });
    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: ['b'], b: ['a'] });
  });

  it('lets a pruned model recover once the server answers for its key again', async () => {
    const store = await import('./relationshipsStore');
    const models = await import('./modelsStore');

    models.setModelsSnapshotForTests({ models: [model('a'), model('b')], status: 'loaded' });
    store.setRelationshipsSnapshotForTests({ relatedKeysByModelKey: { a: ['b'], b: ['a'] } });
    models.setModelsSnapshotForTests({ models: [model('b')], status: 'loaded' });
    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ b: [] });

    // Reinstalled: the library lists it again and a fetch succeeds.
    models.setModelsSnapshotForTests({ models: [model('a'), model('b')], status: 'loaded' });
    api.getRelatedModelKeys.mockResolvedValueOnce(['b']);
    await store.ensureRelatedModelKeysLoaded('a');

    expect(store.getRelationshipsSnapshot().relatedKeysByModelKey).toEqual({ a: ['b'], b: [] });
  });
});
