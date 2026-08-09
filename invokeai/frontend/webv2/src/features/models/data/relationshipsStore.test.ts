import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
  addModelRelationship: vi.fn(),
  getRelatedModelKeys: vi.fn(),
  removeModelRelationship: vi.fn(),
}));

vi.mock('./api', () => api);

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
