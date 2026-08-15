import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ listCustomNodePacks: vi.fn() }));

vi.mock('./api', () => api);

const pack = (name: string, nodeCount = 1) => ({
  name,
  nodeCount,
  nodeTypes: [],
  path: `/custom_nodes/${name}`,
});

const response = (...names: string[]) => ({
  customNodesPath: '/custom_nodes',
  nodePacks: names.map((name) => pack(name)),
});

describe('custom node packs store', () => {
  beforeEach(() => {
    vi.resetModules();
    api.listCustomNodePacks.mockReset();
  });

  it('shares the request between concurrent ensures and retries after an error', async () => {
    api.listCustomNodePacks.mockRejectedValueOnce(new Error('outage')).mockResolvedValueOnce(response('pack-a'));
    const store = await import('./nodesStore');

    const first = store.ensureCustomNodePacksLoaded();
    expect(store.ensureCustomNodePacksLoaded()).toBe(first);
    await first;
    expect(store.getCustomNodesSnapshot()).toMatchObject({ error: 'outage', status: 'error' });

    await store.ensureCustomNodePacksLoaded();
    expect(store.getCustomNodesSnapshot()).toMatchObject({ error: null, status: 'loaded' });
    expect(store.getCustomNodesSnapshot().nodePacks.map((candidate) => candidate.name)).toEqual(['pack-a']);
    expect(api.listCustomNodePacks).toHaveBeenCalledTimes(2);
  });

  it('reruns once when a refresh joins mid-flight', async () => {
    let resolveFirst: ((value: unknown) => void) | undefined;

    api.listCustomNodePacks
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveFirst = resolve;
        })
      )
      .mockResolvedValueOnce(response('pack-a', 'pack-b'));
    const store = await import('./nodesStore');

    const first = store.refreshCustomNodePacks();
    // Joins the in-flight request and queues one trailing rerun.
    expect(store.refreshCustomNodePacks()).toBe(first);

    resolveFirst?.(response('pack-a'));
    await first;
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(api.listCustomNodePacks).toHaveBeenCalledTimes(2);
    expect(store.getCustomNodesSnapshot().nodePacks.map((candidate) => candidate.name)).toEqual(['pack-a', 'pack-b']);
  });

  it('keeps loaded packs and records the error when a refresh fails', async () => {
    api.listCustomNodePacks.mockResolvedValueOnce(response('pack-a')).mockRejectedValueOnce(new Error('outage'));
    const store = await import('./nodesStore');

    await store.refreshCustomNodePacks();
    await store.refreshCustomNodePacks();

    // Contract pinned for the maintenance menu's explicit-refresh surfacing:
    // the stale list keeps rendering, the error waits in the snapshot.
    expect(store.getCustomNodesSnapshot()).toMatchObject({ error: 'outage', status: 'loaded' });
    expect(store.getCustomNodesSnapshot().nodePacks.map((candidate) => candidate.name)).toEqual(['pack-a']);
  });

  it('discards a resolution from a switched-away account', async () => {
    const account = await import('@platform/state/accountLifecycle');

    account.accountLifecycle.activate('user-a');
    const store = await import('./nodesStore');
    let resolveA: ((value: unknown) => void) | undefined;

    api.listCustomNodePacks.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveA = resolve;
      })
    );

    const userARefresh = store.refreshCustomNodePacks();

    account.accountLifecycle.invalidate();
    account.accountLifecycle.activate('user-b');

    resolveA?.(response('stale-pack'));
    await userARefresh;

    expect(store.getCustomNodesSnapshot()).toMatchObject({ nodePacks: [], status: 'idle' });
  });
});
