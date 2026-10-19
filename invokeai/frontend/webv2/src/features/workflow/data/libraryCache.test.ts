import type * as accountLifecycleModule from '@platform/state/accountLifecycle';
import type * as httpModule from '@platform/transport/http';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { WorkflowLibraryListItem, WorkflowLibraryPage, WorkflowRecordDTO } from './api';
import type * as libraryBrowseStoreModule from './libraryBrowseStore';
import type * as libraryCacheModule from './libraryCache';

const api = vi.hoisted(() => ({
  getAllWorkflowTags: vi.fn(),
  getLibraryWorkflowRecord: vi.fn(),
  getWorkflowTagCounts: vi.fn(),
  listLibraryWorkflows: vi.fn(),
}));

const templates = vi.hoisted(() => ({
  getInvocationTemplatesSnapshot: vi.fn(),
  refreshInvocationTemplates: vi.fn(),
}));

vi.mock('./api', () => api);
vi.mock('./templates', () => templates);

let account: typeof accountLifecycleModule;
let browse: typeof libraryBrowseStoreModule;
let cache: typeof libraryCacheModule;
/** Re-imported with the cache so `instanceof ApiError` matches across `vi.resetModules()`. */
let http: typeof httpModule;

const buildItem = (workflowId: string, revision: number): WorkflowLibraryListItem => ({
  category: 'user',
  description: '',
  name: `Workflow ${workflowId}`,
  revision,
  workflow_id: workflowId,
});

const buildPage = (items: WorkflowLibraryListItem[]): WorkflowLibraryPage => ({
  items,
  page: 0,
  pages: 1,
  total: items.length,
});

/** Revision `n` of a workflow holds `n` notes nodes, so the graph a caller receives names its revision. */
const buildRecord = (workflowId: string, revision: number): WorkflowRecordDTO => ({
  ...buildItem(workflowId, revision),
  workflow: {
    edges: [],
    name: `Workflow ${workflowId}`,
    nodes: Array.from({ length: revision }, (_unused, index) => ({
      data: { label: '', notes: '' },
      id: `note-${index}`,
      position: { x: 0, y: 0 },
      type: 'notes',
    })),
  },
});

const createDeferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};

const flushAsyncWork = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

const read = (workflowId: string, expectedRevision?: number, signal?: AbortSignal) =>
  cache.getLibraryWorkflowRecordCached(workflowId, { expectedRevision, signal });

beforeEach(async () => {
  vi.resetModules();
  api.getAllWorkflowTags.mockReset().mockResolvedValue([]);
  api.getWorkflowTagCounts.mockReset().mockResolvedValue({});
  api.listLibraryWorkflows.mockReset().mockResolvedValue(buildPage([]));
  api.getLibraryWorkflowRecord.mockReset();
  templates.getInvocationTemplatesSnapshot
    .mockReset()
    .mockReturnValue({ error: null, status: 'loaded', templates: {} });
  templates.refreshInvocationTemplates.mockReset().mockResolvedValue(undefined);
  cache = await import('./libraryCache');
  browse = await import('./libraryBrowseStore');
  account = await import('@platform/state/accountLifecycle');
  http = await import('@platform/transport/http');
  account.accountLifecycle.activate('user-a');
});

describe('workflow library record freshness', () => {
  const firstPageCalls = () => api.listLibraryWorkflows.mock.calls.filter(([params]) => params.perPage === 20);
  const enrichedNodeCount = () => {
    const enrichment = browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.enrichment;

    return enrichment?.status === 'ready' ? enrichment.nodeCount : null;
  };

  it('refetches a cached record once the list shows a newer revision', async () => {
    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('a', 1)]));
    api.getLibraryWorkflowRecord.mockResolvedValue(buildRecord('a', 1));

    await browse.ensureWorkflowLibraryBrowseLoaded();
    await vi.waitFor(() => expect(enrichedNodeCount()).toBe(1));

    // Opening what the list shows reuses the record enrichment read.
    await expect(read('a', 1)).resolves.toMatchObject({ revision: 1 });
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(1);

    // Another editor saves revision 2; the list revalidates and shows it.
    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('a', 2)]));
    api.getLibraryWorkflowRecord.mockResolvedValue(buildRecord('a', 2));
    await browse.refreshWorkflowLibraryBrowse();

    await vi.waitFor(() => expect(enrichedNodeCount()).toBe(2));
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(2);
    await expect(cache.getLibraryWorkflowCached('a', { expectedRevision: 2 })).resolves.toMatchObject({
      nodes: [{ id: 'note-0' }, { id: 'note-1' }],
    });
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(2);
  });

  it('always fetches for a caller that names no revision', async () => {
    api.getLibraryWorkflowRecord.mockResolvedValueOnce(buildRecord('a', 1)).mockResolvedValueOnce(buildRecord('a', 2));

    await read('a', 1);

    await expect(read('a')).resolves.toMatchObject({ revision: 2 });
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(2);
  });

  it('serves a record newer than the list and has the list refresh exactly once', async () => {
    const listener = vi.fn();
    cache.onWorkflowLibraryCacheInvalidated(listener);
    // The list was read at revision 1, then another editor saved revision 2 before the row was enriched.
    api.listLibraryWorkflows
      .mockResolvedValueOnce(buildPage([buildItem('a', 1)]))
      .mockResolvedValue(buildPage([buildItem('a', 2)]));
    api.getLibraryWorkflowRecord.mockResolvedValue(buildRecord('a', 2));

    await browse.ensureWorkflowLibraryBrowseLoaded();
    await vi.waitFor(() => expect(browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.item.revision).toBe(2));
    await vi.waitFor(() => expect(enrichedNodeCount()).toBe(2));
    // A caller still holding the old row is served the newer record without another report.
    await expect(read('a', 1)).resolves.toMatchObject({ revision: 2 });
    await flushAsyncWork();

    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenCalledWith('a');
    expect(firstPageCalls()).toHaveLength(2);
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(1);
  });

  it('never moves a cached record back to an older revision when reads finish out of order', async () => {
    const older = createDeferred<WorkflowRecordDTO>();
    api.getLibraryWorkflowRecord.mockReturnValueOnce(older.promise).mockResolvedValueOnce(buildRecord('a', 2));

    const slow = read('a');
    await expect(read('a')).resolves.toMatchObject({ revision: 2 });
    older.resolve(buildRecord('a', 1));

    await expect(slow).resolves.toMatchObject({ revision: 2 });
    await expect(read('a', 2)).resolves.toMatchObject({ revision: 2 });
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(2);
  });

  it('drops a record the server reports missing, but keeps it through other failures', async () => {
    api.getLibraryWorkflowRecord.mockResolvedValueOnce(buildRecord('a', 1));
    await read('a', 1);

    const outage = new http.ApiError('{"detail":"Internal Server Error"}', 500);
    api.getLibraryWorkflowRecord.mockRejectedValueOnce(outage);
    await expect(read('a', 2)).rejects.toBe(outage);
    await expect(read('a', 1)).resolves.toMatchObject({ revision: 1 });
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(2);

    const missing = new http.ApiError('{"detail":"Workflow not found"}', 404);
    api.getLibraryWorkflowRecord.mockRejectedValueOnce(missing);
    await expect(read('a', 2)).rejects.toBe(missing);

    // The revision-1 record would satisfy this caller had it survived the 404.
    api.getLibraryWorkflowRecord.mockRejectedValueOnce(missing);
    await expect(read('a', 1)).rejects.toBe(missing);
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(4);
  });

  it('rejects a read its caller aborts mid-flight, storing nothing and keeping the cached record', async () => {
    api.getLibraryWorkflowRecord.mockResolvedValueOnce(buildRecord('a', 1));
    await read('a', 1);

    const late = createDeferred<WorkflowRecordDTO>();
    api.getLibraryWorkflowRecord.mockImplementationOnce(
      (_workflowId: string, signal: AbortSignal) =>
        new Promise((resolve, reject) => {
          signal.addEventListener('abort', () => reject(signal.reason));
          void late.promise.then(resolve);
        })
    );
    const controller = new AbortController();
    const aborted = read('a', 2, controller.signal);

    controller.abort();
    late.resolve(buildRecord('a', 2));

    await expect(aborted).rejects.toMatchObject({ name: 'AbortError' });
    // Revision 1, from the cache: the abort neither stored revision 2 nor evicted what was there.
    await expect(read('a', 1)).resolves.toMatchObject({ revision: 1 });
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(2);
  });

  it('keeps at most 50 records, evicting the least recently used', async () => {
    api.getLibraryWorkflowRecord.mockImplementation((workflowId: string) =>
      Promise.resolve(buildRecord(workflowId, 1))
    );

    for (let index = 0; index < 50; index += 1) {
      await read(`workflow-${index}`, 1);
    }
    // A cache hit makes workflow-0 the most recently used, so the next insertion evicts workflow-1 instead.
    await read('workflow-0', 1);
    await read('workflow-50', 1);
    api.getLibraryWorkflowRecord.mockClear();

    await read('workflow-0', 1);
    await read('workflow-2', 1);
    await read('workflow-50', 1);
    expect(api.getLibraryWorkflowRecord).not.toHaveBeenCalled();

    await read('workflow-1', 1);
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledWith('workflow-1', expect.anything());
  });
});

describe('workflow library invalidation fencing', () => {
  it('keeps a read in flight across an invalidation from repopulating the cache or answering as current', async () => {
    const beforeWrite = createDeferred<WorkflowRecordDTO>();
    const retry = createDeferred<WorkflowRecordDTO>();
    api.getLibraryWorkflowRecord
      .mockReturnValueOnce(beforeWrite.promise)
      .mockReturnValueOnce(retry.promise)
      .mockResolvedValueOnce(buildRecord('a', 2));

    const delayed = read('a', 1);
    cache.invalidateWorkflowLibraryCache('a');
    beforeWrite.resolve(buildRecord('a', 1));
    await vi.waitFor(() => expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(2));

    // Had the pre-write answer been stored, this would be served from the cache.
    await expect(read('a', 1)).resolves.toMatchObject({ revision: 2 });
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(3);

    retry.resolve(buildRecord('a', 2));
    await expect(delayed).resolves.toMatchObject({ revision: 2 });
  });

  it('retries only once when invalidations keep landing mid-read', async () => {
    const first = createDeferred<WorkflowRecordDTO>();
    const second = createDeferred<WorkflowRecordDTO>();
    api.getLibraryWorkflowRecord.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);

    const delayed = read('a', 1);
    cache.invalidateWorkflowLibraryCache();
    first.resolve(buildRecord('a', 1));
    await vi.waitFor(() => expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(2));
    cache.invalidateWorkflowLibraryCache();
    second.resolve(buildRecord('a', 2));

    await expect(delayed).rejects.toBeInstanceOf(cache.WorkflowLibraryChangedDuringReadError);
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(2);
  });

  it('keeps a browse row pending, then enriched, when invalidations exhaust its read', async () => {
    const first = createDeferred<WorkflowRecordDTO>();
    const retry = createDeferred<WorkflowRecordDTO>();
    api.listLibraryWorkflows.mockResolvedValueOnce(buildPage([buildItem('a', 1)]));
    api.getLibraryWorkflowRecord
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(retry.promise)
      .mockResolvedValue(buildRecord('a', 1));

    await browse.ensureWorkflowLibraryBrowseLoaded();
    await vi.waitFor(() => expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(1));
    // The refreshes these invalidations start never land, so nothing but the row's own read can enrich it.
    api.listLibraryWorkflows.mockReturnValue(createDeferred<WorkflowLibraryPage>().promise);

    cache.invalidateWorkflowLibraryCache();
    first.resolve(buildRecord('a', 1));
    await vi.waitFor(() => expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(2));
    cache.invalidateWorkflowLibraryCache();
    retry.resolve(buildRecord('a', 1));

    await vi.waitFor(() =>
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.enrichment).toMatchObject({
        nodeCount: 1,
        status: 'ready',
      })
    );
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(3);
  });

  it('notifies registered listeners when the cache is invalidated', () => {
    const listener = vi.fn();
    const unsubscribe = cache.onWorkflowLibraryCacheInvalidated(listener);

    cache.invalidateWorkflowLibraryCache();
    expect(listener).toHaveBeenCalledTimes(1);

    cache.invalidateWorkflowLibraryCache('workflow-1');
    expect(listener).toHaveBeenLastCalledWith('workflow-1');

    unsubscribe();
    cache.invalidateWorkflowLibraryCache();
    expect(listener).toHaveBeenCalledTimes(2);
  });
});

describe('workflow library account ownership', () => {
  it('clears records synchronously on account invalidation', async () => {
    api.getLibraryWorkflowRecord.mockResolvedValue(buildRecord('workflow-a', 1));
    await read('workflow-a', 1);

    account.accountLifecycle.invalidate();
    account.accountLifecycle.activate('user-b');

    await read('workflow-a', 1);
    expect(api.getLibraryWorkflowRecord).toHaveBeenCalledTimes(2);
  });

  it('rejects delayed record completions from the prior epoch', async () => {
    const delayed = createDeferred<WorkflowRecordDTO>();
    api.getLibraryWorkflowRecord.mockReturnValueOnce(delayed.promise);

    const oldWorkflow = cache.getLibraryWorkflowCached('shared-id', { expectedRevision: 1 });
    const workflowSignal = api.getLibraryWorkflowRecord.mock.calls[0]?.[1] as AbortSignal;

    expect(workflowSignal.aborted).toBe(false);
    account.accountLifecycle.invalidate();
    account.accountLifecycle.activate('user-b');

    expect(workflowSignal.aborted).toBe(true);
    delayed.resolve({ ...buildRecord('shared-id', 1), workflow: { owner: 'a' } });

    await expect(oldWorkflow).rejects.toThrow('no longer active');

    api.getLibraryWorkflowRecord.mockResolvedValueOnce({ ...buildRecord('shared-id', 1), workflow: { owner: 'b' } });
    await expect(cache.getLibraryWorkflowCached('shared-id', { expectedRevision: 1 })).resolves.toEqual({
      id: 'shared-id',
      owner: 'b',
    });
  });
});
