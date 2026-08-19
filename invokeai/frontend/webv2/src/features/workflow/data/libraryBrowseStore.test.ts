import type { InvocationTemplate } from '@features/workflow/core/types';
import type * as accountLifecycleModule from '@platform/state/accountLifecycle';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { WorkflowLibraryListItem, WorkflowLibraryPage } from './api';
import type * as libraryBrowseStoreModule from './libraryBrowseStore';

const api = vi.hoisted(() => ({
  getAllWorkflowTags: vi.fn(),
  getWorkflowTagCounts: vi.fn(),
  listLibraryWorkflows: vi.fn(),
}));

const libraryCache = vi.hoisted(() => ({
  getLibraryWorkflowCached: vi.fn(),
  onWorkflowLibraryCacheInvalidated: vi.fn(),
}));

const templates = vi.hoisted(() => ({
  getInvocationTemplatesSnapshot: vi.fn(),
  refreshInvocationTemplates: vi.fn(),
}));

vi.mock('./api', () => api);
vi.mock('./libraryCache', () => libraryCache);
vi.mock('./templates', () => templates);

const MODEL_LOADER_TEMPLATE: InvocationTemplate = {
  category: 'model',
  classification: 'stable',
  description: '',
  inputs: {
    model: {
      default: undefined,
      description: '',
      exclusiveMaximum: null,
      exclusiveMinimum: null,
      fieldKind: 'input',
      input: 'any',
      maximum: null,
      minimum: null,
      multipleOf: null,
      name: 'model',
      options: null,
      required: true,
      title: 'Model',
      type: { batch: false, cardinality: 'SINGLE', name: 'ModelIdentifierField' },
      uiChoiceLabels: null,
      uiComponent: null,
      uiHidden: false,
      uiModelBase: ['sdxl'],
      uiModelFormat: null,
      uiModelType: ['main'],
      uiOrder: null,
    },
  },
  nodePack: 'invokeai',
  outputs: {},
  outputType: 'model_loader_output',
  tags: [],
  title: 'Main Model',
  type: 'main_model_loader',
  useCache: true,
  version: '1.0.0',
};

const buildItem = (workflowId: string, overrides: Partial<WorkflowLibraryListItem> = {}): WorkflowLibraryListItem => ({
  category: 'user',
  description: '',
  name: `Workflow ${workflowId}`,
  tags: 'lora, upscaling',
  updated_at: '2026-08-18T00:00:00Z',
  workflow_id: workflowId,
  ...overrides,
});

const buildPage = (
  items: WorkflowLibraryListItem[],
  overrides: Partial<Omit<WorkflowLibraryPage, 'items'>> = {}
): WorkflowLibraryPage => ({ items, page: 0, pages: 1, total: items.length, ...overrides });

/** Minimal WorkflowV3 payload: `count` model-loader nodes, each holding a distinct model. */
const buildRawWorkflow = (count: number): Record<string, unknown> => ({
  edges: [],
  name: 'Raw workflow',
  nodes: Array.from({ length: count }, (_unused, index) => ({
    data: {
      inputs: {
        model: {
          label: '',
          name: 'model',
          value: { base: 'sdxl', key: `model-${index}`, name: `Model ${index}`, type: 'main' },
        },
      },
      type: 'main_model_loader',
    },
    id: `node-${index}`,
    position: { x: 0, y: 0 },
    type: 'invocation',
  })),
});

interface Deferred<T> {
  promise: Promise<T>;
  reject: (error: unknown) => void;
  resolve: (value: T) => void;
}

const createDeferred = <T>(): Deferred<T> => {
  let reject!: (error: unknown) => void;
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    reject = rejectPromise;
    resolve = resolvePromise;
  });

  return { promise, reject, resolve };
};

/** Lets every queued microtask (and the coalesced refresh) run before asserting. */
const flushAsyncWork = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

let account: typeof accountLifecycleModule;
let browse: typeof libraryBrowseStoreModule;
let invalidate: () => void;

beforeEach(async () => {
  vi.resetModules();
  api.getAllWorkflowTags.mockReset().mockResolvedValue([]);
  api.getWorkflowTagCounts.mockReset().mockResolvedValue({});
  api.listLibraryWorkflows.mockReset().mockResolvedValue(buildPage([]));
  libraryCache.getLibraryWorkflowCached.mockReset().mockResolvedValue(buildRawWorkflow(1));
  libraryCache.onWorkflowLibraryCacheInvalidated.mockReset().mockImplementation((listener: () => void) => {
    invalidate = listener;
    return () => undefined;
  });
  templates.getInvocationTemplatesSnapshot
    .mockReset()
    .mockReturnValue({ error: null, status: 'loaded', templates: { main_model_loader: MODEL_LOADER_TEMPLATE } });
  templates.refreshInvocationTemplates.mockReset().mockResolvedValue(undefined);

  account = await import('@platform/state/accountLifecycle');
  browse = await import('./libraryBrowseStore');
  account.accountLifecycle.activate('user-a');
});

describe('workflow library page fetching', () => {
  it('sends the active category, tag, and search filter to the list endpoint', async () => {
    await browse.ensureWorkflowLibraryBrowseLoaded();

    expect(api.listLibraryWorkflows).toHaveBeenCalledWith(
      expect.objectContaining({ category: 'user', page: 0, perPage: 20, query: undefined, tags: undefined })
    );

    api.listLibraryWorkflows.mockClear();
    browse.setWorkflowLibraryBrowseFilter({ category: 'default', search: 'upscale', tag: 'lora' });
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().status).toBe('loaded');
    });

    expect(api.listLibraryWorkflows).toHaveBeenCalledWith(
      expect.objectContaining({ category: 'default', page: 0, perPage: 20, query: 'upscale', tags: ['lora'] })
    );
  });

  it('ignores a repeated filter patch that changes nothing', async () => {
    await browse.ensureWorkflowLibraryBrowseLoaded();
    api.listLibraryWorkflows.mockClear();

    browse.setWorkflowLibraryBrowseFilter({ category: 'user', search: '' });

    expect(api.listLibraryWorkflows).not.toHaveBeenCalled();
  });

  it('appends the next page and stops once the last page is loaded', async () => {
    api.listLibraryWorkflows.mockImplementation(({ page }: { page: number }) =>
      Promise.resolve(
        buildPage([buildItem(`page-${page}-a`), buildItem(`page-${page}-b`)], { page, pages: 2, total: 4 })
      )
    );

    await browse.ensureWorkflowLibraryBrowseLoaded();
    expect(browse.getWorkflowLibraryBrowseSnapshot().entries).toHaveLength(2);

    browse.loadNextWorkflowLibraryPage();
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().status).toBe('loaded');
    });

    const snapshot = browse.getWorkflowLibraryBrowseSnapshot();
    expect(snapshot.entries.map((entry) => entry.item.workflow_id)).toEqual([
      'page-0-a',
      'page-0-b',
      'page-1-a',
      'page-1-b',
    ]);
    expect(snapshot.page).toBe(1);

    api.listLibraryWorkflows.mockClear();
    browse.loadNextWorkflowLibraryPage();

    expect(api.listLibraryWorkflows).not.toHaveBeenCalled();
  });

  it('clears the accumulated pages as soon as the filter changes', async () => {
    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('a'), buildItem('b')], { pages: 3, total: 6 }));

    await browse.ensureWorkflowLibraryBrowseLoaded();
    browse.setWorkflowLibraryBrowseFilter({ tag: 'lora' });

    const snapshot = browse.getWorkflowLibraryBrowseSnapshot();
    expect(snapshot.entries).toEqual([]);
    expect(snapshot).toMatchObject({ page: 0, pages: 0, status: 'loading', total: 0 });
  });

  it('drops a stale in-flight page when the filter changes mid-flight', async () => {
    await browse.ensureWorkflowLibraryBrowseLoaded();

    const stale = createDeferred<WorkflowLibraryPage>();
    api.listLibraryWorkflows.mockReturnValueOnce(stale.promise);
    browse.setWorkflowLibraryBrowseFilter({ search: 'stale' });

    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('fresh')]));
    browse.setWorkflowLibraryBrowseFilter({ search: 'fresh' });
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().status).toBe('loaded');
    });

    stale.resolve(buildPage([buildItem('stale')]));
    await flushAsyncWork();

    const snapshot = browse.getWorkflowLibraryBrowseSnapshot();
    expect(snapshot.entries.map((entry) => entry.item.workflow_id)).toEqual(['fresh']);
    expect(snapshot.filter.search).toBe('fresh');
    expect(snapshot.status).toBe('loaded');
  });

  it('records the error message when a page request fails', async () => {
    api.listLibraryWorkflows.mockRejectedValue(new Error('backend down'));

    await browse.ensureWorkflowLibraryBrowseLoaded();

    const snapshot = browse.getWorkflowLibraryBrowseSnapshot();
    expect(snapshot.status).toBe('error');
    expect(snapshot.error).toContain('backend down');
  });

  it('parses each item tag string once into the entry', async () => {
    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('a', { tags: ' lora , , sdxl ' })]));

    await browse.ensureWorkflowLibraryBrowseLoaded();

    expect(browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.tags).toEqual(['lora', 'sdxl']);
  });
});

describe('workflow library tag counts', () => {
  it('stores sorted, non-zero counts for the active category', async () => {
    api.getAllWorkflowTags.mockResolvedValue(['lora', 'sdxl', 'unused']);
    api.getWorkflowTagCounts.mockResolvedValue({ lora: 2, sdxl: 5, unused: 0 });

    await browse.ensureWorkflowLibraryBrowseLoaded();

    expect(api.getAllWorkflowTags).toHaveBeenCalledWith(expect.objectContaining({ categories: ['user'] }));
    expect(api.getWorkflowTagCounts).toHaveBeenCalledWith(
      expect.objectContaining({ categories: ['user'], tags: ['lora', 'sdxl', 'unused'] })
    );
    expect(browse.getWorkflowLibraryBrowseSnapshot().tagCounts).toEqual([
      { count: 5, tag: 'sdxl' },
      { count: 2, tag: 'lora' },
    ]);
  });

  it('refetches counts for the new category and not for a search-only change', async () => {
    await browse.ensureWorkflowLibraryBrowseLoaded();
    api.getAllWorkflowTags.mockClear();

    browse.setWorkflowLibraryBrowseFilter({ search: 'cat' });
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().filter.search).toBe('cat');
    });
    expect(api.getAllWorkflowTags).not.toHaveBeenCalled();

    browse.setWorkflowLibraryBrowseFilter({ category: 'default' });
    await vi.waitFor(() => {
      expect(api.getAllWorkflowTags).toHaveBeenCalledWith(expect.objectContaining({ categories: ['default'] }));
    });
  });
});

describe('workflow library entry enrichment', () => {
  it('fills node count and model requirements for loaded entries', async () => {
    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('a')]));
    libraryCache.getLibraryWorkflowCached.mockResolvedValue(buildRawWorkflow(2));

    await browse.ensureWorkflowLibraryBrowseLoaded();
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.enrichment.status).toBe('ready');
    });

    const enrichment = browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.enrichment;
    expect(enrichment).toMatchObject({ nodeCount: 2, status: 'ready' });
    expect(enrichment?.status === 'ready' && enrichment.requirements.primaryBase).toBe('sdxl');
    expect(enrichment?.status === 'ready' && enrichment.requirements.requirements).toHaveLength(2);
    expect(enrichment?.status === 'ready' && enrichment.document.nodes).toHaveLength(2);
  });

  it('marks a single failed entry as an error without stalling the rest of the pool', async () => {
    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('bad'), buildItem('good')]));
    libraryCache.getLibraryWorkflowCached.mockImplementation((workflowId: string) =>
      workflowId === 'bad' ? Promise.reject(new Error('payload gone')) : Promise.resolve(buildRawWorkflow(1))
    );

    await browse.ensureWorkflowLibraryBrowseLoaded();
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries[1]?.enrichment.status).toBe('ready');
    });

    const failed = browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.enrichment;
    expect(failed?.status).toBe('error');
    expect(failed?.status === 'error' && failed.message).toContain('payload gone');
  });

  it('marks unparseable workflow payloads as errors', async () => {
    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('a')]));
    libraryCache.getLibraryWorkflowCached.mockResolvedValue('not a workflow');

    await browse.ensureWorkflowLibraryBrowseLoaded();
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.enrichment.status).toBe('error');
    });
  });

  it('enriches at most four entries concurrently', async () => {
    const deferrals = new Map<string, Deferred<Record<string, unknown>>>();
    api.listLibraryWorkflows.mockResolvedValue(
      buildPage(Array.from({ length: 6 }, (_unused, index) => buildItem(`item-${index}`)))
    );
    libraryCache.getLibraryWorkflowCached.mockImplementation((workflowId: string) => {
      const deferred = createDeferred<Record<string, unknown>>();

      deferrals.set(workflowId, deferred);

      return deferred.promise;
    });

    await browse.ensureWorkflowLibraryBrowseLoaded();
    await vi.waitFor(() => {
      expect(deferrals.size).toBe(4);
    });

    // The pool must stay saturated at four, never fanning out to all six.
    await flushAsyncWork();
    expect(deferrals.size).toBe(4);

    // Resolutions continue in a microtask, so the map is not mutated mid-loop.
    for (const deferred of deferrals.values()) {
      deferred.resolve(buildRawWorkflow(1));
    }

    await vi.waitFor(() => {
      expect(deferrals.size).toBe(6);
    });
  });

  it('refetches node definitions at most once for a page when the template load fails', async () => {
    templates.getInvocationTemplatesSnapshot.mockReturnValue({
      error: 'Node definitions unavailable.',
      status: 'error',
      templates: {},
    });
    api.listLibraryWorkflows.mockResolvedValue(
      buildPage(Array.from({ length: 6 }, (_unused, index) => buildItem(`item-${index}`)))
    );

    await browse.ensureWorkflowLibraryBrowseLoaded();
    await vi.waitFor(() => {
      expect(
        browse.getWorkflowLibraryBrowseSnapshot().entries.every((entry) => entry.enrichment.status === 'error')
      ).toBe(true);
    });

    // One attempt for the whole batch — never one OpenAPI reparse per worker.
    expect(templates.refreshInvocationTemplates).toHaveBeenCalledTimes(1);
    expect(libraryCache.getLibraryWorkflowCached).not.toHaveBeenCalled();
  });

  it('retries entries that failed to enrich on the next refresh', async () => {
    templates.getInvocationTemplatesSnapshot.mockReturnValue({
      error: 'Node definitions unavailable.',
      status: 'error',
      templates: {},
    });
    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('a')]));

    await browse.ensureWorkflowLibraryBrowseLoaded();
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.enrichment.status).toBe('error');
    });

    templates.getInvocationTemplatesSnapshot.mockReturnValue({
      error: null,
      status: 'loaded',
      templates: { main_model_loader: MODEL_LOADER_TEMPLATE },
    });

    await browse.refreshWorkflowLibraryBrowse();
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.enrichment.status).toBe('ready');
    });
  });

  it('drops enrichment results for entries that left the snapshot', async () => {
    const stalled = createDeferred<Record<string, unknown>>();
    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('gone')]));
    libraryCache.getLibraryWorkflowCached.mockReturnValueOnce(stalled.promise);

    await browse.ensureWorkflowLibraryBrowseLoaded();

    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('kept')]));
    libraryCache.getLibraryWorkflowCached.mockResolvedValue(buildRawWorkflow(3));
    browse.setWorkflowLibraryBrowseFilter({ search: 'kept' });
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.enrichment.status).toBe('ready');
    });

    stalled.resolve(buildRawWorkflow(9));
    await flushAsyncWork();

    const entries = browse.getWorkflowLibraryBrowseSnapshot().entries;
    expect(entries).toHaveLength(1);
    expect(entries[0]?.item.workflow_id).toBe('kept');
    expect(entries[0]?.enrichment).toMatchObject({ nodeCount: 3 });
  });

  it('keeps untouched entry identities stable while enrichment lands', async () => {
    const slow = createDeferred<Record<string, unknown>>();
    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('fast'), buildItem('slow')]));
    libraryCache.getLibraryWorkflowCached.mockImplementation((workflowId: string) =>
      workflowId === 'slow' ? slow.promise : Promise.resolve(buildRawWorkflow(1))
    );

    await browse.ensureWorkflowLibraryBrowseLoaded();
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.enrichment.status).toBe('ready');
    });

    const enrichedFirst = browse.getWorkflowLibraryBrowseSnapshot().entries[0];

    slow.resolve(buildRawWorkflow(1));
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries[1]?.enrichment.status).toBe('ready');
    });

    expect(browse.getWorkflowLibraryBrowseSnapshot().entries[0]).toBe(enrichedFirst);
  });
});

describe('workflow library browse refresh', () => {
  it('coalesces a burst of cache invalidations into one refresh', async () => {
    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('a')]));

    await browse.ensureWorkflowLibraryBrowseLoaded();
    api.listLibraryWorkflows.mockClear();

    invalidate();
    invalidate();
    invalidate();

    await vi.waitFor(() => {
      expect(api.listLibraryWorkflows).toHaveBeenCalledTimes(1);
    });
    await flushAsyncWork();
    expect(api.listLibraryWorkflows).toHaveBeenCalledTimes(1);
  });

  it('does not refresh while the store is idle', async () => {
    invalidate();

    await flushAsyncWork();
    expect(api.listLibraryWorkflows).not.toHaveBeenCalled();
    expect(browse.getWorkflowLibraryBrowseSnapshot().status).toBe('idle');
  });

  it('publishes the refreshed list when an infinite-scroll append races the refresh', async () => {
    const refreshedFirstPage = createDeferred<WorkflowLibraryPage>();
    let firstPageCalls = 0;

    api.listLibraryWorkflows.mockImplementation(({ page, perPage }: { page: number; perPage: number }) => {
      if (perPage === 1) {
        return Promise.resolve(buildPage([], { total: 0 }));
      }

      if (page > 0) {
        return Promise.resolve(buildPage([buildItem('scrolled')], { page, pages: 3, total: 4 }));
      }

      firstPageCalls += 1;

      return firstPageCalls === 1
        ? Promise.resolve(buildPage([buildItem('kept'), buildItem('deleted')], { pages: 3, total: 5 }))
        : refreshedFirstPage.promise;
    });

    await browse.ensureWorkflowLibraryBrowseLoaded();

    // A delete invalidated the cache; the user scrolls while the refresh is in flight.
    const refreshing = browse.refreshWorkflowLibraryBrowse();

    browse.loadNextWorkflowLibraryPage();
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries).toHaveLength(3);
    });

    refreshedFirstPage.resolve(buildPage([buildItem('kept')], { pages: 2, total: 3 }));
    await refreshing;

    const snapshot = browse.getWorkflowLibraryBrowseSnapshot();
    expect(snapshot.entries.map((entry) => entry.item.workflow_id)).toEqual(['kept']);
    expect(snapshot.status).toBe('loaded');
  });

  it('refetches every loaded page and keeps unchanged entry identities', async () => {
    api.listLibraryWorkflows.mockImplementation(({ page }: { page: number }) =>
      Promise.resolve(buildPage([buildItem(`page-${page}`)], { page, pages: 2, total: 2 }))
    );

    await browse.ensureWorkflowLibraryBrowseLoaded();
    browse.loadNextWorkflowLibraryPage();
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries).toHaveLength(2);
    });

    const before = browse.getWorkflowLibraryBrowseSnapshot().entries;
    api.listLibraryWorkflows.mockClear();

    await browse.refreshWorkflowLibraryBrowse();

    expect(api.listLibraryWorkflows.mock.calls.map(([params]) => params.page)).toEqual([0, 1]);

    const after = browse.getWorkflowLibraryBrowseSnapshot().entries;
    expect(after.map((entry) => entry.item.workflow_id)).toEqual(['page-0', 'page-1']);
    expect(after[0]).toBe(before[0]);
    expect(after[1]).toBe(before[1]);
  });

  it('re-enriches a row whose server record changed', async () => {
    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('a')]));
    libraryCache.getLibraryWorkflowCached.mockResolvedValue(buildRawWorkflow(1));

    await browse.ensureWorkflowLibraryBrowseLoaded();
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.enrichment).toMatchObject({ nodeCount: 1 });
    });

    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('a', { updated_at: '2026-08-19T00:00:00Z' })]));
    libraryCache.getLibraryWorkflowCached.mockResolvedValue(buildRawWorkflow(5));

    await browse.refreshWorkflowLibraryBrowse();
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries[0]?.enrichment).toMatchObject({ nodeCount: 5 });
    });
  });
});

describe('workflow library browse account ownership', () => {
  it('resets to the initial snapshot when the account switches', async () => {
    api.listLibraryWorkflows.mockResolvedValue(buildPage([buildItem('a')], { total: 3 }));

    await browse.ensureWorkflowLibraryBrowseLoaded();
    browse.setWorkflowLibraryBrowseFilter({ category: 'default', search: 'cat', tag: 'lora' });
    await vi.waitFor(() => {
      expect(browse.getWorkflowLibraryBrowseSnapshot().entries).toHaveLength(1);
    });

    account.accountLifecycle.activate('user-b');

    expect(browse.getWorkflowLibraryBrowseSnapshot()).toEqual({
      entries: [],
      error: null,
      filter: { category: 'user', search: '', tag: null },
      page: 0,
      pages: 0,
      status: 'idle',
      tagCounts: [],
      total: 0,
      userTotal: null,
    });
  });
});

describe('workflow library user total probe', () => {
  it('probes the user category with a one-row request', async () => {
    api.listLibraryWorkflows.mockImplementation(({ perPage }: { perPage: number }) =>
      Promise.resolve(perPage === 1 ? buildPage([buildItem('a')], { total: 7 }) : buildPage([]))
    );

    await browse.ensureWorkflowLibraryBrowseLoaded();

    expect(api.listLibraryWorkflows).toHaveBeenCalledWith(
      expect.objectContaining({ category: 'user', page: 0, perPage: 1 })
    );
    expect(browse.getWorkflowLibraryBrowseSnapshot().userTotal).toBe(7);
  });

  it('probes only once across repeated ensure calls', async () => {
    await browse.ensureWorkflowLibraryBrowseLoaded();
    await browse.ensureWorkflowLibraryBrowseLoaded();

    expect(api.listLibraryWorkflows.mock.calls.filter(([params]) => params.perPage === 1)).toHaveLength(1);
  });
});
