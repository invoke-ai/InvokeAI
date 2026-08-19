import type { WorkflowTagCount } from '@features/workflow/core/libraryTags';
import type { WorkflowModelRequirementSet } from '@features/workflow/core/modelRequirements';
import type { InvocationTemplates, ProjectGraphState } from '@features/workflow/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { parseWorkflowTags, sortTagCounts } from '@features/workflow/core/libraryTags';
import { extractWorkflowModelRequirements } from '@features/workflow/core/modelRequirements';
import { parseWorkflowJson } from '@features/workflow/core/workflowJson';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { shallowEqual } from '@platform/state/selectorCore';
import { createTrailingSingleFlight } from '@platform/state/singleFlight';
import { getApiErrorMessage } from '@platform/transport/http';

import type { WorkflowLibraryCategory, WorkflowLibraryListItem, WorkflowLibraryPage } from './api';

import { getAllWorkflowTags, getWorkflowTagCounts, listLibraryWorkflows } from './api';
import { getLibraryWorkflowCached, onWorkflowLibraryCacheInvalidated } from './libraryCache';
import { getInvocationTemplatesSnapshot, refreshInvocationTemplates } from './templates';

/**
 * Browse state for the workflow library dialog. The library is backend-owned
 * and can hold thousands of records, so filtering and pagination are server
 * side: every filter change is one page-0 request, infinite scroll appends the
 * next page, and nothing is filtered locally. What the list endpoint cannot
 * answer — node count and model requirements — is enriched in the background
 * from the cached workflow payloads, per entry, so cards fill in as they load
 * without blocking the grid.
 */

export type { WorkflowTagCount } from '@features/workflow/core/libraryTags';

export interface WorkflowLibraryBrowseFilter {
  category: WorkflowLibraryCategory;
  tag: string | null;
  search: string;
}

export type WorkflowLibraryEntryEnrichment =
  | { status: 'pending' }
  | { status: 'error'; message: string }
  | { status: 'ready'; document: ProjectGraphState; nodeCount: number; requirements: WorkflowModelRequirementSet };

export interface WorkflowLibraryEntry {
  item: WorkflowLibraryListItem;
  /** Parsed once from `item.tags` so cards never re-split the raw string. */
  tags: readonly string[];
  enrichment: WorkflowLibraryEntryEnrichment;
}

export interface WorkflowLibraryBrowseSnapshot {
  filter: WorkflowLibraryBrowseFilter;
  status: 'idle' | 'loading' | 'loadingMore' | 'loaded' | 'error';
  /** Accumulated pages for the current filter, in server order. */
  entries: readonly WorkflowLibraryEntry[];
  /** Last loaded page (0-based, mirroring the API). */
  page: number;
  pages: number;
  total: number;
  /** Chip counts for the current category. */
  tagCounts: readonly WorkflowTagCount[];
  /** Total user-category workflows, for the one-time Browse/Yours auto-switch. */
  userTotal: number | null;
  error: string | null;
}

const PER_PAGE = 20;
const ENRICHMENT_CONCURRENCY = 4;

const PENDING_ENRICHMENT: WorkflowLibraryEntryEnrichment = { status: 'pending' };
const EMPTY_ENTRIES: readonly WorkflowLibraryEntry[] = [];
const EMPTY_TAG_COUNTS: readonly WorkflowTagCount[] = [];
const INITIAL_FILTER: WorkflowLibraryBrowseFilter = { category: 'user', search: '', tag: null };

const INITIAL_SNAPSHOT: WorkflowLibraryBrowseSnapshot = {
  entries: EMPTY_ENTRIES,
  error: null,
  filter: INITIAL_FILTER,
  page: 0,
  pages: 0,
  status: 'idle',
  tagCounts: EMPTY_TAG_COUNTS,
  total: 0,
  userTotal: null,
};

const store = createExternalStore<WorkflowLibraryBrowseSnapshot>(INITIAL_SNAPSHOT);

const initialLoadFlight = createTrailingSingleFlight();
const refreshFlight = createTrailingSingleFlight();

/**
 * Bumped only when the filter (or the account) changes. A response tagged with
 * an older generation was requested for a view the user has moved on from, so
 * it is discarded. Two requests for the *same* filter — a cache-invalidation
 * refresh racing an infinite-scroll append — are both current, and both
 * publish: keying staleness to the request order instead would let the append
 * silently swallow the refresh that removed a deleted row.
 */
let filterGeneration = 0;

const isFilterCurrent = (generation: number, owner: AccountScope): boolean =>
  generation === filterGeneration && isAccountScopeCurrent(owner);

// #region Entries

const toEntry = (item: WorkflowLibraryListItem): WorkflowLibraryEntry => ({
  enrichment: PENDING_ENRICHMENT,
  item,
  tags: parseWorkflowTags(item.tags),
});

/**
 * Rebuilds the entry list from a server response while keeping the identity —
 * and the completed enrichment — of every row the server reports unchanged.
 * Selectors and memoized cards depend on those identities holding across
 * refreshes.
 */
const mergeEntries = (
  previous: readonly WorkflowLibraryEntry[],
  items: readonly WorkflowLibraryListItem[]
): WorkflowLibraryEntry[] => {
  const previousById = new Map(previous.map((entry) => [entry.item.workflow_id, entry]));

  return items.map((item) => {
    const existing = previousById.get(item.workflow_id);

    return existing && shallowEqual(existing.item, item) ? existing : toEntry(item);
  });
};

/** Re-arms rows whose enrichment failed so an explicit revalidation retries them. */
const retryFailedEnrichment = (entries: readonly WorkflowLibraryEntry[]): readonly WorkflowLibraryEntry[] =>
  entries.some((entry) => entry.enrichment.status === 'error')
    ? entries.map((entry) =>
        entry.enrichment.status === 'error' ? { ...entry, enrichment: PENDING_ENRICHMENT } : entry
      )
    : entries;

const publishPage = (result: WorkflowLibraryPage, mode: 'append' | 'replace'): void => {
  const previous = store.getSnapshot().entries;
  const merged = mergeEntries(previous, result.items);

  store.patchSnapshot({
    entries: mode === 'append' ? [...previous, ...merged] : merged,
    error: null,
    page: result.page,
    pages: result.pages,
    status: 'loaded',
    total: result.total,
  });

  pumpEnrichment();
};

// #endregion

// #region Enrichment

/**
 * Templates are needed to know which inputs are model fields. They are a
 * session-lived, shared load, so enrichment waits for one shared attempt
 * instead of parsing the schema per entry. A failed attempt is remembered:
 * `refreshInvocationTemplates` refetches and reparses the whole OpenAPI
 * document, so retrying it per entry would turn one outage into a fetch storm.
 * The memo is re-armed by an explicit refresh, a filter change, or an account
 * switch.
 */
let templatesFlight: Promise<void> | null = null;
let hasTemplateLoadFailed = false;

const loadTemplates = async (): Promise<InvocationTemplates> => {
  const snapshot = getInvocationTemplatesSnapshot();

  if (snapshot.status === 'loaded') {
    return snapshot.templates;
  }

  if (hasTemplateLoadFailed) {
    throw new Error('Node definitions are unavailable.');
  }

  templatesFlight ??= refreshInvocationTemplates().finally(() => {
    templatesFlight = null;
  });
  await templatesFlight;

  const settled = getInvocationTemplatesSnapshot();

  if (settled.status !== 'loaded') {
    hasTemplateLoadFailed = true;

    throw new Error(settled.error ?? 'Node definitions are unavailable.');
  }

  return settled.templates;
};

const enrichmentQueue: string[] = [];
const queuedWorkflowIds = new Set<string>();
let activeEnrichmentWorkers = 0;

const findEntryIndex = (workflowId: string): number =>
  store.getSnapshot().entries.findIndex((entry) => entry.item.workflow_id === workflowId);

/** Publishes one entry's enrichment, reusing every other entry object as-is. */
const applyEnrichment = (workflowId: string, enrichment: WorkflowLibraryEntryEnrichment, owner: AccountScope): void => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  const { entries } = store.getSnapshot();
  const index = entries.findIndex((entry) => entry.item.workflow_id === workflowId);
  const existing = entries[index];

  // The row was filtered or paged away while its payload was in flight.
  if (!existing) {
    return;
  }

  const next = entries.slice();

  next[index] = { ...existing, enrichment };
  store.patchSnapshot({ entries: next });
};

const enrichEntry = async (workflowId: string, owner: AccountScope): Promise<void> => {
  try {
    const templates = await loadTemplates();
    const raw = await getLibraryWorkflowCached(workflowId, owner.signal);
    const { document } = parseWorkflowJson(raw);

    applyEnrichment(
      workflowId,
      {
        document,
        nodeCount: document.nodes.length,
        requirements: extractWorkflowModelRequirements(document, templates),
        status: 'ready',
      },
      owner
    );
  } catch (error) {
    // One unreadable workflow marks its own card and never fails the pool.
    applyEnrichment(
      workflowId,
      { message: getApiErrorMessage(error, 'Failed to read this workflow.'), status: 'error' },
      owner
    );
  }
};

const runEnrichmentWorker = async (): Promise<void> => {
  try {
    for (let workflowId = enrichmentQueue.shift(); workflowId !== undefined; workflowId = enrichmentQueue.shift()) {
      queuedWorkflowIds.delete(workflowId);

      // Each item captures the scope it starts under, so a worker that outlives
      // an account switch drops its result instead of writing it.
      if (findEntryIndex(workflowId) !== -1) {
        await enrichEntry(workflowId, captureAccountScope());
      }
    }
  } finally {
    activeEnrichmentWorkers -= 1;
  }
};

/** Queues every still-pending entry and tops the worker pool back up. */
const pumpEnrichment = (): void => {
  for (const entry of store.getSnapshot().entries) {
    const workflowId = entry.item.workflow_id;

    if (entry.enrichment.status === 'pending' && !queuedWorkflowIds.has(workflowId)) {
      queuedWorkflowIds.add(workflowId);
      enrichmentQueue.push(workflowId);
    }
  }

  while (activeEnrichmentWorkers < ENRICHMENT_CONCURRENCY && enrichmentQueue.length > 0) {
    activeEnrichmentWorkers += 1;
    void runEnrichmentWorker();
  }
};

// #endregion

// #region Fetching

const fetchPage = (
  filter: WorkflowLibraryBrowseFilter,
  page: number,
  owner: AccountScope
): Promise<WorkflowLibraryPage> =>
  listLibraryWorkflows({
    category: filter.category,
    page,
    perPage: PER_PAGE,
    query: filter.search || undefined,
    signal: owner.signal,
    tags: filter.tag ? [filter.tag] : undefined,
  });

const loadFirstPage = async (filter: WorkflowLibraryBrowseFilter, owner: AccountScope): Promise<void> => {
  const generation = filterGeneration;

  try {
    const result = await fetchPage(filter, 0, owner);

    if (isFilterCurrent(generation, owner)) {
      publishPage(result, 'replace');
    }
  } catch (error) {
    if (isFilterCurrent(generation, owner)) {
      store.patchSnapshot({ error: getApiErrorMessage(error, 'Failed to load workflows.'), status: 'error' });
    }
  }
};

const loadMorePages = async (filter: WorkflowLibraryBrowseFilter, page: number, owner: AccountScope): Promise<void> => {
  const generation = filterGeneration;

  try {
    const result = await fetchPage(filter, page, owner);

    if (isFilterCurrent(generation, owner)) {
      publishPage(result, 'append');
    }
  } catch (error) {
    if (isFilterCurrent(generation, owner)) {
      store.patchSnapshot({ error: getApiErrorMessage(error, 'Failed to load more workflows.'), status: 'error' });
    }
  }
};

/** Tag chips are best-effort: a failed count leaves the previous chips in place. */
const loadTagCounts = async (category: WorkflowLibraryCategory, owner: AccountScope): Promise<void> => {
  try {
    const tags = await getAllWorkflowTags({ categories: [category], signal: owner.signal });
    const counts =
      tags.length > 0 ? await getWorkflowTagCounts({ categories: [category], signal: owner.signal, tags }) : {};

    // A category switch mid-flight owns the chips now.
    if (!isAccountScopeCurrent(owner) || store.getSnapshot().filter.category !== category) {
      return;
    }

    const tagCounts = sortTagCounts(Object.entries(counts).map(([tag, count]) => ({ count, tag })));

    store.patchSnapshot({ tagCounts: tagCounts.length > 0 ? tagCounts : EMPTY_TAG_COUNTS });
  } catch {
    // Chips are decoration around the grid; the list request owns the error state.
  }
};

/**
 * One-row probe of the user category. Task 6 auto-switches to the bundled
 * defaults on a fresh install, and this answers "does this account have any
 * workflows?" without loading the user list.
 */
const probeUserTotal = async (owner: AccountScope): Promise<void> => {
  if (store.getSnapshot().userTotal !== null) {
    return;
  }

  try {
    const result = await listLibraryWorkflows({ category: 'user', page: 0, perPage: 1, signal: owner.signal });

    if (isAccountScopeCurrent(owner)) {
      store.patchSnapshot({ userTotal: result.total });
    }
  } catch {
    // Leaves `userTotal` null: the dialog simply keeps the category it opened on.
  }
};

// #endregion

// #region Public API

/** Applies a filter patch, resets the accumulated pages, and refetches page 0. */
export const setWorkflowLibraryBrowseFilter = (patch: Partial<WorkflowLibraryBrowseFilter>): void => {
  const snapshot = store.getSnapshot();
  const filter = { ...snapshot.filter, ...patch };

  if (shallowEqual(filter, snapshot.filter)) {
    return;
  }

  const owner = captureAccountScope();
  const isCategoryChanged = filter.category !== snapshot.filter.category;

  // Everything already in flight was requested for the previous filter.
  filterGeneration += 1;
  hasTemplateLoadFailed = false;

  store.patchSnapshot({
    entries: EMPTY_ENTRIES,
    error: null,
    filter,
    page: 0,
    pages: 0,
    status: 'loading',
    tagCounts: isCategoryChanged ? EMPTY_TAG_COUNTS : snapshot.tagCounts,
    total: 0,
  });

  void loadFirstPage(filter, owner);

  if (isCategoryChanged) {
    void loadTagCounts(filter.category, owner);
  }
};

/** Appends the next page; a no-op while a page is in flight or on the last page. */
export const loadNextWorkflowLibraryPage = (): void => {
  const { filter, page, pages, status } = store.getSnapshot();

  if (status !== 'loaded' || page + 1 >= pages) {
    return;
  }

  store.patchSnapshot({ status: 'loadingMore' });
  void loadMorePages(filter, page + 1, captureAccountScope());
};

/** First open: page 0, the category's tag counts, and the user-total probe. */
export const ensureWorkflowLibraryBrowseLoaded = (): Promise<void> => {
  const { status } = store.getSnapshot();

  if (status !== 'idle' && status !== 'error') {
    return initialLoadFlight.inflight() ?? Promise.resolve();
  }

  return initialLoadFlight.run(async () => {
    const owner = captureAccountScope();
    const { filter } = store.getSnapshot();

    hasTemplateLoadFailed = false;
    store.patchSnapshot({ error: null, status: 'loading' });

    await Promise.all([loadFirstPage(filter, owner), loadTagCounts(filter.category, owner), probeUserTotal(owner)]);
  });
};

/** Revalidates every page the user has scrolled through, plus the tag counts. */
export const refreshWorkflowLibraryBrowse = (): Promise<void> =>
  refreshFlight.run(async () => {
    const { filter, page, status } = store.getSnapshot();

    if (status === 'idle') {
      return;
    }

    const owner = captureAccountScope();
    const generation = filterGeneration;

    // An explicit revalidation is also the retry path for a failed enrichment.
    hasTemplateLoadFailed = false;

    try {
      const results = await Promise.all(
        Array.from({ length: page + 1 }, (_unused, index) => fetchPage(filter, index, owner))
      );
      const last = results[results.length - 1];

      if (last && isFilterCurrent(generation, owner)) {
        const items = results.flatMap((result) => result.items);

        store.patchSnapshot({
          entries: mergeEntries(retryFailedEnrichment(store.getSnapshot().entries), items),
          error: null,
          // Deletions can shrink the library below the page the user was on.
          page: Math.min(page, Math.max(0, last.pages - 1)),
          pages: last.pages,
          status: 'loaded',
          total: last.total,
        });
        pumpEnrichment();
      }
    } catch (error) {
      if (isFilterCurrent(generation, owner)) {
        store.patchSnapshot({ error: getApiErrorMessage(error, 'Failed to refresh workflows.'), status: 'error' });
      }
    }

    await loadTagCounts(filter.category, owner);
  });

export const getWorkflowLibraryBrowseSnapshot = (): WorkflowLibraryBrowseSnapshot => store.getSnapshot();

export const useWorkflowLibraryBrowseSelector = store.useSelector;

// #endregion

/**
 * Any local mutation (save, delete, thumbnail) invalidates the library cache
 * and shifts ordering, so the visible pages are refetched. A burst of
 * invalidations from one action collapses into a single refresh.
 */
let isRefreshScheduled = false;

onWorkflowLibraryCacheInvalidated(() => {
  if (isRefreshScheduled || store.getSnapshot().status === 'idle') {
    return;
  }

  isRefreshScheduled = true;
  queueMicrotask(() => {
    isRefreshScheduled = false;

    // The account may have reset between the invalidation and this microtask.
    if (store.getSnapshot().status !== 'idle') {
      void refreshWorkflowLibraryBrowse();
    }
  });
});

registerAccountOwnedResource({
  clear: () => {
    filterGeneration += 1;
    enrichmentQueue.length = 0;
    queuedWorkflowIds.clear();
    templatesFlight = null;
    hasTemplateLoadFailed = false;
    isRefreshScheduled = false;
    initialLoadFlight.reset();
    refreshFlight.reset();
    store.setSnapshot(INITIAL_SNAPSHOT);
  },
  name: 'workflow-library-browse',
});
