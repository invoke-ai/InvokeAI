import type { WorkflowTagCount } from '@features/workflow/core/libraryTags';
import type { WorkflowModelRequirementSet } from '@features/workflow/core/modelRequirements';
import type { InvocationTemplates, ProjectGraphState } from '@features/workflow/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { parseWorkflowTags, sortTagCounts } from '@features/workflow/core/libraryTags';
import { extractWorkflowModelRequirements } from '@features/workflow/core/modelRequirements';
import { parseWorkflowJson } from '@features/workflow/core/workflowJson';
import { createLogger } from '@platform/logging/logger';
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

const libraryLogger = createLogger({ area: 'library', namespace: 'workflows' });

/**
 * Filter and paginate server-side; asynchronously enrich cached payloads per row so node/model details cannot
 * block the grid.
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

// #region View cache

/**
 * Views already browsed (Browse, Yours, a tag or search), kept so switching back shows them at once with their
 * enrichment instead of refetching and re-reading every row. A view older than `VIEW_FRESH_MS` is revalidated in the
 * background and merged in place; library mutations drop the other views.
 */
interface CachedBrowseView {
  entries: readonly WorkflowLibraryEntry[];
  fetchedAt: number;
  page: number;
  pages: number;
  tagCounts: readonly WorkflowTagCount[];
  total: number;
}

const VIEW_CACHE_LIMIT = 8;
const VIEW_FRESH_MS = 30_000;
const viewCache = new Map<string, CachedBrowseView>();
/** When the current view's pages last arrived from the server. */
let currentViewFetchedAt = 0;

const getFilterKey = ({ category, search, tag }: WorkflowLibraryBrowseFilter): string =>
  JSON.stringify([category, search, tag]);

const rememberCurrentView = (): void => {
  const { entries, filter, page, pages, status, tagCounts, total } = store.getSnapshot();

  // An unfetched view is one a library change has outdated; leaving before its refresh lands must not keep it.
  if (status !== 'loaded' || currentViewFetchedAt === 0) {
    return;
  }

  const key = getFilterKey(filter);

  viewCache.delete(key);
  viewCache.set(key, { entries, fetchedAt: currentViewFetchedAt, page, pages, tagCounts, total });

  // Maps iterate in insertion order, so the first key is the least recently left view.
  while (viewCache.size > VIEW_CACHE_LIMIT) {
    const oldest = viewCache.keys().next().value;

    if (oldest === undefined) {
      break;
    }
    viewCache.delete(oldest);
  }
};

// #endregion

const initialLoadFlight = createTrailingSingleFlight();
const refreshFlight = createTrailingSingleFlight();

/** Fence responses by filter/account generation, not request order; same-view append and refresh must both publish. */
let filterGeneration = 0;

const isFilterCurrent = (generation: number, owner: AccountScope): boolean =>
  generation === filterGeneration && isAccountScopeCurrent(owner);

// #region Entries

const toEntry = (item: WorkflowLibraryListItem): WorkflowLibraryEntry => ({
  enrichment: PENDING_ENRICHMENT,
  item,
  tags: parseWorkflowTags(item.tags),
});

/** Preserve unchanged row identity and enrichment across refreshes for selectors and memoized cards. */
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

  if (mode === 'replace') {
    currentViewFetchedAt = Date.now();
  }

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

/**
 * Share one template-load attempt for enrichment and remember failure to prevent fetch storms; explicit
 * refresh/filter/account changes rearm it.
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
    libraryLogger.warn({
      context: { workflowId },
      error,
      message: 'Failed to read a library workflow',
      name: 'workflows.library-read-failed',
    });
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
      libraryLogger.warn({
        error,
        message: 'Failed to load workflows',
        name: 'workflows.library-load-failed',
      });
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
      libraryLogger.warn({
        error,
        message: 'Failed to load more workflows',
        name: 'workflows.library-load-failed',
      });
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

/** Probe one user workflow to decide whether fresh accounts should open bundled defaults. */
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

/**
 * Applies a filter patch. A view browsed before comes back from the cache at once (revalidated in the background once
 * stale); a new one resets the accumulated pages and fetches page 0.
 */
export const setWorkflowLibraryBrowseFilter = (patch: Partial<WorkflowLibraryBrowseFilter>): void => {
  const snapshot = store.getSnapshot();
  const filter = { ...snapshot.filter, ...patch };

  if (shallowEqual(filter, snapshot.filter)) {
    return;
  }

  const owner = captureAccountScope();
  const isCategoryChanged = filter.category !== snapshot.filter.category;

  rememberCurrentView();

  // Everything already in flight was requested for the previous filter.
  filterGeneration += 1;
  hasTemplateLoadFailed = false;

  const cached = viewCache.get(getFilterKey(filter));

  if (cached) {
    currentViewFetchedAt = cached.fetchedAt;
    store.patchSnapshot({
      entries: cached.entries,
      error: null,
      filter,
      page: cached.page,
      pages: cached.pages,
      status: 'loaded',
      tagCounts: cached.tagCounts,
      total: cached.total,
    });
    pumpEnrichment();

    if (Date.now() - cached.fetchedAt > VIEW_FRESH_MS) {
      void refreshWorkflowLibraryBrowse();
    }
    return;
  }

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

        currentViewFetchedAt = Date.now();
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
        libraryLogger.warn({
          error,
          message: 'Failed to refresh workflows',
          name: 'workflows.library-load-failed',
        });
        store.patchSnapshot({ error: getApiErrorMessage(error, 'Failed to refresh workflows.'), status: 'error' });
      }
    }

    await loadTagCounts(filter.category, owner);
  });

export const getWorkflowLibraryBrowseSnapshot = (): WorkflowLibraryBrowseSnapshot => store.getSnapshot();

export const useWorkflowLibraryBrowseSelector = store.useSelector;

// #endregion

/** Coalesce mutation invalidations into one visible-page refresh because local changes can shift ordering. */
let isRefreshScheduled = false;

onWorkflowLibraryCacheInvalidated(() => {
  // A mutation can change any view; the current one refreshes below, the others fetch afresh when revisited.
  viewCache.clear();
  currentViewFetchedAt = 0;

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
    viewCache.clear();
    currentViewFetchedAt = 0;
    initialLoadFlight.reset();
    refreshFlight.reset();
    store.setSnapshot(INITIAL_SNAPSHOT);
  },
  name: 'workflow-library-browse',
});
