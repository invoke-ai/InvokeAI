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

import { getAllWorkflowTags, getWorkflowTagCounts, listLibraryWorkflows, onLibraryWorkflowOpened } from './api';
import {
  getLibraryWorkflowCached,
  onWorkflowLibraryCacheInvalidated,
  WorkflowLibraryChangedDuringReadError,
} from './libraryCache';
import { getInvocationTemplatesSnapshot, refreshInvocationTemplates } from './templates';

const libraryLogger = createLogger({ area: 'library', namespace: 'workflows' });

/**
 * Filter and paginate server-side; asynchronously enrich cached payloads per row so node/model details cannot
 * block the grid.
 */

export type { WorkflowTagCount } from '@features/workflow/core/libraryTags';

/** What the library's own templates are ordered by; bundled templates always list by name. */
export type WorkflowLibrarySort = 'updated' | 'opened' | 'created' | 'name';

export interface WorkflowLibraryBrowseFilter {
  category: WorkflowLibraryCategory;
  /** Public user workflows form a distinct view; absent keeps user listings owner scoped. */
  isPublic?: boolean;
  tag: string | null;
  search: string;
  /** Applies to the user category only; see `getListOrder`. */
  sort: WorkflowLibrarySort;
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
  /** Chip counts for the current category and visibility filter. */
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
const INITIAL_FILTER: WorkflowLibraryBrowseFilter = { category: 'user', search: '', sort: 'updated', tag: null };

const ORDER_BY: Record<WorkflowLibrarySort, 'created_at' | 'name' | 'opened_at' | 'updated_at'> = {
  created: 'created_at',
  name: 'name',
  opened: 'opened_at',
  updated: 'updated_at',
};

/**
 * The list order a filter asks the server for. Bundled templates always list by name: the server rewrites their
 * timestamps at every start, so their dates order nothing. Dates list newest first; names A to Z.
 */
const getListOrder = ({
  category,
  sort,
}: WorkflowLibraryBrowseFilter): { direction: 'ASC' | 'DESC'; orderBy: (typeof ORDER_BY)[WorkflowLibrarySort] } => {
  const effective = category === 'default' ? 'name' : sort;

  return { direction: effective === 'name' ? 'ASC' : 'DESC', orderBy: ORDER_BY[effective] };
};

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
 * Views already browsed (Browse, Yours, Shared, a tag or search), kept so switching back shows them at once with their
 * enrichment instead of refetching and re-reading every row. A view older than `VIEW_FRESH_MS` is revalidated in the
 * background and merged in place; library mutations drop the other views.
 */
interface CachedBrowseView {
  entries: readonly WorkflowLibraryEntry[];
  fetchedAt: number;
  orderBy: ReturnType<typeof getListOrder>['orderBy'];
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

const getFilterKey = (filter: WorkflowLibraryBrowseFilter): string =>
  JSON.stringify([
    filter.category,
    filter.isPublic,
    filter.search,
    filter.tag,
    getListOrder(filter).orderBy,
    getListOrder(filter).direction,
  ]);

const rememberCurrentView = (): void => {
  const { entries, filter, page, pages, status, tagCounts, total } = store.getSnapshot();

  // An unfetched view is one a library change has outdated; leaving before its refresh lands must not keep it.
  if (status !== 'loaded' || currentViewFetchedAt === 0) {
    return;
  }

  const key = getFilterKey(filter);

  viewCache.delete(key);
  viewCache.set(key, {
    entries,
    fetchedAt: currentViewFetchedAt,
    orderBy: getListOrder(filter).orderBy,
    page,
    pages,
    tagCounts,
    total,
  });

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

/**
 * Fences page responses by generation, not request order. A filter change, an account change, or a revalidation of
 * the current view starts a new generation, so a page requested before it cannot publish over what it read.
 */
let listGeneration = 0;
/** The generation of the revalidation reading the current view's pages; infinite scroll waits for it. */
let revalidatingGeneration: number | null = null;

const isListCurrent = (generation: number, owner: AccountScope): boolean =>
  generation === listGeneration && isAccountScopeCurrent(owner);

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

const findEntry = (workflowId: string): WorkflowLibraryEntry | undefined =>
  store.getSnapshot().entries.find((entry) => entry.item.workflow_id === workflowId);

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

const enrichEntry = async (
  { revision, workflow_id: workflowId }: WorkflowLibraryListItem,
  owner: AccountScope
): Promise<void> => {
  try {
    const templates = await loadTemplates();
    const raw = await getLibraryWorkflowCached(workflowId, { expectedRevision: revision, signal: owner.signal });
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
    // The library, not this row, kept changing mid-read: the row stays pending and is read again. This recurs only
    // while invalidations keep landing.
    if (error instanceof WorkflowLibraryChangedDuringReadError) {
      if (isAccountScopeCurrent(owner)) {
        pumpEnrichment();
      }
      return;
    }

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

      const entry = findEntry(workflowId);

      // Each item captures the scope it starts under, so a worker that outlives
      // an account switch drops its result instead of writing it.
      if (entry) {
        await enrichEntry(entry.item, captureAccountScope());
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
    ...getListOrder(filter),
    category: filter.category,
    ...(filter.isPublic === undefined ? {} : { isPublic: filter.isPublic }),
    page,
    perPage: PER_PAGE,
    query: filter.search || undefined,
    signal: owner.signal,
    tags: filter.tag ? [filter.tag] : undefined,
  });

const loadFirstPage = async (filter: WorkflowLibraryBrowseFilter, owner: AccountScope): Promise<void> => {
  const generation = listGeneration;

  try {
    const result = await fetchPage(filter, 0, owner);

    if (isListCurrent(generation, owner)) {
      publishPage(result, 'replace');
    }
  } catch (error) {
    if (isListCurrent(generation, owner)) {
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
  const generation = listGeneration;

  try {
    const result = await fetchPage(filter, page, owner);

    if (isListCurrent(generation, owner)) {
      publishPage(result, 'append');
    }
  } catch (error) {
    if (isListCurrent(generation, owner)) {
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
const loadTagCounts = async (
  category: WorkflowLibraryCategory,
  isPublic: boolean | undefined,
  owner: AccountScope
): Promise<void> => {
  try {
    const publicFilter = isPublic === undefined ? {} : { isPublic };
    const tags = await getAllWorkflowTags({ categories: [category], ...publicFilter, signal: owner.signal });
    const counts =
      tags.length > 0
        ? await getWorkflowTagCounts({ categories: [category], ...publicFilter, signal: owner.signal, tags })
        : {};

    // A category or visibility switch mid-flight owns the chips now.
    const filter = store.getSnapshot().filter;
    if (!isAccountScopeCurrent(owner) || filter.category !== category || filter.isPublic !== isPublic) {
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
  const isTagScopeChanged = isCategoryChanged || filter.isPublic !== snapshot.filter.isPublic;

  rememberCurrentView();

  // Everything already in flight was requested for the previous filter.
  listGeneration += 1;
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
    tagCounts: isTagScopeChanged ? EMPTY_TAG_COUNTS : snapshot.tagCounts,
    total: 0,
  });

  void loadFirstPage(filter, owner);

  if (isTagScopeChanged) {
    void loadTagCounts(filter.category, filter.isPublic, owner);
  }
};

/**
 * Appends the next page; a no-op while a page is in flight or on the last page. During a revalidation the request
 * waits for it and then reads the next page of the revalidated list, because offsets may have shifted.
 */
export const loadNextWorkflowLibraryPage = (): void => {
  const { filter, page, pages, status } = store.getSnapshot();

  if (status !== 'loaded' || page + 1 >= pages) {
    return;
  }

  store.patchSnapshot({ status: 'loadingMore' });

  if (revalidatingGeneration !== listGeneration) {
    void loadMorePages(filter, page + 1, captureAccountScope());
  }
};

/**
 * First open: page 0, the category's tag counts, and the user-total probe, in the account's stored sort so the first
 * request already comes back in order. A later open with another sort switches to it as a filter change.
 */
export const ensureWorkflowLibraryBrowseLoaded = (sort: WorkflowLibrarySort): Promise<void> => {
  const { status } = store.getSnapshot();

  if (status !== 'idle' && status !== 'error') {
    setWorkflowLibraryBrowseFilter({ sort });

    return initialLoadFlight.inflight() ?? Promise.resolve();
  }

  return initialLoadFlight.run(async () => {
    const owner = captureAccountScope();
    const filter = { ...store.getSnapshot().filter, sort };

    store.patchSnapshot({ filter });

    hasTemplateLoadFailed = false;
    store.patchSnapshot({ error: null, status: 'loading' });

    await Promise.all([
      loadFirstPage(filter, owner),
      loadTagCounts(filter.category, filter.isPublic, owner),
      probeUserTotal(owner),
    ]);
  });
};

/**
 * Revalidates every page the user has scrolled through, plus the tag counts. It supersedes a page still in flight and
 * reads that page itself; a scroll during it waits and then appends from the revalidated list.
 */
export const refreshWorkflowLibraryBrowse = (): Promise<void> =>
  refreshFlight.run(async () => {
    const { filter, page, status } = store.getSnapshot();

    if (status === 'idle') {
      return;
    }

    const owner = captureAccountScope();

    listGeneration += 1;

    const generation = listGeneration;
    const lastPage = status === 'loadingMore' ? page + 1 : page;

    revalidatingGeneration = generation;

    // An explicit revalidation is also the retry path for a failed enrichment.
    hasTemplateLoadFailed = false;

    try {
      const results = await Promise.all(
        Array.from({ length: lastPage + 1 }, (_unused, index) => fetchPage(filter, index, owner))
      );
      const last = results[results.length - 1];

      if (last && isListCurrent(generation, owner)) {
        const items = results.flatMap((result) => result.items);
        // A scroll that arrived during this read; one already in flight when it began is part of `lastPage`.
        const isScrollWaiting = status !== 'loadingMore' && store.getSnapshot().status === 'loadingMore';

        currentViewFetchedAt = Date.now();
        revalidatingGeneration = null;
        store.patchSnapshot({
          entries: mergeEntries(retryFailedEnrichment(store.getSnapshot().entries), items),
          error: null,
          // Deletions can shrink the library below the page the user was on.
          page: Math.min(lastPage, Math.max(0, last.pages - 1)),
          pages: last.pages,
          status: 'loaded',
          total: last.total,
        });
        pumpEnrichment();

        if (isScrollWaiting) {
          loadNextWorkflowLibraryPage();
        }
      }
    } catch (error) {
      if (isListCurrent(generation, owner)) {
        revalidatingGeneration = null;
        libraryLogger.warn({
          error,
          message: 'Failed to refresh workflows',
          name: 'workflows.library-load-failed',
        });
        store.patchSnapshot({ error: getApiErrorMessage(error, 'Failed to refresh workflows.'), status: 'error' });
      }
    }

    await loadTagCounts(filter.category, filter.isPublic, owner);
  });

export const getWorkflowLibraryBrowseSnapshot = (): WorkflowLibraryBrowseSnapshot => store.getSnapshot();

export const useWorkflowLibraryBrowseSelector = store.useSelector;

// #endregion

/** Coalesce library changes into one current-view refresh because they can shift ordering. */
let isRefreshScheduled = false;

/** Marks the current view outdated, so leaving it before the refresh lands does not cache it, and refreshes it. */
const scheduleCurrentViewRefresh = (): void => {
  currentViewFetchedAt = 0;

  if (isRefreshScheduled || store.getSnapshot().status === 'idle') {
    return;
  }

  isRefreshScheduled = true;
  queueMicrotask(() => {
    isRefreshScheduled = false;

    // The account may have reset between the change and this microtask.
    if (store.getSnapshot().status !== 'idle') {
      void refreshWorkflowLibraryBrowse();
    }
  });
};

onWorkflowLibraryCacheInvalidated(() => {
  // A mutation can change any view; the current one refreshes, the others fetch afresh when revisited.
  viewCache.clear();
  scheduleCurrentViewRefresh();
});

// Opening a workflow moves it to the top of every view ordered by last opened. The server may record the open after
// the dialog is open again, so the current view refreshes now; hidden ones are dropped and fetched when revisited.
onLibraryWorkflowOpened(() => {
  for (const [key, view] of viewCache) {
    if (view.orderBy === 'opened_at') {
      viewCache.delete(key);
    }
  }

  if (getListOrder(store.getSnapshot().filter).orderBy === 'opened_at') {
    scheduleCurrentViewRefresh();
  }
});

registerAccountOwnedResource({
  clear: () => {
    listGeneration += 1;
    revalidatingGeneration = null;
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
