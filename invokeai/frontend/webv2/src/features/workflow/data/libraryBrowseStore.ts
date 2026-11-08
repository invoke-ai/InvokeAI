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
import { ApiError, getApiErrorMessage } from '@platform/transport/http';

import type { WorkflowLibraryCategory, WorkflowLibraryListItem, WorkflowLibraryPage } from './api';

import { getAllWorkflowTags, getWorkflowTagCounts, listLibraryWorkflows, onLibraryWorkflowOpened } from './api';
import {
  getLibraryWorkflowCached,
  getLibraryWorkflowRecordCached,
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

/** A server item position and the fraction of its row above the grid viewport. */
export interface WorkflowLibraryScrollAnchor {
  index: number;
  fraction: number;
}

export interface WorkflowLibraryBrowseSnapshot {
  filter: WorkflowLibraryBrowseFilter;
  status: 'idle' | 'loading' | 'loaded' | 'error';
  /** Resident metadata window, in server order. */
  startIndex?: number;
  scrollAnchor?: WorkflowLibraryScrollAnchor | null;
  selectedEntry?: WorkflowLibraryEntry | null;
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
const MAX_RESIDENT_ENTRIES = PER_PAGE * 3;
const MAX_DEMAND = 48;
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
 * Views already browsed (Browse, Yours, Shared, a tag or search), kept so switching back shows their metadata at once instead of refetching every row. A view older than `VIEW_FRESH_MS` is revalidated in the
 * background and merged in place; library mutations drop the other views.
 */
interface CachedBrowseView {
  entries: readonly WorkflowLibraryEntry[];
  startIndex: number;
  scrollAnchor: WorkflowLibraryScrollAnchor | null;
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
/** Conservative freshness timestamp for the current metadata window. */
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
    entries: entries.map((entry) =>
      entry.enrichment.status === 'pending' ? entry : { ...entry, enrichment: PENDING_ENRICHMENT }
    ),
    startIndex: store.getSnapshot().startIndex ?? 0,
    scrollAnchor: store.getSnapshot().scrollAnchor ?? null,
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
/** Only metadata published in this list generation can satisfy overlapping range reads. */
let metadataGeneration = -1;

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

const publishPage = (result: WorkflowLibraryPage): void => {
  currentViewFetchedAt = Date.now();
  metadataGeneration = listGeneration;
  store.patchSnapshot({
    entries: mergeEntries(store.getSnapshot().entries, result.items),
    startIndex: result.page * PER_PAGE,
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
  const owner = captureAccountScope();
  const snapshot = getInvocationTemplatesSnapshot();

  if (snapshot.status === 'loaded') {
    return snapshot.templates;
  }

  if (hasTemplateLoadFailed) {
    throw new Error('Node definitions are unavailable.');
  }

  if (!templatesFlight) {
    const flight = refreshInvocationTemplates().finally(() => {
      if (templatesFlight === flight) {
        templatesFlight = null;
      }
    });
    templatesFlight = flight;
  }
  await templatesFlight;
  owner.signal.throwIfAborted();

  const settled = getInvocationTemplatesSnapshot();

  if (settled.status !== 'loaded') {
    hasTemplateLoadFailed = true;

    throw new Error(settled.error ?? 'Node definitions are unavailable.');
  }

  return settled.templates;
};

// Demand belongs to mounted UI leases; selected detail is independent of the metadata window.
let visibleDemand: readonly string[] = [];
let selectedDemand: string | null = null;
let selectedValidation: number | null = null;
let selectedController: AbortController | null = null;
const activeReads = new Map<string, AbortController>();

export const selectWorkflowLibraryEntry = (
  snapshot: Pick<WorkflowLibraryBrowseSnapshot, 'entries' | 'selectedEntry'>,
  workflowId: string | null
): WorkflowLibraryEntry | undefined => {
  const selected = snapshot.selectedEntry?.item.workflow_id === workflowId ? snapshot.selectedEntry : undefined;
  const resident = snapshot.entries.find((entry) => entry.item.workflow_id === workflowId);
  // Independent list and selected reads can complete in either order; neither may lower the revision floor.
  return selected && (!resident || selected.item.revision >= resident.item.revision) ? selected : resident;
};

const findEntry = (workflowId: string): WorkflowLibraryEntry | undefined =>
  selectWorkflowLibraryEntry(store.getSnapshot(), workflowId);

const demandedIds = (): Set<string> =>
  new Set([...(selectedDemand ? [selectedDemand] : []), ...visibleDemand.slice(0, MAX_DEMAND)]);

const trimDocuments = (): void => {
  const demand = demandedIds();
  const snapshot = store.getSnapshot();
  const selectedEntry = selectedDemand ? (findEntry(selectedDemand) ?? null) : null;
  const entries = snapshot.entries.map((entry) =>
    !demand.has(entry.item.workflow_id) && entry.enrichment.status !== 'pending'
      ? { ...entry, enrichment: PENDING_ENRICHMENT }
      : entry
  );
  if (selectedEntry !== snapshot.selectedEntry || entries.some((entry, index) => entry !== snapshot.entries[index])) {
    store.patchSnapshot({ entries, selectedEntry });
  }
  for (const [id, controller] of activeReads) {
    if (!demand.has(id)) {
      controller.abort();
    }
  }
};

const pumpEnrichment = (): void => {
  trimDocuments();
  if (
    selectedDemand &&
    findEntry(selectedDemand)?.enrichment.status === 'pending' &&
    !activeReads.has(selectedDemand) &&
    activeReads.size >= ENRICHMENT_CONCURRENCY
  ) {
    const obsolete = [...activeReads].reverse().find(([id]) => id !== selectedDemand);
    obsolete?.[1].abort();
  }
  for (const workflowId of demandedIds()) {
    if (activeReads.size + (selectedValidation !== null ? 1 : 0) >= ENRICHMENT_CONCURRENCY) {
      break;
    }
    const entry = findEntry(workflowId);
    if (
      !entry ||
      (workflowId === selectedDemand && selectedValidation !== null) ||
      entry.enrichment.status !== 'pending' ||
      activeReads.has(workflowId)
    ) {
      continue;
    }
    const controller = new AbortController();
    const owner = captureAccountScope();
    const generation = listGeneration;
    activeReads.set(workflowId, controller);
    const signal = AbortSignal.any([owner.signal, controller.signal]);
    const publish = (enrichment: WorkflowLibraryEntryEnrichment) => {
      if (
        signal.aborted ||
        !isListCurrent(generation, owner) ||
        !demandedIds().has(workflowId) ||
        findEntry(workflowId)?.item.revision !== entry.item.revision
      ) {
        return;
      }
      const snapshot = store.getSnapshot();
      store.patchSnapshot({
        entries: snapshot.entries.map((candidate) =>
          candidate.item.workflow_id === workflowId && candidate.item.revision === entry.item.revision
            ? { ...candidate, enrichment }
            : candidate
        ),
        selectedEntry: selectedDemand === workflowId ? { ...entry, enrichment } : snapshot.selectedEntry,
      });
    };
    void (async () => {
      try {
        const templates = await loadTemplates();
        signal.throwIfAborted();
        const raw = await getLibraryWorkflowCached(workflowId, { expectedRevision: entry.item.revision, signal });
        signal.throwIfAborted();
        const { document } = parseWorkflowJson(raw);
        publish({
          document,
          nodeCount: document.nodes.length,
          requirements: extractWorkflowModelRequirements(document, templates),
          status: 'ready',
        });
      } catch (error) {
        if (!signal.aborted && !(error instanceof WorkflowLibraryChangedDuringReadError)) {
          publish({ message: getApiErrorMessage(error, 'Failed to read this workflow.'), status: 'error' });
        }
      } finally {
        if (activeReads.get(workflowId) === controller) {
          activeReads.delete(workflowId);
        }
        pumpEnrichment();
      }
    })();
  }
};

export const setWorkflowLibraryVisibleDemand = (ids: readonly string[]): void => {
  visibleDemand = ids.slice(0, MAX_DEMAND);
  pumpEnrichment();
};

export const setWorkflowLibrarySelectedDemand = (id: string | null): void => {
  if (selectedDemand !== id) {
    selectedValidation = null;
    selectedController?.abort();
    selectedController = null;
  }
  selectedDemand = id;
  pumpEnrichment();
};

export const rememberWorkflowLibraryScroll = (
  filter: WorkflowLibraryBrowseFilter,
  scrollAnchor: WorkflowLibraryScrollAnchor
): void => {
  const snapshot = store.getSnapshot();
  // A leaving grid may still report measurements during the new view's commit.
  if (
    snapshot.filter === filter &&
    (snapshot.scrollAnchor?.index !== scrollAnchor.index || snapshot.scrollAnchor.fraction !== scrollAnchor.fraction)
  ) {
    store.patchSnapshot({ scrollAnchor });
  }
};

let windowRequest = 0;
let rangeController: AbortController | null = null;
/** Share bounded metadata reads, but resolve readiness for each caller's requested positions. */
export const loadWorkflowLibraryRange = async (start: number, end: number, force = false): Promise<boolean> => {
  const generation = listGeneration;
  const owner = captureAccountScope();
  const loading = loadWorkflowLibraryWindow(start, end, force);
  const request = windowRequest;
  const loaded = await loading;
  const snapshot = store.getSnapshot();
  const residentStart = snapshot.startIndex ?? 0;
  const residentEnd = residentStart + snapshot.entries.length;
  return (
    loaded &&
    isListCurrent(generation, owner) &&
    request === windowRequest &&
    start >= residentStart &&
    start < residentEnd &&
    Math.min(end, snapshot.total - 1) < residentEnd
  );
};

/** Retain at most 60 consecutive positions. Partial edge pages are transient fetch buffers only. */
const loadWorkflowLibraryWindow = (start: number, end: number, force: boolean): Promise<boolean> => {
  const snapshot = store.getSnapshot();
  if (snapshot.status === 'idle' || snapshot.status === 'loading' || snapshot.pages === 0) {
    return Promise.resolve(false);
  }
  const residentStart = snapshot.startIndex ?? 0;
  const requestedEnd = Math.min(end, snapshot.total - 1);
  const flightCoversDemand =
    rangeFlightWindow !== null &&
    start >= rangeFlightWindow.start &&
    start < rangeFlightWindow.end &&
    requestedEnd < rangeFlightWindow.end;
  // A viewport update inside Retry's window must not replace its forced revalidation with reused pages.
  if (!force && rangeFlightWindow?.force && flightCoversDemand) {
    return rangeFlight!;
  }
  const fetchedAt = currentViewFetchedAt;
  const canReuse =
    !force && metadataGeneration === listGeneration && fetchedAt !== 0 && Date.now() - fetchedAt <= VIEW_FRESH_MS;
  if (
    canReuse &&
    start >= residentStart &&
    start < residentStart + snapshot.entries.length &&
    requestedEnd < residentStart + snapshot.entries.length
  ) {
    if (rangeFlightWindow && !flightCoversDemand) {
      rangeController?.abort();
      windowRequest += 1;
      rangeFlightWindow = null;
      rangeFlight = null;
    }
    return Promise.resolve(true);
  }
  // Most viewports fit aligned pages. A tall viewport can cross four pages while still needing only 60 positions.
  const alignedStart = Math.max(0, Math.min(Math.floor(start / PER_PAGE), snapshot.pages - 3)) * PER_PAGE;
  const windowStart = end < alignedStart + MAX_RESIDENT_ENTRIES ? alignedStart : Math.max(0, start);
  const windowEnd = Math.min(snapshot.total, windowStart + MAX_RESIDENT_ENTRIES);
  const first = Math.floor(windowStart / PER_PAGE);
  const last = Math.max(first, Math.ceil(windowEnd / PER_PAGE) - 1);
  const key = `${listGeneration}:${windowStart}:${windowEnd}:${force}`;
  if (rangeFlightWindow?.key === key && (canReuse || !rangeFlightWindow.reusesMetadata)) {
    return rangeFlight!;
  }
  const flightWindow = { key, start: windowStart, end: windowEnd, force, reusesMetadata: false };
  rangeFlightWindow = flightWindow;
  rangeController?.abort();
  const controller = new AbortController();
  rangeController = controller;
  const request = ++windowRequest;
  const generation = listGeneration;
  const owner = captureAccountScope();
  const flight = (async (): Promise<boolean> => {
    try {
      const reusedPages = new Set<number>();
      let results = await Promise.all(
        Array.from({ length: last - first + 1 }, (_, index) => {
          const page = first + index;
          const offset = page * PER_PAGE - residentStart;
          const size = Math.min(PER_PAGE, snapshot.total - page * PER_PAGE);
          // Reuse complete positions only: a clipped edge page must be read again before extending it.
          if (canReuse && offset >= 0 && offset + size <= snapshot.entries.length) {
            reusedPages.add(page);
            flightWindow.reusesMetadata = true;
            return Promise.resolve({
              page,
              pages: snapshot.pages,
              total: snapshot.total,
              items: snapshot.entries.slice(offset, offset + size).map((entry) => entry.item),
            });
          }
          return fetchPage(snapshot.filter, page, owner, controller.signal);
        })
      );
      if (!isListCurrent(generation, owner) || request !== windowRequest || controller.signal.aborted) {
        return false;
      }
      // A changed total is evidence that offsets shifted; reread reused positions before combining them.
      if (reusedPages.size && results.some((page) => page.total !== snapshot.total || page.pages !== snapshot.pages)) {
        flightWindow.reusesMetadata = false;
        results = await Promise.all(
          results.map((page) =>
            reusedPages.has(page.page) ? fetchPage(snapshot.filter, page.page, owner, controller.signal) : page
          )
        );
        reusedPages.clear();
      }
      let publishedStart = windowStart;
      if (results[0] && windowStart >= results[0].total) {
        results = [await fetchPage(snapshot.filter, Math.max(0, results[0].pages - 1), owner, controller.signal)];
        publishedStart = (results[0]?.page ?? 0) * PER_PAGE;
      }
      if (!isListCurrent(generation, owner) || request !== windowRequest || controller.signal.aborted) {
        return false;
      }
      // A fetched page owns totals; reused pages describe the previous response, not a later server mutation.
      const result =
        results.find((page) => page.total !== snapshot.total || page.pages !== snapshot.pages) ?? results[0];
      if (!result) {
        return false;
      }
      const items = results.flatMap((page) => page.items).slice(publishedStart - (results[0]?.page ?? 0) * PER_PAGE);
      metadataGeneration = generation;
      // A partial read cannot renew the age of positions it did not read, even if the flight crossed expiry.
      currentViewFetchedAt = reusedPages.size ? fetchedAt : Date.now();
      store.patchSnapshot({
        entries: mergeEntries(
          store.getSnapshot().entries,
          items.slice(0, Math.min(MAX_RESIDENT_ENTRIES, Math.max(0, result.total - publishedStart)))
        ),
        startIndex: publishedStart,
        page: Math.min(last, Math.max(0, result.pages - 1)),
        pages: result.pages,
        total: result.total,
        error: null,
        status: 'loaded',
      });
      if (!isListCurrent(generation, owner) || request !== windowRequest || controller.signal.aborted) {
        return false;
      }
      pumpEnrichment();
      return true;
    } catch (error) {
      if (!controller.signal.aborted && isListCurrent(generation, owner) && request === windowRequest) {
        store.patchSnapshot({ error: getApiErrorMessage(error, 'Failed to load workflows.') });
      }
      return false;
    } finally {
      if (request === windowRequest) {
        rangeFlightWindow = null;
        rangeFlight = null;
      }
    }
  })();
  rangeFlight = flight;
  return flight;
};
let rangeFlight: Promise<boolean> | null = null;
let rangeFlightWindow: {
  key: string;
  start: number;
  end: number;
  force: boolean;
  reusesMetadata: boolean;
} | null = null;

const fetchPage = (
  filter: WorkflowLibraryBrowseFilter,
  page: number,
  owner: AccountScope,
  signal?: AbortSignal
): Promise<WorkflowLibraryPage> =>
  listLibraryWorkflows({
    ...getListOrder(filter),
    category: filter.category,
    ...(filter.isPublic === undefined ? {} : { isPublic: filter.isPublic }),
    page,
    perPage: PER_PAGE,
    query: filter.search || undefined,
    signal: signal ? AbortSignal.any([signal, owner.signal]) : owner.signal,
    tags: filter.tag ? [filter.tag] : undefined,
  });

const loadFirstPage = async (filter: WorkflowLibraryBrowseFilter, owner: AccountScope): Promise<void> => {
  const generation = listGeneration;
  const request = windowRequest;

  try {
    const result = await fetchPage(filter, 0, owner);

    if (isListCurrent(generation, owner) && request === windowRequest) {
      publishPage(result);
    }
  } catch (error) {
    if (isListCurrent(generation, owner) && request === windowRequest) {
      libraryLogger.warn({
        error,
        message: 'Failed to load workflows',
        name: 'workflows.library-load-failed',
      });
      store.patchSnapshot({ error: getApiErrorMessage(error, 'Failed to load workflows.'), status: 'error' });
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
 * stale); a new one starts from page 0.
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
  rangeController?.abort();
  rangeFlightWindow = null;
  windowRequest += 1;

  // Everything already in flight was requested for the previous filter.
  listGeneration += 1;
  visibleDemand = [];
  selectedDemand = null;
  selectedValidation = null;
  selectedController?.abort();
  selectedController = null;
  for (const controller of activeReads.values()) {
    controller.abort();
  }
  hasTemplateLoadFailed = false;

  const cached = viewCache.get(getFilterKey(filter));

  if (cached) {
    const isStale = Date.now() - cached.fetchedAt > VIEW_FRESH_MS;
    currentViewFetchedAt = cached.fetchedAt;
    // Revalidation may queue behind another filter; stale positions cannot satisfy overlapping demand.
    metadataGeneration = isStale ? -1 : listGeneration;
    store.patchSnapshot({
      entries: cached.entries,
      startIndex: cached.startIndex,
      scrollAnchor: cached.scrollAnchor,
      selectedEntry: null,
      error: null,
      filter,
      page: cached.page,
      pages: cached.pages,
      status: 'loaded',
      tagCounts: cached.tagCounts,
      total: cached.total,
    });
    pumpEnrichment();

    if (isStale) {
      void refreshWorkflowLibraryBrowse();
    }
    return;
  }

  store.patchSnapshot({
    entries: EMPTY_ENTRIES,
    startIndex: 0,
    scrollAnchor: null,
    selectedEntry: null,
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

/** Revalidate the selected pin independently of the resident list window. */
const revalidateSelected = async (generation: number, owner: AccountScope): Promise<void> => {
  const entry = store.getSnapshot().selectedEntry;
  if (!entry || entry.item.workflow_id !== selectedDemand) {
    return;
  }
  const id = entry.item.workflow_id;
  selectedController?.abort();
  const controller = new AbortController();
  selectedController = controller;
  selectedValidation = generation;
  const current = () => !controller.signal.aborted && isListCurrent(generation, owner) && selectedDemand === id;
  try {
    const record = await getLibraryWorkflowRecordCached(id, { signal: controller.signal });
    if (!current() || record.revision < (findEntry(id)?.item.revision ?? entry.item.revision)) {
      return;
    }
    const item: WorkflowLibraryListItem = {
      ...entry.item,
      workflow_id: record.workflow_id,
      name: record.name,
      revision: record.revision,
      user_id: record.user_id,
      is_public: record.is_public,
      created_at: record.created_at,
      updated_at: record.updated_at,
      opened_at: record.opened_at,
      last_run_at: record.last_run_at,
      thumbnail_url: record.thumbnail_url,
      call_saved_workflow_compatibility: record.call_saved_workflow_compatibility,
      description: typeof record.workflow.description === 'string' ? record.workflow.description : '',
      tags: typeof record.workflow.tags === 'string' ? record.workflow.tags : '',
    };
    store.patchSnapshot({ selectedEntry: toEntry(item) });
  } catch (error) {
    if (!current()) {
      return;
    }
    if (error instanceof ApiError && (error.status === 403 || error.status === 404)) {
      selectedDemand = null;
      store.patchSnapshot({ selectedEntry: null });
    } else {
      store.patchSnapshot({
        selectedEntry: {
          ...(findEntry(id) ?? entry),
          enrichment: { status: 'error', message: getApiErrorMessage(error, 'Failed to read this workflow.') },
        },
      });
    }
  } finally {
    if (selectedController === controller) {
      selectedController = null;
    }
    if (selectedValidation === generation) {
      selectedValidation = null;
    }
  }
};

/**
 * Revalidate resident positions and tag counts. A later range owns the window and fences this list publication.
 */
export const refreshWorkflowLibraryBrowse = (): Promise<void> =>
  refreshFlight.run(async () => {
    const { filter, status } = store.getSnapshot();

    if (status === 'idle') {
      return;
    }

    const owner = captureAccountScope();

    listGeneration += 1;
    for (const controller of activeReads.values()) {
      controller.abort();
    }
    activeReads.clear();
    rangeController?.abort();
    rangeFlightWindow = null;
    const request = ++windowRequest;

    const generation = listGeneration;
    const firstPage = Math.floor((store.getSnapshot().startIndex ?? 0) / PER_PAGE);
    const residentStart = store.getSnapshot().startIndex ?? 0;
    const residentEnd = residentStart + store.getSnapshot().entries.length;
    const lastPage = Math.max(firstPage, Math.ceil(residentEnd / PER_PAGE) - 1);

    // An explicit revalidation is also the retry path for a failed enrichment.
    hasTemplateLoadFailed = false;

    const selectedRead = revalidateSelected(generation, owner);
    try {
      let results = await Promise.all(
        Array.from({ length: lastPage - firstPage + 1 }, (_unused, index) =>
          fetchPage(filter, firstPage + index, owner)
        )
      );
      let publishedStart = residentStart;
      if (results[0] && residentStart >= results[0].total) {
        results = [await fetchPage(filter, Math.max(0, results[0].pages - 1), owner)];
        publishedStart = (results[0]?.page ?? 0) * PER_PAGE;
      }
      const last = results[results.length - 1];

      if (last && isListCurrent(generation, owner) && request === windowRequest) {
        const items = results.flatMap((result) => result.items);
        currentViewFetchedAt = Date.now();
        metadataGeneration = generation;
        store.patchSnapshot({
          startIndex: publishedStart,
          entries: mergeEntries(
            retryFailedEnrichment(store.getSnapshot().entries),
            items.slice(
              publishedStart - (results[0]?.page ?? 0) * PER_PAGE,
              publishedStart -
                (results[0]?.page ?? 0) * PER_PAGE +
                Math.min(MAX_RESIDENT_ENTRIES, Math.max(0, last.total - publishedStart))
            )
          ),
          error: null,
          // Deletions can shrink the library below the page the user was on.
          page: Math.min(lastPage, Math.max(0, last.pages - 1)),
          pages: last.pages,
          status: 'loaded',
          total: last.total,
        });
        await selectedRead;
        pumpEnrichment();
      }
    } catch (error) {
      if (isListCurrent(generation, owner) && request === windowRequest) {
        libraryLogger.warn({
          error,
          message: 'Failed to refresh workflows',
          name: 'workflows.library-load-failed',
        });
        store.patchSnapshot({ error: getApiErrorMessage(error, 'Failed to refresh workflows.'), status: 'error' });
      }
    }

    await selectedRead;
    pumpEnrichment();
    await loadTagCounts(filter.category, filter.isPublic, owner);
  });

export const getWorkflowLibraryBrowseSnapshot = (): WorkflowLibraryBrowseSnapshot => store.getSnapshot();

/** Owner-level resource observations for the large-library fixture, independent of browser heap noise. */
export const getWorkflowLibraryBrowseRetention = () => {
  const snapshot = store.getSnapshot();
  const entries = [
    ...snapshot.entries,
    ...[...viewCache.values()].flatMap((view) => view.entries),
    ...(snapshot.selectedEntry ? [snapshot.selectedEntry] : []),
  ];
  return {
    activeReads: activeReads.size + (selectedValidation !== null ? 1 : 0),
    cachedViews: viewCache.size,
    metadataEntries: entries.length,
    parsedDocuments: new Set(
      entries.filter((entry) => entry.enrichment.status === 'ready').map((entry) => entry.enrichment)
    ).size,
  };
};

export const useWorkflowLibraryBrowseSelector = store.useSelector;

// #endregion

/** Coalesce library changes into one current-view refresh because they can shift ordering. */
let isRefreshScheduled = false;

/** Marks the current view outdated, so leaving it before the refresh lands does not cache it, and refreshes it. */
const scheduleCurrentViewRefresh = (): void => {
  currentViewFetchedAt = 0;
  // Fence existing list publications now, even when another filter holds the refresh flight.
  const controller = rangeController;
  rangeController = null;
  rangeFlightWindow = null;
  rangeFlight = null;
  windowRequest += 1;
  controller?.abort();

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
  metadataGeneration = -1;
  currentViewFetchedAt = 0;
  // A mutation can change any view; the current one refreshes, the others fetch afresh when revisited.
  viewCache.clear();
  for (const controller of activeReads.values()) {
    controller.abort();
  }
  activeReads.clear();
  selectedController?.abort();
  selectedValidation = store.getSnapshot().selectedEntry ? listGeneration + 1 : null;
  scheduleCurrentViewRefresh();
  store.patchSnapshot({
    entries: store.getSnapshot().entries.map((entry) => ({ ...entry, enrichment: PENDING_ENRICHMENT })),
    selectedEntry: store.getSnapshot().selectedEntry
      ? { ...store.getSnapshot().selectedEntry!, enrichment: PENDING_ENRICHMENT }
      : null,
  });
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
    metadataGeneration = -1;
    visibleDemand = [];
    selectedDemand = null;
    selectedValidation = null;
    selectedController?.abort();
    selectedController = null;
    for (const controller of activeReads.values()) {
      controller.abort();
    }
    rangeFlightWindow = null;
    rangeController?.abort();
    rangeController = null;
    windowRequest += 1;
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
