import type {
  GalleryItem,
  GalleryItemKey,
  GalleryItemMutationResult,
  GalleryItemRef,
  GalleryItemsPage,
} from '@features/gallery/core/items';
import type { GalleryBoard } from '@features/gallery/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { InfiniteData, Query, QueryClient, QueryKey } from '@tanstack/react-query';

import { toGalleryItemKey } from '@features/gallery/core/items';
import { pruneImageClusterMembers } from '@features/gallery/core/semanticImageQuery';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { rollBackUnclaimedEntries } from '@platform/state/compareAndSwapRollback';
import { hashKey } from '@tanstack/react-query';

import { ALL_READABLE_BOARDS_ID, isDateBoardId } from './backend';
import {
  fetchGalleryItemsRange,
  GALLERY_PAGE_SIZE,
  galleryKeys,
  getGalleryItemListingKey,
  getGalleryItemListQueries,
  getGalleryItemsFilterFromKey,
  isGallerySinglePageQueryKey,
  type CanonicalGalleryItemsFilter,
} from './queries';

export type GalleryItemCachePatch =
  | { kind: 'delete'; result: GalleryItemMutationResult }
  | { boardId: string; kind: 'move'; result: GalleryItemMutationResult }
  | { kind: 'star'; result: GalleryItemMutationResult; starred: boolean };

let galleryThumbnailRevision = 0;
const galleryThumbnailRevisionListeners = new Set<() => void>();

export const getGalleryThumbnailRevision = (): number => galleryThumbnailRevision;

export const subscribeGalleryThumbnailRevision = (listener: () => void): (() => void) => {
  galleryThumbnailRevisionListeners.add(listener);
  return () => galleryThumbnailRevisionListeners.delete(listener);
};

/** Ask mounted gallery thumbnails to request their URLs again after maintenance repairs them. */
export const refreshGalleryThumbnails = (owner: AccountScope): void => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  galleryThumbnailRevision += 1;
  galleryThumbnailRevisionListeners.forEach((listener) => listener());
};

export const getRefreshedGalleryThumbnailUrl = (url: string, currentRevision: number): string => {
  if (currentRevision === 0 || /^(?:blob|data):/i.test(url)) {
    return url;
  }

  const refreshedUrl = new URL(url, window.location.href);
  refreshedUrl.searchParams.set('gallery_thumbnail_revision', String(currentRevision));
  return refreshedUrl.toString();
};

/** A list window's pages, or one sparse/strip page. */
type GalleryItemsCacheData = InfiniteData<GalleryItemsPage, number> | GalleryItemsPage;

interface ItemCacheRollbackEntry {
  after: GalleryItemsCacheData;
  before: GalleryItemsCacheData;
  queryKey: QueryKey;
}

const isGalleryItemsData = (value: unknown): value is InfiniteData<GalleryItemsPage, number> => {
  if (!value || typeof value !== 'object' || !('pages' in value) || !('pageParams' in value)) {
    return false;
  }

  const data = value as { pages?: unknown; pageParams?: unknown };

  return Array.isArray(data.pages) && Array.isArray(data.pageParams);
};

const isGalleryItemsPage = (value: unknown): value is GalleryItemsPage =>
  typeof value === 'object' &&
  value !== null &&
  Array.isArray((value as { items?: unknown }).items) &&
  typeof (value as { total?: unknown }).total === 'number';

/** The pages a list-family cache entry holds, whichever shape it is. */
const getCachedPages = (query: Query): GalleryItemsPage[] => {
  const data = query.state.data;

  if (isGalleryItemsData(data)) {
    return data.pages;
  }

  return isGallerySinglePageQueryKey(query.queryKey) && isGalleryItemsPage(data) ? [data] : [];
};

const mapPageItems = (
  page: GalleryItemsPage,
  mapItem: (item: GalleryItem) => GalleryItem | null,
  totalDelta = 0
): GalleryItemsPage => {
  let changed = false;
  const items: GalleryItem[] = [];
  const itemIndices: number[] | undefined = page.itemIndices ? [] : undefined;

  for (const [index, item] of page.items.entries()) {
    const nextItem = mapItem(item);

    if (nextItem !== item) {
      changed = true;
    }
    if (nextItem) {
      items.push(nextItem);
      itemIndices?.push(page.itemIndices?.[index] ?? (page.offset ?? 0) + index);
    }
  }

  if (!changed && totalDelta === 0) {
    return page;
  }

  return {
    ...page,
    items: changed ? items : page.items,
    ...(changed && itemIndices ? { itemIndices } : {}),
    total: Math.max(0, page.total - totalDelta),
  };
};

/** Cluster windows ignore board moves; starred-only listings lose items immediately when unstarred. */
const patchRemovesItems = (filter: CanonicalGalleryItemsFilter, patch: GalleryItemCachePatch): boolean => {
  if (patch.kind === 'delete') {
    return true;
  }

  if (patch.kind === 'star') {
    return filter.starred !== undefined && filter.starred !== patch.starred;
  }

  return (
    filter.semantic?.kind !== 'cluster' &&
    filter.boardId !== ALL_READABLE_BOARDS_ID &&
    filter.boardId !== patch.boardId &&
    !isDateBoardId(filter.boardId)
  );
};

const patchItemPage = (
  page: GalleryItemsPage,
  filter: CanonicalGalleryItemsFilter,
  patch: GalleryItemCachePatch,
  itemKeys: ReadonlySet<GalleryItemKey>,
  removedItemCount: number
): GalleryItemsPage => {
  if (patchRemovesItems(filter, patch)) {
    return mapPageItems(page, (item) => (itemKeys.has(toGalleryItemKey(item)) ? null : item), removedItemCount);
  }

  return mapPageItems(page, (item) => {
    if (!itemKeys.has(toGalleryItemKey(item))) {
      return item;
    }

    if (patch.kind === 'star') {
      return item.starred === patch.starred ? item : { ...item, starred: patch.starred };
    }

    if (patch.kind === 'move') {
      return item.boardId === patch.boardId ? item : { ...item, boardId: patch.boardId };
    }

    return item;
  });
};

const patchItemsInfiniteData = (
  data: InfiniteData<GalleryItemsPage, number>,
  filter: CanonicalGalleryItemsFilter,
  patch: GalleryItemCachePatch,
  itemKeys: ReadonlySet<GalleryItemKey>,
  removedItemCount: number
): InfiniteData<GalleryItemsPage, number> => {
  let changed = false;
  const pages = data.pages.map((page) => {
    const nextPage = patchItemPage(page, filter, patch, itemKeys, removedItemCount);
    changed ||= nextPage !== page;

    return nextPage;
  });

  return changed ? { ...data, pages } : data;
};

const patchItemsCacheData = (
  query: Query,
  filter: CanonicalGalleryItemsFilter,
  patch: GalleryItemCachePatch,
  itemKeys: ReadonlySet<GalleryItemKey>,
  removedItemCount: number
): { after: GalleryItemsCacheData; before: GalleryItemsCacheData } | null => {
  const before = query.state.data;

  if (isGalleryItemsData(before)) {
    return { after: patchItemsInfiniteData(before, filter, patch, itemKeys, removedItemCount), before };
  }

  // New items are left to the trailing refetch so their server ordering is preserved.
  if (isGallerySinglePageQueryKey(query.queryKey) && isGalleryItemsPage(before)) {
    return { after: patchItemPage(before, filter, patch, itemKeys, removedItemCount), before };
  }

  return null;
};

const getListingHash = (queryKey: QueryKey): string => hashKey(getGalleryItemListingKey(queryKey));

/**
 * Count the removed items each listing, or each cache entry, holds in its cached pages. Every cached page of a listing
 * reports the same server total, so each must lose the same count: pages left disagreeing would make the sparse views
 * clamp and reconcile against each other.
 */
const countRemovedItems = (
  queries: readonly Query[],
  patch: GalleryItemCachePatch,
  itemKeys: ReadonlySet<GalleryItemKey>,
  getCountKey: (query: Query) => string
): Map<string, number> => {
  const removedKeysByCountKey = new Map<string, Set<GalleryItemKey>>();

  for (const query of queries) {
    const filter = getGalleryItemsFilterFromKey(query.queryKey);

    if (!filter || !patchRemovesItems(filter, patch)) {
      continue;
    }

    const countKey = getCountKey(query);
    const removedKeys = removedKeysByCountKey.get(countKey) ?? new Set<GalleryItemKey>();

    removedKeysByCountKey.set(countKey, removedKeys);
    for (const page of getCachedPages(query)) {
      for (const item of page.items) {
        const key = toGalleryItemKey(item);

        if (itemKeys.has(key)) {
          removedKeys.add(key);
        }
      }
    }
  }

  return new Map([...removedKeysByCountKey].map(([countKey, keys]) => [countKey, keys.size]));
};

/** An optimistic star patch, or the rollback of one, for state that retains star flags outside the item caches. */
export type GalleryItemStarPatchEvent =
  | { itemKeys: ReadonlySet<GalleryItemKey>; kind: 'apply'; patchId: number; starred: boolean }
  | { kind: 'revert'; patchId: number };

type GalleryItemStarPatchListener = (event: GalleryItemStarPatchEvent) => void;

const galleryItemStarPatchListeners = new WeakMap<QueryClient, Set<GalleryItemStarPatchListener>>();
let nextGalleryItemStarPatchId = 0;

/**
 * Observe star patches on `client`'s item caches. An unstarred item leaves starred-only listings, and an item on an
 * evicted page is in no cache at all, so its flag can only be reconciled from the patch itself.
 */
export const subscribeGalleryItemStarPatches = (
  client: QueryClient,
  listener: GalleryItemStarPatchListener
): (() => void) => {
  let listeners = galleryItemStarPatchListeners.get(client);

  if (!listeners) {
    listeners = new Set();
    galleryItemStarPatchListeners.set(client, listeners);
  }

  listeners.add(listener);

  return () => listeners.delete(listener);
};

const emitGalleryItemStarPatch = (client: QueryClient, event: GalleryItemStarPatchEvent): void =>
  galleryItemStarPatchListeners.get(client)?.forEach((listener) => listener(event));

export interface GalleryItemCachePatchOptions {
  /**
   * Which cached pages a removal lowers the total of. `listing` (the default) lowers every cached page of a listing
   * that holds a removed item anywhere, as a first patch must. `holder` lowers only the cache entries that still hold a
   * removed item, for re-applying a patch that already lowered the rest: a page refetched since then is the only one
   * still counting the item.
   */
  totals?: 'holder' | 'listing';
}

/**
 * Applies only backend-confirmed successes. Failed refs are intentionally
 * ignored, and kind-qualified keys prevent same-name images/videos colliding.
 */
export const patchGalleryItemCaches = (
  client: QueryClient,
  patch: GalleryItemCachePatch,
  { totals = 'listing' }: GalleryItemCachePatchOptions = {}
): (() => void) => {
  const itemKeys = new Set(patch.result.succeeded.map(toGalleryItemKey));

  if (itemKeys.size === 0) {
    return () => undefined;
  }

  // The cluster filter's member list is client-owned, so a server refetch can
  // never reconcile it: prune it in the same optimistic step (and restore it
  // with the same rollback) as the list caches it feeds.
  const rollbackClusterMembers =
    patch.kind === 'delete' ? pruneImageClusterMembers(patch.result.succeeded.map(toGalleryItemKey)) : null;
  const rollbackEntries: ItemCacheRollbackEntry[] = [];
  const starPatchId = patch.kind === 'star' ? ++nextGalleryItemStarPatchId : null;

  const queries = getGalleryItemListQueries(client);
  const getCountKey =
    totals === 'listing' ? (query: Query) => getListingHash(query.queryKey) : (query: Query) => query.queryHash;
  const removedCounts = countRemovedItems(queries, patch, itemKeys, getCountKey);

  for (const query of queries) {
    const filter = getGalleryItemsFilterFromKey(query.queryKey);
    const removedItemCount = removedCounts.get(getCountKey(query)) ?? 0;
    const patched = filter ? patchItemsCacheData(query, filter, patch, itemKeys, removedItemCount) : null;

    if (!patched || patched.after === patched.before) {
      continue;
    }

    const applied = client.setQueryData<GalleryItemsCacheData>(query.queryKey, patched.after);

    if (applied) {
      rollbackEntries.push({ after: applied, before: patched.before, queryKey: query.queryKey });
    }
  }

  if (patch.kind === 'star' && starPatchId !== null) {
    emitGalleryItemStarPatch(client, { itemKeys, kind: 'apply', patchId: starPatchId, starred: patch.starred });
  }

  return () => {
    rollbackClusterMembers?.();
    rollBackUnclaimedEntries(
      rollbackEntries,
      (entry) => client.getQueryData<GalleryItemsCacheData>(entry.queryKey),
      (entry) => client.setQueryData(entry.queryKey, entry.before)
    );

    if (starPatchId !== null) {
      emitGalleryItemStarPatch(client, { kind: 'revert', patchId: starPatchId });
    }
  };
};

/** Capture cached source boards before optimistic moves so rejected refs can be restored before refetch. */
export const getGalleryItemBoardIdsFromCaches = (
  client: QueryClient,
  refs: readonly GalleryItemRef[]
): Map<GalleryItemKey, string> => {
  const wanted = new Set(refs.map(toGalleryItemKey));
  const boardIds = new Map<GalleryItemKey, string>();

  for (const query of getGalleryItemListQueries(client)) {
    if (boardIds.size === wanted.size) {
      break;
    }

    for (const page of getCachedPages(query)) {
      for (const item of page.items) {
        const key = toGalleryItemKey(item);

        if (wanted.has(key) && !boardIds.has(key)) {
          boardIds.set(key, item.boardId);
        }
      }
    }
  }

  return boardIds;
};

/**
 * Capture each cached starred flag before mutation; failed batches must restore actual prior values rather than
 * invert the request.
 */
export const getGalleryItemStarredFromCaches = (
  client: QueryClient,
  refs: readonly GalleryItemRef[]
): Map<GalleryItemKey, boolean> => {
  const wanted = new Set(refs.map(toGalleryItemKey));
  const starred = new Map<GalleryItemKey, boolean>();

  for (const query of getGalleryItemListQueries(client)) {
    if (starred.size === wanted.size) {
      break;
    }

    for (const page of getCachedPages(query)) {
      for (const item of page.items) {
        const key = toGalleryItemKey(item);

        if (wanted.has(key) && !starred.has(key)) {
          starred.set(key, item.starred);
        }
      }
    }
  }

  return starred;
};

interface BoardCacheRollbackEntry {
  after: GalleryBoard[];
  before: GalleryBoard[];
  queryKey: QueryKey;
}

const isGalleryBoardsData = (value: unknown): value is GalleryBoard[] =>
  Array.isArray(value) && value.every((board) => typeof board === 'object' && board !== null && 'id' in board);

/**
 * Patches one board across every cached board list and returns a rollback
 * that restores the prior lists — skipping any list something else has
 * written to since, the same conflict rule as `patchGalleryItemCaches`.
 */
export const patchGalleryBoardCaches = (
  client: QueryClient,
  boardId: string,
  changes: Partial<Pick<GalleryBoard, 'archived' | 'name' | 'projectId'>>
): (() => void) => {
  const owner = captureAccountScope();
  const rollbackEntries: BoardCacheRollbackEntry[] = [];

  for (const query of client.getQueryCache().findAll({ queryKey: galleryKeys.boardsForAccount(owner) })) {
    const before = query.state.data;

    if (!isGalleryBoardsData(before)) {
      continue;
    }

    let changed = false;
    const after = before.map((board) => {
      if (board.id !== boardId) {
        return board;
      }

      changed = true;
      return { ...board, ...changes };
    });

    if (!changed) {
      continue;
    }

    const applied = client.setQueryData<GalleryBoard[]>(query.queryKey, after);

    if (applied) {
      rollbackEntries.push({ after: applied, before, queryKey: query.queryKey });
    }
  }

  return () =>
    rollBackUnclaimedEntries(
      rollbackEntries,
      (entry) => client.getQueryData<InfiniteData<GalleryItemsPage, number>>(entry.queryKey),
      (entry) => client.setQueryData(entry.queryKey, entry.before)
    );
};

/** The contiguous row range a window's pages cover, or null for a shape the rebuild cannot reason about. */
const getGalleryWindowSpan = (
  data: InfiniteData<GalleryItemsPage, number>
): { offset: number; rowCount: number } | null => {
  const [firstOffset] = data.pageParams;

  if (typeof firstOffset !== 'number') {
    return null;
  }

  const isContiguous = data.pageParams.every(
    (pageParam, index) => pageParam === firstOffset + index * GALLERY_PAGE_SIZE
  );

  return isContiguous ? { offset: firstOffset, rowCount: data.pageParams.length * GALLERY_PAGE_SIZE } : null;
};

/**
 * Swaps an active window's pages atomically from one span-sized read. False
 * falls back to the collapse: the read failed or the entry changed meanwhile.
 */
const rebuildGalleryItemWindow = async (client: QueryClient, owner: AccountScope, query: Query): Promise<boolean> => {
  const filter = getGalleryItemsFilterFromKey(query.queryKey);
  const before = query.state.data;

  if (!filter || !isGalleryItemsData(before)) {
    return false;
  }

  const span = getGalleryWindowSpan(before);

  if (!span) {
    return false;
  }

  // Name-hydrated windows fetch videos one by one; re-reading a video-heavy
  // span every mutation would cost more than the collapse ever did.
  if (
    (filter.semantic !== undefined || isDateBoardId(filter.boardId)) &&
    before.pages.reduce((count, page) => count + page.items.filter((item) => item.kind === 'video').length, 0) >
      GALLERY_PAGE_SIZE
  ) {
    return false;
  }

  let result: GalleryItemsPage;

  try {
    result = await fetchGalleryItemsRange(client, owner, filter, {
      limit: span.rowCount,
      offset: span.offset,
      signal: owner.signal,
      includeAbsolutePositions: true,
    });
  } catch {
    return false;
  }

  const liveQuery = client.getQueryCache().get(query.queryHash);

  // A page fetch that started during the span read snapshotted the old pages
  // and will land after this swap; only an idle, untouched entry may take it.
  if (liveQuery?.state.data !== before || liveQuery.state.fetchStatus !== 'idle') {
    return false;
  }

  const loadedRangeCount = Math.min(span.rowCount, Math.max(0, result.total - span.offset));
  const pageCount = Math.max(1, Math.ceil(loadedRangeCount / GALLERY_PAGE_SIZE));
  const pages: GalleryItemsPage[] = Array.from({ length: pageCount }, (_, pageIndex) => {
    const pageOffset = span.offset + pageIndex * GALLERY_PAGE_SIZE;
    const pageEnd = pageOffset + GALLERY_PAGE_SIZE;
    const items: GalleryItem[] = [];
    const itemIndices: number[] | undefined = result.itemIndices ? [] : undefined;

    result.items.forEach((item, index) => {
      const itemIndex = result.itemIndices?.[index] ?? (result.offset ?? span.offset) + index;

      if (itemIndex >= pageOffset && itemIndex < pageEnd) {
        items.push(item);
        itemIndices?.push(itemIndex);
      }
    });

    return { items, ...(itemIndices ? { itemIndices } : {}), offset: pageOffset, total: result.total };
  });

  // TanStack never stores zero pages; an emptied span keeps one empty page.
  if (pages.length === 0) {
    pages.push({ items: [], total: result.total });
  }

  client.setQueryData<InfiniteData<GalleryItemsPage, number>>(query.queryKey, {
    pageParams: pages.map((_, index) => span.offset + index * GALLERY_PAGE_SIZE),
    pages,
  });

  return true;
};

/** Collapses a window to its anchor page, so its refetch replays one request. */
const collapseGalleryItemWindowToAnchor = (client: QueryClient, query: Query): void => {
  // A transient entry (anchored windows carry gcTime 0) may have been
  // collected while a rebuild awaited; writing to its key would resurrect it.
  if (client.getQueryCache().get(query.queryHash) !== query) {
    return;
  }

  const data = query.state.data;

  if (!isGalleryItemsData(data) || data.pages.length <= 1) {
    return;
  }

  const anchorOffset =
    (query.queryKey[5] === 'anchor' || query.queryKey[5] === 'infinite') && typeof query.queryKey[6] === 'number'
      ? query.queryKey[6]
      : 0;
  const anchorIndex = Math.max(0, data.pageParams.indexOf(anchorOffset));

  client.setQueryData<InfiniteData<GalleryItemsPage, number>>(query.queryKey, {
    pageParams: [data.pageParams[anchorIndex] ?? anchorOffset],
    pages: [data.pages[anchorIndex] ?? data.pages[0]],
  });
};

const runGalleryInvalidation = async (
  client: QueryClient,
  owner: AccountScope,
  includeBoards: boolean
): Promise<void> => {
  // Date-board pages and lazy range selection share these names. Mark them
  // stale before active pages refetch so they cannot hydrate stale refs.
  await client.cancelQueries({ queryKey: galleryKeys.itemNamesForAccount(owner) });
  await client.invalidateQueries({
    queryKey: galleryKeys.itemNamesForAccount(owner),
    refetchType: 'none',
  });
  await client.cancelQueries({ queryKey: galleryKeys.itemListsForAccount(owner) });

  const rebuiltQueryHashes = new Set<string>();

  // Rebuild active windows in one span to preserve viewport rows; unobserved windows collapse to their pinned
  // page.
  for (const query of getGalleryItemListQueries(client, owner)) {
    const data = query.state.data;

    if (!isGalleryItemsData(data) || data.pages.length <= 1) {
      continue;
    }

    if (query.isActive() && (await rebuildGalleryItemWindow(client, owner, query))) {
      rebuiltQueryHashes.add(query.queryHash);
      continue;
    }

    collapseGalleryItemWindowToAnchor(client, query);
  }

  await client.invalidateQueries({
    predicate: (query) => !rebuiltQueryHashes.has(query.queryHash),
    queryKey: galleryKeys.itemListsForAccount(owner),
  });

  if (includeBoards) {
    await client.invalidateQueries({ queryKey: galleryKeys.boardsForAccount(owner) });
  }
};

interface GalleryInvalidationState {
  includeBoards: boolean;
  promise: Promise<void> | null;
  requested: boolean;
}

const galleryInvalidations = new WeakMap<QueryClient, Map<string, GalleryInvalidationState>>();

/**
 * Coalesce same-tick invalidations with at most one trailing pass to avoid cancelling and restarting observed
 * refetches.
 */
const scheduleGalleryInvalidation = (
  client: QueryClient,
  owner: AccountScope,
  includeBoards: boolean
): Promise<void> => {
  const ownerKey = hashKey(galleryKeys.itemListsForAccount(owner));
  const clientStates = galleryInvalidations.get(client) ?? new Map<string, GalleryInvalidationState>();
  const state = clientStates.get(ownerKey) ?? {
    includeBoards: false,
    promise: null,
    requested: false,
  };

  galleryInvalidations.set(client, clientStates);
  clientStates.set(ownerKey, state);
  state.includeBoards ||= includeBoards;
  state.requested = true;

  if (!state.promise) {
    state.promise = (async () => {
      try {
        // Let a synchronous burst collapse into one pass.
        await Promise.resolve();

        while (state.requested) {
          state.requested = false;
          const shouldInvalidateBoards = state.includeBoards;

          state.includeBoards = false;
          await runGalleryInvalidation(client, owner, shouldInvalidateBoards);
        }
      } finally {
        state.promise = null;
        clientStates.delete(ownerKey);
      }
    })();
  }

  return state.promise;
};

export const invalidateGalleryItems = (
  client: QueryClient,
  owner: AccountScope = captureAccountScope()
): Promise<void> => scheduleGalleryInvalidation(client, owner, false);

export const invalidateGallery = (client: QueryClient, owner: AccountScope = captureAccountScope()): Promise<void> =>
  scheduleGalleryInvalidation(client, owner, true);
