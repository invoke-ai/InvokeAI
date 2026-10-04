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
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { rollBackUnclaimedEntries } from '@platform/state/compareAndSwapRollback';
import { hashKey } from '@tanstack/react-query';

import { ALL_READABLE_BOARDS_ID, isDateBoardId } from './backend';
import {
  fetchGalleryItemsRange,
  GALLERY_PAGE_SIZE,
  galleryKeys,
  getGalleryItemListQueries,
  getGalleryItemsFilterFromKey,
  isGalleryStarredStripQueryKey,
  type CanonicalGalleryItemsFilter,
} from './queries';

export type GalleryItemCachePatch =
  | { kind: 'delete'; result: GalleryItemMutationResult }
  | { boardId: string; kind: 'move'; result: GalleryItemMutationResult }
  | { kind: 'star'; result: GalleryItemMutationResult; starred: boolean };

/** A list window's pages, or the starred strip's single page. */
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

  return isGalleryStarredStripQueryKey(query.queryKey) && isGalleryItemsPage(data) ? [data] : [];
};

const mapPageItems = (
  page: GalleryItemsPage,
  mapItem: (item: GalleryItem) => GalleryItem | null,
  totalDelta = 0
): GalleryItemsPage => {
  let changed = false;
  const items: GalleryItem[] = [];

  for (const item of page.items) {
    const nextItem = mapItem(item);

    if (nextItem !== item) {
      changed = true;
    }
    if (nextItem) {
      items.push(nextItem);
    }
  }

  if (!changed && totalDelta === 0) {
    return page;
  }

  return {
    ...page,
    items: changed ? items : page.items,
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

const countRemovedItems = (page: GalleryItemsPage, itemKeys: ReadonlySet<GalleryItemKey>): number =>
  page.items.filter((item) => itemKeys.has(toGalleryItemKey(item))).length;

const patchItemsInfiniteData = (
  data: InfiniteData<GalleryItemsPage, number>,
  filter: CanonicalGalleryItemsFilter,
  patch: GalleryItemCachePatch,
  itemKeys: ReadonlySet<GalleryItemKey>
): InfiniteData<GalleryItemsPage, number> => {
  const removedItemKeys = new Set<GalleryItemKey>();

  if (patchRemovesItems(filter, patch)) {
    for (const page of data.pages) {
      for (const item of page.items) {
        const key = toGalleryItemKey(item);

        if (itemKeys.has(key)) {
          removedItemKeys.add(key);
        }
      }
    }
  }

  let changed = false;
  const pages = data.pages.map((page) => {
    const nextPage = patchItemPage(page, filter, patch, itemKeys, removedItemKeys.size);
    changed ||= nextPage !== page;

    return nextPage;
  });

  return changed ? { ...data, pages } : data;
};

const patchItemsCacheData = (
  query: Query,
  filter: CanonicalGalleryItemsFilter,
  patch: GalleryItemCachePatch,
  itemKeys: ReadonlySet<GalleryItemKey>
): { after: GalleryItemsCacheData; before: GalleryItemsCacheData } | null => {
  const before = query.state.data;

  if (isGalleryItemsData(before)) {
    return { after: patchItemsInfiniteData(before, filter, patch, itemKeys), before };
  }

  // A newly starred item is left to the trailing refetch, which knows where
  // it belongs chronologically in the strip.
  if (isGalleryStarredStripQueryKey(query.queryKey) && isGalleryItemsPage(before)) {
    return { after: patchItemPage(before, filter, patch, itemKeys, countRemovedItems(before, itemKeys)), before };
  }

  return null;
};

/**
 * Applies only backend-confirmed successes. Failed refs are intentionally
 * ignored, and kind-qualified keys prevent same-name images/videos colliding.
 */
export const patchGalleryItemCaches = (client: QueryClient, patch: GalleryItemCachePatch): (() => void) => {
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

  for (const query of getGalleryItemListQueries(client)) {
    const filter = getGalleryItemsFilterFromKey(query.queryKey);
    const patched = filter ? patchItemsCacheData(query, filter, patch, itemKeys) : null;

    if (!patched || patched.after === patched.before) {
      continue;
    }

    const applied = client.setQueryData<GalleryItemsCacheData>(query.queryKey, patched.after);

    if (applied) {
      rollbackEntries.push({ after: applied, before: patched.before, queryKey: query.queryKey });
    }
  }

  return () => {
    rollbackClusterMembers?.();
    rollBackUnclaimedEntries(
      rollbackEntries,
      (entry) => client.getQueryData<GalleryItemsCacheData>(entry.queryKey),
      (entry) => client.setQueryData(entry.queryKey, entry.before)
    );
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
  changes: Partial<Pick<GalleryBoard, 'archived' | 'name'>>
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

  const pages: GalleryItemsPage[] = [];

  for (let index = 0; index < result.items.length; index += GALLERY_PAGE_SIZE) {
    pages.push({ items: result.items.slice(index, index + GALLERY_PAGE_SIZE), total: result.total });
  }

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
