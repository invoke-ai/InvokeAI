import type {
  GalleryItem,
  GalleryItemKey,
  GalleryItemMutationResult,
  GalleryItemRef,
  GalleryItemsPage,
} from '@features/gallery/core/items';
import type { GalleryBoard, GalleryGeneralAccess } from '@features/gallery/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { InfiniteData, Query, QueryClient, QueryKey } from '@tanstack/react-query';

import { canOnBoard } from '@features/gallery/core/boardAccess';
import { toGalleryItemKey } from '@features/gallery/core/items';
import { pruneImageClusterMembers } from '@features/gallery/core/semanticImageQuery';
import { queryClient } from '@platform/query/client';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { rollBackUnclaimedEntries } from '@platform/state/compareAndSwapRollback';
import { hashKey } from '@tanstack/react-query';

import { ALL_READABLE_BOARDS_ID, isDateBoardId } from './backend';
import {
  fetchGalleryItemsRange,
  GALLERY_PAGE_SIZE,
  galleryBoardsOptions,
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

/** Every cached page of a listing must lose the same count, or sparse views clamp against each other. */
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

export type GalleryItemStarPatchEvent =
  | { itemKeys: ReadonlySet<GalleryItemKey>; kind: 'apply'; patchId: number; starred: boolean }
  | { kind: 'revert'; patchId: number };

type GalleryItemStarPatchListener = (event: GalleryItemStarPatchEvent) => void;

const galleryItemStarPatchListeners = new WeakMap<QueryClient, Set<GalleryItemStarPatchListener>>();
let nextGalleryItemStarPatchId = 0;

/** An item on an evicted page is in no cache, so its flag can only be reconciled from the patch itself. */
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
  /** `holder` lowers only entries still holding a removed item, to re-apply a patch that lowered the rest. */
  totals?: 'holder' | 'listing';
}

/** Applies only backend-confirmed successes; kind-qualified keys keep same-name media apart. */
export const patchGalleryItemCaches = (
  client: QueryClient,
  patch: GalleryItemCachePatch,
  { totals = 'listing' }: GalleryItemCachePatchOptions = {}
): (() => void) => {
  const itemKeys = new Set(patch.result.succeeded.map(toGalleryItemKey));

  if (itemKeys.size === 0) {
    return () => undefined;
  }

  // The cluster member list is client-owned, so no refetch reconciles it: prune and roll it back with the list caches.
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

/** Captured before optimistic moves so rejected refs can be restored before refetch. */
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

/** Failed batches restore actual prior values, never invert. */
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

export const getGalleryProjectBoardsFromCaches = (client: QueryClient, projectId: string): GalleryBoard[] => {
  const boards = new Map<string, GalleryBoard>();

  for (const query of client
    .getQueryCache()
    .findAll({ queryKey: galleryKeys.boardsForAccount(captureAccountScope()) })) {
    const data = query.state.data;

    if (!isGalleryBoardsData(data)) {
      continue;
    }

    for (const board of data) {
      if (board.projectId === projectId && !boards.has(board.id)) {
        boards.set(board.id, board);
      }
    }
  }

  return [...boards.values()];
};

const isArchivedBoardsListing = (query: Query): boolean =>
  (query.queryKey.at(-1) as { includeArchived?: boolean } | undefined)?.includeArchived === true;

const getCachedGalleryBoardListings = (client: QueryClient): Query[] =>
  client
    .getQueryCache()
    .findAll({ queryKey: galleryKeys.boardsForAccount(captureAccountScope()) })
    // A listing whose refetch failed keeps deciding from its last data, rather than sending every result to the inbox.
    .filter((query) => isGalleryBoardsData(query.state.data))
    // Of two read in the same instant, the one not invalidated since wins.
    .sort(
      (left, right) =>
        right.state.dataUpdatedAt - left.state.dataUpdatedAt ||
        Number(left.state.isInvalidated) - Number(right.state.isInvalidated)
    );

/** Explains a `canAddToCachedGalleryBoard` refusal from the listing that decided, not a panel's older view. */
export const getCachedGalleryBoard = (boardId: string, client: QueryClient = queryClient): GalleryBoard | undefined => {
  for (const listing of getCachedGalleryBoardListings(client)) {
    const board = (listing.state.data as GalleryBoard[]).find((candidate) => candidate.id === boardId);
    if (board) {
      return board;
    }
  }
  return undefined;
};

/**
 * Newest cached listing decides. A board listed with `add_media` is a destination even archived; a board no listing
 * holds is not, but absence counts only from a listing that would have held it. Failed refetches keep their data.
 */
export const canAddToCachedGalleryBoard = (boardId: string, client: QueryClient = queryClient): boolean => {
  let isMissingFromNewerLiveListing = false;

  for (const listing of getCachedGalleryBoardListings(client)) {
    const board = (listing.state.data as GalleryBoard[]).find((candidate) => candidate.id === boardId);
    if (board) {
      return !(isMissingFromNewerLiveListing && !board.archived) && canOnBoard(board, 'add_media');
    }
    if (isArchivedBoardsListing(listing)) {
      return false;
    }
    isMissingFromNewerLiveListing = true;
  }
  return false;
};

export const getGalleryBoardDestinationIds = (client: QueryClient = queryClient): readonly string[] => {
  const decided = new Set<string>();
  const destinations: string[] = [];
  let hasNewerLiveListing = false;
  for (const listing of getCachedGalleryBoardListings(client)) {
    for (const board of listing.state.data as GalleryBoard[]) {
      if (decided.has(board.id)) {
        continue;
      }
      decided.add(board.id);
      if ((!hasNewerLiveListing || board.archived) && canOnBoard(board, 'add_media')) {
        destinations.push(board.id);
      }
    }
    if (isArchivedBoardsListing(listing)) {
      break;
    }
    hasNewerLiveListing = true;
  }
  return destinations;
};

const GALLERY_BOARD_DESTINATIONS_STALE_MS = 60_000;

/**
 * Null while any all-boards listing is cached, whatever its age, so synchronous callers decide from it; stale ones are
 * refreshed in the background, never awaited. With nothing cached and no read possible, results go to the inbox.
 */
export const ensureGalleryBoardDestinations = (client: QueryClient = queryClient): Promise<void> | null => {
  const options = galleryBoardsOptions({ includeArchived: true });
  // Any listing including archived boards will do, whatever its order.
  const listings = client
    .getQueryCache()
    .findAll({ queryKey: galleryKeys.boardsForAccount(captureAccountScope()) })
    .filter((query) => isArchivedBoardsListing(query) && isGalleryBoardsData(query.state.data));

  if (listings.length === 0) {
    return client
      .fetchQuery({ ...options, staleTime: GALLERY_BOARD_DESTINATIONS_STALE_MS })
      .then(() => undefined)
      .catch(() => {
        // Unanswered: the cached listings decide.
      });
  }

  const now = Date.now();
  const hasCurrentListing = listings.some(
    (query) => !query.state.isInvalidated && now - query.state.dataUpdatedAt < GALLERY_BOARD_DESTINATIONS_STALE_MS
  );

  if (!hasCurrentListing) {
    void client.prefetchQuery({ ...options, staleTime: GALLERY_BOARD_DESTINATIONS_STALE_MS });
  }

  return null;
};

/** Refetches the all-boards listing at once: nothing observes it, and `ensureGalleryBoardDestinations` reads cache. */
export const invalidateGalleryBoardDestinations = async (
  client: QueryClient,
  owner: AccountScope = captureAccountScope()
): Promise<void> => {
  await client.invalidateQueries({ queryKey: galleryKeys.boardsForAccount(owner) });
  if (!isAccountScopeCurrent(owner)) {
    return;
  }
  const isReadInFlight = client
    .getQueryCache()
    .findAll({ queryKey: galleryKeys.boardsForAccount(owner) })
    .some((query) => isArchivedBoardsListing(query) && query.state.fetchStatus === 'fetching');
  if (!isReadInFlight) {
    await client.prefetchQuery({
      ...galleryBoardsOptions({ includeArchived: true }),
      staleTime: GALLERY_BOARD_DESTINATIONS_STALE_MS,
    });
  }
};

/** In-flight listing reads predate the board and would drop it on landing, so they are cancelled first. */
export const addGalleryBoardToCaches = async (client: QueryClient, board: GalleryBoard): Promise<void> => {
  const owner = captureAccountScope();
  await client.cancelQueries({ queryKey: galleryKeys.boardsForAccount(owner) });
  for (const query of client.getQueryCache().findAll({ queryKey: galleryKeys.boardsForAccount(owner) })) {
    const boards = query.state.data;
    if (isGalleryBoardsData(boards) && !boards.some((candidate) => candidate.id === board.id)) {
      client.setQueryData<GalleryBoard[]>(query.queryKey, [...boards, board]);
    }
  }
};

/** The rollback skips lists written since, the same conflict rule as `patchGalleryItemCaches`. */
export const patchGalleryBoardCaches = (
  client: QueryClient,
  boardId: string,
  changes: Partial<Pick<GalleryBoard, 'archived' | 'name' | 'projectId'>> & { generalAccess?: GalleryGeneralAccess }
): (() => void) => {
  const { generalAccess, ...fields } = changes;
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
      return {
        ...board,
        ...fields,
        ...(generalAccess !== undefined && board.access ? { access: { ...board.access, generalAccess } } : {}),
      };
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

/** False (failed read, changed entry) collapses instead. */
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

  // Name-hydrated windows fetch videos singly; re-reading a video-heavy span costs more than collapsing.
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

  // A page fetch started during the span read will land after this swap; only an idle, untouched entry may take it.
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

  if (pages.length === 0) {
    pages.push({ items: [], total: result.total });
  }

  client.setQueryData<InfiniteData<GalleryItemsPage, number>>(query.queryKey, {
    pageParams: pages.map((_, index) => span.offset + index * GALLERY_PAGE_SIZE),
    pages,
  });

  return true;
};

const collapseGalleryItemWindowToAnchor = (client: QueryClient, query: Query): void => {
  // A transient entry (gcTime 0 on anchored windows) may be collected while a rebuild awaits; a write would revive it.
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
  // Mark shared name caches stale first so date-board pages and range selection cannot hydrate stale refs.
  await client.cancelQueries({ queryKey: galleryKeys.itemNamesForAccount(owner) });
  await client.invalidateQueries({
    queryKey: galleryKeys.itemNamesForAccount(owner),
    refetchType: 'none',
  });
  await client.cancelQueries({ queryKey: galleryKeys.itemListsForAccount(owner) });

  const rebuiltQueryHashes = new Set<string>();

  // Active windows rebuild in one span to preserve viewport rows.
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

/** Coalesces same-tick invalidations so observed refetches are not restarted. */
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
