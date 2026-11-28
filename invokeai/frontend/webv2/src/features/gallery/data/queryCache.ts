import type {
  GalleryItem,
  GalleryItemKey,
  GalleryItemMutationResult,
  GalleryItemRef,
  GalleryItemsPage,
} from '@features/gallery/core/items';
import type { GalleryMarks, GalleryMarksPatch } from '@features/gallery/core/marks';
import type { GalleryBoard, GalleryGeneralAccess } from '@features/gallery/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { InfiniteData, Query, QueryClient, QueryKey } from '@tanstack/react-query';

import { canOnBoard } from '@features/gallery/core/boardAccess';
import { compareGalleryItems, toGalleryItemKey } from '@features/gallery/core/items';
import { applyGalleryMarksPatch, getGalleryItemMarks, matchesGalleryListingMarks } from '@features/gallery/core/marks';
import { pruneImageClusterMembers } from '@features/gallery/core/semanticImageQuery';
import { queryClient } from '@platform/query/client';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { rollBackUnclaimedEntries } from '@platform/state/compareAndSwapRollback';
import { hashKey, partialMatchKey } from '@tanstack/react-query';

import { ALL_READABLE_BOARDS_ID, isDateBoardId } from './backend';
import {
  fetchGalleryItemsRange,
  GALLERY_PAGE_SIZE,
  GALLERY_STARRED_STRIP_LIMIT,
  galleryBoardsOptions,
  galleryKeys,
  getGalleryItemListingKey,
  getGalleryItemListQueries,
  getGalleryItemsFilterFromKey,
  isGallerySinglePageQueryKey,
  isGalleryStarredStripQueryKey,
  type CanonicalGalleryItemsFilter,
} from './queries';

export type GalleryItemCachePatch =
  | { kind: 'delete'; result: GalleryItemMutationResult }
  | { boardId: string; kind: 'move'; result: GalleryItemMutationResult }
  | { kind: 'marks'; patch: GalleryMarksPatch; result: GalleryItemMutationResult };

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

/** The item as the patched listing holds it, or null when the patch takes it out of that listing. */
const getPatchedItem = (
  filter: CanonicalGalleryItemsFilter,
  patch: GalleryItemCachePatch,
  item: GalleryItem
): GalleryItem | null => {
  if (patch.kind === 'delete') {
    return null;
  }

  if (patch.kind === 'marks') {
    const marked = applyGalleryMarksPatch(item, patch.patch);

    // A semantic listing keeps no marks filter, so its canonical filter has none to fail.
    return matchesGalleryListingMarks(filter, getGalleryItemMarks(marked)) ? marked : null;
  }

  if (!listsBoard(filter, patch.boardId)) {
    return null;
  }

  return item.boardId === patch.boardId ? item : { ...item, boardId: patch.boardId };
};

const listsBoard = (filter: CanonicalGalleryItemsFilter, boardId: string): boolean =>
  filter.semantic?.kind === 'cluster' ||
  filter.boardId === ALL_READABLE_BOARDS_ID ||
  filter.boardId === boardId ||
  isDateBoardId(filter.boardId);

const patchItemPage = (
  page: GalleryItemsPage,
  filter: CanonicalGalleryItemsFilter,
  patch: GalleryItemCachePatch,
  itemKeys: ReadonlySet<GalleryItemKey>,
  removedItemCount: number
): GalleryItemsPage =>
  mapPageItems(
    page,
    (item) => (itemKeys.has(toGalleryItemKey(item)) ? getPatchedItem(filter, patch, item) : item),
    removedItemCount
  );

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

const patchItemsCacheValue = (
  queryKey: QueryKey,
  before: unknown,
  filter: CanonicalGalleryItemsFilter,
  patch: GalleryItemCachePatch,
  itemKeys: ReadonlySet<GalleryItemKey>,
  removedItemCount: number
): { after: GalleryItemsCacheData; before: GalleryItemsCacheData } | null => {
  if (isGalleryItemsData(before)) {
    return { after: patchItemsInfiniteData(before, filter, patch, itemKeys, removedItemCount), before };
  }

  // New items are left to the trailing refetch so their server ordering is preserved.
  if (isGallerySinglePageQueryKey(queryKey) && isGalleryItemsPage(before)) {
    return { after: patchItemPage(before, filter, patch, itemKeys, removedItemCount), before };
  }

  return null;
};

const patchItemsCacheData = (
  query: Query,
  filter: CanonicalGalleryItemsFilter,
  patch: GalleryItemCachePatch,
  itemKeys: ReadonlySet<GalleryItemKey>,
  removedItemCount: number
): { after: GalleryItemsCacheData; before: GalleryItemsCacheData } | null =>
  patchItemsCacheValue(query.queryKey, query.state.data, filter, patch, itemKeys, removedItemCount);

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

    if (!filter) {
      continue;
    }

    const countKey = getCountKey(query);
    const removedKeys = removedKeysByCountKey.get(countKey) ?? new Set<GalleryItemKey>();

    removedKeysByCountKey.set(countKey, removedKeys);
    for (const page of getCachedPages(query)) {
      for (const item of page.items) {
        const key = toGalleryItemKey(item);

        if (itemKeys.has(key) && getPatchedItem(filter, patch, item) === null) {
          removedKeys.add(key);
        }
      }
    }
  }

  return new Map([...removedKeysByCountKey].map(([countKey, keys]) => [countKey, keys.size]));
};

export type GalleryItemMarksPatchEvent =
  | { itemKeys: ReadonlySet<GalleryItemKey>; kind: 'apply'; patch: GalleryMarksPatch; patchId: number }
  | { kind: 'revert'; patchId: number };

type GalleryItemMarksPatchListener = (event: GalleryItemMarksPatchEvent) => void;

const galleryItemMarksPatchListeners = new WeakMap<QueryClient, Set<GalleryItemMarksPatchListener>>();
let nextGalleryItemMarksPatchId = 0;

/** An item on an evicted page is in no cache, so its marks can only be reconciled from the patch itself. */
export const subscribeGalleryItemMarksPatches = (
  client: QueryClient,
  listener: GalleryItemMarksPatchListener
): (() => void) => {
  let listeners = galleryItemMarksPatchListeners.get(client);

  if (!listeners) {
    listeners = new Set();
    galleryItemMarksPatchListeners.set(client, listeners);
  }

  listeners.add(listener);

  return () => listeners.delete(listener);
};

const emitGalleryItemMarksPatch = (client: QueryClient, event: GalleryItemMarksPatchEvent): void =>
  galleryItemMarksPatchListeners.get(client)?.forEach((listener) => listener(event));

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
  const marksPatchId = patch.kind === 'marks' ? ++nextGalleryItemMarksPatchId : null;

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

  if (patch.kind === 'marks' && marksPatchId !== null) {
    emitGalleryItemMarksPatch(client, { itemKeys, kind: 'apply', patch: patch.patch, patchId: marksPatchId });
  }

  const restoreMarksFences = patch.kind === 'marks' ? null : narrowGalleryItemMarksFences(client, patch, itemKeys);

  return () => {
    restoreMarksFences?.();
    rollbackClusterMembers?.();
    rollBackUnclaimedEntries(
      rollbackEntries,
      (entry) => client.getQueryData<GalleryItemsCacheData>(entry.queryKey),
      (entry) => client.setQueryData(entry.queryKey, entry.before)
    );

    if (marksPatchId !== null) {
      emitGalleryItemMarksPatch(client, { kind: 'revert', patchId: marksPatchId });
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
export const getGalleryItemMarksFromCaches = (
  client: QueryClient,
  refs: readonly GalleryItemRef[]
): Map<GalleryItemKey, GalleryMarks> => {
  const wanted = new Set(refs.map(toGalleryItemKey));
  const marks = new Map<GalleryItemKey, GalleryMarks>();

  for (const query of getGalleryItemListQueries(client)) {
    if (marks.size === wanted.size) {
      break;
    }

    for (const page of getCachedPages(query)) {
      for (const item of page.items) {
        const key = toGalleryItemKey(item);

        if (wanted.has(key) && !marks.has(key)) {
          marks.set(key, getGalleryItemMarks(item));
        }
      }
    }
  }

  return marks;
};

/** An item a marks write targets, as a cached listing held it before the write, and the listings that held it. */
interface GalleryMarkedItem {
  item: GalleryItem;
  /** The filter of each listing that held it, by `getListingScope`. */
  scopes: ReadonlyMap<string, CanonicalGalleryItemsFilter>;
}

/** Listings with equal scopes differ only in marks, so an item one holds belongs to any other whose marks it passes. */
const getListingScope = (filter: CanonicalGalleryItemsFilter): string => {
  const scope: Partial<CanonicalGalleryItemsFilter> = { ...filter };

  delete scope.colorLabels;
  delete scope.flags;
  delete scope.maxRating;
  delete scope.minRating;
  delete scope.starred;

  return hashKey([scope]);
};

const getMarkedItems = (
  queries: readonly Query[],
  itemKeys: ReadonlySet<GalleryItemKey>
): Map<GalleryItemKey, GalleryMarkedItem> => {
  const marked = new Map<GalleryItemKey, { item: GalleryItem; scopes: Map<string, CanonicalGalleryItemsFilter> }>();

  for (const query of queries) {
    const filter = getGalleryItemsFilterFromKey(query.queryKey);

    if (!filter) {
      continue;
    }

    for (const page of getCachedPages(query)) {
      for (const item of page.items) {
        const key = toGalleryItemKey(item);

        if (itemKeys.has(key)) {
          const entry = marked.get(key) ?? { item, scopes: new Map<string, CanonicalGalleryItemsFilter>() };

          entry.scopes.set(getListingScope(filter), filter);
          marked.set(key, entry);
        }
      }
    }
  }

  return marked;
};

interface GalleryCachedValue {
  queryKey: QueryKey;
  value: unknown;
}

const getValuePages = ({ queryKey, value }: GalleryCachedValue): { page: GalleryItemsPage; start: number }[] => {
  if (isGalleryItemsData(value)) {
    return value.pages.map((page, index) => ({ page, start: page.offset ?? value.pageParams[index] ?? 0 }));
  }

  return isGallerySinglePageQueryKey(queryKey) && isGalleryItemsPage(value)
    ? [{ page: value, start: value.offset ?? (typeof queryKey[6] === 'number' ? queryKey[6] : 0) }]
    : [];
};

const getPageIndices = (page: GalleryItemsPage, start: number): number[] =>
  page.itemIndices ?? page.items.map((_, index) => start + index);

/** The indices a page answers for: an index in it that holds no item has none, or lost it to a patch. */
const getPageCapacity = (queryKey: QueryKey): number =>
  isGalleryStarredStripQueryKey(queryKey) ? GALLERY_STARRED_STRIP_LIMIT : GALLERY_PAGE_SIZE;

const isRangeLoaded = (ranges: readonly (readonly [number, number])[], from: number, to: number): boolean => {
  let reached = from;

  for (const [start, end] of [...ranges].sort(([left], [right]) => left - right)) {
    if (start <= reached && end > reached) {
      reached = end;
    }
  }

  return reached >= to;
};

/**
 * The index an item takes in a listing, decided from loaded pages alone: null when an index no page answers for lies
 * between its loaded neighbours, since an unloaded item there could sort on either side of it. It takes a vacant index
 * there when one is left (a patch removed its item), else shifts the indices from its own up by one.
 */
const findListingInsertIndex = (
  values: readonly GalleryCachedValue[],
  item: GalleryItem,
  orderDir: CanonicalGalleryItemsFilter['orderDir']
): { index: number; shift: boolean } | null => {
  const slots = new Map<number, GalleryItem>();
  const ranges: (readonly [number, number])[] = [];
  let total: number | undefined;

  for (const cached of values) {
    for (const { page, start } of getValuePages(cached)) {
      total ??= page.total;
      ranges.push([start, start + getPageCapacity(cached.queryKey)]);
      getPageIndices(page, start).forEach((index, position) => {
        const slotItem = page.items[position];

        if (slotItem && !slots.has(index)) {
          slots.set(index, slotItem);
        }
      });
    }
  }

  if (total === undefined) {
    return null;
  }

  const ordered = [...slots].sort(([left], [right]) => left - right);
  const nextPosition = ordered.findIndex(([, candidate]) => compareGalleryItems(item, candidate, { orderDir }) < 0);
  const previous = nextPosition === -1 ? ordered.at(-1) : ordered[nextPosition - 1];
  const next = nextPosition === -1 ? undefined : ordered[nextPosition];
  const from = previous ? previous[0] + 1 : 0;
  const to = next ? next[0] : total;

  return isRangeLoaded(ranges, from, to) ? { index: from, shift: next !== undefined && from >= to } : null;
};

/** Raises the value's totals by one and puts the item at `at` in the page answering for it, if any. */
const insertIntoValue = (
  cached: GalleryCachedValue,
  item: GalleryItem,
  at: { index: number; shift: boolean } | null
): unknown => {
  const capacity = getPageCapacity(cached.queryKey);
  const isStrip = isGalleryStarredStripQueryKey(cached.queryKey);
  const patchPage = (page: GalleryItemsPage, start: number): GalleryItemsPage => {
    const indices = getPageIndices(page, start);
    const shifted = at?.shift ? indices.map((index) => (index >= at.index ? index + 1 : index)) : indices;

    if (!at || at.index < start || at.index >= start + capacity) {
      return { ...page, ...(page.itemIndices ? { itemIndices: shifted } : {}), total: page.total + 1 };
    }

    const position = indices.filter((index) => index < at.index).length;
    const items = [...page.items.slice(0, position), item, ...page.items.slice(position)];

    return {
      ...page,
      // The strip stays as long as the server would list it.
      items: isStrip ? items.slice(0, capacity) : items,
      ...(page.itemIndices
        ? { itemIndices: [...shifted.slice(0, position), at.index, ...shifted.slice(position)] }
        : {}),
      total: page.total + 1,
    };
  };

  if (isGalleryItemsData(cached.value)) {
    const data = cached.value;

    return {
      ...data,
      pages: data.pages.map((page, index) => patchPage(page, page.offset ?? data.pageParams[index] ?? 0)),
    };
  }

  const [single] = getValuePages(cached);

  return single ? patchPage(single.page, single.start) : cached.value;
};

const holdsItem = (cached: GalleryCachedValue, key: GalleryItemKey): boolean =>
  getValuePages(cached).some(({ page }) => page.items.some((item) => toGalleryItemKey(item) === key));

/**
 * Puts each marked item into a listing of `values` (one listing's cached entries) its new marks bring it into, where
 * the loaded pages decide its place. `unplaced` reports an item that entered the listing with no place found;
 * `countUnplaced` still raises the totals for one, as every entry of a listing must agree on its total.
 */
const insertMarkedItems = (
  values: readonly GalleryCachedValue[],
  filter: CanonicalGalleryItemsFilter,
  marked: ReadonlyMap<GalleryItemKey, GalleryMarkedItem>,
  patch: GalleryMarksPatch,
  { countUnplaced }: { countUnplaced: boolean }
): { unplaced: boolean; values: GalleryCachedValue[] } => {
  const scope = getListingScope(filter);
  let current = [...values];
  let unplaced = false;
  let held: Set<GalleryItemKey> | null = null;

  for (const [key, { item, scopes }] of marked) {
    const markedItem = applyGalleryMarksPatch(item, patch);

    if (
      filter.semantic !== undefined ||
      !scopes.has(scope) ||
      matchesGalleryListingMarks(filter, getGalleryItemMarks(item)) ||
      !matchesGalleryListingMarks(filter, getGalleryItemMarks(markedItem)) ||
      (held ??= new Set(
        current.flatMap((cached) => getValuePages(cached).flatMap(({ page }) => page.items.map(toGalleryItemKey)))
      )).has(key)
    ) {
      continue;
    }

    const at = findListingInsertIndex(current, markedItem, filter.orderDir);
    const next = current.map((cached) => ({ ...cached, value: insertIntoValue(cached, markedItem, at) }));
    const placed = next.some((cached) => holdsItem(cached, key));

    unplaced ||= !placed;
    if (placed) {
      held.add(key);
    }
    if (placed || countUnplaced) {
      current = next;
    }
  }

  return { unplaced, values: current };
};

/** Moves marked items into the cached listings they now belong to; `unplaced` while a shown listing could not. */
const insertMarkedItemsIntoCaches = (
  client: QueryClient,
  marked: ReadonlyMap<GalleryItemKey, GalleryMarkedItem>,
  patch: GalleryMarksPatch
): { rollback: () => void; unplaced: boolean } => {
  const listings = new Map<string, { filter: CanonicalGalleryItemsFilter; queries: Query[] }>();

  for (const query of getGalleryItemListQueries(client)) {
    const filter = getGalleryItemsFilterFromKey(query.queryKey);

    if (filter && getValuePages({ queryKey: query.queryKey, value: query.state.data }).length > 0) {
      const listing = listings.get(getListingHash(query.queryKey)) ?? { filter, queries: [] };

      listing.queries.push(query);
      listings.set(getListingHash(query.queryKey), listing);
    }
  }

  const rollbackEntries: ItemCacheRollbackEntry[] = [];
  let unplaced = false;

  for (const { filter, queries } of listings.values()) {
    const before = queries.map((query) => ({ queryKey: query.queryKey, value: query.state.data }));
    const result = insertMarkedItems(before, filter, marked, patch, { countUnplaced: true });

    unplaced ||= result.unplaced && queries.some((query) => query.isActive());
    result.values.forEach((cached, index) => {
      const previous = before[index]?.value;

      if (cached.value === previous) {
        return;
      }

      const applied = client.setQueryData<GalleryItemsCacheData>(
        cached.queryKey,
        cached.value as GalleryItemsCacheData
      );

      if (applied) {
        rollbackEntries.push({ after: applied, before: previous as GalleryItemsCacheData, queryKey: cached.queryKey });
      }
    });
  }

  return {
    rollback: () =>
      rollBackUnclaimedEntries(
        rollbackEntries,
        (entry) => client.getQueryData<GalleryItemsCacheData>(entry.queryKey),
        (entry) => client.setQueryData(entry.queryKey, entry.before)
      ),
    unplaced,
  };
};

/** Patches every cached listing for a marks write, moving items between listings; rolled back as one. */
const paintGalleryItemMarks = (
  client: QueryClient,
  refs: readonly GalleryItemRef[],
  patch: GalleryMarksPatch
): { marked: Map<GalleryItemKey, GalleryMarkedItem>; rollback: () => void; unplaced: boolean } => {
  const marked = getMarkedItems(getGalleryItemListQueries(client), new Set(refs.map(toGalleryItemKey)));
  const rollbackCaches = patchGalleryItemCaches(client, {
    kind: 'marks',
    patch,
    result: { failed: [], succeeded: [...refs] },
  });
  const inserted = insertMarkedItemsIntoCaches(client, marked, patch);

  return {
    marked,
    // Insertions first: they were written over the patch, whose rollback skips entries that no longer show it.
    rollback: () => {
      inserted.rollback();
      rollbackCaches();
    },
    unplaced: inserted.unplaced,
  };
};

interface GalleryItemMarksFenceEntry {
  /** Narrowed to the items the server marked once it answers. */
  itemKeys: ReadonlySet<GalleryItemKey>;
  /** Narrowed by deletions and moves, so a read that omits an item for those reasons never gets it back. */
  marked: Map<GalleryItemKey, GalleryMarkedItem>;
  owner: AccountScope;
  patch: GalleryMarksPatch;
  /**
   * Null until the server answers; then the listing reads in flight, which may predate it. Later reads stand as read,
   * and the next listing invalidation drops the entry.
   */
  settledReads: ReadonlySet<Promise<unknown>> | null;
}

interface GalleryItemMarksRefetchState {
  boards: boolean;
  timer: ReturnType<typeof setTimeout> | null;
  /** A write in the burst asked for no delay; a later write's delay does not postpone it. */
  urgent: boolean;
}

const galleryItemMarksFences = new WeakMap<QueryClient, Map<number, GalleryItemMarksFenceEntry>>();
const galleryItemMarksRefetches = new WeakMap<QueryClient, WeakMap<AccountScope, GalleryItemMarksRefetchState>>();
let nextGalleryItemMarksFenceId = 0;

/**
 * Re-applies, in write order, the marks the server may not have had when this listing data was read. Without `read`,
 * the data was read after the last invalidation dropped the entries settled before it.
 */
const applyGalleryItemMarksFence = (
  client: QueryClient,
  queryKey: QueryKey,
  data: unknown,
  read?: Promise<unknown>
): void => {
  const entries = galleryItemMarksFences.get(client);

  if (!entries?.size) {
    return;
  }

  const owner = captureAccountScope();
  const filter = getGalleryItemsFilterFromKey(queryKey);

  if (!filter || !partialMatchKey(queryKey, galleryKeys.itemListsForAccount(owner))) {
    return;
  }

  let current = data;

  for (const entry of entries.values()) {
    if (entry.owner !== owner || (read && entry.settledReads && !entry.settledReads.has(read))) {
      continue;
    }

    const patch: GalleryItemCachePatch = { kind: 'marks', patch: entry.patch, result: { failed: [], succeeded: [] } };
    // Only items this data still holds lower its totals, so data read after the server had the marks stays as it is.
    const removed = new Set(
      (isGalleryItemsData(current) ? current.pages : isGalleryItemsPage(current) ? [current] : [])
        .flatMap((page) => page.items)
        .filter((item) => entry.itemKeys.has(toGalleryItemKey(item)) && getPatchedItem(filter, patch, item) === null)
        .map(toGalleryItemKey)
    ).size;

    current = patchItemsCacheValue(queryKey, current, filter, patch, entry.itemKeys, removed)?.after ?? current;
    // Only where this data decides the place: data read after the server had the mark may list it elsewhere.
    current =
      insertMarkedItems(
        [{ queryKey, value: current }],
        filter,
        new Map([...entry.marked].filter(([key]) => entry.itemKeys.has(key))),
        entry.patch,
        { countUnplaced: false }
      ).values[0]?.value ?? current;
  }

  if (current !== data) {
    client.setQueryData(queryKey, current);
  }
};

const getGalleryItemMarksFence = (client: QueryClient): Map<number, GalleryItemMarksFenceEntry> => {
  let entries = galleryItemMarksFences.get(client);

  if (!entries) {
    entries = new Map();
    galleryItemMarksFences.set(client, entries);
    client.getQueryCache().subscribe((event) => {
      if (event.type === 'updated' && event.action.type === 'success' && !event.action.manual) {
        applyGalleryItemMarksFence(client, event.query.queryKey, event.query.state.data, event.query.promise);
      }
    });
  }

  return entries;
};

const moveMarkedItem = ({ item, scopes }: GalleryMarkedItem, boardId: string): GalleryMarkedItem => ({
  item: { ...item, boardId },
  scopes: new Map(
    [...scopes.values()].map((filter) => {
      const moved = listsBoard(filter, boardId) ? filter : { ...filter, boardId };

      return [getListingScope(moved), moved];
    })
  ),
});

/** Takes deleted or moved items out of the listings the fence would put them into; the rollback puts back each one. */
const narrowGalleryItemMarksFences = (
  client: QueryClient,
  patch: Exclude<GalleryItemCachePatch, { kind: 'marks' }>,
  itemKeys: ReadonlySet<GalleryItemKey>
): (() => void) => {
  const restores: (() => void)[] = [];

  for (const entry of galleryItemMarksFences.get(client)?.values() ?? []) {
    for (const key of itemKeys) {
      const prior = entry.marked.get(key);

      if (!prior) {
        continue;
      }

      const next = patch.kind === 'move' ? moveMarkedItem(prior, patch.boardId) : null;

      if (next) {
        entry.marked.set(key, next);
      } else {
        entry.marked.delete(key);
      }
      restores.push(() => {
        if ((entry.marked.get(key) ?? null) === next) {
          entry.marked.set(key, prior);
        }
      });
    }
  }

  return () => restores.forEach((restore) => restore());
};

/** Entries the server has answered are in any listing read after the invalidation that just cancelled older reads. */
const releaseSettledGalleryItemMarks = (client: QueryClient, owner: AccountScope): void => {
  const entries = galleryItemMarksFences.get(client);

  for (const [id, entry] of entries ?? []) {
    if ((entry.owner === owner && entry.settledReads !== null) || !isAccountScopeCurrent(entry.owner)) {
      entries?.delete(id);
    }
  }
};

/** One listing refetch once no marks are pending and `delayMs` has passed since the last write settled. */
const scheduleGalleryItemMarksRefetch = (
  client: QueryClient,
  owner: AccountScope,
  { boards, delayMs }: { boards: boolean; delayMs: number }
): void => {
  const states = galleryItemMarksRefetches.get(client) ?? new WeakMap<AccountScope, GalleryItemMarksRefetchState>();
  const state = states.get(owner) ?? { boards: false, timer: null, urgent: false };

  galleryItemMarksRefetches.set(client, states);
  states.set(owner, state);
  state.boards ||= boards;
  state.urgent ||= delayMs === 0;
  if (state.timer !== null) {
    clearTimeout(state.timer);
  }
  state.timer = setTimeout(
    () => {
      state.timer = null;

      const isPending = [...(galleryItemMarksFences.get(client)?.values() ?? [])].some(
        (entry) => entry.owner === owner && entry.settledReads === null
      );

      // A pending write schedules again when it settles.
      if (!isAccountScopeCurrent(owner) || isPending) {
        return;
      }

      const includeBoards = state.boards;

      state.boards = false;
      state.urgent = false;
      void scheduleGalleryInvalidation(client, owner, includeBoards).catch(() => undefined);
    },
    state.urgent ? 0 : delayMs
  );
};

/** Items a write failed for, with the marks to put back: those it set, as they were before it. */
export interface GalleryItemMarksRestore {
  patch: GalleryMarksPatch;
  refs: readonly GalleryItemRef[];
}

export interface GalleryItemMarksCacheWrite {
  /**
   * The server has the marks, except for `failed` items: the write keeps its place among pending writes for the rest,
   * and each failed item gets its prior marks back unless a later pending write sets the same marks.
   */
  settle: (failed?: readonly GalleryItemMarksRestore[]) => void;
  /** Restores the listings that still show this write and stops keeping it over refetches. */
  rollback: () => void;
}

/**
 * Optimistic marks for every cached listing, kept over any listing read that lands before the server has them, so a
 * stale refetch never undoes a newer mark. An item its marks move into another loaded listing (the Bookmarked strip and
 * the grid) goes there at its place; one whose place the loaded pages cannot tell is left to a refetch without delay.
 * Each write that settles or rolls back schedules one coalesced refetch: after `refetchDelayMs` with no write pending,
 * including boards when any write in the burst asked for them.
 */
export const writeGalleryItemMarks = (
  client: QueryClient,
  refs: readonly GalleryItemRef[],
  patch: GalleryMarksPatch,
  { boards, refetchDelayMs }: { boards: boolean; refetchDelayMs: number }
): GalleryItemMarksCacheWrite => {
  const owner = captureAccountScope();
  const entries = getGalleryItemMarksFence(client);
  const id = ++nextGalleryItemMarksFenceId;

  for (const [staleId, staleEntry] of entries) {
    if (!isAccountScopeCurrent(staleEntry.owner)) {
      entries.delete(staleId);
    }
  }

  const painted = paintGalleryItemMarks(client, refs, patch);
  const entry: GalleryItemMarksFenceEntry = {
    itemKeys: new Set(refs.map(toGalleryItemKey)),
    marked: painted.marked,
    owner,
    patch,
    settledReads: null,
  };
  let isOpen = true;

  entries.set(id, entry);

  const isSetLater = (key: GalleryItemKey, restore: GalleryMarksPatch): boolean =>
    [...entries].some(
      ([laterId, later]) =>
        laterId > id &&
        later.owner === owner &&
        later.itemKeys.has(key) &&
        (Object.keys(restore) as (keyof GalleryMarksPatch)[]).some((field) => later.patch[field] !== undefined)
    );

  return {
    rollback: () => {
      if (!isOpen) {
        return;
      }
      isOpen = false;
      entries.delete(id);
      painted.rollback();
      scheduleGalleryItemMarksRefetch(client, owner, { boards, delayMs: 0 });
    },
    settle: (failed = []) => {
      if (!isOpen) {
        return;
      }
      isOpen = false;

      const failedKeys = new Set(failed.flatMap(({ refs: failedRefs }) => failedRefs.map(toGalleryItemKey)));

      entry.itemKeys = new Set([...entry.itemKeys].filter((key) => !failedKeys.has(key)));
      entry.settledReads = new Set(getGalleryItemListQueries(client, owner).flatMap((query) => query.promise ?? []));
      for (const restore of failed) {
        const restorable = restore.refs.filter((ref) => !isSetLater(toGalleryItemKey(ref), restore.patch));

        if (restorable.length > 0 && Object.keys(restore.patch).length > 0) {
          paintGalleryItemMarks(client, restorable, restore.patch);
        }
      }
      scheduleGalleryItemMarksRefetch(client, owner, {
        boards,
        delayMs: painted.unplaced || failed.length > 0 ? 0 : refetchDelayMs,
      });
    },
  };
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

  const rebuilt: InfiniteData<GalleryItemsPage, number> = {
    pageParams: pages.map((_, index) => span.offset + index * GALLERY_PAGE_SIZE),
    pages,
  };

  client.setQueryData(query.queryKey, rebuilt);
  applyGalleryItemMarksFence(client, query.queryKey, rebuilt);

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
  releaseSettledGalleryItemMarks(client, owner);

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
