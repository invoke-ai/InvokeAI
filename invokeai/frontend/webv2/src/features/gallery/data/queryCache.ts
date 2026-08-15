import type {
  GalleryItem,
  GalleryItemKey,
  GalleryItemMutationResult,
  GalleryItemRef,
  GalleryItemsPage,
} from '@features/gallery/core/items';
import type { GalleryBoard } from '@features/gallery/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { InfiniteData, QueryClient, QueryKey } from '@tanstack/react-query';

import { toGalleryItemKey } from '@features/gallery/core/items';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { hashKey } from '@tanstack/react-query';

import { ALL_READABLE_BOARDS_ID, isDateBoardId } from './backend';
import {
  galleryKeys,
  getGalleryItemListQueries,
  getGalleryItemsFilterFromKey,
  type CanonicalGalleryItemsFilter,
} from './queries';

export type GalleryItemCachePatch =
  | { kind: 'delete'; result: GalleryItemMutationResult }
  | { boardId: string; kind: 'move'; result: GalleryItemMutationResult }
  | { kind: 'star'; result: GalleryItemMutationResult; starred: boolean };

interface ItemCacheRollbackEntry {
  after: InfiniteData<GalleryItemsPage, number>;
  before: InfiniteData<GalleryItemsPage, number>;
  queryKey: QueryKey;
}

const isGalleryItemsData = (value: unknown): value is InfiniteData<GalleryItemsPage, number> => {
  if (!value || typeof value !== 'object' || !('pages' in value) || !('pageParams' in value)) {
    return false;
  }

  const data = value as { pages?: unknown; pageParams?: unknown };

  return Array.isArray(data.pages) && Array.isArray(data.pageParams);
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

const patchItemPage = (
  page: GalleryItemsPage,
  filter: CanonicalGalleryItemsFilter,
  patch: GalleryItemCachePatch,
  itemKeys: ReadonlySet<GalleryItemKey>,
  removedItemCount: number
): GalleryItemsPage => {
  if (patch.kind === 'star') {
    return mapPageItems(page, (item) => {
      if (!itemKeys.has(toGalleryItemKey(item)) || item.starred === patch.starred) {
        return item;
      }

      return { ...item, starred: patch.starred };
    });
  }

  if (patch.kind === 'delete') {
    return mapPageItems(page, (item) => (itemKeys.has(toGalleryItemKey(item)) ? null : item), removedItemCount);
  }

  const keepsMovedItems =
    filter.boardId === ALL_READABLE_BOARDS_ID || filter.boardId === patch.boardId || isDateBoardId(filter.boardId);

  return mapPageItems(
    page,
    (item) => {
      if (!itemKeys.has(toGalleryItemKey(item))) {
        return item;
      }

      if (!keepsMovedItems) {
        return null;
      }

      return item.boardId === patch.boardId ? item : { ...item, boardId: patch.boardId };
    },
    keepsMovedItems ? 0 : removedItemCount
  );
};

const patchItemsInfiniteData = (
  data: InfiniteData<GalleryItemsPage, number>,
  filter: CanonicalGalleryItemsFilter,
  patch: GalleryItemCachePatch,
  itemKeys: ReadonlySet<GalleryItemKey>
): InfiniteData<GalleryItemsPage, number> => {
  const removesItems =
    patch.kind === 'delete' ||
    (patch.kind === 'move' &&
      filter.boardId !== ALL_READABLE_BOARDS_ID &&
      filter.boardId !== patch.boardId &&
      !isDateBoardId(filter.boardId));
  const removedItemKeys = new Set<GalleryItemKey>();

  if (removesItems) {
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

/**
 * Applies only backend-confirmed successes. Failed refs are intentionally
 * ignored, and kind-qualified keys prevent same-name images/videos colliding.
 */
export const patchGalleryItemCaches = (client: QueryClient, patch: GalleryItemCachePatch): (() => void) => {
  const itemKeys = new Set(patch.result.succeeded.map(toGalleryItemKey));

  if (itemKeys.size === 0) {
    return () => undefined;
  }

  const rollbackEntries: ItemCacheRollbackEntry[] = [];

  for (const query of getGalleryItemListQueries(client)) {
    const before = query.state.data;
    const filter = getGalleryItemsFilterFromKey(query.queryKey);

    if (!filter || !isGalleryItemsData(before)) {
      continue;
    }

    const after = patchItemsInfiniteData(before, filter, patch, itemKeys);

    if (after === before) {
      continue;
    }

    const applied = client.setQueryData<InfiniteData<GalleryItemsPage, number>>(query.queryKey, after);

    if (applied) {
      rollbackEntries.push({ after: applied, before, queryKey: query.queryKey });
    }
  }

  return () => {
    for (const { after, before, queryKey } of rollbackEntries) {
      if (client.getQueryData(queryKey) === after) {
        client.setQueryData(queryKey, before);
      }
    }
  };
};

/**
 * Reads the board each requested item currently sits on, from whichever list
 * cache holds it. Optimistic moves capture this before patching so a rejected
 * ref can be put back without waiting for the reconciling refetch.
 */
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

    const data = query.state.data;

    if (!isGalleryItemsData(data)) {
      continue;
    }

    for (const page of data.pages) {
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
 * Reads the starred flag each requested item currently has, from whichever
 * list cache holds it. Optimistic star/unstar captures this before patching
 * so a totally-failed batch can restore each item's actual prior flag rather
 * than blanket-inverting the whole request.
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

    const data = query.state.data;

    if (!isGalleryItemsData(data)) {
      continue;
    }

    for (const page of data.pages) {
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

  return () => {
    for (const { after, before, queryKey } of rollbackEntries) {
      if (client.getQueryData(queryKey) === after) {
        client.setQueryData(queryKey, before);
      }
    }
  };
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

  // Refetching an infinite query replays every retained page. Collapse each
  // logical window to its pinned page first; users can explicitly load the
  // surrounding pages again, while a mutation costs one list request.
  for (const query of getGalleryItemListQueries(client, owner)) {
    const data = query.state.data;

    if (!isGalleryItemsData(data) || data.pages.length <= 1) {
      continue;
    }

    const anchorOffset =
      query.queryKey[5] === 'anchor' && typeof query.queryKey[6] === 'number' ? query.queryKey[6] : 0;
    const anchorIndex = Math.max(0, data.pageParams.indexOf(anchorOffset));

    client.setQueryData<InfiniteData<GalleryItemsPage, number>>(query.queryKey, {
      pageParams: [data.pageParams[anchorIndex] ?? anchorOffset],
      pages: [data.pages[anchorIndex] ?? data.pages[0]],
    });
  }

  await client.invalidateQueries({ queryKey: galleryKeys.itemListsForAccount(owner) });

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
 * Coalesces same-tick mutation/result bursts and performs at most one trailing
 * pass when another request arrives during an active invalidation. This avoids
 * repeatedly cancelling and restarting the same observed Gallery refetch.
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
