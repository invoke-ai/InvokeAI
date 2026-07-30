import type {
  GalleryItem,
  GalleryItemKey,
  GalleryItemMutationResult,
  GalleryItemsPage,
} from '@features/gallery/core/items';
import type { GalleryImage, GalleryImagesPage } from '@features/gallery/core/types';
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
  getGalleryImageListQueries,
  getGalleryImagesFilterFromKey,
  type CanonicalGalleryItemsFilter,
  type CanonicalGalleryImagesFilter,
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

    if (patch.kind === 'star' && filter.starredFirst) {
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

/** TODO(Task 5/7): Remove after image actions use confirmed mixed mutation results. */
export type GalleryImageCachePatch =
  | { imageNames: readonly string[]; kind: 'delete' }
  | { boardId: string; imageNames: readonly string[]; kind: 'move' }
  | { imageNames: readonly string[]; kind: 'star'; starred: boolean };

interface CacheRollbackEntry {
  after: InfiniteData<GalleryImagesPage, number>;
  before: InfiniteData<GalleryImagesPage, number>;
  queryKey: QueryKey;
}

const isGalleryImagesData = (value: unknown): value is InfiniteData<GalleryImagesPage, number> => {
  if (!value || typeof value !== 'object' || !('pages' in value) || !('pageParams' in value)) {
    return false;
  }

  const data = value as { pages?: unknown; pageParams?: unknown };

  return Array.isArray(data.pages) && Array.isArray(data.pageParams);
};

const mapPageImages = (
  page: GalleryImagesPage,
  mapImage: (image: GalleryImage) => GalleryImage | null,
  totalDelta = 0
): GalleryImagesPage => {
  let changed = false;
  const images: GalleryImage[] = [];

  for (const image of page.images) {
    const nextImage = mapImage(image);

    if (nextImage !== image) {
      changed = true;
    }
    if (nextImage) {
      images.push(nextImage);
    }
  }

  if (!changed && totalDelta === 0) {
    return page;
  }

  return {
    ...page,
    images: changed ? images : page.images,
    total: Math.max(0, page.total - totalDelta),
  };
};

const patchPage = (
  page: GalleryImagesPage,
  filter: CanonicalGalleryImagesFilter,
  patch: GalleryImageCachePatch,
  imageNames: ReadonlySet<string>,
  removedImageCount: number
): GalleryImagesPage => {
  if (patch.kind === 'star') {
    return mapPageImages(page, (image) => {
      if (!imageNames.has(image.imageName) || image.starred === patch.starred) {
        return image;
      }

      return { ...image, starred: patch.starred };
    });
  }

  if (patch.kind === 'delete') {
    return mapPageImages(page, (image) => (imageNames.has(image.imageName) ? null : image), removedImageCount);
  }

  const keepsMovedImages =
    filter.boardId === ALL_READABLE_BOARDS_ID || filter.boardId === patch.boardId || isDateBoardId(filter.boardId);

  return mapPageImages(
    page,
    (image) => {
      if (!imageNames.has(image.imageName)) {
        return image;
      }

      if (!keepsMovedImages) {
        return null;
      }

      return image.boardId === patch.boardId ? image : { ...image, boardId: patch.boardId };
    },
    keepsMovedImages ? 0 : removedImageCount
  );
};

const patchInfiniteData = (
  data: InfiniteData<GalleryImagesPage, number>,
  filter: CanonicalGalleryImagesFilter,
  patch: GalleryImageCachePatch
): InfiniteData<GalleryImagesPage, number> => {
  const imageNames = new Set(patch.imageNames);
  const removesImages =
    patch.kind === 'delete' ||
    (patch.kind === 'move' &&
      filter.boardId !== ALL_READABLE_BOARDS_ID &&
      filter.boardId !== patch.boardId &&
      !isDateBoardId(filter.boardId));
  const removedImageNames = new Set<string>();

  if (removesImages) {
    for (const page of data.pages) {
      for (const image of page.images) {
        if (imageNames.has(image.imageName)) {
          removedImageNames.add(image.imageName);
        }
      }
    }
  }

  let changed = false;
  const pages = data.pages.map((page) => {
    const nextPage = patchPage(page, filter, patch, imageNames, removedImageNames.size);
    changed ||= nextPage !== page;

    return nextPage;
  });

  return changed ? { ...data, pages } : data;
};

/**
 * Applies a small mutation to every currently cached page for this account.
 * The returned rollback is concurrency-safe: it only restores an entry while
 * the optimistic value is still the current cache value.
 */
export const patchGalleryImageCaches = (client: QueryClient, patch: GalleryImageCachePatch): (() => void) => {
  if (patch.imageNames.length === 0) {
    return () => undefined;
  }

  const rollbackEntries: CacheRollbackEntry[] = [];

  for (const query of getGalleryImageListQueries(client)) {
    const before = query.state.data;
    const filter = getGalleryImagesFilterFromKey(query.queryKey);

    if (!filter || !isGalleryImagesData(before)) {
      continue;
    }

    // A starred-first query cannot be patched coherently in place: changing
    // the flag may move an image across pages. Leave that cache untouched and
    // let the post-mutation invalidation rebuild its server-defined order.
    if (patch.kind === 'star' && filter.starredFirst) {
      continue;
    }

    const after = patchInfiniteData(before, filter, patch);

    if (after === before) {
      continue;
    }

    const applied = client.setQueryData<InfiniteData<GalleryImagesPage, number>>(query.queryKey, after);

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
  includeBoards: boolean,
  includeLegacyImages: boolean
): Promise<void> => {
  // Date-board pages and lazy range selection share these names. Mark them
  // stale before active pages refetch so they cannot hydrate stale refs.
  await client.cancelQueries({ queryKey: galleryKeys.itemNamesForAccount(owner) });
  await client.invalidateQueries({
    queryKey: galleryKeys.itemNamesForAccount(owner),
    refetchType: 'none',
  });
  await client.cancelQueries({ queryKey: galleryKeys.itemListsForAccount(owner) });
  if (includeLegacyImages) {
    await client.cancelQueries({ queryKey: galleryKeys.legacyImageListsForAccount(owner) });
  }

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

  if (includeLegacyImages) {
    for (const query of getGalleryImageListQueries(client, owner)) {
      const data = query.state.data;

      if (!isGalleryImagesData(data) || data.pages.length <= 1) {
        continue;
      }

      const anchorOffset =
        query.queryKey[5] === 'anchor' && typeof query.queryKey[6] === 'number' ? query.queryKey[6] : 0;
      const anchorIndex = Math.max(0, data.pageParams.indexOf(anchorOffset));

      client.setQueryData<InfiniteData<GalleryImagesPage, number>>(query.queryKey, {
        pageParams: [data.pageParams[anchorIndex] ?? anchorOffset],
        pages: [data.pages[anchorIndex] ?? data.pages[0]],
      });
    }
  }

  await client.invalidateQueries({ queryKey: galleryKeys.itemListsForAccount(owner) });

  if (includeLegacyImages) {
    await client.invalidateQueries({ queryKey: galleryKeys.legacyImageListsForAccount(owner) });
  }

  if (includeBoards) {
    await client.invalidateQueries({ queryKey: galleryKeys.boardsForAccount(owner) });
  }
};

interface GalleryInvalidationState {
  includeBoards: boolean;
  includeLegacyImages: boolean;
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
  includeBoards: boolean,
  includeLegacyImages: boolean
): Promise<void> => {
  const ownerKey = hashKey(galleryKeys.itemListsForAccount(owner));
  const clientStates = galleryInvalidations.get(client) ?? new Map<string, GalleryInvalidationState>();
  const state = clientStates.get(ownerKey) ?? {
    includeBoards: false,
    includeLegacyImages: false,
    promise: null,
    requested: false,
  };

  galleryInvalidations.set(client, clientStates);
  clientStates.set(ownerKey, state);
  state.includeBoards ||= includeBoards;
  state.includeLegacyImages ||= includeLegacyImages;
  state.requested = true;

  if (!state.promise) {
    state.promise = (async () => {
      try {
        // Let a synchronous burst collapse into one pass.
        await Promise.resolve();

        while (state.requested) {
          state.requested = false;
          const shouldInvalidateBoards = state.includeBoards;
          const shouldInvalidateLegacyImages = state.includeLegacyImages;

          state.includeBoards = false;
          state.includeLegacyImages = false;
          await runGalleryInvalidation(client, owner, shouldInvalidateBoards, shouldInvalidateLegacyImages);
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
): Promise<void> => scheduleGalleryInvalidation(client, owner, false, false);

/**
 * TODO(Task 5/7): Remove when legacy image consumers leave their isolated
 * query domain.
 */
export const invalidateGalleryImages = (
  client: QueryClient,
  owner: AccountScope = captureAccountScope()
): Promise<void> => scheduleGalleryInvalidation(client, owner, false, true);

export const invalidateGallery = (client: QueryClient, owner: AccountScope = captureAccountScope()): Promise<void> =>
  scheduleGalleryInvalidation(client, owner, true, true);
