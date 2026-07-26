import type { GalleryImage, GalleryImagesPage } from '@features/gallery/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { InfiniteData, QueryClient, QueryKey } from '@tanstack/react-query';

import { captureAccountScope } from '@platform/state/accountLifecycle';
import { hashKey } from '@tanstack/react-query';

import { ALL_READABLE_BOARDS_ID, isDateBoardId } from './backend';
import {
  galleryKeys,
  getGalleryImageListQueries,
  getGalleryImagesFilterFromKey,
  type CanonicalGalleryImagesFilter,
} from './queries';

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
  includeBoards: boolean
): Promise<void> => {
  // Date-board page queries depend on the supporting names query. Mark that
  // stale first so an active page refetch cannot observe the old name list.
  await client.cancelQueries({ queryKey: galleryKeys.dateBoardNamesForAccount(owner) });
  await client.invalidateQueries({
    queryKey: galleryKeys.dateBoardNamesForAccount(owner),
    refetchType: 'none',
  });
  await client.cancelQueries({ queryKey: galleryKeys.imageListsForAccount(owner) });

  // Refetching an infinite query replays every retained page. Collapse each
  // logical window to its pinned page first; users can explicitly load the
  // surrounding pages again, while a mutation costs one list request.
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

  await client.invalidateQueries({ queryKey: galleryKeys.imageListsForAccount(owner) });

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
  const ownerKey = hashKey(galleryKeys.imageListsForAccount(owner));
  const clientStates = galleryInvalidations.get(client) ?? new Map<string, GalleryInvalidationState>();
  const state = clientStates.get(ownerKey) ?? { includeBoards: false, promise: null, requested: false };

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
        // No await occurs between the final loop check and this cleanup, so a
        // caller cannot observe a resolved runner while leaving requested work
        // stranded behind it.
        state.promise = null;
        clientStates.delete(ownerKey);
      }
    })();
  }

  return state.promise;
};

export const invalidateGalleryImages = (
  client: QueryClient,
  owner: AccountScope = captureAccountScope()
): Promise<void> => scheduleGalleryInvalidation(client, owner, false);

export const invalidateGallery = (client: QueryClient, owner: AccountScope = captureAccountScope()): Promise<void> =>
  scheduleGalleryInvalidation(client, owner, true);
