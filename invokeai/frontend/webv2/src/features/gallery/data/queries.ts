import type { GalleryItem, GalleryItemsPage } from '@features/gallery/core/items';
import type {
  GalleryBoardOrderBy,
  GalleryImage,
  GalleryImagesPage,
  GalleryOrderDir,
  GalleryView,
} from '@features/gallery/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { toGalleryItemKey } from '@features/gallery/core/items';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import {
  hashKey,
  infiniteQueryOptions,
  queryOptions,
  type InfiniteData,
  type QueryClient,
  type QueryKey,
} from '@tanstack/react-query';

import {
  type GalleryItemNames,
  hydrateGalleryDateBoardItemPage,
  isDateBoardId,
  listGalleryBoards,
  listGalleryDateBoardItemNames,
  listGalleryDateBoards,
  listGalleryItemNames,
  listGalleryItems,
  listPaletteImages,
} from './backend';

export const GALLERY_PAGE_SIZE = 60;
export const GALLERY_MAX_INFINITE_PAGES = 10;
export const GALLERY_MAX_ROWS = GALLERY_PAGE_SIZE * GALLERY_MAX_INFINITE_PAGES;

export interface GalleryBoardsQuery {
  includeArchived?: boolean;
  includeDateBoards?: boolean;
  orderBy?: GalleryBoardOrderBy;
  orderDir?: GalleryOrderDir;
}

interface CanonicalGalleryBoardsQuery {
  includeArchived: boolean;
  includeDateBoards: boolean;
  orderBy: GalleryBoardOrderBy;
  orderDir: GalleryOrderDir;
}

export interface GalleryItemsFilter {
  boardId: string;
  /** Inclusive lower-bound calendar day (YYYY-MM-DD) on created_at. */
  createdFrom?: string;
  /** Inclusive upper-bound calendar day (YYYY-MM-DD) on created_at. */
  createdTo?: string;
  galleryView: GalleryView;
  orderDir?: GalleryOrderDir;
  searchTerm: string;
  starredFirst?: boolean;
}

export interface CanonicalGalleryItemsFilter {
  boardId: string;
  createdFrom?: string;
  createdTo?: string;
  galleryView: GalleryView;
  orderDir: GalleryOrderDir;
  searchTerm: string;
  starredFirst: boolean;
}

export type GalleryItemsWindow = { kind: 'anchor'; offset: number } | { kind: 'infinite' };

interface GalleryAccountKey {
  accountId: string | null;
  epoch: number;
}

type GalleryItemsInfiniteQueryKey = readonly [
  'gallery',
  'items',
  'list',
  GalleryAccountKey,
  CanonicalGalleryItemsFilter,
];

type GalleryItemsAnchorQueryKey = readonly [...GalleryItemsInfiniteQueryKey, 'anchor', number];

export type GalleryItemsListQueryKey = GalleryItemsAnchorQueryKey | GalleryItemsInfiniteQueryKey;

type LegacyGalleryImagesInfiniteQueryKey = readonly [
  'gallery',
  'legacy-images',
  'list',
  GalleryAccountKey,
  CanonicalGalleryItemsFilter,
];

type LegacyGalleryImagesAnchorQueryKey = readonly [...LegacyGalleryImagesInfiniteQueryKey, 'anchor', number];

type LegacyGalleryImagesListQueryKey = LegacyGalleryImagesAnchorQueryKey | LegacyGalleryImagesInfiniteQueryKey;

const canonicalizeBoardsQuery = (query: GalleryBoardsQuery): CanonicalGalleryBoardsQuery => ({
  includeArchived: query.includeArchived ?? false,
  includeDateBoards: query.includeDateBoards ?? false,
  orderBy: query.orderBy ?? 'created_at',
  orderDir: query.orderDir ?? 'DESC',
});

export const canonicalizeGalleryItemsFilter = (filter: GalleryItemsFilter): CanonicalGalleryItemsFilter => ({
  boardId: filter.boardId,
  ...(filter.createdFrom ? { createdFrom: filter.createdFrom } : {}),
  ...(filter.createdTo ? { createdTo: filter.createdTo } : {}),
  galleryView: filter.galleryView,
  orderDir: filter.orderDir ?? 'DESC',
  searchTerm: filter.searchTerm.trim(),
  starredFirst: filter.starredFirst ?? false,
});

const getAccountKey = (owner: AccountScope): GalleryAccountKey => ({
  accountId: owner.accountId,
  epoch: owner.epoch,
});

const normalizePageOffset = (offset: number): number =>
  Math.max(0, Math.floor(offset / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE);

const getWindowKey = (window: GalleryItemsWindow): readonly [] | readonly ['anchor', number] =>
  window.kind === 'infinite' ? [] : ([window.kind, normalizePageOffset(window.offset)] as const);

export const galleryKeys = {
  all: ['gallery'] as const,
  boardsRoot: () => [...galleryKeys.all, 'boards'] as const,
  boardsForAccount: (owner: AccountScope) => [...galleryKeys.boardsRoot(), getAccountKey(owner)] as const,
  boards: (owner: AccountScope, query: CanonicalGalleryBoardsQuery) =>
    [...galleryKeys.boardsForAccount(owner), query] as const,
  itemsRoot: () => [...galleryKeys.all, 'items'] as const,
  itemListsRoot: () => [...galleryKeys.itemsRoot(), 'list'] as const,
  itemListsForAccount: (owner: AccountScope) => [...galleryKeys.itemListsRoot(), getAccountKey(owner)] as const,
  items: (
    owner: AccountScope,
    filter: CanonicalGalleryItemsFilter,
    window: GalleryItemsWindow = { kind: 'infinite' }
  ): GalleryItemsListQueryKey =>
    [...galleryKeys.itemListsForAccount(owner), filter, ...getWindowKey(window)] as GalleryItemsListQueryKey,
  itemNamesRoot: () => [...galleryKeys.itemsRoot(), 'names'] as const,
  itemNamesForAccount: (owner: AccountScope) => [...galleryKeys.itemNamesRoot(), getAccountKey(owner)] as const,
  itemNames: (owner: AccountScope, filter: CanonicalGalleryItemsFilter) =>
    [...galleryKeys.itemNamesForAccount(owner), filter] as const,

  /**
   * TODO(Task 5/7): Remove the legacy image query domain after Gallery state,
   * projection, and preview consumers use GalleryItem throughout.
   */
  legacyImagesRoot: () => [...galleryKeys.all, 'legacy-images'] as const,
  legacyImageListsRoot: () => [...galleryKeys.legacyImagesRoot(), 'list'] as const,
  legacyImageListsForAccount: (owner: AccountScope) =>
    [...galleryKeys.legacyImageListsRoot(), getAccountKey(owner)] as const,
  legacyImages: (
    owner: AccountScope,
    filter: CanonicalGalleryItemsFilter,
    window: GalleryItemsWindow = { kind: 'infinite' }
  ): LegacyGalleryImagesListQueryKey =>
    [
      ...galleryKeys.legacyImageListsForAccount(owner),
      filter,
      ...getWindowKey(window),
    ] as LegacyGalleryImagesListQueryKey,
};

const galleryItemNamesOptionsForOwner = (owner: AccountScope, filter: CanonicalGalleryItemsFilter) =>
  queryOptions({
    queryFn: async ({ signal }) => {
      const requestSignal = AbortSignal.any([signal, owner.signal]);
      const result = await (isDateBoardId(filter.boardId)
        ? listGalleryDateBoardItemNames({ ...filter, signal: requestSignal })
        : listGalleryItemNames({ ...filter, signal: requestSignal }));

      assertAccountScopeCurrent(owner);
      requestSignal.throwIfAborted();

      return result;
    },
    queryKey: galleryKeys.itemNames(owner, filter),
    staleTime: 60_000,
  });

/**
 * Lazy item-name query options. Constructing these does not subscribe or
 * request; range selection fetches them explicitly on first Shift-click.
 */
export const galleryItemNamesOptions = (inputFilter: GalleryItemsFilter) => {
  const owner = captureAccountScope();

  return galleryItemNamesOptionsForOwner(owner, canonicalizeGalleryItemsFilter(inputFilter));
};

const dateBoardNamesConsumers = new WeakMap<QueryClient, Map<string, number>>();

/**
 * A date-name request is shared by infinite and paginated consumers. One
 * consumer may stop waiting immediately without cancelling work still needed
 * by another; the final departing consumer owns cancellation.
 */
const fetchSharedDateBoardNames = (
  client: QueryClient,
  queryKey: QueryKey,
  signal: AbortSignal,
  fetchNames: () => Promise<GalleryItemNames>
): Promise<GalleryItemNames> => {
  const queryHash = hashKey(queryKey);
  const consumers = dateBoardNamesConsumers.get(client) ?? new Map<string, number>();

  dateBoardNamesConsumers.set(client, consumers);
  consumers.set(queryHash, (consumers.get(queryHash) ?? 0) + 1);

  return new Promise((resolve, reject) => {
    let settled = false;
    const release = (cancelIfLast: boolean) => {
      const remainingConsumers = Math.max(0, (consumers.get(queryHash) ?? 1) - 1);

      if (remainingConsumers === 0) {
        consumers.delete(queryHash);
        if (cancelIfLast) {
          void client.cancelQueries({ exact: true, queryKey });
        }
      } else {
        consumers.set(queryHash, remainingConsumers);
      }
    };
    const onAbort = () => {
      if (settled) {
        return;
      }

      settled = true;
      signal.removeEventListener('abort', onAbort);
      release(true);
      reject(signal.reason ?? new DOMException('The operation was aborted.', 'AbortError'));
    };
    const settle = (complete: () => void) => {
      if (settled) {
        return;
      }

      settled = true;
      signal.removeEventListener('abort', onAbort);
      release(false);
      complete();
    };

    signal.addEventListener('abort', onAbort, { once: true });
    if (signal.aborted) {
      onAbort();
      return;
    }

    let namesPromise: Promise<GalleryItemNames>;

    try {
      namesPromise = fetchNames();
    } catch (error: unknown) {
      settle(() => reject(error));
      return;
    }

    void namesPromise.then(
      (names) => settle(() => resolve(names)),
      (error: unknown) => settle(() => reject(error))
    );
  });
};

export const galleryBoardsOptions = (query: GalleryBoardsQuery = {}) => {
  const owner = captureAccountScope();
  const canonicalQuery = canonicalizeBoardsQuery(query);

  return queryOptions({
    queryFn: async ({ signal }) => {
      const requestSignal = AbortSignal.any([signal, owner.signal]);
      const [boards, dateBoards] = await Promise.all([
        listGalleryBoards({ ...canonicalQuery, signal: requestSignal }),
        canonicalQuery.includeDateBoards ? listGalleryDateBoards(requestSignal) : Promise.resolve([]),
      ]);

      assertAccountScopeCurrent(owner);
      requestSignal.throwIfAborted();

      return [...boards, ...dateBoards];
    },
    queryKey: galleryKeys.boards(owner, canonicalQuery),
    staleTime: 60_000,
  });
};

const getNextPageParam = (
  window: GalleryItemsWindow,
  lastPage: Pick<GalleryItemsPage, 'total'>,
  lastPageParam: number
): number | undefined => {
  const nextOffset = lastPageParam + GALLERY_PAGE_SIZE;
  const isInsideWindow = window.kind === 'anchor' || nextOffset < GALLERY_MAX_ROWS;

  return isInsideWindow && nextOffset < lastPage.total ? nextOffset : undefined;
};

export const galleryItemsInfiniteOptions = (
  inputFilter: GalleryItemsFilter,
  window: GalleryItemsWindow = { kind: 'infinite' }
) => {
  const owner = captureAccountScope();
  const filter = canonicalizeGalleryItemsFilter(inputFilter);
  const normalizedWindow =
    window.kind === 'infinite' ? window : ({ ...window, offset: normalizePageOffset(window.offset) } as const);
  const initialPageParam = normalizedWindow.kind === 'infinite' ? 0 : normalizedWindow.offset;

  return infiniteQueryOptions<
    GalleryItemsPage,
    Error,
    InfiniteData<GalleryItemsPage, number>,
    GalleryItemsListQueryKey,
    number
  >({
    ...(normalizedWindow.kind === 'infinite' ? {} : { gcTime: 0 }),
    getNextPageParam: (lastPage, allPages, lastPageParam) =>
      allPages.length >= GALLERY_MAX_INFINITE_PAGES
        ? undefined
        : getNextPageParam(normalizedWindow, lastPage, lastPageParam),
    getPreviousPageParam: (_firstPage, allPages, firstPageParam) =>
      allPages.length < GALLERY_MAX_INFINITE_PAGES && firstPageParam >= GALLERY_PAGE_SIZE
        ? firstPageParam - GALLERY_PAGE_SIZE
        : undefined,
    initialPageParam,
    maxPages: GALLERY_MAX_INFINITE_PAGES,
    queryFn: async ({ client, pageParam, signal }) => {
      const requestSignal = AbortSignal.any([signal, owner.signal]);
      let result: GalleryItemsPage;

      if (isDateBoardId(filter.boardId)) {
        const namesOptions = galleryItemNamesOptionsForOwner(owner, filter);
        const names = await fetchSharedDateBoardNames(client, namesOptions.queryKey, requestSignal, () =>
          client.fetchQuery(namesOptions)
        );

        assertAccountScopeCurrent(owner);
        requestSignal.throwIfAborted();
        result = await hydrateGalleryDateBoardItemPage({
          ...names,
          limit: GALLERY_PAGE_SIZE,
          offset: pageParam,
          signal: requestSignal,
        });
      } else {
        result = await listGalleryItems({
          ...filter,
          limit: GALLERY_PAGE_SIZE,
          offset: pageParam,
          signal: requestSignal,
        });
      }

      assertAccountScopeCurrent(owner);
      requestSignal.throwIfAborted();

      return result.items.length <= GALLERY_PAGE_SIZE
        ? result
        : { ...result, items: result.items.slice(0, GALLERY_PAGE_SIZE) };
    },
    queryKey: galleryKeys.items(owner, filter, normalizedWindow),
    staleTime: 60_000,
  });
};

export const flattenGalleryItemsData = (data: InfiniteData<GalleryItemsPage, number> | undefined): GalleryItem[] => {
  if (!data) {
    return [];
  }

  const itemKeys = new Set<string>();
  const items: GalleryItem[] = [];

  for (const page of data.pages) {
    for (const item of page.items) {
      const key = toGalleryItemKey(item);

      if (itemKeys.has(key)) {
        continue;
      }

      itemKeys.add(key);
      items.push(item);

      if (items.length === GALLERY_MAX_ROWS) {
        return items;
      }
    }
  }

  return items;
};

export const getGalleryItemsFilterFromKey = (queryKey: QueryKey): CanonicalGalleryItemsFilter | null => {
  if (
    queryKey[0] !== 'gallery' ||
    queryKey[1] !== 'items' ||
    queryKey[2] !== 'list' ||
    !queryKey[3] ||
    typeof queryKey[3] !== 'object' ||
    !queryKey[4] ||
    typeof queryKey[4] !== 'object'
  ) {
    return null;
  }

  return queryKey[4] as CanonicalGalleryItemsFilter;
};

export const getGalleryItemListQueries = (client: QueryClient, owner: AccountScope = captureAccountScope()) =>
  client.getQueryCache().findAll({ queryKey: galleryKeys.itemListsForAccount(owner) });

/**
 * TODO(Task 5/7): Remove this image-page compatibility query after Gallery
 * state/projection and Preview consume `galleryItemsInfiniteOptions`.
 * It is isolated under `gallery/legacy-images` and never changes the mixed
 * query's GalleryItemsPage result.
 */
export const galleryImagesInfiniteOptions = (
  inputFilter: GalleryItemsFilter,
  window: GalleryItemsWindow = { kind: 'infinite' }
) => {
  const owner = captureAccountScope();
  const filter = canonicalizeGalleryItemsFilter(inputFilter);
  const normalizedWindow =
    window.kind === 'infinite' ? window : ({ ...window, offset: normalizePageOffset(window.offset) } as const);
  const initialPageParam = normalizedWindow.kind === 'infinite' ? 0 : normalizedWindow.offset;

  return infiniteQueryOptions<
    GalleryImagesPage,
    Error,
    InfiniteData<GalleryImagesPage, number>,
    LegacyGalleryImagesListQueryKey,
    number
  >({
    ...(normalizedWindow.kind === 'infinite' ? {} : { gcTime: 0 }),
    getNextPageParam: (lastPage, allPages, lastPageParam) =>
      allPages.length >= GALLERY_MAX_INFINITE_PAGES
        ? undefined
        : getNextPageParam(normalizedWindow, lastPage, lastPageParam),
    getPreviousPageParam: (_firstPage, allPages, firstPageParam) =>
      allPages.length < GALLERY_MAX_INFINITE_PAGES && firstPageParam >= GALLERY_PAGE_SIZE
        ? firstPageParam - GALLERY_PAGE_SIZE
        : undefined,
    initialPageParam,
    maxPages: GALLERY_MAX_INFINITE_PAGES,
    queryFn: async ({ pageParam, signal }) => {
      const requestSignal = AbortSignal.any([signal, owner.signal]);
      const result = await listPaletteImages({
        ...filter,
        limit: GALLERY_PAGE_SIZE,
        offset: pageParam,
        signal: requestSignal,
      });

      assertAccountScopeCurrent(owner);
      requestSignal.throwIfAborted();

      return result.images.length <= GALLERY_PAGE_SIZE
        ? result
        : { ...result, images: result.images.slice(0, GALLERY_PAGE_SIZE) };
    },
    queryKey: galleryKeys.legacyImages(owner, filter, normalizedWindow),
    staleTime: 60_000,
  });
};

/** TODO(Task 5/7): Remove with `galleryImagesInfiniteOptions`. */
export const flattenGalleryImagesData = (data: InfiniteData<GalleryImagesPage, number> | undefined): GalleryImage[] => {
  if (!data) {
    return [];
  }

  const imageNames = new Set<string>();
  const images: GalleryImage[] = [];

  for (const page of data.pages) {
    for (const image of page.images) {
      if (imageNames.has(image.imageName)) {
        continue;
      }

      imageNames.add(image.imageName);
      images.push(image);

      if (images.length === GALLERY_MAX_ROWS) {
        return images;
      }
    }
  }

  return images;
};

/** TODO(Task 5/7): Remove legacy image cache discovery. */
export const getGalleryImageListQueries = (client: QueryClient, owner: AccountScope = captureAccountScope()) =>
  client.getQueryCache().findAll({ queryKey: galleryKeys.legacyImageListsForAccount(owner) });

/** TODO(Task 5/7): Remove legacy image key guard. */
export const getGalleryImagesFilterFromKey = (queryKey: QueryKey): CanonicalGalleryItemsFilter | null => {
  if (
    queryKey[0] !== 'gallery' ||
    queryKey[1] !== 'legacy-images' ||
    queryKey[2] !== 'list' ||
    !queryKey[4] ||
    typeof queryKey[4] !== 'object'
  ) {
    return null;
  }

  return queryKey[4] as CanonicalGalleryItemsFilter;
};

/** TODO(Task 5/7): Remove legacy type aliases with the image-only consumers. */
export type GalleryImagesFilter = GalleryItemsFilter;
export type CanonicalGalleryImagesFilter = CanonicalGalleryItemsFilter;
export type GalleryImagesWindow = GalleryItemsWindow;
export type GalleryImagesListQueryKey = LegacyGalleryImagesListQueryKey;
export const canonicalizeGalleryImagesFilter = canonicalizeGalleryItemsFilter;
