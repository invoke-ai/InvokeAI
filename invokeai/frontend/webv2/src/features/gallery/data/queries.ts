import type {
  GalleryBoardOrderBy,
  GalleryImage,
  GalleryImagesPage,
  GalleryOrderDir,
  GalleryView,
} from '@features/gallery/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';

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
  type GalleryDateBoardImageNames,
  hydrateGalleryDateBoardImagePage,
  isDateBoardId,
  listGalleryBoards,
  listGalleryDateBoardImageNames,
  listGalleryDateBoards,
  listGalleryImages,
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

export interface GalleryImagesFilter {
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

export interface CanonicalGalleryImagesFilter {
  boardId: string;
  createdFrom?: string;
  createdTo?: string;
  galleryView: GalleryView;
  orderDir: GalleryOrderDir;
  searchTerm: string;
  starredFirst: boolean;
}

export type GalleryImagesWindow = { kind: 'anchor'; offset: number } | { kind: 'infinite' };

interface GalleryAccountKey {
  accountId: string | null;
  epoch: number;
}

type GalleryImagesInfiniteQueryKey = readonly [
  'gallery',
  'images',
  'list',
  GalleryAccountKey,
  CanonicalGalleryImagesFilter,
];

type GalleryImagesAnchorQueryKey = readonly [...GalleryImagesInfiniteQueryKey, 'anchor', number];

export type GalleryImagesListQueryKey = GalleryImagesAnchorQueryKey | GalleryImagesInfiniteQueryKey;

const canonicalizeBoardsQuery = (query: GalleryBoardsQuery): CanonicalGalleryBoardsQuery => ({
  includeArchived: query.includeArchived ?? false,
  includeDateBoards: query.includeDateBoards ?? false,
  orderBy: query.orderBy ?? 'created_at',
  orderDir: query.orderDir ?? 'DESC',
});

export const canonicalizeGalleryImagesFilter = (filter: GalleryImagesFilter): CanonicalGalleryImagesFilter => ({
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

const getWindowKey = (window: GalleryImagesWindow): readonly [] | readonly ['anchor', number] =>
  window.kind === 'infinite' ? [] : ([window.kind, normalizePageOffset(window.offset)] as const);

export const galleryKeys = {
  all: ['gallery'] as const,
  boardsRoot: () => [...galleryKeys.all, 'boards'] as const,
  boardsForAccount: (owner: AccountScope) => [...galleryKeys.boardsRoot(), getAccountKey(owner)] as const,
  boards: (owner: AccountScope, query: CanonicalGalleryBoardsQuery) =>
    [...galleryKeys.boardsForAccount(owner), query] as const,
  imagesRoot: () => [...galleryKeys.all, 'images'] as const,
  imageListsRoot: () => [...galleryKeys.imagesRoot(), 'list'] as const,
  imageListsForAccount: (owner: AccountScope) => [...galleryKeys.imageListsRoot(), getAccountKey(owner)] as const,
  images: (
    owner: AccountScope,
    filter: CanonicalGalleryImagesFilter,
    window: GalleryImagesWindow = { kind: 'infinite' }
  ): GalleryImagesListQueryKey =>
    [...galleryKeys.imageListsForAccount(owner), filter, ...getWindowKey(window)] as GalleryImagesListQueryKey,
  dateBoardNamesRoot: () => [...galleryKeys.imagesRoot(), 'date-names'] as const,
  dateBoardNamesForAccount: (owner: AccountScope) =>
    [...galleryKeys.dateBoardNamesRoot(), getAccountKey(owner)] as const,
  dateBoardNames: (owner: AccountScope, filter: CanonicalGalleryImagesFilter) =>
    [...galleryKeys.dateBoardNamesForAccount(owner), filter] as const,
};

const galleryDateBoardNamesOptions = (owner: AccountScope, filter: CanonicalGalleryImagesFilter) =>
  queryOptions({
    queryFn: async ({ signal }) => {
      const requestSignal = AbortSignal.any([signal, owner.signal]);
      const result = await listGalleryDateBoardImageNames({ ...filter, signal: requestSignal });

      assertAccountScopeCurrent(owner);
      requestSignal.throwIfAborted();

      return result;
    },
    queryKey: galleryKeys.dateBoardNames(owner, filter),
    staleTime: 60_000,
  });

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
  fetchNames: () => Promise<GalleryDateBoardImageNames>
): Promise<GalleryDateBoardImageNames> => {
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

    let namesPromise: Promise<GalleryDateBoardImageNames>;

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
  window: GalleryImagesWindow,
  lastPage: GalleryImagesPage,
  lastPageParam: number
): number | undefined => {
  const nextOffset = lastPageParam + GALLERY_PAGE_SIZE;

  const isInsideWindow = window.kind === 'anchor' || nextOffset < GALLERY_MAX_ROWS;

  return isInsideWindow && nextOffset < lastPage.total ? nextOffset : undefined;
};

export const galleryImagesInfiniteOptions = (
  inputFilter: GalleryImagesFilter,
  window: GalleryImagesWindow = { kind: 'infinite' }
) => {
  const owner = captureAccountScope();
  const filter = canonicalizeGalleryImagesFilter(inputFilter);
  const normalizedWindow =
    window.kind === 'infinite' ? window : ({ ...window, offset: normalizePageOffset(window.offset) } as const);
  const initialPageParam = normalizedWindow.kind === 'infinite' ? 0 : normalizedWindow.offset;

  return infiniteQueryOptions<
    GalleryImagesPage,
    Error,
    InfiniteData<GalleryImagesPage, number>,
    GalleryImagesListQueryKey,
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
      let result: GalleryImagesPage;

      if (isDateBoardId(filter.boardId)) {
        const namesOptions = galleryDateBoardNamesOptions(owner, filter);
        const names = await fetchSharedDateBoardNames(client, namesOptions.queryKey, requestSignal, () =>
          client.fetchQuery(namesOptions)
        );

        assertAccountScopeCurrent(owner);
        requestSignal.throwIfAborted();
        result = await hydrateGalleryDateBoardImagePage({
          ...names,
          limit: GALLERY_PAGE_SIZE,
          offset: pageParam,
          signal: requestSignal,
        });
      } else {
        result = await listGalleryImages({
          ...filter,
          limit: GALLERY_PAGE_SIZE,
          offset: pageParam,
          signal: requestSignal,
        });
      }

      assertAccountScopeCurrent(owner);
      requestSignal.throwIfAborted();

      return result.images.length <= GALLERY_PAGE_SIZE
        ? result
        : { ...result, images: result.images.slice(0, GALLERY_PAGE_SIZE) };
    },
    queryKey: galleryKeys.images(owner, filter, normalizedWindow),
    staleTime: 60_000,
  });
};

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

export const getGalleryImagesFilterFromKey = (queryKey: QueryKey): CanonicalGalleryImagesFilter | null => {
  if (
    queryKey[0] !== 'gallery' ||
    queryKey[1] !== 'images' ||
    queryKey[2] !== 'list' ||
    !queryKey[4] ||
    typeof queryKey[4] !== 'object'
  ) {
    return null;
  }

  return queryKey[4] as CanonicalGalleryImagesFilter;
};

export const getGalleryImageListQueries = (client: QueryClient, owner: AccountScope = captureAccountScope()) =>
  client.getQueryCache().findAll({ queryKey: galleryKeys.imageListsForAccount(owner) });
