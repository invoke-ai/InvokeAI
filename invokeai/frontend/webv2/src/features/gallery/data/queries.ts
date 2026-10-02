import type { GalleryItem, GalleryItemsPage } from '@features/gallery/core/items';
import type { GallerySemanticQuery, GallerySemanticReference } from '@features/gallery/core/semanticImageQuery';
import type { GallerySettings } from '@features/gallery/core/settings';
import type { GalleryBoardOrderBy, GalleryOrderDir, GalleryView } from '@features/gallery/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { toGalleryItemKey } from '@features/gallery/core/items';
import {
  GALLERY_MAX_INFINITE_PAGES,
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
  GALLERY_STARRED_STRIP_LIMIT,
} from '@features/gallery/core/paging';
import { toGallerySemanticQuery } from '@features/gallery/core/semanticImageQuery';
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
  fetchImageIndexAvailability,
  hydrateGalleryDateBoardItemPage,
  isDateBoardId,
  listGalleryBoards,
  listGalleryDateBoardItemNames,
  listGalleryDateBoards,
  listGalleryItemNames,
  listGalleryItems,
  listSemanticGalleryItemNames,
} from './backend';

export { GALLERY_MAX_INFINITE_PAGES, GALLERY_MAX_ROWS, GALLERY_PAGE_SIZE, GALLERY_STARRED_STRIP_LIMIT };

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
  /**
   * When set, items come from semantic image-similarity search over the board
   * (relevance order) instead of the board listing; order/starred controls do
   * not apply to a ranked result set.
   */
  semanticQuery?: GallerySemanticReference | null;
  /** true = only starred items, false = only unstarred; absent = all. */
  starred?: boolean;
}

export interface CanonicalGalleryItemsFilter {
  boardId: string;
  createdFrom?: string;
  createdTo?: string;
  galleryView: GalleryView;
  orderDir: GalleryOrderDir;
  searchTerm: string;
  /** Label-free semantic reference: a file query is keyed by its registry id. */
  semantic?: GallerySemanticQuery;
  starred?: boolean;
}

/**
 * Offset anchors a page-aligned infinite window with GALLERY_MAX_ROWS reach. Board, search, and view changes reset
 * it to zero.
 */
export type GalleryItemsWindow = { kind: 'anchor'; offset: number } | { kind: 'infinite'; offset?: number };

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

type GalleryItemsAnchorQueryKey = readonly [...GalleryItemsInfiniteQueryKey, 'anchor' | 'infinite', number];

/** The bounded starred strip: one `GalleryItemsPage`, not an infinite window. */
type GalleryItemsStripQueryKey = readonly [...GalleryItemsInfiniteQueryKey, 'strip'];

export type GalleryItemsListQueryKey =
  | GalleryItemsAnchorQueryKey
  | GalleryItemsInfiniteQueryKey
  | GalleryItemsStripQueryKey;

const canonicalizeBoardsQuery = (query: GalleryBoardsQuery): CanonicalGalleryBoardsQuery => ({
  includeArchived: query.includeArchived ?? false,
  includeDateBoards: query.includeDateBoards ?? false,
  orderBy: query.orderBy ?? 'created_at',
  orderDir: query.orderDir ?? 'DESC',
});

export const canonicalizeGalleryItemsFilter = (filter: GalleryItemsFilter): CanonicalGalleryItemsFilter => {
  const semantic = filter.semanticQuery ? toGallerySemanticQuery(filter.semanticQuery) : undefined;

  if (semantic) {
    // Pin irrelevant filter fields to canonical values: rankings depend only on the query and its board (a cluster is
    // a fixed member list with no board), and distinct keys would repeat uploads or downloads.
    return {
      boardId: semantic.kind === 'cluster' ? '' : filter.boardId,
      galleryView: 'images',
      orderDir: 'DESC',
      searchTerm: '',
      semantic,
    };
  }

  return {
    boardId: filter.boardId,
    ...(filter.createdFrom ? { createdFrom: filter.createdFrom } : {}),
    ...(filter.createdTo ? { createdTo: filter.createdTo } : {}),
    galleryView: filter.galleryView,
    orderDir: filter.orderDir ?? 'DESC',
    searchTerm: filter.searchTerm.trim(),
    ...(filter.starred !== undefined ? { starred: filter.starred } : {}),
  };
};

const getAccountKey = (owner: AccountScope): GalleryAccountKey => ({
  accountId: owner.accountId,
  epoch: owner.epoch,
});

const normalizePageOffset = (offset: number): number =>
  Math.max(0, Math.floor(offset / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE);

const getWindowKey = (
  window: GalleryItemsWindow
): readonly [] | readonly ['anchor', number] | readonly ['infinite', number] => {
  if (window.kind === 'infinite') {
    const offset = normalizePageOffset(window.offset ?? 0);

    // Preserve the zero-offset key so base-window consumers share one cache entry.
    return offset === 0 ? [] : (['infinite', offset] as const);
  }

  return [window.kind, normalizePageOffset(window.offset)] as const;
};

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
  starredStrip: (owner: AccountScope, filter: CanonicalGalleryItemsFilter): GalleryItemsStripQueryKey =>
    [...galleryKeys.itemListsForAccount(owner), filter, 'strip'] as const,
  itemNamesRoot: () => [...galleryKeys.itemsRoot(), 'names'] as const,
  itemNamesForAccount: (owner: AccountScope) => [...galleryKeys.itemNamesRoot(), getAccountKey(owner)] as const,
  itemNames: (owner: AccountScope, filter: CanonicalGalleryItemsFilter) =>
    [...galleryKeys.itemNamesForAccount(owner), filter] as const,
  imageIndexAvailability: (owner: AccountScope) => [...galleryKeys.all, 'image-index', getAccountKey(owner)] as const,
};

const galleryItemNamesOptionsForOwner = (owner: AccountScope, filter: CanonicalGalleryItemsFilter) =>
  queryOptions({
    queryFn: async ({ signal }) => {
      const requestSignal = AbortSignal.any([signal, owner.signal]);
      const result = await (filter.semantic
        ? listSemanticGalleryItemNames({ boardId: filter.boardId, query: filter.semantic, signal: requestSignal })
        : isDateBoardId(filter.boardId)
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
 * A name-list request (date boards and semantic searches) is shared by
 * infinite and paginated consumers. One consumer may stop waiting immediately
 * without cancelling work still needed by another; the final departing
 * consumer owns cancellation.
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

/**
 * Share one range reader between pages and rebuilds. Name-list filters hydrate slices of a shared fetch to avoid
 * repeating semantic uploads; clamp to limit.
 */
export const fetchGalleryItemsRange = async (
  client: QueryClient,
  owner: AccountScope,
  filter: CanonicalGalleryItemsFilter,
  { limit, offset, signal }: { limit: number; offset: number; signal: AbortSignal }
): Promise<GalleryItemsPage> => {
  let result: GalleryItemsPage;

  if (filter.semantic || isDateBoardId(filter.boardId)) {
    const namesOptions = galleryItemNamesOptionsForOwner(owner, filter);
    const names = await fetchSharedDateBoardNames(client, namesOptions.queryKey, signal, () =>
      client.fetchQuery(namesOptions)
    );

    assertAccountScopeCurrent(owner);
    signal.throwIfAborted();
    result = await hydrateGalleryDateBoardItemPage({ ...names, limit, offset, signal });
  } else {
    result = await listGalleryItems({ ...filter, limit, offset, signal });
  }

  assertAccountScopeCurrent(owner);
  signal.throwIfAborted();

  return result.items.length <= limit ? result : { ...result, items: result.items.slice(0, limit) };
};

/**
 * Poll missing-model and failed statuses only. Settled status refreshes on a new stale consumer, including after
 * layout changes.
 */
export const IMAGE_INDEX_UNAVAILABLE_POLL_MS = 30_000;
const IMAGE_INDEX_STALE_MS = 5 * 60_000;

export const imageIndexAvailabilityOptions = () => {
  const owner = captureAccountScope();

  return queryOptions({
    queryFn: async ({ signal }) => {
      const availability = await fetchImageIndexAvailability(AbortSignal.any([signal, owner.signal]));

      assertAccountScopeCurrent(owner);

      return availability;
    },
    queryKey: galleryKeys.imageIndexAvailability(owner),
    refetchInterval: (query) =>
      query.state.status === 'error' || query.state.data?.state === 'model_missing'
        ? IMAGE_INDEX_UNAVAILABLE_POLL_MS
        : false,
    staleTime: IMAGE_INDEX_STALE_MS,
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

/** The boards query the gallery grid resolves its selected board against. */
export const getGalleryListingBoardsQuery = (settings: GallerySettings): GalleryBoardsQuery => ({
  includeArchived: settings.showArchivedBoards,
  includeDateBoards: settings.showDateBoards,
  orderBy: settings.boardOrderBy,
  orderDir: settings.boardOrderDir,
});

const getNextPageParam = (
  window: GalleryItemsWindow,
  lastPage: Pick<GalleryItemsPage, 'total'>,
  lastPageParam: number
): number | undefined => {
  const nextOffset = lastPageParam + GALLERY_PAGE_SIZE;
  const isInsideWindow =
    window.kind === 'anchor' || nextOffset < normalizePageOffset(window.offset ?? 0) + GALLERY_MAX_ROWS;

  return isInsideWindow && nextOffset < lastPage.total ? nextOffset : undefined;
};

export const galleryItemsInfiniteOptions = (
  inputFilter: GalleryItemsFilter,
  window: GalleryItemsWindow = { kind: 'infinite' }
) => {
  const owner = captureAccountScope();
  const filter = canonicalizeGalleryItemsFilter(inputFilter);
  const normalizedWindow =
    window.kind === 'infinite'
      ? ({ kind: 'infinite', offset: normalizePageOffset(window.offset ?? 0) } as const)
      : ({ ...window, offset: normalizePageOffset(window.offset) } as const);
  const initialPageParam = normalizedWindow.offset;
  const isBaseInfiniteWindow = normalizedWindow.kind === 'infinite' && normalizedWindow.offset === 0;

  return infiniteQueryOptions<
    GalleryItemsPage,
    Error,
    InfiniteData<GalleryItemsPage, number>,
    GalleryItemsListQueryKey,
    number
  >({
    // Anchored windows (paginated pages and deep infinite reveals) are
    // transient views; only the base window's cache is worth keeping around.
    ...(isBaseInfiniteWindow ? {} : { gcTime: 0 }),
    getNextPageParam: (lastPage, allPages, lastPageParam) =>
      allPages.length >= GALLERY_MAX_INFINITE_PAGES
        ? undefined
        : getNextPageParam(normalizedWindow, lastPage, lastPageParam),
    getPreviousPageParam: (_firstPage, allPages, firstPageParam) => {
      // Infinite windows cannot prepend above their anchor without shifting the viewport. Paginated windows may
      // prepend because consumers select by pageParam.
      const lowestPageParam = normalizedWindow.kind === 'infinite' ? normalizedWindow.offset : 0;

      return allPages.length < GALLERY_MAX_INFINITE_PAGES && firstPageParam - GALLERY_PAGE_SIZE >= lowestPageParam
        ? firstPageParam - GALLERY_PAGE_SIZE
        : undefined;
    },
    initialPageParam,
    maxPages: GALLERY_MAX_INFINITE_PAGES,
    queryFn: ({ client, pageParam, signal }) =>
      fetchGalleryItemsRange(client, owner, filter, {
        limit: GALLERY_PAGE_SIZE,
        offset: pageParam,
        signal: AbortSignal.any([signal, owner.signal]),
      }),
    queryKey: galleryKeys.items(owner, filter, normalizedWindow),
    staleTime: 60_000,
  });
};

/**
 * The starred strip shares the list key family (and so the account-wide
 * invalidation and mutation patching) with the listing it sits above, keyed
 * on that listing's filter plus `starred: true`.
 */
export const galleryStarredStripOptions = (inputFilter: GalleryItemsFilter) => {
  const owner = captureAccountScope();
  const filter: CanonicalGalleryItemsFilter = { ...canonicalizeGalleryItemsFilter(inputFilter), starred: true };

  return queryOptions({
    queryFn: ({ client, signal }) =>
      fetchGalleryItemsRange(client, owner, filter, {
        limit: GALLERY_STARRED_STRIP_LIMIT,
        offset: 0,
        signal: AbortSignal.any([signal, owner.signal]),
      }),
    queryKey: galleryKeys.starredStrip(owner, filter),
    staleTime: 60_000,
  });
};

export const isGalleryStarredStripQueryKey = (queryKey: QueryKey): boolean => queryKey[5] === 'strip';

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
