import type { GalleryItem, GalleryItemRef, GalleryItemsPage } from '@features/gallery/core/items';
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
  type Query,
  type QueryClient,
  type QueryKey,
  type QueryObserverOptions,
  QueryObserver,
} from '@tanstack/react-query';

import {
  type GalleryItemLocation,
  type GalleryItemNames,
  fetchImageIndexAvailability,
  getGalleryItemLocation,
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
export { isDateBoardId };

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

export type GalleryItemsPageQueryKey = readonly [...GalleryItemsInfiniteQueryKey, 'page', number];

/** The bounded starred strip: one `GalleryItemsPage`, not an infinite window. */
type GalleryItemsStripQueryKey = readonly [...GalleryItemsInfiniteQueryKey, 'strip'];

export type GalleryItemsListQueryKey =
  | GalleryItemsAnchorQueryKey
  | GalleryItemsInfiniteQueryKey
  | GalleryItemsPageQueryKey
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
  itemPage: (owner: AccountScope, filter: CanonicalGalleryItemsFilter, offset: number): GalleryItemsPageQueryKey =>
    [...galleryKeys.itemListsForAccount(owner), filter, 'page', normalizePageOffset(offset)] as const,
  itemTotal: (owner: AccountScope, filter: CanonicalGalleryItemsFilter) =>
    [...galleryKeys.itemListsForAccount(owner), filter, 'total'] as const,
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

const MAX_INACTIVE_PAGES_PER_LISTING = 10;
const MAX_CONCURRENT_PAGE_FETCHES = 4;

interface PendingPageFetch<T> {
  signal: AbortSignal;
  run: () => Promise<T>;
  resolve: (result: T) => void;
  reject: (error: unknown) => void;
  onAbort: () => void;
  started: boolean;
}

interface GalleryPageLifecycle {
  lastUse: WeakMap<Query, number>;
  nextUse: number;
  queue: PendingPageFetch<unknown>[];
  activeFetches: number;
  unsubscribe: () => void;
}

const lifecycles = new WeakMap<QueryClient, GalleryPageLifecycle>();

const getSparsePageIdentity = (queryKey: QueryKey): { listingKey: QueryKey; listingHash: string } | null => {
  if (
    queryKey.length !== 7 ||
    queryKey[0] !== 'gallery' ||
    queryKey[1] !== 'items' ||
    queryKey[2] !== 'list' ||
    !queryKey[3] ||
    typeof queryKey[3] !== 'object' ||
    !queryKey[4] ||
    typeof queryKey[4] !== 'object' ||
    queryKey[5] !== 'page' ||
    typeof queryKey[6] !== 'number'
  ) {
    return null;
  }

  const listingKey = queryKey.slice(0, 5);

  return { listingKey, listingHash: hashKey(listingKey) };
};

const ensureLifecycle = (client: QueryClient): GalleryPageLifecycle => {
  const existing = lifecycles.get(client);

  if (existing) {
    return existing;
  }

  const lifecycle: GalleryPageLifecycle = {
    lastUse: new WeakMap(),
    nextUse: 0,
    queue: [],
    activeFetches: 0,
    unsubscribe: () => undefined,
  };

  const touch = (query: Query) => lifecycle.lastUse.set(query, ++lifecycle.nextUse);

  const pruneListing = (listingKey: QueryKey) => {
    const candidates = client
      .getQueryCache()
      .findAll({ queryKey: listingKey })
      .filter((query) => {
        const identity = getSparsePageIdentity(query.queryKey);

        if (!identity || identity.listingHash !== hashKey(listingKey)) {
          return false;
        }

        return (
          query.state.fetchStatus === 'idle' &&
          (query.state.data !== undefined || query.state.status === 'error') &&
          query.getObserversCount() === 0
        );
      });

    if (candidates.length <= MAX_INACTIVE_PAGES_PER_LISTING) {
      return;
    }

    candidates.sort((left, right) => {
      const leftUse = lifecycle.lastUse.get(left) ?? left.state.dataUpdatedAt;
      const rightUse = lifecycle.lastUse.get(right) ?? right.state.dataUpdatedAt;

      return leftUse - rightUse || Number(left.queryKey[6]) - Number(right.queryKey[6]);
    });

    for (const query of candidates.slice(0, candidates.length - MAX_INACTIVE_PAGES_PER_LISTING)) {
      client.removeQueries({ exact: true, queryKey: query.queryKey });
    }
  };

  const seedExistingPages = () => {
    const existingPages = client
      .getQueryCache()
      .findAll({ queryKey: ['gallery', 'items', 'list'] })
      .filter((query) => getSparsePageIdentity(query.queryKey))
      .sort((left, right) => left.state.dataUpdatedAt - right.state.dataUpdatedAt);

    for (const query of existingPages) {
      if (!lifecycle.lastUse.has(query)) {
        touch(query);
      }
    }
  };

  seedExistingPages();
  lifecycle.unsubscribe = client.getQueryCache().subscribe((event) => {
    if (event.type === 'removed') {
      return;
    }

    const identity = getSparsePageIdentity(event.query.queryKey);

    if (!identity) {
      return;
    }

    if (
      event.type === 'added' ||
      event.type === 'observerAdded' ||
      event.type === 'observerRemoved' ||
      (event.type === 'updated' && event.action.type === 'success')
    ) {
      touch(event.query);
    }

    pruneListing(identity.listingKey);
  });

  lifecycles.set(client, lifecycle);

  return lifecycle;
};

const schedulePageFetch = <T>(
  lifecycle: GalleryPageLifecycle,
  signal: AbortSignal,
  run: () => Promise<T>
): Promise<T> =>
  new Promise<T>((resolve, reject) => {
    if (signal.aborted) {
      reject(signal.reason ?? new DOMException('The operation was aborted.', 'AbortError'));
      return;
    }

    const task: PendingPageFetch<T> = {
      signal,
      run,
      resolve,
      reject,
      started: false,
      onAbort: () => {
        if (task.started) {
          return;
        }

        const index = lifecycle.queue.indexOf(task as PendingPageFetch<unknown>);

        if (index !== -1) {
          lifecycle.queue.splice(index, 1);
        }

        signal.removeEventListener('abort', task.onAbort);
        reject(signal.reason ?? new DOMException('The operation was aborted.', 'AbortError'));
      },
    };

    signal.addEventListener('abort', task.onAbort, { once: true });
    lifecycle.queue.push(task as PendingPageFetch<unknown>);

    const pump = () => {
      while (lifecycle.activeFetches < MAX_CONCURRENT_PAGE_FETCHES && lifecycle.queue.length > 0) {
        const next = lifecycle.queue.shift();

        if (!next) {
          return;
        }

        if (next.signal.aborted) {
          next.onAbort();
          continue;
        }

        next.started = true;
        lifecycle.activeFetches += 1;
        void next
          .run()
          .then(
            (result) => {
              next.signal.removeEventListener('abort', next.onAbort);
              if (next.signal.aborted) {
                next.reject(next.signal.reason ?? new DOMException('The operation was aborted.', 'AbortError'));
              } else {
                next.resolve(result);
              }
            },
            (error: unknown) => {
              next.signal.removeEventListener('abort', next.onAbort);
              next.reject(error);
            }
          )
          .finally(() => {
            lifecycle.activeFetches -= 1;
            pump();
          });
      }
    };

    pump();
  });

/** Query deduplicates same-key calls before they enter this per-client bounded scheduler. */
const fetchGalleryPageWithLifecycle = <T>(
  client: QueryClient,
  signal: AbortSignal,
  run: () => Promise<T>
): Promise<T> => schedulePageFetch(ensureLifecycle(client), signal, run);

/** One Query-owned page at an absolute offset. Shared by every consumer of the same account/listing/page. */
export const galleryItemsPageOptions = (inputFilter: GalleryItemsFilter, offset: number) => {
  const owner = captureAccountScope();
  const filter = canonicalizeGalleryItemsFilter(inputFilter);
  const pageOffset = normalizePageOffset(offset);

  return queryOptions({
    queryFn: ({ client, signal }) => {
      const requestSignal = AbortSignal.any([signal, owner.signal]);

      return fetchGalleryPageWithLifecycle(client, requestSignal, () =>
        fetchGalleryItemsRange(client, owner, filter, {
          limit: GALLERY_PAGE_SIZE,
          offset: pageOffset,
          signal: requestSignal,
          includeAbsolutePositions: true,
        })
      );
    },
    queryKey: galleryKeys.itemPage(owner, filter, pageOffset),
    staleTime: 60_000,
  });
};

/** A bounded count-only read shared by every page of one account-scoped listing. */
export const galleryItemsTotalOptions = (inputFilter: GalleryItemsFilter) => {
  const owner = captureAccountScope();
  const filter = canonicalizeGalleryItemsFilter(inputFilter);

  return queryOptions({
    queryFn: async ({ client, signal }) => {
      const requestSignal = AbortSignal.any([signal, owner.signal]);
      const page = await fetchGalleryItemsRange(client, owner, filter, {
        limit: 0,
        offset: 0,
        signal: requestSignal,
      });

      return page.total;
    },
    queryKey: galleryKeys.itemTotal(owner, filter),
    staleTime: 60_000,
  });
};

const galleryItemLocationKey = (
  owner: AccountScope,
  filter: ReturnType<typeof canonicalizeGalleryItemsFilter>,
  ref: GalleryItemRef
) => ['gallery', 'item-location', getAccountKey(owner), filter, ref] as const;

interface SharedQueryConsumerState {
  count: number;
  unsubscribeObserver: () => void;
}

const sharedQueryConsumers = new WeakMap<QueryClient, Map<string, SharedQueryConsumerState>>();

/** Stop one caller's wait immediately; cancel the Query only after its final caller leaves. */
const fetchSharedQuery = <T, TQueryKey extends QueryKey>(
  client: QueryClient,
  options: QueryObserverOptions<T, Error, T, T, TQueryKey>,
  signal: AbortSignal | undefined,
  cancelQueryWhenUnused: boolean,
  fetch: () => Promise<T>
): Promise<T> => {
  if (signal?.aborted) {
    return Promise.reject(signal.reason ?? new DOMException('The operation was aborted.', 'AbortError'));
  }

  const queryHash = hashKey(options.queryKey);
  const consumers = sharedQueryConsumers.get(client) ?? new Map<string, SharedQueryConsumerState>();

  sharedQueryConsumers.set(client, consumers);
  let sharedState = consumers.get(queryHash);

  if (!sharedState) {
    // The explicit Query observer keeps TanStack from aborting a signal-aware read when its last UI observer leaves
    // but an imperative reveal/location caller still awaits the shared request.
    const observer = new QueryObserver(client, { ...options, enabled: false });

    sharedState = { count: 0, unsubscribeObserver: observer.subscribe(() => undefined) };
    consumers.set(queryHash, sharedState);
  }
  sharedState.count += 1;

  return new Promise((resolve, reject) => {
    let settled = false;
    const release = (cancelIfLast: boolean) => {
      const state = consumers.get(queryHash);
      const remainingConsumers = Math.max(0, (state?.count ?? 1) - 1);

      if (remainingConsumers === 0) {
        consumers.delete(queryHash);
        state?.unsubscribeObserver();
        const query = client.getQueryCache().find({ exact: true, queryKey: options.queryKey });

        if (cancelIfLast && cancelQueryWhenUnused && (query?.getObserversCount() ?? 0) === 0) {
          void client.cancelQueries({ exact: true, queryKey: options.queryKey });
        }
        if (consumers.size === 0) {
          sharedQueryConsumers.delete(client);
        }
      } else {
        state!.count = remainingConsumers;
      }
    };
    const onAbort = () => {
      if (settled) {
        return;
      }

      settled = true;
      signal?.removeEventListener('abort', onAbort);
      release(true);
      reject(signal?.reason ?? new DOMException('The operation was aborted.', 'AbortError'));
    };
    const settle = (complete: () => void) => {
      if (settled) {
        return;
      }

      settled = true;
      signal?.removeEventListener('abort', onAbort);
      release(false);
      complete();
    };

    signal?.addEventListener('abort', onAbort, { once: true });
    if (signal?.aborted) {
      onAbort();
      return;
    }

    let request: Promise<T>;

    try {
      request = fetch();
    } catch (error: unknown) {
      settle(() => reject(error));
      return;
    }

    void request.then(
      (value) => settle(() => resolve(value)),
      (error: unknown) => settle(() => reject(error))
    );
  });
};

/** Fetch one shared page while tracking every imperative consumer that cannot be seen by Query observers. */
export const fetchGalleryItemsPage = (
  queryClient: QueryClient,
  inputFilter: GalleryItemsFilter,
  offset: number,
  { signal, staleTime }: { signal?: AbortSignal; staleTime?: number } = {}
): Promise<GalleryItemsPage> => {
  const pageOptions = galleryItemsPageOptions(inputFilter, offset);

  return fetchSharedQuery(queryClient, pageOptions, signal, true, () =>
    queryClient.fetchQuery(staleTime === undefined ? pageOptions : { ...pageOptions, staleTime })
  );
};

const galleryItemLocationOptionsForOwner = (
  owner: AccountScope,
  inputFilter: GalleryItemsFilter,
  ref: GalleryItemRef
) => {
  const filter = canonicalizeGalleryItemsFilter(inputFilter);

  if (filter.semantic) {
    throw new TypeError('Semantic gallery results do not have an ordinary listing location.');
  }

  return queryOptions({
    queryFn: async ({ signal }) => {
      const requestSignal = AbortSignal.any([signal, owner.signal]);
      const location = await getGalleryItemLocation({ ...filter, ...ref, signal: requestSignal });

      assertAccountScopeCurrent(owner);
      requestSignal.throwIfAborted();

      return location;
    },
    queryKey: galleryItemLocationKey(owner, filter, ref),
    retry: false,
    staleTime: 0,
  });
};

/** Account-fenced location for one item in an ordinary, fully filtered gallery listing. */
export const galleryItemLocationOptions = (inputFilter: GalleryItemsFilter, ref: GalleryItemRef) =>
  galleryItemLocationOptionsForOwner(captureAccountScope(), inputFilter, ref);

export interface VerifiedGalleryItemPage {
  index: number;
  offset: number;
  page: GalleryItemsPage;
  total: number;
}

const isOwnerKey = (key: unknown, owner: AccountScope): boolean =>
  Boolean(
    key &&
    typeof key === 'object' &&
    'accountId' in key &&
    'epoch' in key &&
    key.accountId === owner.accountId &&
    key.epoch === owner.epoch
  );

const pageContainsLocation = (
  page: GalleryItemsPage,
  requestedOffset: number,
  location: GalleryItemLocation,
  ref: GalleryItemRef
): boolean => {
  if (
    !Number.isSafeInteger(location.index) ||
    !Number.isSafeInteger(location.total) ||
    location.index < 0 ||
    location.index >= location.total ||
    location.total !== page.total ||
    (page.offset !== undefined && page.offset !== requestedOffset) ||
    location.kind !== ref.kind ||
    location.name !== ref.name
  ) {
    return false;
  }

  const indices = page.itemIndices ?? page.items.map((_, index) => (page.offset ?? requestedOffset) + index);
  const localIndex = indices.indexOf(location.index);
  const item: GalleryItem | undefined = localIndex >= 0 ? page.items[localIndex] : undefined;

  return item?.kind === ref.kind && item.name === ref.name;
};

const isValidLocation = (location: GalleryItemLocation, ref: GalleryItemRef): boolean =>
  location.kind === ref.kind &&
  location.name === ref.name &&
  Number.isSafeInteger(location.index) &&
  Number.isSafeInteger(location.total) &&
  location.index >= 0 &&
  location.index < location.total;

/**
 * Resolve the target's current rank and fetch only its aligned page. A mismatch means the listing shifted between
 * the rank and page reads; refresh both once, then leave failure to the caller without changing Gallery state.
 */
export const fetchVerifiedGalleryItemPage = async (
  queryClient: QueryClient,
  inputFilter: GalleryItemsFilter,
  ref: GalleryItemRef,
  owner: AccountScope = captureAccountScope(),
  signal: AbortSignal = owner.signal
): Promise<VerifiedGalleryItemPage | null> => {
  const filter = canonicalizeGalleryItemsFilter(inputFilter);
  const requestSignal = AbortSignal.any([signal, owner.signal]);

  if (filter.semantic) {
    throw new TypeError('Semantic gallery results do not have an ordinary listing location.');
  }

  const fenceError = <T>(request: Promise<T>): Promise<T> =>
    request.catch((error: unknown) => {
      // Account lifetime errors remain authoritative even though the owner signal also cancels Query work.
      assertAccountScopeCurrent(owner);
      requestSignal.throwIfAborted();
      throw error;
    });

  for (let attempt = 0; attempt < 2; attempt += 1) {
    assertAccountScopeCurrent(owner);
    requestSignal.throwIfAborted();
    const locationOptions = galleryItemLocationOptionsForOwner(owner, filter, ref);

    const location = await fenceError(
      fetchSharedQuery(queryClient, locationOptions, requestSignal, true, () =>
        queryClient.fetchQuery({ ...locationOptions, staleTime: 0 })
      )
    );

    assertAccountScopeCurrent(owner);
    requestSignal.throwIfAborted();

    if (!isValidLocation(location, ref)) {
      continue;
    }

    const offset = Math.floor(location.index / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE;
    const pageOptions = galleryItemsPageOptions(filter, offset);

    if (!isOwnerKey(pageOptions.queryKey[3], owner)) {
      throw new Error('Gallery account changed before the target page could be fetched.');
    }

    // A cached page can have a fresh staleTime while an external insert/delete has shifted its offsets. Always read
    // the one resolved page from the backend before treating a locator result as verified.
    const page = await fenceError(
      fetchGalleryItemsPage(queryClient, filter, offset, {
        signal: requestSignal,
        staleTime: 0,
      })
    );

    assertAccountScopeCurrent(owner);
    requestSignal.throwIfAborted();

    if (pageContainsLocation(page, offset, location, ref)) {
      return { index: location.index, offset, page, total: location.total };
    }

    if (attempt === 0) {
      await queryClient.invalidateQueries({ exact: true, queryKey: pageOptions.queryKey, refetchType: 'none' });
    }
  }

  return null;
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
  {
    limit,
    offset,
    signal,
    includeAbsolutePositions = false,
  }: { limit: number; offset: number; signal: AbortSignal; includeAbsolutePositions?: boolean }
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

  if (!includeAbsolutePositions) {
    return { items: result.items.slice(0, limit), total: result.total };
  }

  const itemIndices = result.itemIndices ?? result.items.map((_, index) => offset + index);

  if (itemIndices.length !== result.items.length) {
    throw new TypeError('Gallery page item indices must stay aligned with its items.');
  }

  const items = result.items.slice(0, limit);
  const truncatedIndices = itemIndices.slice(0, limit);

  if (truncatedIndices.some((index) => !Number.isSafeInteger(index) || index < offset || index >= offset + limit)) {
    throw new RangeError('Gallery page item indices must stay within the requested range.');
  }

  return { ...result, items, itemIndices: truncatedIndices, offset };
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
      query.state.status === 'error' ||
      query.state.data?.state === 'model_missing' ||
      query.state.data?.state === 'switching'
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
    queryFn: ({ client, signal }) => {
      const requestSignal = AbortSignal.any([signal, owner.signal]);

      return fetchGalleryPageWithLifecycle(client, requestSignal, () =>
        fetchGalleryItemsRange(client, owner, filter, {
          limit: GALLERY_STARRED_STRIP_LIMIT,
          offset: 0,
          signal: requestSignal,
        })
      );
    },
    queryKey: galleryKeys.starredStrip(owner, filter),
    staleTime: 60_000,
  });
};

export const isGalleryStarredStripQueryKey = (queryKey: QueryKey): boolean => queryKey[5] === 'strip';

export const isGallerySinglePageQueryKey = (queryKey: QueryKey): boolean =>
  queryKey[5] === 'page' || isGalleryStarredStripQueryKey(queryKey);

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
