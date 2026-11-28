import type { GalleryItem, GalleryItemsPage } from '@features/gallery/core/items';

import { AccountScopeExpiredError, accountLifecycle } from '@platform/state/accountLifecycle';
import { hashKey, InfiniteQueryObserver, QueryClient, QueryObserver } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const backend = vi.hoisted(() => ({
  hydrateGalleryDateBoardItemPage: vi.fn(),
  isDateBoardId: vi.fn(),
  listGalleryBoards: vi.fn(),
  listGalleryDateBoardItemNames: vi.fn(),
  listGalleryDateBoards: vi.fn(),
  listGalleryItemNames: vi.fn(),
  listGalleryItems: vi.fn(),
  listPaletteImages: vi.fn(),
  listSemanticGalleryItemNames: vi.fn(),
}));

vi.mock('./backend', () => backend);

import {
  flattenGalleryItemsData,
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
  GALLERY_STARRED_STRIP_LIMIT,
  galleryBoardsOptions,
  galleryItemNamesOptions,
  galleryItemsPageOptions,
  galleryItemsInfiniteOptions,
  galleryStarredStripOptions,
  getGalleryItemListQueries,
  getFreshGalleryItemsPage,
  isGalleryItemsPageAwaitingRefresh,
  getGalleryItemsFilterFromKey,
  isGalleryStarredStripQueryKey,
  canonicalizeGalleryItemsFilter,
  type GalleryItemsFilter,
} from './queries';
import { invalidateGalleryItems, patchGalleryItemCaches } from './queryCache';

const createQueryClient = (): QueryClient =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

const createItem = (index: number, prefix = 'image'): GalleryItem => ({
  boardId: 'board-1',
  category: 'general',
  createdAt: new Date(index * 1_000).toISOString(),
  fullUrl: `/images/${prefix}-${index}`,
  height: 64,
  isIntermediate: false,
  kind: 'image',
  name: `${prefix}-${index}`,
  sourceQueueItemId: 'backend-gallery',
  colorLabel: null,
  starred: false,
  rating: 0,
  rejected: false,
  thumbnailUrl: `/images/${prefix}-${index}/thumbnail`,
  width: 64,
});

const createPage = ({
  count = GALLERY_PAGE_SIZE,
  offset,
  prefix,
  total,
}: {
  count?: number;
  offset: number;
  prefix?: string;
  total: number;
}): GalleryItemsPage => ({
  items: Array.from({ length: count }, (_, index) => createItem(offset + index, prefix)),
  total,
});

const baseFilter: GalleryItemsFilter = {
  boardId: 'board-1',
  galleryView: 'images',
  orderDir: 'DESC',
  searchTerm: 'portrait',
};

describe('Gallery item query read model', () => {
  beforeEach(() => {
    accountLifecycle.activate('gallery-query-test');
    backend.hydrateGalleryDateBoardItemPage.mockReset();
    backend.isDateBoardId.mockReset();
    backend.listGalleryBoards.mockReset();
    backend.listGalleryDateBoardItemNames.mockReset();
    backend.listGalleryDateBoards.mockReset();
    backend.listGalleryItemNames.mockReset();
    backend.listGalleryItems.mockReset();
    backend.listPaletteImages.mockReset();
    backend.listSemanticGalleryItemNames.mockReset();

    backend.isDateBoardId.mockImplementation((boardId: string) => boardId.startsWith('by_date:'));
    backend.listGalleryItems.mockResolvedValue({ items: [], total: 0 });
  });

  afterEach(() => {
    accountLifecycle.invalidate();
  });

  it('coalesces board readers through one query key and includes requested virtual boards', async () => {
    const queryClient = createQueryClient();
    const board = { id: 'none', kind: 'uncategorized', name: '' };
    const dateBoard = { id: 'by_date:2026-07-18', kind: 'date', name: 'July 18' };
    const options = galleryBoardsOptions({ includeDateBoards: true, orderDir: 'DESC' });

    backend.listGalleryBoards.mockResolvedValue([board]);
    backend.listGalleryDateBoards.mockResolvedValue([dateBoard]);

    await expect(Promise.all([queryClient.fetchQuery(options), queryClient.fetchQuery(options)])).resolves.toEqual([
      [board, dateBoard],
      [board, dateBoard],
    ]);
    expect(backend.listGalleryBoards).toHaveBeenCalledOnce();
    expect(backend.listGalleryDateBoards).toHaveBeenCalledOnce();
  });

  it.each(['invalidation', 'failed refresh'] as const)(
    'requires a successful page read after %s despite patches, rollback, and cancellation',
    async (suspension) => {
      const client = createQueryClient();
      const options = galleryItemsPageOptions(baseFilter, 0);
      const queryHash = hashKey(options.queryKey);
      const page = createPage({ count: 2, offset: 0, total: 2 });
      backend.listGalleryItems.mockResolvedValue(page);
      client.setQueryData(options.queryKey, page);
      expect(getFreshGalleryItemsPage(client, queryHash)).toBeUndefined();
      expect(isGalleryItemsPageAwaitingRefresh(client, queryHash)).toBe(false);
      await client.fetchQuery({ ...options, staleTime: 0 });
      const observer = new QueryObserver(client, options);
      const unsubscribe = observer.subscribe(() => undefined);
      const readFreshPage = () => getFreshGalleryItemsPage(client, queryHash);
      expect(readFreshPage()?.items).toEqual(page.items);

      if (suspension === 'invalidation') {
        await client.invalidateQueries({ exact: true, queryKey: options.queryKey, refetchType: 'none' });
      } else {
        backend.listGalleryItems.mockRejectedValueOnce(new Error('page unavailable'));
        await expect(client.fetchQuery({ ...options, staleTime: 0 })).rejects.toThrow('page unavailable');
      }
      expect(readFreshPage()).toBeUndefined();
      expect(isGalleryItemsPageAwaitingRefresh(client, queryHash)).toBe(true);

      const applyStar = () =>
        patchGalleryItemCaches(client, {
          kind: 'marks',
          patch: { flag: 'pick' },
          result: { failed: [], succeeded: [{ kind: 'image', name: page.items[0]!.name }] },
        });
      const rollback = applyStar();
      expect(client.getQueryData<GalleryItemsPage>(options.queryKey)?.items[0]?.starred).toBe(true);
      expect(readFreshPage()).toBeUndefined();
      rollback();
      expect(client.getQueryData<GalleryItemsPage>(options.queryKey)?.items[0]?.starred).toBe(false);
      expect(readFreshPage()).toBeUndefined();

      let resolvePending: ((page: GalleryItemsPage) => void) | undefined;
      backend.listGalleryItems.mockImplementationOnce(
        () =>
          new Promise<GalleryItemsPage>((resolve) => {
            resolvePending = resolve;
          })
      );
      const pending = client.fetchQuery({ ...options, staleTime: 0 }).catch(() => undefined);
      const rollbackPending = applyStar();
      await client.cancelQueries({ exact: true, queryKey: options.queryKey });
      await pending;
      resolvePending?.(page);
      // Cancelling the read still preserves the optimistic write, but cannot make its old page authoritative.
      expect(client.getQueryData<GalleryItemsPage>(options.queryKey)?.items[0]?.starred).toBe(true);
      expect(readFreshPage()).toBeUndefined();
      rollbackPending();
      expect(readFreshPage()).toBeUndefined();

      await client.fetchQuery({ ...options, staleTime: 0 });
      expect(readFreshPage()?.items).toEqual(page.items);
      expect(isGalleryItemsPageAwaitingRefresh(client, queryHash)).toBe(false);
      unsubscribe();
      client.clear();
      expect(readFreshPage()).toBeUndefined();
    }
  );

  it('keys sparse pages by canonical account, listing, and absolute offset identity', () => {
    const page = galleryItemsPageOptions(baseFilter, 60);
    const samePage = galleryItemsPageOptions({ ...baseFilter, searchTerm: ' portrait ' }, 119);

    expect(samePage.queryKey).toEqual(page.queryKey);
    expect(galleryItemsPageOptions(baseFilter, 120).queryKey).not.toEqual(page.queryKey);
    for (const changedFilter of [
      { boardId: 'board-2' },
      { galleryView: 'assets' as const },
      { searchTerm: 'landscape' },
      { createdFrom: '2026-07-01' },
      { orderDir: 'ASC' as const },
      { starred: true },
    ]) {
      expect(galleryItemsPageOptions({ ...baseFilter, ...changedFilter }, 60).queryKey).not.toEqual(page.queryKey);
    }

    const semantic = galleryItemsPageOptions(
      { ...baseFilter, semanticQuery: { fileId: 'external-1', kind: 'file', label: 'portrait.png' } },
      60
    );
    expect(
      galleryItemsPageOptions(
        { ...baseFilter, semanticQuery: { fileId: 'external-1', kind: 'file', label: 'renamed.png' } },
        60
      ).queryKey
    ).toEqual(semantic.queryKey);
    expect(
      galleryItemsPageOptions(
        { ...baseFilter, semanticQuery: { fileId: 'external-2', kind: 'file', label: 'portrait.png' } },
        60
      ).queryKey
    ).not.toEqual(semantic.queryKey);

    accountLifecycle.activate('gallery-query-test');

    expect(galleryItemsPageOptions(baseFilter, 60).queryKey).not.toEqual(page.queryKey);
  });

  it('deduplicates concurrent requests for the same sparse page', async () => {
    const queryClient = createQueryClient();
    let resolvePage: ((page: GalleryItemsPage) => void) | undefined;
    backend.listGalleryItems.mockImplementation(
      () =>
        new Promise<GalleryItemsPage>((resolve) => {
          resolvePage = resolve;
        })
    );
    const options = galleryItemsPageOptions(baseFilter, 60);
    const firstRequest = queryClient.fetchQuery(options);
    const secondRequest = queryClient.fetchQuery(galleryItemsPageOptions(baseFilter, 60));
    const expectedPage = {
      ...createPage({ offset: 60, total: 240 }),
      itemIndices: Array.from({ length: GALLERY_PAGE_SIZE }, (_, index) => 60 + index),
      offset: 60,
    };

    await vi.waitFor(() => expect(backend.listGalleryItems).toHaveBeenCalledOnce());
    resolvePage?.(createPage({ offset: 60, total: 240 }));

    await expect(Promise.all([firstRequest, secondRequest])).resolves.toEqual([expectedPage, expectedPage]);
    expect(backend.listGalleryItems).toHaveBeenCalledOnce();
  });

  it('rejects a sparse page result after its captured account lifetime expires', async () => {
    const queryClient = createQueryClient();
    let requestSignal: AbortSignal | undefined;
    let resolvePage: ((page: GalleryItemsPage) => void) | undefined;
    backend.listGalleryItems.mockImplementation(
      ({ signal }: { signal: AbortSignal }) =>
        new Promise<GalleryItemsPage>((resolve) => {
          requestSignal = signal;
          resolvePage = resolve;
        })
    );
    const options = galleryItemsPageOptions(baseFilter, 60);
    const request = queryClient.fetchQuery(options);

    await vi.waitFor(() => expect(backend.listGalleryItems).toHaveBeenCalledOnce());
    accountLifecycle.activate('next-gallery-query-test');
    resolvePage?.(createPage({ offset: 60, total: 240 }));

    await expect(request).rejects.toBeInstanceOf(AccountScopeExpiredError);
    expect(requestSignal?.aborted).toBe(true);
    expect(queryClient.getQueryData(options.queryKey)).toBeUndefined();
  });

  it('routes a semantic reference through one shared ranked name list and keys it by label-free identity', async () => {
    const queryClient = createQueryClient();
    const semanticFilter: GalleryItemsFilter = {
      ...baseFilter,
      semanticQuery: { fileId: 'external-3', kind: 'file', label: 'cat.png' },
    };
    const rankedNames = {
      items: [{ kind: 'image', name: 'ranked.png' }],
      total: 1,
    };

    backend.listSemanticGalleryItemNames.mockResolvedValue(rankedNames);
    backend.hydrateGalleryDateBoardItemPage.mockResolvedValue({ items: [], total: 1 });

    const options = galleryItemsInfiniteOptions(semanticFilter);

    await queryClient.fetchInfiniteQuery(options);
    expect(backend.listSemanticGalleryItemNames).toHaveBeenCalledWith(
      expect.objectContaining({ query: { fileId: 'external-3', kind: 'file' } })
    );
    expect(backend.hydrateGalleryDateBoardItemPage).toHaveBeenCalledWith(
      expect.objectContaining({ items: rankedNames.items, limit: GALLERY_PAGE_SIZE, offset: 0, total: 1 })
    );
    expect(backend.listGalleryItems).not.toHaveBeenCalled();

    // Range selection reads the same cached ranked list: a dropped file must not re-upload per consumer or page.
    await expect(queryClient.fetchQuery(galleryItemNamesOptions(semanticFilter))).resolves.toEqual(rankedNames);
    expect(backend.listSemanticGalleryItemNames).toHaveBeenCalledOnce();
    expect(backend.listGalleryItemNames).not.toHaveBeenCalled();

    // The label is presentation, not identity: relabels reuse the cache entry while a different registered file (or no
    // reference at all) does not.
    expect(
      galleryItemsInfiniteOptions({
        ...semanticFilter,
        semanticQuery: { fileId: 'external-3', kind: 'file', label: 'renamed.png' },
      }).queryKey
    ).toEqual(options.queryKey);
    expect(
      galleryItemsInfiniteOptions({
        ...semanticFilter,
        semanticQuery: { fileId: 'external-4', kind: 'file', label: 'cat.png' },
      }).queryKey
    ).not.toEqual(options.queryKey);
    expect(galleryItemsInfiniteOptions(baseFilter).queryKey).not.toEqual(options.queryKey);
  });

  it('anchors an infinite window at its offset, sharing the base key only at offset 0', async () => {
    // Base windows retain shared historical keys; deep reveals receive distinct transient window entries.
    expect(galleryItemsInfiniteOptions(baseFilter, { kind: 'infinite', offset: 0 }).queryKey).toEqual(
      galleryItemsInfiniteOptions(baseFilter).queryKey
    );
    expect(galleryItemsInfiniteOptions(baseFilter, { kind: 'infinite', offset: 6000 }).queryKey).not.toEqual(
      galleryItemsInfiniteOptions(baseFilter).queryKey
    );

    const queryClient = createQueryClient();
    backend.listGalleryItems.mockImplementation(({ offset }: { offset: number }) =>
      Promise.resolve(createPage({ offset, total: 20_000 }))
    );
    const options = galleryItemsInfiniteOptions(baseFilter, { kind: 'infinite', offset: 6000 });
    const observer = new InfiniteQueryObserver(queryClient, options);

    try {
      await observer.refetch();
      expect(observer.getCurrentResult().data?.pageParams).toEqual([6000]);

      // The GALLERY_MAX_ROWS reach applies from the anchor, not from 0.
      for (let fetches = 0; fetches < 12; fetches += 1) {
        await observer.fetchNextPage();
      }

      const pageParams = observer.getCurrentResult().data?.pageParams ?? [];

      expect(pageParams[0]).toBe(6000);
      expect(pageParams[pageParams.length - 1]).toBe(6000 + GALLERY_MAX_ROWS - GALLERY_PAGE_SIZE);
    } finally {
      observer.destroy();
    }
  });

  it('does not create item-list cache entries for repeated invalidation events', async () => {
    const queryClient = createQueryClient();
    backend.listGalleryItems.mockResolvedValue(createPage({ count: 1, offset: 0, total: 1 }));
    const options = galleryItemsInfiniteOptions(baseFilter);

    await queryClient.fetchInfiniteQuery(options);
    for (let event = 0; event < 100; event += 1) {
      await invalidateGalleryItems(queryClient);
    }

    expect(getGalleryItemListQueries(queryClient)).toHaveLength(1);
    expect(backend.listGalleryItems).toHaveBeenCalledOnce();
  });

  it('aborts the old request when an observer switches filters and isolates late completion', async () => {
    const queryClient = createQueryClient();
    const oldOptions = galleryItemsInfiniteOptions({ ...baseFilter, boardId: 'board-old' });
    const newOptions = galleryItemsInfiniteOptions({ ...baseFilter, boardId: 'board-new' });
    let oldRequestSignal: AbortSignal | undefined;
    let resolveOldRequest: ((page: GalleryItemsPage) => void) | undefined;

    backend.listGalleryItems.mockImplementation(({ boardId, signal }: { boardId: string; signal: AbortSignal }) => {
      if (boardId === 'board-old') {
        oldRequestSignal = signal;
        return new Promise<GalleryItemsPage>((resolve) => {
          resolveOldRequest = resolve;
        });
      }

      return Promise.resolve(createPage({ count: 1, offset: 0, prefix: 'new', total: 1 }));
    });

    const observer = new InfiniteQueryObserver(queryClient, oldOptions);
    const unsubscribe = observer.subscribe(() => undefined);

    try {
      await vi.waitFor(() => expect(backend.listGalleryItems).toHaveBeenCalledTimes(1));
      observer.setOptions(newOptions);
      await vi.waitFor(() => expect(flattenGalleryItemsData(observer.getCurrentResult().data)[0]?.name).toBe('new-0'));
      expect(oldRequestSignal?.aborted).toBe(true);

      resolveOldRequest?.(createPage({ count: 1, offset: 0, prefix: 'old', total: 1 }));
      await Promise.resolve();
      expect(queryClient.getQueryData(oldOptions.queryKey)).toBeUndefined();
    } finally {
      unsubscribe();
      observer.destroy();
    }
  });

  it('does not cancel a shared date-name request when one list consumer is cancelled', async () => {
    const queryClient = createQueryClient();
    let namesSignal: AbortSignal | undefined;
    let resolveNames: ((value: { items: { kind: 'image'; name: string }[]; total: number }) => void) | undefined;
    backend.listGalleryDateBoardItemNames.mockImplementation(
      ({ signal }: { signal: AbortSignal }) =>
        new Promise<{ items: { kind: 'image'; name: string }[]; total: number }>((resolve) => {
          namesSignal = signal;
          resolveNames = resolve;
        })
    );
    backend.hydrateGalleryDateBoardItemPage.mockResolvedValue(
      createPage({ count: 1, offset: 0, prefix: 'shared-date', total: 1 })
    );
    const filter = { ...baseFilter, boardId: 'by_date:2026-07-18' };
    const infiniteOptions = galleryItemsInfiniteOptions(filter);
    const pageOptions = galleryItemsInfiniteOptions(filter, { kind: 'anchor', offset: 0 });
    const infiniteRequest = queryClient.fetchInfiniteQuery(infiniteOptions);
    const pageRequest = queryClient.fetchInfiniteQuery(pageOptions);

    await vi.waitFor(() => {
      expect(backend.listGalleryDateBoardItemNames).toHaveBeenCalledOnce();
    });
    await queryClient.cancelQueries({ exact: true, queryKey: pageOptions.queryKey });

    expect(namesSignal?.aborted).toBe(false);
    resolveNames?.({ items: [{ kind: 'image', name: 'shared-date-0' }], total: 1 });
    await expect(infiniteRequest).resolves.toMatchObject({
      pages: [{ items: [{ name: 'shared-date-0' }], total: 1 }],
    });
    await pageRequest.catch(() => undefined);
  });
});

describe('canonicalizeGalleryItemsFilter under a semantic query', () => {
  const reference = { fileId: 'external-1-abc', kind: 'file', label: 'shot.png' } as const;

  it('ignores the controls a ranked result set does not answer to', () => {
    // Semantic results depend only on reference and board; other filters must not repeat uploads or downloads.
    const base = canonicalizeGalleryItemsFilter({
      boardId: 'board-a',
      galleryView: 'images',
      searchTerm: '',
      semanticQuery: reference,
    });

    for (const variant of [
      { galleryView: 'assets' as const },
      { orderDir: 'ASC' as const },
      { starred: true },
      { createdFrom: '2026-01-01' },
    ]) {
      expect(
        canonicalizeGalleryItemsFilter({
          boardId: 'board-a',
          galleryView: 'images',
          searchTerm: '',
          semanticQuery: reference,
          ...variant,
        })
      ).toEqual(base);
    }
  });
});

describe('galleryStarredStripOptions', () => {
  beforeEach(() => {
    backend.hydrateGalleryDateBoardItemPage.mockReset();
    backend.isDateBoardId.mockReset();
    backend.listGalleryDateBoardItemNames.mockReset();
    backend.listGalleryItems.mockReset();
    accountLifecycle.activate('strip-query-test');
    backend.isDateBoardId.mockImplementation((boardId: string) => boardId.startsWith('by_date:'));
  });

  it('reads one bounded starred-only range under the listing key family', async () => {
    const queryClient = createQueryClient();
    backend.listGalleryItems.mockResolvedValue(createPage({ count: 2, offset: 0, total: 9 }));

    const options = galleryStarredStripOptions(baseFilter);
    const page = await queryClient.fetchQuery(options);

    expect(page.total).toBe(9);
    // The strip lives in the list key family (so account-wide invalidation and mutation patching reach it) under the
    // listing's filter plus `starred`.
    expect(getGalleryItemListQueries(queryClient).map((query) => query.queryKey)).toEqual([options.queryKey]);
    expect(getGalleryItemsFilterFromKey(options.queryKey)).toEqual({
      ...canonicalizeGalleryItemsFilter(baseFilter),
      starred: true,
    });
    expect(isGalleryStarredStripQueryKey(options.queryKey)).toBe(true);
    expect(isGalleryStarredStripQueryKey(galleryItemsInfiniteOptions(baseFilter).queryKey)).toBe(false);
    expect(backend.listGalleryItems).toHaveBeenCalledOnce();
    expect(backend.listGalleryItems.mock.calls[0]?.[0]).toMatchObject({
      boardId: 'board-1',
      limit: GALLERY_STARRED_STRIP_LIMIT,
      offset: 0,
      starred: true,
    });
  });
});
