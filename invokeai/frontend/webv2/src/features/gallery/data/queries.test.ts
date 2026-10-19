import type { GalleryItem, GalleryItemsPage } from '@features/gallery/core/items';
import type { GallerySemanticReference } from '@features/gallery/core/semanticImageQuery';

import { AccountScopeExpiredError, accountLifecycle } from '@platform/state/accountLifecycle';
import { InfiniteQueryObserver, QueryClient, QueryObserver, type InfiniteData } from '@tanstack/react-query';
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
  galleryItemsTotalOptions,
  galleryItemsInfiniteOptions,
  galleryStarredStripOptions,
  getGalleryItemListQueries,
  IMAGE_INDEX_UNAVAILABLE_POLL_MS,
  imageIndexAvailabilityOptions,
  getGalleryItemsFilterFromKey,
  isGalleryStarredStripQueryKey,
  canonicalizeGalleryItemsFilter,
  type GalleryItemsFilter,
} from './queries';
import { invalidateGalleryItems } from './queryCache';

const OFFSETS = Array.from({ length: 10 }, (_, index) => index * GALLERY_PAGE_SIZE);

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
  starred: false,
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

  it('fetches one normalized sparse page through the shared range reader', async () => {
    const queryClient = createQueryClient();
    backend.listGalleryItems.mockImplementation(({ limit, offset }: { limit: number; offset: number }) =>
      Promise.resolve(createPage({ count: limit, offset, total: 240 }))
    );
    const options = galleryItemsPageOptions(baseFilter, 119);

    const page = await queryClient.fetchQuery(options);

    expect(backend.listGalleryItems).toHaveBeenCalledWith(expect.objectContaining({ limit: 60, offset: 60 }));
    expect(page).toMatchObject({ itemIndices: Array.from({ length: 60 }, (_, index) => 60 + index), offset: 60 });
    expect(getGalleryItemListQueries(queryClient).map((query) => query.queryKey)).toEqual([options.queryKey]);
  });

  it('shares a count-only read on the account and listing key without fetching item rows', async () => {
    const queryClient = createQueryClient();
    let total = 62;
    backend.listGalleryItems.mockImplementation(() => Promise.resolve({ items: [], total }));
    const options = galleryItemsTotalOptions(baseFilter);

    await expect(
      Promise.all([
        queryClient.fetchQuery(options),
        queryClient.fetchQuery(galleryItemsTotalOptions({ ...baseFilter, searchTerm: ' portrait ' })),
      ])
    ).resolves.toEqual([62, 62]);

    expect(backend.listGalleryItems).toHaveBeenCalledOnce();
    expect(backend.listGalleryItems).toHaveBeenCalledWith(expect.objectContaining({ limit: 0, offset: 0 }));
    expect(options.queryKey).toEqual(galleryItemsTotalOptions(baseFilter).queryKey);
    expect(options.queryKey).not.toEqual(galleryItemsTotalOptions({ ...baseFilter, boardId: 'board-2' }).queryKey);
    total = 63;
    await invalidateGalleryItems(queryClient);
    await expect(queryClient.fetchQuery(options)).resolves.toBe(63);
    expect(backend.listGalleryItems).toHaveBeenCalledTimes(2);
    accountLifecycle.activate('gallery-query-test-count-transition');
    expect(galleryItemsTotalOptions(baseFilter).queryKey).not.toEqual(options.queryKey);
  });

  it('discovers date-board totals through its shared name metadata without hydrating rows', async () => {
    const queryClient = createQueryClient();
    const dateFilter = { ...baseFilter, boardId: 'by_date:2026-07-18' };
    backend.listGalleryDateBoardItemNames.mockResolvedValue({
      items: [{ kind: 'image', name: 'date-image' }],
      total: 1,
    });
    backend.hydrateGalleryDateBoardItemPage.mockResolvedValue({ items: [], offset: 0, total: 1 });

    await expect(queryClient.fetchQuery(galleryItemsTotalOptions(dateFilter))).resolves.toBe(1);
    expect(backend.listGalleryDateBoardItemNames).toHaveBeenCalledOnce();
    expect(backend.hydrateGalleryDateBoardItemPage).toHaveBeenCalledWith(
      expect.objectContaining({ limit: 0, offset: 0, total: 1 })
    );
  });

  it('keeps short final API pages aligned to their requested absolute offset', async () => {
    const queryClient = createQueryClient();
    backend.listGalleryItems.mockResolvedValue(createPage({ count: 2, offset: 60, total: 62 }));

    const page = await queryClient.fetchQuery(galleryItemsPageOptions(baseFilter, 60));

    expect(page).toMatchObject({ itemIndices: [60, 61], offset: 60, total: 62 });
  });

  it('truncates an overlong result with its absolute indices still aligned', async () => {
    const queryClient = createQueryClient();
    backend.listGalleryItems.mockResolvedValue(createPage({ count: GALLERY_PAGE_SIZE + 2, offset: 60, total: 240 }));

    const page = await queryClient.fetchQuery(galleryItemsPageOptions(baseFilter, 60));

    expect(page.items).toHaveLength(GALLERY_PAGE_SIZE);
    expect(page.itemIndices).toEqual(Array.from({ length: GALLERY_PAGE_SIZE }, (_, index) => 60 + index));
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

  it('keeps date-board sparse pages on the existing shared hydration path', async () => {
    const queryClient = createQueryClient();
    backend.listGalleryDateBoardItemNames.mockResolvedValue({
      items: Array.from({ length: 180 }, (_, index) => ({ kind: 'image' as const, name: `date-${index}` })),
      total: 180,
    });
    backend.hydrateGalleryDateBoardItemPage.mockResolvedValue(createPage({ offset: 60, total: 180 }));

    await queryClient.fetchQuery(galleryItemsPageOptions({ ...baseFilter, boardId: 'by_date:2026-07-18' }, 119));

    expect(backend.listGalleryDateBoardItemNames).toHaveBeenCalledOnce();
    expect(backend.hydrateGalleryDateBoardItemPage).toHaveBeenCalledWith(
      expect.objectContaining({ limit: 60, offset: 60 })
    );
    expect(backend.listGalleryItems).not.toHaveBeenCalled();
  });

  it('refetches active sparse pages during Gallery list invalidation', async () => {
    const queryClient = createQueryClient();
    backend.listGalleryItems.mockImplementation(({ offset }: { offset: number }) =>
      Promise.resolve(createPage({ offset, prefix: `page-${backend.listGalleryItems.mock.calls.length}`, total: 240 }))
    );
    const observer = new QueryObserver(queryClient, galleryItemsPageOptions(baseFilter, 60));
    const unsubscribe = observer.subscribe(() => undefined);

    try {
      await vi.waitFor(() => expect(backend.listGalleryItems).toHaveBeenCalledOnce());
      await invalidateGalleryItems(queryClient);

      expect(backend.listGalleryItems).toHaveBeenCalledTimes(2);
      expect(observer.getCurrentResult().data?.items[0]?.name).toBe('page-2-60');
    } finally {
      unsubscribe();
      observer.destroy();
    }
  });

  it('loads ten fixed pages into one bounded logical query', async () => {
    const queryClient = createQueryClient();
    backend.listGalleryItems.mockImplementation(({ offset }: { offset: number }) =>
      Promise.resolve(createPage({ offset, total: 1_000 }))
    );
    const options = galleryItemsInfiniteOptions(baseFilter);
    const observer = new InfiniteQueryObserver(queryClient, options);

    try {
      await observer.refetch();
      for (let page = 1; page < OFFSETS.length; page += 1) {
        await observer.fetchNextPage();
      }

      const data = observer.getCurrentResult().data;
      const cachedData = queryClient.getQueryData<InfiniteData<GalleryItemsPage, number>>(options.queryKey);

      expect(backend.listGalleryItems.mock.calls.map(([request]) => request.offset)).toEqual(OFFSETS);
      expect(data?.pageParams).toEqual(OFFSETS);
      expect(data?.pages.flatMap((page) => page.items)).toHaveLength(GALLERY_MAX_ROWS);
      expect(flattenGalleryItemsData(data)).toHaveLength(GALLERY_MAX_ROWS);
      expect(flattenGalleryItemsData(cachedData)).toHaveLength(GALLERY_MAX_ROWS);
      expect(observer.getCurrentResult().hasNextPage).toBe(false);
      expect(getGalleryItemListQueries(queryClient)).toHaveLength(1);
    } finally {
      observer.destroy();
    }
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

    // Range selection reads the same cached ranked list: a dropped-file
    // reference must not re-upload its blob once per consumer or per page.
    await expect(queryClient.fetchQuery(galleryItemNamesOptions(semanticFilter))).resolves.toEqual(rankedNames);
    expect(backend.listSemanticGalleryItemNames).toHaveBeenCalledOnce();
    expect(backend.listGalleryItemNames).not.toHaveBeenCalled();

    // The label is presentation, not identity: relabels reuse the cache entry
    // while a different registered file (or no reference at all) does not.
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

  it('preserves sparse absolute indices from semantic hydration', async () => {
    const queryClient = createQueryClient();
    const semanticFilter: GalleryItemsFilter = {
      ...baseFilter,
      semanticQuery: { imageName: 'reference.png', kind: 'image' },
    };
    const itemIndices = [60, 62];
    const items = [createItem(60, 'semantic'), createItem(62, 'semantic')];

    backend.listSemanticGalleryItemNames.mockResolvedValue({
      items: Array.from({ length: 63 }, (_, index) => ({ kind: 'image' as const, name: `rank-${index}` })),
      total: 63,
    });
    backend.hydrateGalleryDateBoardItemPage.mockResolvedValue({ items, itemIndices, offset: 60, total: 63 });

    const page = await queryClient.fetchQuery(galleryItemsPageOptions(semanticFilter, 60));

    expect(page).toEqual({ items, itemIndices, offset: 60, total: 63 });
  });

  it('keeps semantic filters in the key while page params stay inside one cache entry', async () => {
    const baseKey = galleryItemsInfiniteOptions(baseFilter).queryKey;

    expect(galleryItemsInfiniteOptions({ ...baseFilter, searchTerm: ' portrait ' }).queryKey).toEqual(baseKey);
    expect(galleryItemsInfiniteOptions({ ...baseFilter, boardId: 'board-2' }).queryKey).not.toEqual(baseKey);
    expect(galleryItemsInfiniteOptions({ ...baseFilter, galleryView: 'assets' }).queryKey).not.toEqual(baseKey);
    expect(galleryItemsInfiniteOptions({ ...baseFilter, searchTerm: 'landscape' }).queryKey).not.toEqual(baseKey);
    expect(galleryItemsInfiniteOptions({ ...baseFilter, createdFrom: '2026-07-01' }).queryKey).not.toEqual(baseKey);

    const queryClient = createQueryClient();
    backend.listGalleryItems.mockImplementation(({ offset }: { offset: number }) =>
      Promise.resolve(createPage({ offset, total: 120 }))
    );
    const options = galleryItemsInfiniteOptions(baseFilter);
    const observer = new InfiniteQueryObserver(queryClient, options);

    try {
      await observer.refetch();
      await observer.fetchNextPage();

      expect(observer.getCurrentResult().data?.pageParams).toEqual([0, 60]);
      expect(getGalleryItemListQueries(queryClient)).toHaveLength(1);
      expect(getGalleryItemListQueries(queryClient)[0]?.queryKey).toEqual(baseKey);
    } finally {
      observer.destroy();
    }
  });

  it('releases inactive anchor windows immediately', async () => {
    const queryClient = createQueryClient();
    backend.listGalleryItems.mockImplementation(({ offset }: { offset: number }) =>
      Promise.resolve(createPage({ offset, total: 1_000 }))
    );

    for (const offset of Array.from({ length: 11 }, (_, index) => index * GALLERY_PAGE_SIZE)) {
      await queryClient.fetchInfiniteQuery(galleryItemsInfiniteOptions(baseFilter, { kind: 'anchor', offset }));
    }

    await vi.waitFor(() => {
      expect(getGalleryItemListQueries(queryClient).length).toBeLessThanOrEqual(1);
    });
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

  it('never grows an anchored infinite window above its anchor, but still lets paginated anchors', () => {
    // Infinite windows must not prepend and shift the viewport. Paginated consumers select by pageParam, allowing
    // Preview to load backward safely.
    const page: GalleryItemsPage = { items: [], total: 20_000 };
    const onePage = [page];
    const anchored = galleryItemsInfiniteOptions(baseFilter, { kind: 'infinite', offset: 6000 });
    const base = galleryItemsInfiniteOptions(baseFilter);
    const paginated = galleryItemsInfiniteOptions(baseFilter, { kind: 'anchor', offset: 6000 });

    expect(anchored.getPreviousPageParam?.(page, onePage, 6000, [6000])).toBeUndefined();
    expect(anchored.getPreviousPageParam?.(page, onePage, 6060, [6060])).toBe(6000);
    expect(base.getPreviousPageParam?.(page, onePage, 0, [0])).toBeUndefined();
    expect(base.getPreviousPageParam?.(page, onePage, GALLERY_PAGE_SIZE, [GALLERY_PAGE_SIZE])).toBe(0);
    expect(paginated.getPreviousPageParam?.(page, onePage, 6000, [6000])).toBe(6000 - GALLERY_PAGE_SIZE);
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

  it('fetches one date-board ref list while hydrating multiple fixed pages', async () => {
    const queryClient = createQueryClient();
    const refs = Array.from({ length: 180 }, (_, index) => ({ kind: 'image' as const, name: `date-${index}` }));
    backend.listGalleryDateBoardItemNames.mockResolvedValue({ items: refs, total: refs.length });
    backend.hydrateGalleryDateBoardItemPage.mockImplementation(
      ({ limit, offset, total }: { limit: number; offset: number; total: number }) =>
        Promise.resolve({
          ...createPage({ count: limit, offset, prefix: 'date', total }),
          itemIndices: Array.from({ length: limit }, (_, index) => offset + index),
          offset,
        })
    );
    const options = galleryItemsInfiniteOptions({ ...baseFilter, boardId: 'by_date:2026-07-18' });
    const observer = new InfiniteQueryObserver(queryClient, options);

    try {
      await observer.refetch();
      await observer.fetchNextPage();
      await observer.fetchNextPage();

      expect(backend.listGalleryDateBoardItemNames).toHaveBeenCalledOnce();
      expect(backend.hydrateGalleryDateBoardItemPage.mock.calls.map(([request]) => request.offset)).toEqual([
        0, 60, 120,
      ]);
      const data = observer.getCurrentResult().data;

      expect(flattenGalleryItemsData(data)).toHaveLength(180);
      expect(data?.pages.map((page) => Object.keys(page).sort())).toEqual(
        Array.from({ length: 3 }, () => ['items', 'total'])
      );
      expect(backend.listGalleryItems).not.toHaveBeenCalled();

      backend.listGalleryDateBoardItemNames.mockResolvedValueOnce({
        items: [{ kind: 'image', name: 'refreshed' }, ...refs],
        total: refs.length + 1,
      });
      await invalidateGalleryItems(queryClient);
      await observer.refetch();
      expect(backend.listGalleryDateBoardItemNames).toHaveBeenCalledTimes(2);
    } finally {
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
    // Semantic results depend only on the reference and its board; unrelated filters must not repeat blob uploads
    // or remote downloads.
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

  it('keys a search by the board it ranks within, but a map cluster by its members alone', () => {
    const inBoard = (boardId: string, semanticQuery: GallerySemanticReference) =>
      canonicalizeGalleryItemsFilter({ boardId, galleryView: 'images', searchTerm: '', semanticQuery });
    const cluster = { clusterId: 'cluster-1', kind: 'cluster', label: 'Cats' } as const;

    expect(inBoard('board-b', reference)).not.toEqual(inBoard('board-a', reference));
    expect(inBoard('board-b', cluster)).toEqual(inBoard('board-a', cluster));
  });

  it('still distinguishes one reference from another', () => {
    const other = canonicalizeGalleryItemsFilter({
      boardId: 'board-a',
      galleryView: 'images',
      searchTerm: '',
      semanticQuery: { fileId: 'external-2-def', kind: 'file', label: 'shot.png' },
    });

    expect(other).not.toEqual(
      canonicalizeGalleryItemsFilter({
        boardId: 'board-a',
        galleryView: 'images',
        searchTerm: '',
        semanticQuery: reference,
      })
    );
  });

  it('leaves a non-semantic filter keyed on all of its controls', () => {
    const onBoardA = canonicalizeGalleryItemsFilter({ boardId: 'board-a', galleryView: 'images', searchTerm: '' });
    const onBoardB = canonicalizeGalleryItemsFilter({ boardId: 'board-b', galleryView: 'images', searchTerm: '' });

    expect(onBoardA).not.toEqual(onBoardB);
  });

  it('keys the starred filter only when it is set', () => {
    const unfiltered = canonicalizeGalleryItemsFilter({ boardId: 'board-a', galleryView: 'images', searchTerm: '' });
    const starredOnly = canonicalizeGalleryItemsFilter({
      boardId: 'board-a',
      galleryView: 'images',
      searchTerm: '',
      starred: true,
    });

    expect(unfiltered).not.toHaveProperty('starred');
    expect(starredOnly).toEqual({ ...unfiltered, starred: true });
  });
});

describe('imageIndexAvailabilityOptions', () => {
  const pollFor = (state: { status: 'error' | 'success'; data?: { modelName: string | null; state: string } }) => {
    const { refetchInterval } = imageIndexAvailabilityOptions();

    return typeof refetchInterval === 'function'
      ? refetchInterval({ state } as unknown as Parameters<typeof refetchInterval>[0])
      : refetchInterval;
  };

  it('polls only while the answer can still change on its own', () => {
    // Retry missing-model and failed statuses; settled readiness must not poll for the whole session.
    expect(pollFor({ data: { modelName: 'clip', state: 'model_missing' }, status: 'success' })).toBe(
      IMAGE_INDEX_UNAVAILABLE_POLL_MS
    );
    expect(pollFor({ data: { modelName: null, state: 'switching' }, status: 'success' })).toBe(
      IMAGE_INDEX_UNAVAILABLE_POLL_MS
    );
    expect(pollFor({ status: 'error' })).toBe(IMAGE_INDEX_UNAVAILABLE_POLL_MS);
    expect(pollFor({ data: { modelName: null, state: 'ready' }, status: 'success' })).toBe(false);
    expect(pollFor({ data: { modelName: null, state: 'disabled' }, status: 'success' })).toBe(false);
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
    // The strip lives in the list key family (so account-wide invalidation
    // and mutation patching reach it) under the listing's filter plus `starred`.
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

  it('routes a date board through its starred-only name list', async () => {
    const queryClient = createQueryClient();
    const refs = [{ kind: 'image' as const, name: 'starred-0' }];
    backend.listGalleryDateBoardItemNames.mockResolvedValue({ items: refs, total: 1 });
    backend.hydrateGalleryDateBoardItemPage.mockResolvedValue({
      ...createPage({ count: 1, offset: 0, total: 1 }),
      itemIndices: [0],
      offset: 0,
    });

    const page = await queryClient.fetchQuery(
      galleryStarredStripOptions({ ...baseFilter, boardId: 'by_date:2026-07-18' })
    );

    expect(page).toEqual({ items: [createItem(0)], total: 1 });

    expect(backend.listGalleryDateBoardItemNames.mock.calls[0]?.[0]).toMatchObject({
      boardId: 'by_date:2026-07-18',
      starred: true,
    });
    expect(backend.hydrateGalleryDateBoardItemPage).toHaveBeenCalledWith(
      expect.objectContaining({ items: refs, limit: GALLERY_STARRED_STRIP_LIMIT, offset: 0 })
    );
    expect(backend.listGalleryItems).not.toHaveBeenCalled();
  });
});
