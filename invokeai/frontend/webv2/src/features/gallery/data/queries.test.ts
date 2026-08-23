import type { GalleryItem, GalleryItemsPage } from '@features/gallery/core/items';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { InfiniteQueryObserver, QueryClient, type InfiniteData } from '@tanstack/react-query';
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
  galleryBoardsOptions,
  galleryItemNamesOptions,
  galleryItemsInfiniteOptions,
  getGalleryItemListQueries,
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
  starredFirst: true,
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
      starredCount: 0,
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
    // A zero-offset window must keep the historical key so every existing
    // consumer shares one cache entry; a deep window (a reveal past the base
    // reach) is its own transient entry starting at its own page.
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

  it('fetches one date-board ref list while hydrating multiple fixed pages', async () => {
    const queryClient = createQueryClient();
    const refs = Array.from({ length: 180 }, (_, index) => ({ kind: 'image' as const, name: `date-${index}` }));
    backend.listGalleryDateBoardItemNames.mockResolvedValue({ items: refs, starredCount: 0, total: refs.length });
    backend.hydrateGalleryDateBoardItemPage.mockImplementation(
      ({ limit, offset, total }: { limit: number; offset: number; total: number }) =>
        Promise.resolve(createPage({ count: limit, offset, prefix: 'date', total }))
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
      expect(flattenGalleryItemsData(observer.getCurrentResult().data)).toHaveLength(180);
      expect(backend.listGalleryItems).not.toHaveBeenCalled();

      backend.listGalleryDateBoardItemNames.mockResolvedValueOnce({
        items: [{ kind: 'image', name: 'refreshed' }, ...refs],
        starredCount: 0,
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
    let resolveNames:
      | ((value: { items: { kind: 'image'; name: string }[]; starredCount: number; total: number }) => void)
      | undefined;
    backend.listGalleryDateBoardItemNames.mockImplementation(
      ({ signal }: { signal: AbortSignal }) =>
        new Promise<{ items: { kind: 'image'; name: string }[]; starredCount: number; total: number }>((resolve) => {
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
    resolveNames?.({ items: [{ kind: 'image', name: 'shared-date-0' }], starredCount: 0, total: 1 });
    await expect(infiniteRequest).resolves.toMatchObject({
      pages: [{ items: [{ name: 'shared-date-0' }], total: 1 }],
    });
    await pageRequest.catch(() => undefined);
  });
});

describe('canonicalizeGalleryItemsFilter under a semantic query', () => {
  const reference = { fileId: 'external-1-abc', kind: 'file', label: 'shot.png' } as const;

  it('ignores the controls a ranked result set does not answer to', () => {
    // The semantic branch sends only the reference, so board, view, order,
    // starred-first and the date range change nothing about the response.
    // While they stayed in the key, clicking a board minted a fresh key and
    // re-ran the search — re-uploading the dropped blob, or making the server
    // re-download a remote URL, to render byte-identical results.
    const base = canonicalizeGalleryItemsFilter({
      boardId: 'board-a',
      galleryView: 'images',
      searchTerm: '',
      semanticQuery: reference,
    });

    for (const variant of [
      { boardId: 'board-b' },
      { galleryView: 'assets' as const },
      { orderDir: 'ASC' as const },
      { starredFirst: true },
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
});
