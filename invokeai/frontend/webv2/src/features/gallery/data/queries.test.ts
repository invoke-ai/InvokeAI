import type { GalleryImage, GalleryImagesPage } from '@features/gallery/core/types';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { InfiniteQueryObserver, QueryClient, type InfiniteData } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const backend = vi.hoisted(() => ({
  hydrateGalleryDateBoardImagePage: vi.fn(),
  isDateBoardId: vi.fn(),
  listGalleryBoards: vi.fn(),
  listGalleryDateBoardImageNames: vi.fn(),
  listGalleryDateBoards: vi.fn(),
  listGalleryImages: vi.fn(),
}));

vi.mock('./backend', () => backend);

import {
  flattenGalleryImagesData,
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
  galleryBoardsOptions,
  galleryImagesInfiniteOptions,
  getGalleryImageListQueries,
  type GalleryImagesFilter,
} from './queries';
import { invalidateGalleryImages } from './queryCache';

const OFFSETS = Array.from({ length: 10 }, (_, index) => index * GALLERY_PAGE_SIZE);

const createQueryClient = (): QueryClient =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

const createImage = (index: number, prefix = 'image'): GalleryImage => ({
  boardId: 'board-1',
  height: 64,
  imageCategory: 'general',
  imageName: `${prefix}-${index}`,
  imageUrl: `/images/${prefix}-${index}`,
  queuedAt: new Date(index * 1_000).toISOString(),
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
}): GalleryImagesPage => ({
  images: Array.from({ length: count }, (_, index) => createImage(offset + index, prefix)),
  total,
});

const baseFilter: GalleryImagesFilter = {
  boardId: 'board-1',
  galleryView: 'images',
  orderDir: 'DESC',
  searchTerm: 'portrait',
  starredFirst: true,
};

describe('Gallery query read model', () => {
  beforeEach(() => {
    accountLifecycle.activate('gallery-query-test');
    backend.hydrateGalleryDateBoardImagePage.mockReset();
    backend.isDateBoardId.mockReset();
    backend.listGalleryBoards.mockReset();
    backend.listGalleryDateBoardImageNames.mockReset();
    backend.listGalleryDateBoards.mockReset();
    backend.listGalleryImages.mockReset();

    backend.isDateBoardId.mockImplementation((boardId: string) => boardId.startsWith('by_date:'));
    backend.listGalleryImages.mockResolvedValue({ images: [], total: 0 });
  });

  afterEach(() => {
    accountLifecycle.invalidate();
  });

  it('coalesces board readers through one query key and includes requested virtual boards', async () => {
    const queryClient = createQueryClient();
    const board = { id: 'none', kind: 'uncategorized', name: 'Uncategorized' };
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
    backend.listGalleryImages.mockImplementation(({ offset }: { offset: number }) =>
      Promise.resolve(createPage({ offset, total: 1_000 }))
    );
    const options = galleryImagesInfiniteOptions(baseFilter);
    const observer = new InfiniteQueryObserver(queryClient, options);

    try {
      await observer.refetch();
      for (let page = 1; page < OFFSETS.length; page += 1) {
        await observer.fetchNextPage();
      }

      const data = observer.getCurrentResult().data;
      const cachedData = queryClient.getQueryData<InfiniteData<GalleryImagesPage, number>>(options.queryKey);

      expect(backend.listGalleryImages.mock.calls.map(([request]) => request.offset)).toEqual(OFFSETS);
      expect(backend.listGalleryImages.mock.calls.map(([request]) => request.limit)).toEqual(
        OFFSETS.map(() => GALLERY_PAGE_SIZE)
      );
      expect(data?.pageParams).toEqual(OFFSETS);
      expect(data?.pages.flatMap((page) => page.images)).toHaveLength(GALLERY_MAX_ROWS);
      expect(flattenGalleryImagesData(data)).toHaveLength(GALLERY_MAX_ROWS);
      expect(flattenGalleryImagesData(cachedData)).toHaveLength(GALLERY_MAX_ROWS);
      expect(observer.getCurrentResult().hasNextPage).toBe(false);
      expect(getGalleryImageListQueries(queryClient)).toHaveLength(1);
    } finally {
      observer.destroy();
    }
  });

  it('keeps semantic filters in the key while infinite page params stay inside one cache entry', async () => {
    const baseKey = galleryImagesInfiniteOptions(baseFilter).queryKey;

    expect(galleryImagesInfiniteOptions({ ...baseFilter, searchTerm: ' portrait ' }).queryKey).toEqual(baseKey);
    expect(galleryImagesInfiniteOptions({ ...baseFilter, boardId: 'board-2' }).queryKey).not.toEqual(baseKey);
    expect(galleryImagesInfiniteOptions({ ...baseFilter, galleryView: 'assets' }).queryKey).not.toEqual(baseKey);
    expect(galleryImagesInfiniteOptions({ ...baseFilter, searchTerm: 'landscape' }).queryKey).not.toEqual(baseKey);
    expect(galleryImagesInfiniteOptions({ ...baseFilter, createdFrom: '2026-07-01' }).queryKey).not.toEqual(baseKey);
    expect(
      galleryImagesInfiniteOptions({
        ...baseFilter,
        createdFrom: '2026-07-01',
        createdTo: '2026-07-15',
      }).queryKey
    ).not.toEqual(
      galleryImagesInfiniteOptions({
        ...baseFilter,
        createdFrom: '2026-07-01',
        createdTo: '2026-07-16',
      }).queryKey
    );

    const queryClient = createQueryClient();
    backend.listGalleryImages.mockImplementation(({ offset }: { offset: number }) =>
      Promise.resolve(createPage({ offset, total: 120 }))
    );
    const options = galleryImagesInfiniteOptions(baseFilter);
    const observer = new InfiniteQueryObserver(queryClient, options);

    try {
      await observer.refetch();
      await observer.fetchNextPage();

      expect(observer.getCurrentResult().data?.pageParams).toEqual([0, 60]);
      expect(getGalleryImageListQueries(queryClient)).toHaveLength(1);
      expect(getGalleryImageListQueries(queryClient)[0]?.queryKey).toEqual(baseKey);
    } finally {
      observer.destroy();
    }
  });

  it.each([{ kind: 'anchor' as const }])(
    'releases inactive $kind windows immediately instead of accumulating page keys',
    async ({ kind }) => {
      const queryClient = createQueryClient();
      backend.listGalleryImages.mockImplementation(({ offset }: { offset: number }) =>
        Promise.resolve(createPage({ offset, total: 1_000 }))
      );

      for (const offset of Array.from({ length: 11 }, (_, index) => index * GALLERY_PAGE_SIZE)) {
        await queryClient.fetchInfiniteQuery(galleryImagesInfiniteOptions(baseFilter, { kind, offset }));
      }

      await vi.waitFor(() => {
        expect(getGalleryImageListQueries(queryClient).length).toBeLessThanOrEqual(1);
      });
      const cachedRows = getGalleryImageListQueries(queryClient).reduce((total, query) => {
        const data = query.state.data as InfiniteData<GalleryImagesPage, number> | undefined;

        return total + (data?.pages.reduce((count, page) => count + page.images.length, 0) ?? 0);
      }, 0);

      expect(cachedRows).toBeLessThanOrEqual(GALLERY_PAGE_SIZE);
    }
  );

  it('does not create image-list cache entries for repeated invalidation events', async () => {
    const queryClient = createQueryClient();
    backend.listGalleryImages.mockResolvedValue(createPage({ count: 1, offset: 0, total: 1 }));
    const options = galleryImagesInfiniteOptions(baseFilter);

    await queryClient.fetchInfiniteQuery(options);
    expect(getGalleryImageListQueries(queryClient)).toHaveLength(1);

    for (let event = 0; event < 100; event += 1) {
      await invalidateGalleryImages(queryClient);
    }

    expect(getGalleryImageListQueries(queryClient)).toHaveLength(1);
    expect(getGalleryImageListQueries(queryClient)[0]?.queryKey).toEqual(options.queryKey);
    expect(backend.listGalleryImages).toHaveBeenCalledOnce();
  });

  it('aborts the old request when an observer switches filters and isolates its late completion', async () => {
    const queryClient = createQueryClient();
    const oldOptions = galleryImagesInfiniteOptions({ ...baseFilter, boardId: 'board-old' });
    const newOptions = galleryImagesInfiniteOptions({ ...baseFilter, boardId: 'board-new' });
    let oldRequestSignal: AbortSignal | undefined;
    let resolveOldRequest: ((page: GalleryImagesPage) => void) | undefined;
    let oldBackendCompleted = false;

    backend.listGalleryImages.mockImplementation(({ boardId, signal }: { boardId: string; signal: AbortSignal }) => {
      if (boardId === 'board-old') {
        oldRequestSignal = signal;

        return new Promise<GalleryImagesPage>((resolve) => {
          resolveOldRequest = (page) => {
            oldBackendCompleted = true;
            resolve(page);
          };
        });
      }

      return Promise.resolve(createPage({ count: 1, offset: 0, prefix: 'new', total: 1 }));
    });

    const observer = new InfiniteQueryObserver(queryClient, oldOptions);
    const unsubscribe = observer.subscribe(() => undefined);

    try {
      await vi.waitFor(() => {
        expect(backend.listGalleryImages).toHaveBeenCalledTimes(1);
      });

      observer.setOptions(newOptions);

      await vi.waitFor(() => {
        expect(flattenGalleryImagesData(observer.getCurrentResult().data)[0]?.imageName).toBe('new-0');
      });
      expect(oldRequestSignal?.aborted).toBe(true);

      resolveOldRequest?.(createPage({ count: 1, offset: 0, prefix: 'old', total: 1 }));
      await vi.waitFor(() => {
        expect(oldBackendCompleted).toBe(true);
      });

      expect(queryClient.getQueryData(oldOptions.queryKey)).toBeUndefined();
      expect(
        flattenGalleryImagesData(
          queryClient.getQueryData<InfiniteData<GalleryImagesPage, number>>(newOptions.queryKey)
        )[0]?.imageName
      ).toBe('new-0');
    } finally {
      unsubscribe();
      observer.destroy();
    }
  });

  it('fetches a date-board name list once while hydrating multiple fixed pages', async () => {
    const queryClient = createQueryClient();
    const imageNames = Array.from({ length: 180 }, (_, index) => `date-${index}`);
    backend.listGalleryDateBoardImageNames.mockResolvedValue({ imageNames, total: imageNames.length });
    backend.hydrateGalleryDateBoardImagePage.mockImplementation(
      ({
        imageNames: names,
        limit,
        offset,
        total,
      }: {
        imageNames: string[];
        limit: number;
        offset: number;
        total: number;
      }) =>
        Promise.resolve({
          images: names.slice(offset, offset + limit).map((_, index) => createImage(offset + index, 'date')),
          total,
        })
    );
    const options = galleryImagesInfiniteOptions({
      ...baseFilter,
      boardId: 'by_date:2026-07-18',
      createdFrom: '2026-07-01',
      createdTo: '2026-07-31',
    });
    const observer = new InfiniteQueryObserver(queryClient, options);

    try {
      await observer.refetch();
      await observer.fetchNextPage();
      await observer.fetchNextPage();

      expect(backend.listGalleryDateBoardImageNames).toHaveBeenCalledOnce();
      expect(backend.hydrateGalleryDateBoardImagePage.mock.calls.map(([request]) => request.offset)).toEqual([
        0, 60, 120,
      ]);
      expect(backend.hydrateGalleryDateBoardImagePage.mock.calls.map(([request]) => request.limit)).toEqual([
        60, 60, 60,
      ]);
      expect(observer.getCurrentResult().data?.pageParams).toEqual([0, 60, 120]);
      expect(flattenGalleryImagesData(observer.getCurrentResult().data)).toHaveLength(180);
      expect(backend.listGalleryImages).not.toHaveBeenCalled();

      backend.listGalleryDateBoardImageNames.mockResolvedValueOnce({
        imageNames: [`refreshed-date-image`, ...imageNames],
        total: imageNames.length + 1,
      });
      await invalidateGalleryImages(queryClient);
      await observer.refetch();

      expect(backend.listGalleryDateBoardImageNames).toHaveBeenCalledTimes(2);
    } finally {
      observer.destroy();
    }
  });

  it('does not cancel a shared date-name request when one list consumer is cancelled', async () => {
    const queryClient = createQueryClient();
    let namesSignal: AbortSignal | undefined;
    let resolveNames: ((value: { imageNames: string[]; total: number }) => void) | undefined;
    backend.listGalleryDateBoardImageNames.mockImplementation(
      ({ signal }: { signal: AbortSignal }) =>
        new Promise<{ imageNames: string[]; total: number }>((resolve) => {
          namesSignal = signal;
          resolveNames = resolve;
        })
    );
    backend.hydrateGalleryDateBoardImagePage.mockResolvedValue(
      createPage({ count: 1, offset: 0, prefix: 'shared-date', total: 1 })
    );
    const filter = { ...baseFilter, boardId: 'by_date:2026-07-18' };
    const infiniteOptions = galleryImagesInfiniteOptions(filter);
    const pageOptions = galleryImagesInfiniteOptions(filter, { kind: 'anchor', offset: 0 });
    const infiniteRequest = queryClient.fetchInfiniteQuery(infiniteOptions);
    const pageRequest = queryClient.fetchInfiniteQuery(pageOptions);

    await vi.waitFor(() => {
      expect(queryClient.getQueryState(infiniteOptions.queryKey)?.fetchStatus).toBe('fetching');
      expect(queryClient.getQueryState(pageOptions.queryKey)?.fetchStatus).toBe('fetching');
      expect(backend.listGalleryDateBoardImageNames).toHaveBeenCalledOnce();
    });

    await queryClient.cancelQueries({ exact: true, queryKey: pageOptions.queryKey });

    expect(namesSignal?.aborted).toBe(false);
    resolveNames?.({ imageNames: ['shared-date-0'], total: 1 });
    await expect(infiniteRequest).resolves.toMatchObject({
      pages: [{ images: [{ imageName: 'shared-date-0' }], total: 1 }],
    });
    await pageRequest.catch(() => undefined);
  });
});
