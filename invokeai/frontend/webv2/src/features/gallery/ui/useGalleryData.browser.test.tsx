import type { GalleryItem } from '@features/gallery/core/items';
import type { GalleryImage } from '@features/gallery/core/types';

import { getGallerySettings } from '@features/gallery/core/settings';
import { fetchGalleryItemsPage } from '@features/gallery/data/queries';
import { invalidateGallery, patchGalleryItemCaches } from '@features/gallery/data/queryCache';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, useEffect } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryData } from './useGalleryData';

import { useGalleryData } from './useGalleryData';

const mocks = vi.hoisted(() => ({ listGalleryBoards: vi.fn(), listGalleryItems: vi.fn() }));

vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  listGalleryBoards: mocks.listGalleryBoards,
  listGalleryItems: mocks.listGalleryItems,
}));

const TOTAL = 12_000;
const settings = getGallerySettings({ paginationMode: 'infinite' });
const paginatedSettings = getGallerySettings({ paginationMode: 'paginated' });
let latestData: GalleryData | null = null;

const readRequests = () =>
  mocks.listGalleryItems.mock.calls.map(([request]) => {
    const { limit, offset } = request as { limit: number; offset: number };
    return { limit, offset };
  });

const createItem = (index: number): GalleryItem => ({
  boardId: 'none',
  category: 'general',
  createdAt: new Date(Date.UTC(2026, 0, 1, 0, 0, index)).toISOString(),
  fullUrl: `/full/${index}`,
  height: 64,
  isIntermediate: false,
  kind: 'image',
  name: `image-${index}.png`,
  starred: false,
  thumbnailUrl: `/thumb/${index}`,
  width: 64,
});

const recentImages: GalleryImage[] = [
  {
    boardId: 'none',
    height: 64,
    imageCategory: 'general',
    imageName: 'recent.png',
    imageUrl: '/full/recent.png',
    queuedAt: new Date(1_000).toISOString(),
    sourceQueueItemId: 'recent-generation',
    starred: false,
    thumbnailUrl: '/thumb/recent.png',
    width: 64,
  },
];

const Probe = ({
  page = 0,
  paginated = false,
  searchTerm = '',
  recentImages = [],
}: { page?: number; paginated?: boolean; recentImages?: GalleryImage[]; searchTerm?: string } = {}) => {
  const data = useGalleryData({
    galleryView: 'images',
    page,
    projectBoardId: null,
    recentImages,
    searchTerm,
    selectedBoardId: null,
    settings: paginated ? paginatedSettings : settings,
    sparseViewport: true,
  });
  useEffect(() => {
    latestData = data;
  }, [data]);

  return (
    <>
      <output data-testid="total">{data.total ?? 'unknown'}</output>
      <output data-testid="page-offsets">{[...(data.sparseListing?.pageStates.keys() ?? [])].join(',')}</output>
      <output data-testid="anchor">{data.sparseListing?.itemSlots.get(6_000)?.name ?? 'missing'}</output>
      <output data-testid="items">{data.items?.map((item) => item.name).join(',') ?? 'unknown'}</output>
    </>
  );
};

const renderProbe = (props: Parameters<typeof Probe>[0] = {}) =>
  act(() =>
    root?.render(
      <QueryClientProvider client={queryClient!}>
        <Probe {...props} />
      </QueryClientProvider>
    )
  );

const findPageQuery = (offset: number) =>
  queryClient
    ?.getQueryCache()
    .findAll({ queryKey: ['gallery', 'items', 'list'] })
    .find((query) => query.queryKey[5] === 'page' && query.queryKey[6] === offset);

const mockListingTotal = (getTotal: () => number) =>
  mocks.listGalleryItems.mockImplementation(({ offset, limit }: { offset: number; limit: number }) => {
    const total = getTotal();

    return Promise.resolve({
      items: Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, index) =>
        createItem(offset + index)
      ),
      total,
    });
  });

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  accountLifecycle.activate('sparse-gallery-query-test');
  mocks.listGalleryBoards.mockImplementation(() => Promise.resolve([]));
  mocks.listGalleryItems.mockImplementation(({ offset, limit }: { offset: number; limit: number }) =>
    Promise.resolve({
      items: Array.from({ length: Math.max(0, Math.min(limit, TOTAL - offset)) }, (_, index) =>
        createItem(offset + index)
      ),
      total: TOTAL,
    })
  );
  latestData = null;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  queryClient?.clear();
  host = null;
  root = null;
  queryClient = null;
});

describe('useGalleryData sparse page subscriptions', () => {
  it('fetches only the selected paginated page and clamps later requests to the known final page', async () => {
    const total = 62;
    mocks.listGalleryItems.mockImplementation(({ offset, limit }: { offset: number; limit: number }) =>
      Promise.resolve({
        items: Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, index) =>
          createItem(offset + index)
        ),
        total,
      })
    );

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe page={1} paginated />
        </QueryClientProvider>
      )
    );

    await vi.waitFor(() => expect(latestData?.items).toHaveLength(2));
    expect(readRequests()).toEqual([
      { limit: 0, offset: 0 },
      { limit: 60, offset: 60 },
    ]);
    expect([...latestData!.sparseListing!.itemSlots.keys()]).toEqual([0, 1]);

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe page={5} paginated />
        </QueryClientProvider>
      )
    );

    expect(readRequests()).toEqual([
      { limit: 0, offset: 0 },
      { limit: 60, offset: 60 },
    ]);
    expect([...latestData!.sparseListing!.pageStates.keys()]).toEqual([60]);
    expect(latestData?.sparseListing?.itemSlots.get(0)?.name).toBe('image-60.png');
    expect(latestData?.sparseListing?.itemSlots.get(1)?.name).toBe('image-61.png');
  });

  it('reconciles a stale infinite total from a pinned reveal page', async () => {
    let total = 60;
    mocks.listGalleryItems.mockImplementation(({ offset, limit }: { offset: number; limit: number }) =>
      Promise.resolve({
        items: Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, index) =>
          createItem(offset + index)
        ),
        total,
      })
    );

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe />
        </QueryClientProvider>
      )
    );
    await vi.waitFor(() => expect(latestData?.total).toBe(60));

    // Another client adds a 61st item; Find in Gallery verified it at index 60.
    total = 61;
    await act(() => latestData!.pinRevealIndex!(60));

    await vi.waitFor(() => expect(latestData?.total).toBe(61));
    expect(latestData?.sparseListing?.itemSlots.get(60)?.name).toBe('image-60.png');
  });

  it('keeps an empty cold paginated listing subscribed to page zero', async () => {
    mocks.listGalleryItems.mockResolvedValue({ items: [], total: 0 });

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe page={5} paginated />
        </QueryClientProvider>
      )
    );

    await vi.waitFor(() => expect(latestData?.total).toBe(0));
    expect(readRequests()).toEqual([
      { limit: 0, offset: 0 },
      { limit: 60, offset: 0 },
    ]);
    expect([...latestData!.sparseListing!.pageStates.keys()]).toEqual([0]);
    expect(latestData?.items).toEqual([]);
  });

  it.each([
    ['infinite', false],
    ['numbered', true],
  ] as const)('refetches an empty %s listing after gallery invalidation', async (_mode, paginated) => {
    let total = 0;
    mocks.listGalleryItems.mockImplementation(({ offset, limit }: { offset: number; limit: number }) =>
      Promise.resolve({
        items: Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, index) =>
          createItem(offset + index)
        ),
        total,
      })
    );

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe page={paginated ? 5 : 0} paginated={paginated} />
        </QueryClientProvider>
      )
    );
    await vi.waitFor(() => expect(latestData?.total).toBe(0));
    expect(latestData?.items).toEqual([]);

    total = 1;
    await act(() => invalidateGallery(queryClient!));
    await vi.waitFor(() => {
      expect(latestData?.total).toBe(1);
      expect(latestData?.items?.map((item) => item.name)).toEqual(['image-0.png']);
    });
  });

  it('discovers total at page zero, then fetches only aligned pages around a distant range', async () => {
    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe />
        </QueryClientProvider>
      )
    );

    await vi.waitFor(() => expect(host?.querySelector('[data-testid="total"]')?.textContent).toBe(String(TOTAL)));
    expect(readRequests()).toEqual([{ limit: 60, offset: 0 }]);

    await act(() => latestData?.setVisibleRange?.({ endIndexExclusive: 6_120, startIndex: 6_000 }));

    await vi.waitFor(() => {
      expect(readRequests()).toEqual([
        { limit: 60, offset: 0 },
        { limit: 60, offset: 6_000 },
        { limit: 60, offset: 6_060 },
      ]);
    });

    const activePageOffsets = queryClient
      ?.getQueryCache()
      .getAll()
      .filter((query) => query.queryKey[5] === 'page' && query.getObserversCount() > 0)
      .map((query) => query.queryKey[6] as number)
      .sort((left, right) => left - right);

    expect(activePageOffsets).toEqual([6_000, 6_060]);
  });

  it('discovers a cold later page count before fetching only the clamped item page', async () => {
    const total = 62;
    mocks.listGalleryItems.mockImplementation(({ offset, limit }: { offset: number; limit: number }) =>
      Promise.resolve({
        items: Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, index) =>
          createItem(offset + index)
        ),
        total,
      })
    );

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe page={5} paginated />
        </QueryClientProvider>
      )
    );

    await vi.waitFor(() => expect(latestData?.items).toHaveLength(2));
    expect(readRequests()).toEqual([
      { limit: 0, offset: 0 },
      { limit: 60, offset: 60 },
    ]);
    expect([...latestData!.sparseListing!.pageStates.keys()]).toEqual([60]);
  });

  it('does not restore a stale discovered count while a shrunken listing loads its clamped page', async () => {
    let resolveShrunkPage!: (page: { items: GalleryItem[]; total: number }) => void;
    let resolveFirstPage!: (page: { items: GalleryItem[]; total: number }) => void;
    mocks.listGalleryItems.mockImplementation(({ limit, offset }: { limit: number; offset: number }) => {
      if (limit === 0) {
        return Promise.resolve({ items: [], total: 121 });
      }

      if (offset === 120) {
        return new Promise((resolve) => {
          resolveShrunkPage = resolve;
        });
      }

      if (offset === 0) {
        return new Promise((resolve) => {
          resolveFirstPage = resolve;
        });
      }

      throw new Error(`Unexpected item page at offset ${offset}`);
    });

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe page={2} paginated />
        </QueryClientProvider>
      )
    );
    await vi.waitFor(() => expect(readRequests()).toContainEqual({ limit: 0, offset: 0 }));
    await vi.waitFor(() => expect(readRequests()).toContainEqual({ limit: 60, offset: 120 }));

    let renderError: unknown;
    try {
      await act(async () => {
        resolveShrunkPage({ items: [], total: 60 });
        await Promise.resolve();
      });
    } catch (error) {
      renderError = error;
    }

    expect(renderError).toBeUndefined();
    await vi.waitFor(() => expect(readRequests()).toContainEqual({ limit: 60, offset: 0 }));
    expect(latestData?.total).toBe(60);
    expect(readRequests().filter(({ limit, offset }) => limit === 60 && offset === 120)).toHaveLength(1);

    await act(async () => {
      resolveFirstPage({
        items: Array.from({ length: 60 }, (_, index) => createItem(index)),
        total: 60,
      });
      await Promise.resolve();
    });
    await vi.waitFor(() => expect(latestData?.items).toHaveLength(60));
    expect(latestData?.total).toBe(60);
    expect(readRequests().filter(({ limit, offset }) => limit === 60 && offset === 120)).toHaveLength(1);
  });

  it('settles on the clamped page when a removal empties the last paginated page', async () => {
    const total = 61;
    mocks.listGalleryItems.mockImplementation(({ offset, limit }: { offset: number; limit: number }) =>
      Promise.resolve({
        items: Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, index) =>
          createItem(offset + index)
        ),
        total,
      })
    );

    // Page zero is cached from an earlier visit.
    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe paginated />
        </QueryClientProvider>
      )
    );
    await vi.waitFor(() => expect(latestData?.items).toHaveLength(60));
    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe page={1} paginated />
        </QueryClientProvider>
      )
    );
    await vi.waitFor(() => expect(latestData?.items?.map((item) => item.name)).toEqual(['image-60.png']));

    let renderError: unknown;
    try {
      await act(async () => {
        patchGalleryItemCaches(queryClient!, {
          boardId: 'elsewhere',
          kind: 'move',
          result: { failed: [], succeeded: [{ kind: 'image', name: 'image-60.png' }] },
        });
        await Promise.resolve();
      });
    } catch (error) {
      renderError = error;
    }

    expect(renderError).toBeUndefined();
    await vi.waitFor(() => expect(latestData?.total).toBe(60));
    expect(latestData?.items).toHaveLength(60);
  });

  it('reconciles a stale page total by refetching instead of flipping between clamped pages', async () => {
    // The server has not applied the move yet, so every read still reports 61 items.
    mockListingTotal(() => 61);
    await renderProbe({ paginated: true });
    await vi.waitFor(() => expect(latestData?.items).toHaveLength(60));
    await renderProbe({ page: 1, paginated: true });
    await vi.waitFor(() => expect(latestData?.items?.map((item) => item.name)).toEqual(['image-60.png']));
    await act(async () => {
      patchGalleryItemCaches(queryClient!, {
        boardId: 'elsewhere',
        kind: 'move',
        result: { failed: [], succeeded: [{ kind: 'image', name: 'image-60.png' }] },
      });
      await Promise.resolve();
    });
    await vi.waitFor(() => expect(latestData?.total).toBe(60));

    let renderError: unknown;
    try {
      await act(async () => {
        // A refetch of the shown page lands before the mutation and reports the old total.
        await queryClient!.invalidateQueries({ exact: true, queryKey: findPageQuery(0)!.queryKey });
      });
    } catch (error) {
      renderError = error;
    }

    expect(renderError).toBeUndefined();
    await vi.waitFor(() => {
      expect(latestData?.total).toBe(61);
      expect(latestData?.items?.map((item) => item.name)).toEqual(['image-60.png']);
    });
    expect(readRequests().filter(({ limit, offset }) => limit === 60 && offset === 0)).toHaveLength(3);
    expect(readRequests().filter(({ limit, offset }) => limit === 60 && offset === 60)).toHaveLength(2);
  });

  it('opens a verified paginated page past a stale retained total', async () => {
    let total = 60;
    mockListingTotal(() => total);
    await renderProbe({ paginated: true });
    await vi.waitFor(() => expect(latestData?.total).toBe(60));

    // Another client adds a 61st item; Find in Gallery verifies it on page 1 before selecting that page.
    total = 61;
    await act(() => fetchGalleryItemsPage(queryClient!, latestData!.filter, 60, { staleTime: 0 }));
    await renderProbe({ page: 1, paginated: true });

    await vi.waitFor(() => expect(latestData?.items?.map((item) => item.name)).toEqual(['image-60.png']));
    expect(latestData?.total).toBe(61);
  });

  it('does not reconcile page totals after an optimistic removal across subscribed pages', async () => {
    await renderProbe();
    await vi.waitFor(() => expect(latestData?.total).toBe(TOTAL));
    await act(() => latestData?.setVisibleRange?.({ endIndexExclusive: 120, startIndex: 0 }));
    await vi.waitFor(() => {
      expect(latestData?.sparseListing?.itemSlots.get(60)?.name).toBe('image-60.png');
      expect(latestData?.isLoadingItems).toBe(false);
    });
    const requestCount = readRequests().length;

    await act(async () => {
      patchGalleryItemCaches(queryClient!, {
        kind: 'delete',
        result: { failed: [], succeeded: [{ kind: 'image', name: 'image-5.png' }] },
      });
      await new Promise((resolve) => {
        setTimeout(resolve, 50);
      });
    });

    expect(latestData?.total).toBe(TOTAL - 1);
    expect(readRequests()).toHaveLength(requestCount);
  });

  it('reads a failed page again when it is subscribed after leaving the view, but not while it stays visible', async () => {
    let distantPageReads = 0;
    mocks.listGalleryItems.mockImplementation(({ offset, limit }: { offset: number; limit: number }) => {
      if (offset === 6_000 && ++distantPageReads === 1) {
        return Promise.reject(new Error('temporary page failure'));
      }

      return Promise.resolve({
        items: Array.from({ length: limit }, (_, index) => createItem(offset + index)),
        total: TOTAL,
      });
    });
    await renderProbe();
    await vi.waitFor(() => expect(latestData?.total).toBe(TOTAL));
    await act(() => latestData?.setVisibleRange?.({ endIndexExclusive: 6_060, startIndex: 6_000 }));
    await vi.waitFor(() =>
      expect(latestData?.sparseListing?.pageStates.get(6_000)?.error?.message).toBe('temporary page failure')
    );

    await act(() => invalidateGallery(queryClient!));
    expect(distantPageReads).toBe(1);

    await act(() => latestData?.setVisibleRange?.({ endIndexExclusive: 60, startIndex: 0 }));
    await act(() => latestData?.setVisibleRange?.({ endIndexExclusive: 6_060, startIndex: 6_000 }));

    await vi.waitFor(() => expect(latestData?.sparseListing?.itemSlots.get(6_000)?.name).toBe('image-6000.png'));
    expect(distantPageReads).toBe(2);
  });

  it('surfaces count errors and retries count discovery before loading a clamped page', async () => {
    let attempts = 0;
    mocks.listGalleryItems.mockImplementation(({ offset, limit }: { offset: number; limit: number }) => {
      attempts += 1;
      if (attempts === 1) {
        return Promise.reject(new Error('count unavailable'));
      }

      return Promise.resolve({
        items: Array.from({ length: Math.max(0, Math.min(limit, 62 - offset)) }, (_, index) =>
          createItem(offset + index)
        ),
        total: 62,
      });
    });

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe page={5} paginated />
        </QueryClientProvider>
      )
    );
    await vi.waitFor(() => expect(latestData?.queryError?.message).toBe('count unavailable'));
    expect([...latestData!.sparseListing!.pageStates.keys()]).toEqual([0]);
    await act(async () => {
      await latestData?.sparseListing?.pageStates.get(0)?.retry();
    });
    await vi.waitFor(() => expect(latestData?.items).toHaveLength(2));
    expect(readRequests()).toEqual([
      { limit: 0, offset: 0 },
      { limit: 0, offset: 0 },
      { limit: 60, offset: 60 },
    ]);
  });

  it.each(['filter', 'account'] as const)('does not apply a retry result after a %s transition', async (transition) => {
    let activeScope = 'old';
    let oldPageCallCount = 0;
    let resolveOldRetry: (() => void) | undefined;
    const requestsByScope: { limit: number; offset: number; scope: string }[] = [];
    mocks.listGalleryItems.mockImplementation(
      ({ limit, offset, searchTerm }: { limit: number; offset: number; searchTerm: string }) => {
        const scope = searchTerm === 'new listing' || activeScope === 'new' ? 'new' : 'old';
        requestsByScope.push({ limit, offset, scope });

        if (scope === 'new') {
          return Promise.resolve({
            items: Array.from({ length: Math.max(0, Math.min(limit, 62 - offset)) }, (_, index) => ({
              ...createItem(offset + index),
              name: `new-${offset + index}.png`,
            })),
            total: 62,
          });
        }

        if (limit === 0) {
          return Promise.resolve({ items: [], total: 62 });
        }

        if (offset === 60 && oldPageCallCount === 0) {
          oldPageCallCount += 1;
          return Promise.reject(new Error('temporary page failure'));
        }

        if (offset === 60) {
          oldPageCallCount += 1;
        }

        return new Promise((resolve) => {
          resolveOldRetry = () =>
            resolve({
              items: Array.from({ length: 2 }, (_, index) => ({
                ...createItem(offset + index),
                name: `stale-${offset + index}.png`,
              })),
              total: 62,
            });
        });
      }
    );

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe page={1} paginated />
        </QueryClientProvider>
      )
    );
    await vi.waitFor(() => expect(latestData?.queryError?.message).toBe('temporary page failure'));

    act(() => {
      void latestData?.sparseListing?.pageStates.get(60)?.retry();
    });
    await vi.waitFor(() => expect(oldPageCallCount).toBe(2));

    if (transition === 'account') {
      await act(() => {
        activeScope = 'new';
        accountLifecycle.activate('gallery-query-retry-new-account');
        root?.render(
          <QueryClientProvider client={queryClient!}>
            <Probe page={1} paginated />
          </QueryClientProvider>
        );
      });
    } else {
      await act(() =>
        root?.render(
          <QueryClientProvider client={queryClient!}>
            <Probe page={1} paginated searchTerm="new listing" />
          </QueryClientProvider>
        )
      );
    }
    await vi.waitFor(() => expect(latestData?.items?.map((item) => item.name)).toEqual(['new-61.png', 'new-60.png']));

    await act(() => resolveOldRetry?.());
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });
    expect(latestData?.items?.map((item) => item.name)).toEqual(['new-61.png', 'new-60.png']);
    expect(latestData?.total).toBe(62);
    const newScopeRequests = requestsByScope.filter(({ scope }) => scope === 'new');
    expect(newScopeRequests.map(({ limit, offset }) => ({ limit, offset }))).toEqual([
      { limit: 0, offset: 0 },
      { limit: 60, offset: 60 },
    ]);
  });

  it('keeps an authoritative recent out of the overlay after its page is evicted and reloaded', async () => {
    const authoritativeItem = { ...createItem(0), name: 'recent.png' };
    const pageCalls = new Map<number, number>();
    let resolvePageZero: (() => void) | undefined;
    mocks.listGalleryItems.mockImplementation(({ offset, limit }: { offset: number; limit: number }) => {
      pageCalls.set(offset, (pageCalls.get(offset) ?? 0) + 1);
      const items =
        offset === 0
          ? [authoritativeItem, ...Array.from({ length: limit - 1 }, (_, index) => createItem(index + 1))]
          : Array.from({ length: limit }, (_, index) => createItem(offset + index));

      if (offset === 0 && pageCalls.get(offset) === 1) {
        return new Promise((resolve) => {
          resolvePageZero = () => resolve({ items, total: TOTAL });
        });
      }

      return Promise.resolve({ items, total: TOTAL });
    });

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe recentImages={recentImages} />
        </QueryClientProvider>
      )
    );
    await vi.waitFor(() => {
      expect(latestData?.items?.filter((item) => item.name === 'recent.png')).toHaveLength(1);
      expect(latestData?.sparseListing?.recentItems.map((item) => item.name)).toEqual(['recent.png']);
    });

    await act(() => resolvePageZero?.());
    await vi.waitFor(() => expect(latestData?.sparseListing?.recentItems).toEqual([]));
    expect(latestData?.items?.filter((item) => item.name === 'recent.png')).toHaveLength(1);
    expect(latestData?.sparseListing?.itemSlots.get(0)?.name).toBe('recent.png');
    expect(latestData?.total).toBe(TOTAL);

    await act(() => latestData?.setVisibleRange?.({ endIndexExclusive: 6_060, startIndex: 6_000 }));
    await vi.waitFor(() => expect(latestData?.sparseListing?.pageStates.has(6_000)).toBe(true));
    const pageZero = queryClient
      ?.getQueryCache()
      .findAll({ queryKey: ['gallery', 'items', 'list'] })
      .find((query) => query.queryKey[5] === 'page' && query.queryKey[6] === 0);
    expect(pageZero?.getObserversCount()).toBe(0);
    if (pageZero) {
      await act(() => queryClient?.removeQueries({ exact: true, queryKey: pageZero.queryKey }));
    }

    await act(() => latestData?.setVisibleRange?.({ endIndexExclusive: 60, startIndex: 0 }));
    await vi.waitFor(() => {
      expect(pageCalls.get(0)).toBe(2);
      expect(latestData?.items?.filter((item) => item.name === 'recent.png')).toHaveLength(1);
    });
    expect(latestData?.items?.filter((item) => item.name === 'recent.png')).toHaveLength(1);
    expect(latestData?.sparseListing?.itemSlots.get(0)?.name).toBe('recent.png');
    expect(latestData?.sparseListing?.recentItems).toEqual([]);
    expect(latestData?.total).toBe(TOTAL);
  });

  it('uses an indexed distant range while the initial page is still pending', async () => {
    let releaseCountDiscovery: (() => void) | undefined;
    mocks.listGalleryItems.mockImplementation(({ offset, limit }: { offset: number; limit: number }) => {
      if (offset === 0) {
        return new Promise((resolve) => {
          releaseCountDiscovery = () =>
            resolve({
              items: Array.from({ length: limit }, (_, index) => createItem(index)),
              total: TOTAL,
            });
        });
      }

      return Promise.resolve({
        items: Array.from({ length: Math.max(0, Math.min(limit, TOTAL - offset)) }, (_, index) =>
          createItem(offset + index)
        ),
        total: TOTAL,
      });
    });

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe />
        </QueryClientProvider>
      )
    );
    await vi.waitFor(() => expect(readRequests()).toEqual([{ limit: 60, offset: 0 }]));

    await act(() => latestData?.setVisibleRange?.({ endIndexExclusive: 6_060, startIndex: 6_000 }));
    await vi.waitFor(() => {
      expect(readRequests()).toEqual([
        { limit: 60, offset: 0 },
        { limit: 60, offset: 6_000 },
      ]);
    });
    const activePageOffsets = queryClient
      ?.getQueryCache()
      .getAll()
      .filter((query) => query.queryKey[5] === 'page' && query.getObserversCount() > 0)
      .map((query) => query.queryKey[6] as number);
    expect(activePageOffsets).toEqual([6_000]);

    releaseCountDiscovery?.();
  });

  it('keeps the logical sparse range stable while reconciling page totals', async () => {
    const retryPageResolvers = new Map<number, () => void>();
    const pageCallCounts = new Map<number, number>();
    const createPage = (offset: number, limit: number, total: number) => ({
      items: Array.from({ length: Math.max(0, Math.min(limit, total - offset)) }, (_, index) =>
        createItem(offset + index)
      ),
      total,
    });
    mocks.listGalleryItems.mockImplementation(({ offset, limit }: { offset: number; limit: number }) => {
      const callCount = (pageCallCounts.get(offset) ?? 0) + 1;
      pageCallCounts.set(offset, callCount);

      if (offset === 0) {
        return Promise.resolve(createPage(offset, limit, TOTAL));
      }

      if (callCount === 1 || callCount === 3) {
        return Promise.resolve(createPage(offset, limit, offset === 6_000 ? 11_000 : 10_000));
      }

      if (callCount === 2 || callCount === 4) {
        return new Promise((resolve) => {
          retryPageResolvers.set(offset, () => {
            resolve(createPage(offset, limit, 11_000));
          });
        });
      }

      return Promise.resolve(createPage(offset, limit, 11_000));
    });

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe />
        </QueryClientProvider>
      )
    );
    await vi.waitFor(() => expect(host?.querySelector('[data-testid="total"]')?.textContent).toBe(String(TOTAL)));

    await act(() => latestData?.setVisibleRange?.({ endIndexExclusive: 6_120, startIndex: 6_000 }));
    await vi.waitFor(() => expect(retryPageResolvers.size).toBe(2));

    expect(host?.querySelector('[data-testid="total"]')?.textContent).toBe(String(TOTAL));
    expect(host?.querySelector('[data-testid="page-offsets"]')?.textContent).toBe('6000,6060');
    expect(host?.querySelector('[data-testid="anchor"]')?.textContent).toBe('image-6000.png');

    await act(() => {
      retryPageResolvers.forEach((resolve) => resolve());
    });
    retryPageResolvers.clear();
    await vi.waitFor(() => expect(host?.querySelector('[data-testid="total"]')?.textContent).toBe('11000'));

    expect(host?.querySelector('[data-testid="page-offsets"]')?.textContent).toBe('6000,6060');
    expect(host?.querySelector('[data-testid="anchor"]')?.textContent).toBe('image-6000.png');

    await act(async () => {
      const activePages = latestData?.sparseListing?.pageStates;
      await Promise.all([activePages?.get(6_000)?.retry(), activePages?.get(6_060)?.retry()]);
    });
    await vi.waitFor(() => expect(retryPageResolvers.size).toBe(2));

    expect(host?.querySelector('[data-testid="total"]')?.textContent).toBe('11000');
    expect(host?.querySelector('[data-testid="page-offsets"]')?.textContent).toBe('6000,6060');
    expect(host?.querySelector('[data-testid="anchor"]')?.textContent).toBe('image-6000.png');

    await act(() => {
      retryPageResolvers.forEach((resolve) => resolve());
    });
    await vi.waitFor(() => expect(latestData?.isLoadingItems).toBe(false));
    await new Promise((resolve) => {
      setTimeout(resolve, 50);
    });

    expect(host?.querySelector('[data-testid="total"]')?.textContent).toBe('11000');
    expect(host?.querySelector('[data-testid="page-offsets"]')?.textContent).toBe('6000,6060');
    expect(host?.querySelector('[data-testid="anchor"]')?.textContent).toBe('image-6000.png');
    expect(pageCallCounts.get(6_000)).toBe(4);
    expect(pageCallCounts.get(6_060)).toBe(4);
  });
});
