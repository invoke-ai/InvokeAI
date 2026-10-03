import type { GalleryItem } from '@features/gallery/core/items';

import { getGallerySettings } from '@features/gallery/core/settings';
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

const Probe = ({ page = 0, paginated = false }: { page?: number; paginated?: boolean } = {}) => {
  const data = useGalleryData({
    galleryView: 'images',
    page,
    projectBoardId: null,
    recentImages: [],
    searchTerm: '',
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
    </>
  );
};

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
    expect(mocks.listGalleryItems.mock.calls.map(([request]) => (request as { offset: number }).offset)).toEqual([60]);
    expect([...latestData!.sparseListing!.itemSlots.keys()]).toEqual([0, 1]);

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe page={5} paginated />
        </QueryClientProvider>
      )
    );

    expect(mocks.listGalleryItems.mock.calls.map(([request]) => (request as { offset: number }).offset)).toEqual([60]);
    expect([...latestData!.sparseListing!.pageStates.keys()]).toEqual([60]);
    expect(latestData?.sparseListing?.itemSlots.get(0)?.name).toBe('image-60.png');
    expect(latestData?.sparseListing?.itemSlots.get(1)?.name).toBe('image-61.png');
  });

  it('keeps an empty listing at page zero when a later page is requested', async () => {
    mocks.listGalleryItems.mockResolvedValue({ items: [], total: 0 });

    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe paginated />
        </QueryClientProvider>
      )
    );

    await vi.waitFor(() => expect(latestData?.total).toBe(0));
    await act(() =>
      root?.render(
        <QueryClientProvider client={queryClient!}>
          <Probe page={5} paginated />
        </QueryClientProvider>
      )
    );

    expect(mocks.listGalleryItems.mock.calls.map(([request]) => (request as { offset: number }).offset)).toEqual([0]);
    expect([...latestData!.sparseListing!.pageStates.keys()]).toEqual([0]);
    expect(latestData?.items).toEqual([]);
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
    expect(mocks.listGalleryItems.mock.calls.map(([request]) => (request as { offset: number }).offset)).toEqual([0]);

    await act(() => latestData?.setVisibleRange?.({ endIndexExclusive: 6_120, startIndex: 6_000 }));

    await vi.waitFor(() => {
      expect(mocks.listGalleryItems.mock.calls.map(([request]) => (request as { offset: number }).offset)).toEqual([
        0, 6_000, 6_060,
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

  it('uses an indexed distant range before page-zero count discovery completes', async () => {
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
    await vi.waitFor(() =>
      expect(mocks.listGalleryItems.mock.calls.map(([request]) => (request as { offset: number }).offset)).toEqual([0])
    );

    await act(() => latestData?.setVisibleRange?.({ endIndexExclusive: 6_060, startIndex: 6_000 }));
    await vi.waitFor(() => {
      expect(mocks.listGalleryItems.mock.calls.map(([request]) => (request as { offset: number }).offset)).toEqual([
        0, 6_000,
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

  it('reconciles same-total conflicts across generations without moving the visible anchor', async () => {
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
