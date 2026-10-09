import type { GalleryItem, GalleryItemsPage } from '@features/gallery/core/items';

import { abortGalleryLocatorRequests, createGalleryLocatorRequest } from '@features/gallery/core/selection';
import { AccountScopeExpiredError, accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryObserver } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const backend = vi.hoisted(() => ({
  fetchImageIndexAvailability: vi.fn(),
  getGalleryItemLocation: vi.fn(),
  hydrateGalleryDateBoardItemPage: vi.fn(),
  isDateBoardId: vi.fn(),
  listGalleryBoards: vi.fn(),
  listGalleryDateBoardItemNames: vi.fn(),
  listGalleryDateBoards: vi.fn(),
  listGalleryItemNames: vi.fn(),
  listGalleryItems: vi.fn(),
  listSemanticGalleryItemNames: vi.fn(),
}));

vi.mock('./backend', () => backend);

import {
  fetchGalleryItemsPage,
  fetchVerifiedGalleryItemPage,
  GALLERY_PAGE_SIZE,
  galleryItemsPageOptions,
  type GalleryItemsFilter,
} from './queries';

const filter: GalleryItemsFilter = {
  boardId: 'board-1',
  createdFrom: '2026-07-01',
  createdTo: '2026-07-31',
  galleryView: 'assets',
  orderDir: 'ASC',
  searchTerm: 'portrait',
  starred: false,
};

const ref = { kind: 'video' as const, name: 'target.mp4' };

const createItem = (name: string): GalleryItem => ({
  boardId: 'board-1',
  category: 'user',
  createdAt: '2026-07-15T12:00:00Z',
  durationSeconds: 4,
  fullUrl: `/videos/${name}`,
  height: 64,
  isIntermediate: false,
  kind: 'video',
  name,
  starred: false,
  thumbnailUrl: `/videos/${name}/thumbnail`,
  width: 64,
});

const createPage = (offset: number, total: number, index: number, name = ref.name): GalleryItemsPage => ({
  items: [createItem(name)],
  itemIndices: [index],
  offset,
  total,
});

const createQueryClient = (): QueryClient => new QueryClient({ defaultOptions: { queries: { retry: false } } });

beforeEach(() => {
  accountLifecycle.activate('gallery-location-test');
  backend.fetchImageIndexAvailability.mockReset();
  backend.getGalleryItemLocation.mockReset();
  backend.hydrateGalleryDateBoardItemPage.mockReset();
  backend.isDateBoardId.mockReset().mockReturnValue(false);
  backend.listGalleryBoards.mockReset();
  backend.listGalleryDateBoardItemNames.mockReset();
  backend.listGalleryDateBoards.mockReset();
  backend.listGalleryItemNames.mockReset();
  backend.listGalleryItems.mockReset();
  backend.listSemanticGalleryItemNames.mockReset();
});

afterEach(() => accountLifecycle.invalidate());

describe('fetchVerifiedGalleryItemPage', () => {
  it('aborts locator lifetimes across Gallery navigation entry points', () => {
    const first = createGalleryLocatorRequest();
    const second = createGalleryLocatorRequest();

    abortGalleryLocatorRequests();

    expect(first.signal.aborted).toBe(true);
    expect(second.signal.aborted).toBe(true);
    first.release();
    second.release();
  });

  it('uses the filtered location and fetches only the aligned 60-item page, including sparse hydrated slots', async () => {
    backend.getGalleryItemLocation.mockResolvedValue({ ...ref, index: 127, total: 200 });
    backend.listGalleryItems.mockResolvedValue(createPage(120, 200, 127));
    const queryClient = createQueryClient();

    await expect(fetchVerifiedGalleryItemPage(queryClient, filter, ref)).resolves.toMatchObject({
      index: 127,
      offset: 120,
      page: { itemIndices: [127], items: [{ kind: 'video', name: 'target.mp4' }] },
      total: 200,
    });

    expect(backend.getGalleryItemLocation).toHaveBeenCalledWith(
      expect.objectContaining({ ...filter, ...ref, signal: expect.any(AbortSignal) })
    );
    expect(backend.listGalleryItems).toHaveBeenCalledOnce();
    expect(backend.listGalleryItems.mock.calls[0]?.[0]).toMatchObject({ limit: GALLERY_PAGE_SIZE, offset: 120 });
    expect(backend.listGalleryItemNames).not.toHaveBeenCalled();
    queryClient.clear();
  });

  it('refreshes a cached target page before treating an external locator rank as verified', async () => {
    backend.getGalleryItemLocation.mockResolvedValue({ ...ref, index: 127, total: 200 });
    backend.listGalleryItems.mockResolvedValue(createPage(120, 200, 127));
    const queryClient = createQueryClient();
    const pageOptions = galleryItemsPageOptions(filter, 120);

    queryClient.setQueryData(pageOptions.queryKey, createPage(120, 200, 127));
    await expect(fetchVerifiedGalleryItemPage(queryClient, filter, ref)).resolves.toMatchObject({ index: 127 });

    expect(backend.listGalleryItems).toHaveBeenCalledOnce();
    queryClient.clear();
  });

  it('re-resolves once and refetches the same page when its hydrated slot disagrees', async () => {
    backend.getGalleryItemLocation
      .mockResolvedValueOnce({ ...ref, index: 122, total: 200 })
      .mockResolvedValueOnce({ ...ref, index: 121, total: 200 });
    backend.listGalleryItems
      .mockResolvedValueOnce(createPage(120, 200, 121, 'other.mp4'))
      .mockResolvedValueOnce(createPage(120, 200, 121));
    const queryClient = createQueryClient();

    await expect(fetchVerifiedGalleryItemPage(queryClient, filter, ref)).resolves.toMatchObject({ index: 121 });

    expect(backend.getGalleryItemLocation).toHaveBeenCalledTimes(2);
    expect(backend.listGalleryItems).toHaveBeenCalledTimes(2);
    expect(backend.listGalleryItems.mock.calls.map(([request]) => request.offset)).toEqual([120, 120]);
    queryClient.clear();
  });

  it('stops after one failed alignment instead of retrying indefinitely', async () => {
    backend.getGalleryItemLocation.mockResolvedValue({ ...ref, index: 61, total: 120 });
    backend.listGalleryItems.mockResolvedValue(createPage(60, 120, 60, 'other.mp4'));
    const queryClient = createQueryClient();

    await expect(fetchVerifiedGalleryItemPage(queryClient, filter, ref)).resolves.toBeNull();

    expect(backend.getGalleryItemLocation).toHaveBeenCalledTimes(2);
    expect(backend.listGalleryItems).toHaveBeenCalledTimes(2);
    queryClient.clear();
  });

  it('reads again when a gallery invalidation cancels its page read', async () => {
    backend.getGalleryItemLocation.mockResolvedValue({ ...ref, index: 127, total: 200 });
    backend.listGalleryItems
      .mockImplementationOnce(
        () =>
          new Promise(() => {
            // Held until the invalidation cancels it.
          })
      )
      .mockResolvedValueOnce(createPage(120, 200, 127));
    const queryClient = createQueryClient();
    const verified = fetchVerifiedGalleryItemPage(queryClient, filter, ref);

    await vi.waitFor(() => expect(backend.listGalleryItems).toHaveBeenCalledOnce());
    await queryClient.cancelQueries({ queryKey: ['gallery', 'items', 'list'] });

    await expect(verified).resolves.toMatchObject({ index: 127, offset: 120 });
    expect(backend.getGalleryItemLocation).toHaveBeenCalledTimes(2);
    expect(backend.listGalleryItems).toHaveBeenCalledTimes(2);
    queryClient.clear();
  });

  it('abandons location work owned by an account epoch after that account changes', async () => {
    let resolveLocation!: (location: { kind: 'video'; name: string; index: number; total: number }) => void;
    backend.getGalleryItemLocation.mockReturnValue(
      new Promise((resolve) => {
        resolveLocation = resolve;
      })
    );
    const queryClient = createQueryClient();
    const operation = fetchVerifiedGalleryItemPage(queryClient, filter, ref);

    await vi.waitFor(() => expect(backend.getGalleryItemLocation).toHaveBeenCalledOnce());
    accountLifecycle.activate('gallery-location-next-account');
    resolveLocation({ ...ref, index: 1, total: 5 });

    await expect(operation).rejects.toBeInstanceOf(AccountScopeExpiredError);
    expect(backend.listGalleryItems).not.toHaveBeenCalled();
    queryClient.clear();
  });

  it('aborts a superseded item-location request before it can fetch a page', async () => {
    let locationSignal: AbortSignal | undefined;
    backend.getGalleryItemLocation.mockImplementation(
      ({ signal }: { signal: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          locationSignal = signal;
          signal.addEventListener('abort', () => reject(signal.reason), { once: true });
        })
    );
    const queryClient = createQueryClient();
    const locator = new AbortController();
    const request = fetchVerifiedGalleryItemPage(queryClient, filter, ref, undefined, locator.signal);

    await vi.waitFor(() => expect(backend.getGalleryItemLocation).toHaveBeenCalledOnce());
    locator.abort();

    await expect(request).rejects.toMatchObject({ name: 'AbortError' });
    expect(locationSignal?.aborted).toBe(true);
    expect(backend.listGalleryItems).not.toHaveBeenCalled();
    queryClient.clear();
  });

  it('starts a fresh same-target lookup after the prior locator is superseded', async () => {
    let firstSignal: AbortSignal | undefined;
    backend.getGalleryItemLocation.mockImplementation(({ signal }: { signal: AbortSignal }) => {
      if (backend.getGalleryItemLocation.mock.calls.length > 1) {
        return Promise.resolve({ ...ref, index: 127, total: 200 });
      }

      firstSignal = signal;

      return new Promise((_resolve, reject) => {
        signal.addEventListener('abort', () => reject(signal.reason), { once: true });
      });
    });
    backend.listGalleryItems.mockResolvedValue(createPage(120, 200, 127));
    const queryClient = createQueryClient();
    const firstLocator = new AbortController();
    const firstRequest = fetchVerifiedGalleryItemPage(queryClient, filter, ref, undefined, firstLocator.signal);

    await vi.waitFor(() => expect(backend.getGalleryItemLocation).toHaveBeenCalledOnce());
    firstLocator.abort();
    await expect(firstRequest).rejects.toMatchObject({ name: 'AbortError' });
    expect(firstSignal?.aborted).toBe(true);

    const secondRequest = fetchVerifiedGalleryItemPage(queryClient, filter, ref);

    await expect(secondRequest).resolves.toMatchObject({ index: 127, offset: 120 });
    expect(backend.getGalleryItemLocation).toHaveBeenCalledTimes(2);
    expect(backend.listGalleryItems).toHaveBeenCalledOnce();
    queryClient.clear();
  });

  it('cancels a running page when its locator is the last unobserved caller', async () => {
    backend.getGalleryItemLocation.mockResolvedValue({ ...ref, index: 127, total: 200 });
    let pageSignal: AbortSignal | undefined;
    let resolvePage!: (page: GalleryItemsPage) => void;
    backend.listGalleryItems.mockImplementation(
      ({ signal }: { signal: AbortSignal }) =>
        new Promise((resolve) => {
          pageSignal = signal;
          resolvePage = resolve;
        })
    );
    const queryClient = createQueryClient();
    const pageOptions = galleryItemsPageOptions(filter, 120);
    const locator = new AbortController();
    const request = fetchVerifiedGalleryItemPage(queryClient, filter, ref, undefined, locator.signal);

    await vi.waitFor(() => expect(backend.listGalleryItems).toHaveBeenCalledOnce());
    locator.abort();

    await expect(request).rejects.toMatchObject({ name: 'AbortError' });
    expect(pageSignal?.aborted).toBe(true);
    expect(backend.getGalleryItemLocation).toHaveBeenCalledOnce();
    expect(backend.listGalleryItems).toHaveBeenCalledOnce();
    expect(queryClient.getQueryData(pageOptions.queryKey)).toBeUndefined();
    resolvePage(createPage(120, 200, 127));
    queryClient.clear();
  });

  it('keeps an unobserved Preview page caller alive when its locator is superseded', async () => {
    backend.getGalleryItemLocation.mockResolvedValue({ ...ref, index: 127, total: 200 });
    let pageSignal: AbortSignal | undefined;
    let resolvePage!: (page: GalleryItemsPage) => void;
    backend.listGalleryItems.mockImplementation(
      ({ signal }: { signal: AbortSignal }) =>
        new Promise((resolve) => {
          pageSignal = signal;
          resolvePage = resolve;
        })
    );
    const queryClient = createQueryClient();
    const pageOptions = galleryItemsPageOptions(filter, 120);
    const fetchQuery = vi.spyOn(queryClient, 'fetchQuery');
    const previewRequest = new AbortController();
    const previewPage = fetchGalleryItemsPage(queryClient, filter, 120, { signal: previewRequest.signal });
    const locator = new AbortController();
    const locatorPage = fetchVerifiedGalleryItemPage(queryClient, filter, ref, undefined, locator.signal);

    try {
      await vi.waitFor(() => expect(backend.listGalleryItems).toHaveBeenCalledOnce());
      await vi.waitFor(() =>
        expect(fetchQuery).toHaveBeenCalledWith(
          expect.objectContaining({ queryKey: pageOptions.queryKey, staleTime: 0 })
        )
      );
      locator.abort();

      await expect(locatorPage).rejects.toMatchObject({ name: 'AbortError' });
      expect(pageSignal?.aborted).toBe(false);
      resolvePage(createPage(120, 200, 127));
      await expect(previewPage).resolves.toMatchObject({ itemIndices: [127], offset: 120 });
      expect(pageSignal?.aborted).toBe(false);
      expect(queryClient.getQueryData(pageOptions.queryKey)).toMatchObject({ itemIndices: [127] });
    } finally {
      previewRequest.abort();
      queryClient.clear();
    }
  });

  it('does not cancel another unobserved page fetch for an already-aborted wrapper caller', async () => {
    let pageSignal: AbortSignal | undefined;
    let resolvePage!: (page: GalleryItemsPage) => void;
    backend.listGalleryItems.mockImplementation(
      ({ signal }: { signal: AbortSignal }) =>
        new Promise((resolve) => {
          pageSignal = signal;
          resolvePage = resolve;
        })
    );
    const queryClient = createQueryClient();
    const pageOptions = galleryItemsPageOptions(filter, 120);
    const otherCaller = queryClient.fetchQuery(pageOptions);
    const alreadyAborted = new AbortController();

    await vi.waitFor(() => expect(backend.listGalleryItems).toHaveBeenCalledOnce());
    alreadyAborted.abort();

    await expect(
      fetchGalleryItemsPage(queryClient, filter, 120, { signal: alreadyAborted.signal })
    ).rejects.toMatchObject({ name: 'AbortError' });
    expect(pageSignal?.aborted).toBe(false);

    resolvePage(createPage(120, 200, 127));
    await expect(otherCaller).resolves.toMatchObject({ itemIndices: [127] });
    expect(pageSignal?.aborted).toBe(false);
    queryClient.clear();
  });

  it('keeps an observed Gallery page request alive when its locator is superseded', async () => {
    let resolveLocation!: (location: { kind: 'video'; name: string; index: number; total: number }) => void;
    backend.getGalleryItemLocation.mockReturnValue(
      new Promise((resolve) => {
        resolveLocation = resolve;
      })
    );
    let resolvePage!: (page: GalleryItemsPage) => void;
    let pageSignal: AbortSignal | undefined;
    backend.listGalleryItems.mockImplementation(
      ({ signal }: { signal: AbortSignal }) =>
        new Promise((resolve) => {
          pageSignal = signal;
          resolvePage = resolve;
        })
    );
    const queryClient = createQueryClient();
    const pageOptions = galleryItemsPageOptions(filter, 120);
    const observer = new QueryObserver(queryClient, pageOptions);
    const unsubscribe = observer.subscribe(() => undefined);
    const locator = new AbortController();
    const fetchQuery = vi.spyOn(queryClient, 'fetchQuery');

    try {
      await vi.waitFor(() => expect(backend.listGalleryItems).toHaveBeenCalledOnce());
      const request = fetchVerifiedGalleryItemPage(queryClient, filter, ref, undefined, locator.signal);
      await vi.waitFor(() => expect(backend.getGalleryItemLocation).toHaveBeenCalledOnce());
      resolveLocation({ ...ref, index: 127, total: 200 });
      await vi.waitFor(() =>
        expect(
          queryClient.getQueryCache().find({ exact: true, queryKey: pageOptions.queryKey })?.getObserversCount()
        ).toBe(1)
      );
      await vi.waitFor(() =>
        expect(fetchQuery).toHaveBeenCalledWith(
          expect.objectContaining({ queryKey: pageOptions.queryKey, staleTime: 0 })
        )
      );
      locator.abort();

      await expect(request).rejects.toMatchObject({ name: 'AbortError' });
      expect(pageSignal?.aborted).toBe(false);
      resolvePage(createPage(120, 200, 127));
      await vi.waitFor(() => expect(observer.getCurrentResult().data?.itemIndices).toEqual([127]));
      expect(pageSignal?.aborted).toBe(false);
    } finally {
      unsubscribe();
      observer.destroy();
      queryClient.clear();
    }
  });

  it('removes a superseded locator page from the bounded scheduler queue when no caller remains', async () => {
    const refs = Array.from({ length: 5 }, (_, index) => ({ kind: 'video' as const, name: `target-${index}.mp4` }));
    backend.getGalleryItemLocation.mockImplementation(({ name }: { name: string }) => {
      const index = Number(name.match(/-(\d+)\.mp4$/)?.[1] ?? 0);

      return Promise.resolve({ kind: 'video', name, index: index * GALLERY_PAGE_SIZE + 1, total: 400 });
    });
    const pendingPages: Array<{
      offset: number;
      resolve: (page: GalleryItemsPage) => void;
    }> = [];
    backend.listGalleryItems.mockImplementation(
      ({ offset }: { offset: number }) =>
        new Promise((resolve) => {
          pendingPages.push({ offset, resolve });
        })
    );
    const queryClient = createQueryClient();
    const locators = refs.map(() => new AbortController());
    const requests = refs.map((target, index) =>
      fetchVerifiedGalleryItemPage(queryClient, filter, target, undefined, locators[index]?.signal).then(
        (result) => ({ result }),
        (error: unknown) => ({ error })
      )
    );

    await vi.waitFor(() => expect(backend.getGalleryItemLocation).toHaveBeenCalledTimes(5));
    await vi.waitFor(() => expect(pendingPages).toHaveLength(4));
    locators[4]?.abort();

    await expect(requests[4]).resolves.toMatchObject({ error: { name: 'AbortError' } });
    pendingPages.slice(0, 4).forEach(({ offset, resolve }) => {
      const pageIndex = offset / GALLERY_PAGE_SIZE;
      const target = refs[pageIndex];
      resolve(createPage(offset, 400, offset + 1, target?.name));
    });
    await expect(Promise.all(requests.slice(0, 4))).resolves.toHaveLength(4);

    expect(backend.listGalleryItems).toHaveBeenCalledTimes(4);
    expect(backend.listGalleryItems.mock.calls.map(([request]) => request.offset)).toEqual([0, 60, 120, 180]);
    queryClient.clear();
  });
});
