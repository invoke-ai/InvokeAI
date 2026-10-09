import type { GalleryItem, GalleryItemsPage } from '@features/gallery/core/items';

import { AccountScopeExpiredError, accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryObserver } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const backend = vi.hoisted(() => ({ listGalleryItems: vi.fn() }));

vi.mock('./backend', () => ({
  isDateBoardId: (boardId: string) => boardId.startsWith('by_date:'),
  listGalleryItems: backend.listGalleryItems,
}));

import { planGalleryPageOffsets } from '@features/gallery/ui/galleryGridLayout';

import {
  GALLERY_PAGE_SIZE,
  fetchGalleryItemsPage,
  galleryItemsPageOptions,
  galleryStarredStripOptions,
  type GalleryItemsFilter,
} from './queries';

const filter: GalleryItemsFilter = {
  boardId: 'board-1',
  galleryView: 'images',
  orderDir: 'DESC',
  searchTerm: '',
};

const createPage = (offset: number): GalleryItemsPage => ({
  items: Array.from({ length: GALLERY_PAGE_SIZE }, (_, index) => createItem(offset + index)),
  itemIndices: Array.from({ length: GALLERY_PAGE_SIZE }, (_, index) => offset + index),
  offset,
  total: 6_000,
});

const createItem = (index: number): GalleryItem => ({
  boardId: 'board-1',
  category: 'general',
  createdAt: new Date(index * 1_000).toISOString(),
  fullUrl: `/images/${index}`,
  height: 64,
  isIntermediate: false,
  kind: 'image',
  name: `image-${index}`,
  sourceQueueItemId: 'backend-gallery',
  starred: false,
  thumbnailUrl: `/images/${index}/thumbnail`,
  width: 64,
});

const createQueryClient = (): QueryClient =>
  new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 60_000 } } });

const pageQueries = (client: QueryClient) =>
  client
    .getQueryCache()
    .findAll({ queryKey: ['gallery', 'items', 'list'] })
    .filter((query) => query.queryKey[5] === 'page');

const fetchPage = (client: QueryClient, pageIndex: number, pageFilter = filter) =>
  client.fetchQuery(galleryItemsPageOptions(pageFilter, pageIndex * GALLERY_PAGE_SIZE));

describe('Gallery sparse page lifecycle', () => {
  beforeEach(() => {
    accountLifecycle.activate('gallery-page-lifecycle-test');
    backend.listGalleryItems.mockReset();
    backend.listGalleryItems.mockImplementation(({ offset }: { offset: number }) =>
      Promise.resolve(createPage(offset))
    );
  });

  afterEach(() => {
    accountLifecycle.invalidate();
  });

  it('retains only the ten most recently used inactive pages per listing', async () => {
    const client = createQueryClient();

    await Promise.all(Array.from({ length: 100 }, (_, index) => fetchPage(client, index)));

    const retained = pageQueries(client);
    const retainedItems = retained.flatMap((query) => (query.state.data as GalleryItemsPage).items);

    expect(retained).toHaveLength(10);
    expect(retained.map((query) => query.queryKey[6])).toEqual([
      5400, 5460, 5520, 5580, 5640, 5700, 5760, 5820, 5880, 5940,
    ]);
    expect(retainedItems).toHaveLength(10 * GALLERY_PAGE_SIZE);

    client.clear();
  });

  it('does not retain a page whose read fails with nothing observing it', async () => {
    const client = createQueryClient();
    backend.listGalleryItems.mockRejectedValue(new Error('temporary failure'));

    for (let index = 0; index < 25; index += 1) {
      await expect(fetchPage(client, index)).rejects.toThrow('temporary failure');
    }

    expect(pageQueries(client)).toHaveLength(0);
    client.clear();
  });

  it.each([37, 6_000, 600_000])('bounds retained page data while browsing a %i-item listing', async (total) => {
    const client = createQueryClient();
    backend.listGalleryItems.mockImplementation(({ offset }: { offset: number }) =>
      Promise.resolve({
        ...createPage(offset),
        items: Array.from({ length: Math.min(GALLERY_PAGE_SIZE, total - offset) }, (_, index) =>
          createItem(offset + index)
        ),
        itemIndices: Array.from({ length: Math.min(GALLERY_PAGE_SIZE, total - offset) }, (_, index) => offset + index),
        total,
      })
    );

    const pageCount = Math.ceil(total / GALLERY_PAGE_SIZE);
    const sampledPages = Math.min(pageCount, 100);
    for (let sample = 0; sample < sampledPages; sample += 1) {
      const pageIndex = Math.floor((sample * (pageCount - 1)) / Math.max(1, sampledPages - 1));
      await fetchPage(client, pageIndex);

      const retained = pageQueries(client);
      expect(retained).toHaveLength(Math.min(sample + 1, 10));
      expect(retained.flatMap((query) => (query.state.data as GalleryItemsPage).items).length).toBeLessThanOrEqual(
        Math.min(total, 10 * GALLERY_PAGE_SIZE)
      );
    }

    expect(backend.listGalleryItems).toHaveBeenCalledTimes(sampledPages);
    expect(pageQueries(client).flatMap((query) => (query.state.data as GalleryItemsPage).items)).toHaveLength(
      Math.min(total, 10 * GALLERY_PAGE_SIZE)
    );
    if (pageCount > 10) {
      expect(client.getQueryData(galleryItemsPageOptions(filter, 0).queryKey)).toBeUndefined();
      backend.listGalleryItems.mockClear();
      await fetchPage(client, 0);
      expect(backend.listGalleryItems.mock.calls.map(([request]) => request.offset)).toEqual([0]);
      expect(pageQueries(client)).toHaveLength(10);
    }
    client.clear();
  });

  it('requests only new viewport pages for a page scroll, fling, distant jump, and backward reload', async () => {
    const client = createQueryClient();
    backend.listGalleryItems.mockImplementation(({ offset }: { offset: number }) =>
      Promise.resolve({ ...createPage(offset), total: 600_000 })
    );
    let unsubscribePrevious: Array<() => void> = [];
    const browseRange = async (startIndex: number, expectedRequests: number[]) => {
      const offsets = planGalleryPageOffsets({
        endIndexExclusive: startIndex + 120,
        startIndex,
        total: 600_000,
      });
      const observers = offsets.map(
        (offset) => new QueryObserver(client, { ...galleryItemsPageOptions(filter, offset), enabled: false })
      );
      const unsubscribes = observers.map((observer) => observer.subscribe(() => undefined));
      unsubscribePrevious.forEach((unsubscribe) => unsubscribe());
      backend.listGalleryItems.mockClear();
      await Promise.all(offsets.map((offset) => client.fetchQuery(galleryItemsPageOptions(filter, offset))));
      expect(backend.listGalleryItems.mock.calls.map(([request]) => request.offset)).toEqual(expectedRequests);
      expect(observers.every((observer) => observer.getCurrentResult().data?.items.length === 60)).toBe(true);
      expect(pageQueries(client).filter((query) => query.getObserversCount() === 0).length).toBeLessThanOrEqual(10);
      unsubscribePrevious = unsubscribes;
    };

    await browseRange(0, [0, 60]);
    await browseRange(60, [120]);
    await browseRange(6_000, [6_000, 6_060]);
    await browseRange(540_000, [540_000, 540_060]);
    for (const start of [12_000, 18_000, 24_000, 30_000, 36_000]) {
      await browseRange(start, [start, start + 60]);
    }
    expect(client.getQueryData(galleryItemsPageOptions(filter, 0).queryKey)).toBeUndefined();
    await browseRange(0, [0, 60]);
    unsubscribePrevious.forEach((unsubscribe) => unsubscribe());
    expect(pageQueries(client)).toHaveLength(10);
    client.clear();
  });

  it('keeps retention limits separate for listings whose canonical filters differ', async () => {
    const client = createQueryClient();
    const starredFilter: GalleryItemsFilter = { ...filter, starred: true };

    await Promise.all([
      ...Array.from({ length: 12 }, (_, index) => fetchPage(client, index, filter)),
      ...Array.from({ length: 12 }, (_, index) => fetchPage(client, index, starredFilter)),
    ]);

    expect(
      pageQueries(client).filter((query) => (query.queryKey[4] as { starred?: boolean }).starred === undefined)
    ).toHaveLength(10);
    expect(
      pageQueries(client).filter((query) => (query.queryKey[4] as { starred?: boolean }).starred === true)
    ).toHaveLength(10);
    client.clear();
  });

  it('protects observed pages while pruning inactive pages, then prunes after observation ends', async () => {
    const client = createQueryClient();

    await fetchPage(client, 0);
    const options = galleryItemsPageOptions(filter, 0);
    const observer = new QueryObserver(client, { ...options, enabled: false });
    const unsubscribe = observer.subscribe(() => undefined);

    await Promise.all(Array.from({ length: 12 }, (_, index) => fetchPage(client, index + 1)));

    const observedKey = options.queryKey;
    const evictedKey = galleryItemsPageOptions(filter, GALLERY_PAGE_SIZE).queryKey;

    expect(pageQueries(client).map((query) => query.queryKey[6])).toContain(0);
    expect(client.getQueryData<GalleryItemsPage>(observedKey)?.items).toHaveLength(GALLERY_PAGE_SIZE);
    expect(client.getQueryData(evictedKey)).toBeUndefined();
    expect(pageQueries(client).filter((query) => query.getObserversCount() === 0)).toHaveLength(10);

    unsubscribe();

    expect(pageQueries(client)).toHaveLength(10);
    expect(pageQueries(client).every((query) => query.getObserversCount() === 0)).toBe(true);
    client.clear();
  });

  it('retains every sparse page while more than ten pages are observed', async () => {
    const client = createQueryClient();
    const offsets = Array.from({ length: 14 }, (_, index) => index * GALLERY_PAGE_SIZE);
    const unsubscribes = offsets.map((offset) =>
      new QueryObserver(client, { ...galleryItemsPageOptions(filter, offset), enabled: false }).subscribe(
        () => undefined
      )
    );

    await Promise.all(offsets.map((offset) => client.fetchQuery(galleryItemsPageOptions(filter, offset))));

    expect(pageQueries(client)).toHaveLength(14);
    expect(pageQueries(client).every((query) => query.state.data !== undefined && query.getObserversCount() > 0)).toBe(
      true
    );

    unsubscribes.forEach((unsubscribe) => unsubscribe());
    expect(pageQueries(client)).toHaveLength(10);
    client.clear();
  });

  it('loads every page covering a wide visible range with at most four concurrent reads', async () => {
    const client = createQueryClient();
    const startIndex = 17;
    const endIndexExclusive = 757;
    const offsets = planGalleryPageOffsets({ endIndexExclusive, startIndex, total: 6_000 });
    const requests: Array<{ offset: number; resolve: () => void }> = [];
    let active = 0;
    let peak = 0;
    let started = 0;

    backend.listGalleryItems.mockImplementation(({ offset }: { offset: number }) => {
      started += 1;
      active += 1;
      peak = Math.max(peak, active);

      return new Promise<GalleryItemsPage>((resolve) => {
        requests.push({
          offset,
          resolve: () => {
            active -= 1;
            resolve(createPage(offset));
          },
        });
      });
    });

    const visibleOffsets = offsets;
    const results = visibleOffsets.map((offset) => client.fetchQuery(galleryItemsPageOptions(filter, offset)));

    expect(visibleOffsets).toEqual(Array.from({ length: 13 }, (_, index) => index * GALLERY_PAGE_SIZE));
    await vi.waitFor(() => expect(started).toBe(4));
    for (let completed = 1; completed <= visibleOffsets.length; completed += 1) {
      requests.shift()?.resolve();
      if (completed < visibleOffsets.length) {
        await vi.waitFor(() => expect(started).toBe(Math.min(visibleOffsets.length, 4 + completed)));
      }
    }

    const pages = await Promise.all(results);
    const visibleIndices = pages
      .flatMap((page) => page.itemIndices ?? [])
      .filter((index) => index >= startIndex && index < endIndexExclusive)
      .sort((left, right) => left - right);

    expect(backend.listGalleryItems.mock.calls.map(([request]) => request.offset)).toEqual(visibleOffsets);
    expect(pages.map((page) => page.offset)).toEqual(visibleOffsets);
    expect(visibleIndices).toEqual(
      Array.from({ length: endIndexExclusive - startIndex }, (_, index) => startIndex + index)
    );
    expect(peak).toBe(4);
    expect(started).toBe(visibleOffsets.length);
    client.clear();
  });

  it('shares the four-request limit between visible pages and the starred strip', async () => {
    const client = createQueryClient();
    const pending: Array<{ resolve: () => void }> = [];
    let active = 0;
    let peak = 0;
    let started = 0;
    backend.listGalleryItems.mockImplementation(({ offset }: { offset: number }) => {
      started += 1;
      active += 1;
      peak = Math.max(peak, active);
      return new Promise<GalleryItemsPage>((resolve) => {
        pending.push({
          resolve: () => {
            active -= 1;
            resolve(createPage(offset));
          },
        });
      });
    });

    const pageRequests = Array.from({ length: 4 }, (_, index) =>
      client.fetchQuery(galleryItemsPageOptions(filter, index * GALLERY_PAGE_SIZE))
    );
    await vi.waitFor(() => expect(started).toBe(4));
    const stripRequest = client.fetchQuery(galleryStarredStripOptions(filter));
    await Promise.resolve();
    expect(started).toBe(4);

    pending.shift()?.resolve();
    await vi.waitFor(() => expect(started).toBe(5));
    while (pending.length) {
      pending.shift()?.resolve();
    }
    await Promise.all([...pageRequests, stripRequest]);

    expect(peak).toBe(4);
    expect(started).toBe(5);
    client.clear();
  });

  it('removes aborted queued pages and propagates cancellation to running work', async () => {
    const client = createQueryClient();
    const resolvers: Array<() => void> = [];
    const signals: AbortSignal[] = [];

    backend.listGalleryItems.mockImplementation(({ offset, signal }: { offset: number; signal: AbortSignal }) => {
      signals.push(signal);

      return new Promise<GalleryItemsPage>((resolve) => {
        resolvers.push(() => resolve(createPage(offset)));
      });
    });

    const options = Array.from({ length: 5 }, (_, index) => galleryItemsPageOptions(filter, index * GALLERY_PAGE_SIZE));
    const results = options.map((query) => client.fetchQuery(query).catch((error: unknown) => error));

    await vi.waitFor(() => expect(signals).toHaveLength(4));
    await client.cancelQueries({ exact: true, queryKey: options[0].queryKey });
    await client.cancelQueries({ exact: true, queryKey: options[4].queryKey });

    expect(signals[0]?.aborted).toBe(true);
    expect(signals).toHaveLength(4);

    for (const resolve of resolvers) {
      resolve();
    }
    await Promise.all(results);

    expect(backend.listGalleryItems).toHaveBeenCalledTimes(4);
    client.clear();
  });

  it('aborts queued and running pages on account epoch rotation and fences late results', async () => {
    const client = createQueryClient();
    const pending: Array<{ offset: number; signal: AbortSignal; resolve: (page: GalleryItemsPage) => void }> = [];

    backend.listGalleryItems.mockImplementation(
      ({ offset, signal }: { offset: number; signal: AbortSignal }) =>
        new Promise<GalleryItemsPage>((resolve) => {
          pending.push({ offset, signal, resolve });
        })
    );

    const oldOptions = Array.from({ length: 5 }, (_, index) =>
      galleryItemsPageOptions(filter, index * GALLERY_PAGE_SIZE)
    );
    const oldResults = oldOptions.map((options) => client.fetchQuery(options).catch((error: unknown) => error));

    await vi.waitFor(() => expect(pending).toHaveLength(4));
    accountLifecycle.activate('gallery-page-lifecycle-test');
    expect(pending.every((request) => request.signal.aborted)).toBe(true);

    const freshOptions = galleryItemsPageOptions(filter, 0);
    expect(freshOptions.queryKey).not.toEqual(oldOptions[0]?.queryKey);
    const freshResult = client.fetchQuery(freshOptions);

    pending.forEach((request) => request.resolve(createPage(request.offset)));
    await Promise.all(oldResults);

    await vi.waitFor(() => expect(pending).toHaveLength(5));
    expect(pending[4]?.offset).toBe(0);
    pending[4]?.resolve(createPage(0));

    const freshPage = await freshResult;

    expect(freshPage.offset).toBe(0);
    expect(freshPage.itemIndices).toHaveLength(GALLERY_PAGE_SIZE);
    expect(freshPage.itemIndices?.[0]).toBe(0);
    await expect(oldResults[0]).resolves.toBeInstanceOf(AccountScopeExpiredError);
    expect(client.getQueryData(oldOptions[0]?.queryKey ?? [])).toBeUndefined();
    expect(client.getQueryData(freshOptions.queryKey)).toEqual(freshPage);
    client.clear();
  });

  it('refetches an LRU-pruned page directly at its requested offset', async () => {
    const client = createQueryClient();

    await Promise.all(Array.from({ length: 12 }, (_, index) => fetchPage(client, index)));
    expect(pageQueries(client).some((query) => query.queryKey[6] === 0)).toBe(false);

    backend.listGalleryItems.mockClear();
    await fetchPage(client, 0);

    expect(backend.listGalleryItems).toHaveBeenCalledTimes(1);
    expect(backend.listGalleryItems).toHaveBeenCalledWith(expect.objectContaining({ offset: 0 }));
    client.clear();
  });

  it('shares one page fetch among three consumer-equivalent observers', async () => {
    const client = createQueryClient();
    let resolve!: (page: GalleryItemsPage) => void;
    backend.listGalleryItems.mockImplementation(
      () =>
        new Promise<GalleryItemsPage>((complete) => {
          resolve = complete;
        })
    );

    const equivalentFilters: GalleryItemsFilter[] = [filter, { ...filter, searchTerm: '   ' }, { ...filter }];
    const observers = equivalentFilters.map(
      (equivalentFilter) => new QueryObserver(client, galleryItemsPageOptions(equivalentFilter, 0))
    );
    const unsubscribes = observers.map((observer) => observer.subscribe(() => undefined));

    await vi.waitFor(() => expect(backend.listGalleryItems).toHaveBeenCalledOnce());
    resolve(createPage(0));

    await vi.waitFor(() => {
      expect(observers.every((observer) => observer.getCurrentResult().data?.offset === 0)).toBe(true);
    });

    expect(observers[1]?.getCurrentQuery()).toBe(observers[0]?.getCurrentQuery());
    expect(observers[2]?.getCurrentQuery()).toBe(observers[0]?.getCurrentQuery());
    expect(backend.listGalleryItems).toHaveBeenCalledOnce();
    unsubscribes.forEach((unsubscribe) => unsubscribe());
    observers.forEach((observer) => observer.destroy());
    client.clear();
  });

  it('keeps an imperative page read alive after its final UI observer leaves', async () => {
    const client = createQueryClient();
    let resolve!: (page: GalleryItemsPage) => void;
    let requestSignal: AbortSignal | undefined;
    backend.listGalleryItems.mockImplementation(
      ({ signal }: { signal: AbortSignal }) =>
        new Promise<GalleryItemsPage>((complete) => {
          resolve = complete;
          requestSignal = signal;
        })
    );

    const options = galleryItemsPageOptions(filter, 0);
    const observer = new QueryObserver(client, { ...options, enabled: false });
    const unsubscribe = observer.subscribe(() => undefined);
    const imperativeRead = fetchGalleryItemsPage(client, filter, 0);

    await vi.waitFor(() => expect(backend.listGalleryItems).toHaveBeenCalledOnce());
    unsubscribe();

    expect(requestSignal?.aborted).toBe(false);
    resolve(createPage(0));
    await expect(imperativeRead).resolves.toMatchObject({ offset: 0 });

    observer.destroy();
    client.clear();
  });
});
