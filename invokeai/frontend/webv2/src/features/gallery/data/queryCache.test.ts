import type { GalleryItem, GalleryItemMutationResult, GalleryItemsPage } from '@features/gallery/core/items';
import type { GalleryBoard } from '@features/gallery/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { InfiniteQueryObserver, QueryClient, type InfiniteData } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  ALL_READABLE_BOARDS_ID,
  hydrateGalleryDateBoardItemPage,
  listGalleryBoards,
  listGalleryDateBoardItemNames,
  listGalleryItems,
} from './backend';
import {
  canonicalizeGalleryItemsFilter,
  galleryItemsInfiniteOptions,
  galleryKeys,
  type GalleryItemsFilter,
} from './queries';
import {
  addGalleryBoardToCaches,
  canAddToCachedGalleryBoard,
  getGalleryBoardDestinationIds,
  ensureGalleryBoardDestinations,
  getCachedGalleryBoard,
  getGalleryItemBoardIdsFromCaches,
  getGalleryItemMarksFromCaches,
  invalidateGallery,
  invalidateGalleryBoardDestinations,
  invalidateGalleryItems,
  patchGalleryBoardCaches,
  patchGalleryItemCaches,
  writeGalleryItemMarks,
} from './queryCache';

vi.mock('./backend', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  hydrateGalleryDateBoardItemPage: vi.fn(),
  listGalleryBoards: vi.fn(),
  listGalleryDateBoardItemNames: vi.fn(),
  listGalleryItems: vi.fn(),
}));

type GalleryItemsData = InfiniteData<GalleryItemsPage, number>;

const createClient = (): QueryClient =>
  new QueryClient({
    defaultOptions: {
      queries: {
        gcTime: Infinity,
        retry: false,
      },
    },
  });

const createItem = (
  name: string,
  boardId = 'board-1',
  starred = false,
  kind: 'image' | 'video' = 'image'
): GalleryItem =>
  kind === 'image'
    ? {
        boardId,
        category: 'general',
        createdAt: '2026-07-26T00:00:00.000Z',
        fullUrl: `/api/v1/images/i/${name}/full`,
        height: 768,
        isIntermediate: false,
        kind,
        name,
        sourceQueueItemId: 'backend-gallery',
        colorLabel: null,
        starred,
        rating: 0 as const,
        rejected: false,
        thumbnailUrl: `/api/v1/images/i/${name}/thumbnail`,
        width: 512,
      }
    : {
        boardId,
        category: 'general',
        createdAt: '2026-07-26T00:00:00.000Z',
        durationSeconds: 1,
        fullUrl: `/api/v1/videos/i/${name}/full`,
        height: 768,
        isIntermediate: false,
        kind,
        name,
        colorLabel: null,
        rating: 0 as const,
        rejected: false,
        starred,
        thumbnailUrl: `/api/v1/videos/i/${name}/thumbnail`,
        width: 512,
      };

const createData = (pages: GalleryItem[][]): GalleryItemsData => {
  const total = pages.reduce((count, items) => count + items.length, 0);

  return {
    pageParams: pages.map((_, index) => index * 60),
    pages: pages.map((items) => ({ items, total })),
  };
};

const getItemsKey = (boardId: string, owner: AccountScope = captureAccountScope()) =>
  galleryKeys.items(owner, canonicalizeGalleryItemsFilter({ boardId, galleryView: 'images', searchTerm: '' }));

const getPageKey = (boardId: string, offset = 60, owner: AccountScope = captureAccountScope()) =>
  galleryKeys.itemPage(
    owner,
    canonicalizeGalleryItemsFilter({ boardId, galleryView: 'images', searchTerm: '' }),
    offset
  );

const getData = (client: QueryClient, queryKey: ReturnType<typeof getItemsKey>): GalleryItemsData => {
  const data = client.getQueryData<GalleryItemsData>(queryKey);

  expect(data).toBeDefined();

  return data as GalleryItemsData;
};

const toRef = ({ kind, name }: GalleryItem) => ({ kind, name });

const getResult = (
  succeeded: GalleryItemMutationResult['succeeded'],
  failed: GalleryItemMutationResult['failed'] = []
): GalleryItemMutationResult => ({ failed, succeeded });

describe('Gallery item cache patches', () => {
  beforeEach(() => {
    accountLifecycle.activate('gallery-query-cache-test');
  });

  it('patches a qualified mark in place and rolls back without rebuilding untouched pages or page params', () => {
    const client = createClient();
    const target = createItem('target.png');
    const samePageUntouched = createItem('same-page-untouched.png');
    const untouchedPageItem = createItem('untouched-page.mp4', 'board-1', false, 'video');
    const key = getItemsKey('board-1');

    client.setQueryData(key, createData([[target, samePageUntouched], [untouchedPageItem]]));
    const before = getData(client, key);
    const firstPageBefore = before.pages[0];
    const untouchedPageBefore = before.pages[1];
    const pageParamsBefore = before.pageParams;

    const rollback = patchGalleryItemCaches(client, {
      kind: 'marks',
      patch: { flag: 'pick' },
      result: getResult([{ kind: 'image', name: target.name }]),
    });
    const after = getData(client, key);

    expect(after).not.toBe(before);
    expect(after.pageParams).toBe(pageParamsBefore);
    expect(after.pages[0]).not.toBe(firstPageBefore);
    expect(after.pages[1]).toBe(untouchedPageBefore);
    expect(after.pages[0]?.items[0]).toEqual({ ...target, starred: true });
    expect(after.pages[0]?.items[1]).toBe(samePageUntouched);

    rollback();

    const rolledBack = getData(client, key);

    expect(rolledBack).toEqual(before);
    expect(rolledBack.pageParams).toBe(pageParamsBefore);
    expect(rolledBack.pages[1]).toBe(untouchedPageBefore);
    expect(rolledBack.pages[0]?.items[1]).toBe(samePageUntouched);
  });

  it('takes marked items out of every listing their new marks fail, lowering each listing once', () => {
    const client = createClient();
    const rated = (name: string, rating: 3 | 4) => ({ ...createItem(name), rating });
    const first = rated('first.png', 4);
    const second = rated('second.png', 3);
    const kept = rated('kept.png', 4);
    const filtered = canonicalizeGalleryItemsFilter({
      boardId: 'board-1',
      galleryView: 'images',
      minRating: 3,
      searchTerm: '',
      starred: false,
    });
    const windowKey = galleryKeys.items(captureAccountScope(), filtered);
    const pageKey = galleryKeys.itemPage(captureAccountScope(), filtered, 60);
    const unfilteredKey = getItemsKey('board-1');
    const result = getResult([
      { kind: 'image', name: first.name },
      { kind: 'image', name: second.name },
    ]);

    client.setQueryData(windowKey, createData([[first, kept]]));
    client.setQueryData<GalleryItemsPage>(pageKey, { items: [second], itemIndices: [60], offset: 60, total: 61 });
    client.setQueryData(unfilteredKey, createData([[first, second, kept]]));
    const rollback = patchGalleryItemCaches(client, { kind: 'marks', patch: { rating: 2 }, result });

    // One listing, two cached entries: each loses both removed items from its total.
    expect(getData(client, windowKey).pages[0]).toEqual({ items: [kept], total: 0 });
    expect(client.getQueryData<GalleryItemsPage>(pageKey)).toEqual({
      items: [],
      itemIndices: [],
      offset: 60,
      total: 59,
    });
    expect(getData(client, unfilteredKey).pages[0]?.items).toEqual([
      { ...first, rating: 2 },
      { ...second, rating: 2 },
      kept,
    ]);

    rollback();

    expect(getData(client, windowKey).pages[0]).toEqual({ items: [first, kept], total: 2 });
    expect(getData(client, unfilteredKey).pages[0]?.items).toEqual([first, second, kept]);
  });

  it('keeps a bookmarked item in a flag listing that still lists it, and drops it from the grid partition', () => {
    const client = createClient();
    const target = { ...createItem('target.png'), rejected: true };
    const flagsKey = galleryKeys.items(
      captureAccountScope(),
      canonicalizeGalleryItemsFilter({
        boardId: 'board-1',
        flags: ['pick', 'reject'],
        galleryView: 'images',
        searchTerm: '',
      })
    );
    const gridKey = galleryKeys.items(
      captureAccountScope(),
      canonicalizeGalleryItemsFilter({ boardId: 'board-1', galleryView: 'images', searchTerm: '', starred: false })
    );
    const result = getResult([{ kind: 'image', name: target.name }]);

    client.setQueryData(flagsKey, createData([[target]]));
    client.setQueryData(gridKey, createData([[target]]));
    patchGalleryItemCaches(client, { kind: 'marks', patch: { flag: 'pick' }, result });

    expect(getData(client, flagsKey).pages[0]).toEqual({
      items: [{ ...target, rejected: false, starred: true }],
      total: 1,
    });
    expect(getData(client, gridKey).pages[0]).toEqual({ items: [], total: 0 });
  });

  it('deletes matching qualified items across pages and keeps each page total consistent', () => {
    const client = createClient();
    const firstTarget = createItem('shared-name', 'board-1', false, 'image');
    const firstUntouched = createItem('shared-name', 'board-1', false, 'video');
    const secondTarget = createItem('second-target.mp4', 'board-1', false, 'video');
    const secondUntouched = createItem('second-untouched.png');
    const key = getItemsKey('board-1');

    client.setQueryData(
      key,
      createData([
        [firstTarget, firstUntouched],
        [secondTarget, secondUntouched],
      ])
    );
    const before = getData(client, key);

    patchGalleryItemCaches(client, {
      kind: 'delete',
      result: getResult([
        { kind: 'image', name: firstTarget.name },
        { kind: 'video', name: secondTarget.name },
      ]),
    });
    const after = getData(client, key);

    expect(after.pageParams).toBe(before.pageParams);
    expect(after.pages.map((page) => page.items)).toEqual([[firstUntouched], [secondUntouched]]);
    expect(after.pages.map((page) => page.total)).toEqual([2, 2]);
  });

  it('removes moved items from source boards while all-items and date views retain updated items', () => {
    const client = createClient();
    const target = createItem('moved.mp4', 'board-1', false, 'video');
    const untouched = createItem('untouched.png');
    const sourceKey = getItemsKey('board-1');
    const allKey = getItemsKey(ALL_READABLE_BOARDS_ID);
    const dateKey = getItemsKey('by_date:2026-07-26');
    const result = getResult([{ kind: 'video', name: target.name }]);

    client.setQueryData(sourceKey, createData([[target], [untouched]]));
    client.setQueryData(allKey, createData([[target], [untouched]]));
    client.setQueryData(dateKey, createData([[target], [untouched]]));

    patchGalleryItemCaches(client, { boardId: 'board-2', kind: 'move', result });

    const source = getData(client, sourceKey);
    const all = getData(client, allKey);
    const date = getData(client, dateKey);

    expect(source.pages.flatMap((page) => page.items)).toEqual([untouched]);
    expect(source.pages.map((page) => page.total)).toEqual([1, 1]);
    for (const retained of [all, date]) {
      expect(retained.pages.flatMap((page) => page.items)).toEqual([{ ...target, boardId: 'board-2' }, untouched]);
      expect(retained.pages.map((page) => page.total)).toEqual([2, 2]);
    }
  });

  describe('sparse page entries', () => {
    it('patches marks, delete, and move results and restores each page on rollback', () => {
      const client = createClient();
      const target = createItem('sparse-target.png', 'board-1', false);
      const deleted = createItem('sparse-deleted.png', 'board-1', false, 'video');
      const moved = createItem('sparse-moved.png', 'board-1', false);
      const other = createItem('sparse-other.png', 'board-1', false);
      const key = getPageKey('board-1');
      const before: GalleryItemsPage = {
        items: [target, deleted, moved, other],
        itemIndices: [60, 62, 64, 65],
        offset: 60,
        total: 12,
      };

      client.setQueryData(key, before);

      expect(getGalleryItemMarksFromCaches(client, [{ kind: 'image', name: target.name }])).toEqual(
        new Map([[`image:${target.name}`, { colorLabel: null, flag: 'none', rating: 0 }]])
      );
      expect(getGalleryItemBoardIdsFromCaches(client, [{ kind: 'image', name: moved.name }])).toEqual(
        new Map([[`image:${moved.name}`, 'board-1']])
      );

      const rollbackStar = patchGalleryItemCaches(client, {
        kind: 'marks',
        patch: { flag: 'pick' },
        result: getResult([{ kind: 'image', name: target.name }]),
      });
      expect(client.getQueryData<GalleryItemsPage>(key)?.items[0]).toEqual({ ...target, starred: true });
      expect(client.getQueryData<GalleryItemsPage>(key)?.itemIndices).toEqual([60, 62, 64, 65]);
      expect(client.getQueryData<GalleryItemsPage>(key)?.total).toBe(12);
      rollbackStar();
      expect(client.getQueryData<GalleryItemsPage>(key)).toEqual(before);

      const rollbackDelete = patchGalleryItemCaches(client, {
        kind: 'delete',
        result: getResult([{ kind: 'video', name: deleted.name }]),
      });
      expect(client.getQueryData<GalleryItemsPage>(key)).toEqual({
        items: [target, moved, other],
        itemIndices: [60, 64, 65],
        offset: 60,
        total: 11,
      });
      rollbackDelete();
      expect(client.getQueryData<GalleryItemsPage>(key)).toEqual(before);

      const rollbackMove = patchGalleryItemCaches(client, {
        boardId: 'board-2',
        kind: 'move',
        result: getResult([{ kind: 'image', name: moved.name }]),
      });
      expect(client.getQueryData<GalleryItemsPage>(key)).toEqual({
        items: [target, deleted, other],
        itemIndices: [60, 62, 65],
        offset: 60,
        total: 11,
      });
      rollbackMove();
      expect(client.getQueryData<GalleryItemsPage>(key)).toEqual(before);
    });
    it('re-applies a confirmed removal only to pages a mid-flight refetch restored the item to', () => {
      const client = createClient();
      const firstKey = getPageKey('board-1', 0);
      const lastKey = getPageKey('board-1', 60);
      const firstPage: GalleryItemsPage = { items: [createItem('first.png')], itemIndices: [0], offset: 0, total: 61 };
      const lastPage: GalleryItemsPage = { items: [createItem('last.png')], itemIndices: [60], offset: 60, total: 61 };
      const removedLastPage: GalleryItemsPage = { items: [], itemIndices: [], offset: 60, total: 60 };
      const result = getResult([{ kind: 'image', name: 'last.png' }]);

      client.setQueryData(firstKey, firstPage);
      client.setQueryData(lastKey, lastPage);
      patchGalleryItemCaches(client, { kind: 'delete', result });
      // The last page is read again before the server applies the deletion, so it still holds the item and the total.
      client.setQueryData(lastKey, lastPage);

      patchGalleryItemCaches(client, { kind: 'delete', result }, { totals: 'holder' });

      expect(client.getQueryData<GalleryItemsPage>(firstKey)).toEqual({ ...firstPage, total: 60 });
      expect(client.getQueryData<GalleryItemsPage>(lastKey)).toEqual(removedLastPage);
    });
  });

  describe('starred strip entries', () => {
    const getStripKey = (boardId: string) =>
      galleryKeys.starredStrip(
        captureAccountScope(),
        canonicalizeGalleryItemsFilter({ boardId, galleryView: 'images', searchTerm: '', starred: true })
      );
    const getStrip = (client: QueryClient, key: ReturnType<typeof getStripKey>): GalleryItemsPage => {
      const page = client.getQueryData<GalleryItemsPage>(key);

      if (!page) {
        throw new Error('missing strip');
      }

      return page;
    };

    it('drops an unstarred item and its count at once, and rolls the strip back', () => {
      const client = createClient();
      const target = createItem('target.png', 'board-1', true);
      const other = createItem('other.png', 'board-1', true);
      const key = getStripKey('board-1');
      const before: GalleryItemsPage = { items: [target, other], total: 5 };

      client.setQueryData(key, before);
      const rollback = patchGalleryItemCaches(client, {
        kind: 'marks',
        patch: { flag: 'none' },
        result: getResult([{ kind: 'image', name: target.name }]),
      });

      expect(getStrip(client, key)).toEqual({ items: [other], total: 4 });

      rollback();

      expect(getStrip(client, key)).toEqual(before);
    });
  });

  it('does not let rollback clobber a later concurrent cache update', () => {
    const client = createClient();
    const target = createItem('target.png');
    const key = getItemsKey('board-1');

    client.setQueryData(key, createData([[target]]));
    const rollback = patchGalleryItemCaches(client, {
      kind: 'marks',
      patch: { flag: 'pick' },
      result: getResult([{ kind: 'image', name: target.name }]),
    });
    const concurrent = client.setQueryData<GalleryItemsData>(key, (current) => {
      if (!current) {
        return current;
      }

      return {
        ...current,
        pages: current.pages.map((page) => ({
          ...page,
          items: page.items.map((item) =>
            item.kind === target.kind && item.name === target.name ? { ...item, boardId: 'board-concurrent' } : item
          ),
        })),
      };
    });

    rollback();

    expect(getData(client, key)).toBe(concurrent);
    expect(getData(client, key).pages[0]?.items[0]).toMatchObject({
      boardId: 'board-concurrent',
      kind: 'image',
      name: target.name,
      starred: true,
    });
  });
});

describe('getGalleryItemBoardIdsFromCaches', () => {
  beforeEach(() => {
    accountLifecycle.activate('gallery-query-cache-test');
  });

  it('reads each requested item’s current board from whichever list holds it', () => {
    const client = createClient();

    client.setQueryData(getItemsKey('board-1'), createData([[createItem('a.png', 'board-1')]]));
    client.setQueryData(
      getItemsKey(ALL_READABLE_BOARDS_ID),
      createData([[createItem('b.mp4', 'board-2', false, 'video')]])
    );

    const boardIds = getGalleryItemBoardIdsFromCaches(client, [
      { kind: 'image', name: 'a.png' },
      { kind: 'video', name: 'b.mp4' },
      { kind: 'image', name: 'unloaded.png' },
    ]);

    expect(boardIds.get('image:a.png')).toBe('board-1');
    expect(boardIds.get('video:b.mp4')).toBe('board-2');
    expect(boardIds.has('image:unloaded.png')).toBe(false);
  });
});

const createBoard = (id: string, overrides: Partial<GalleryBoard> = {}): GalleryBoard => ({
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id,
  imageCount: 1,
  kind: 'board',
  name: `Board ${id}`,
  isInbox: false,
  projectId: null,
  videoCount: 0,
  ...overrides,
});

describe('patchGalleryBoardCaches', () => {
  beforeEach(() => {
    accountLifecycle.activate('gallery-query-cache-test');
  });

  const getBoardsKey = (owner: AccountScope = captureAccountScope()) =>
    galleryKeys.boards(owner, {
      includeArchived: false,
      includeDateBoards: true,
      orderBy: 'created_at',
      orderDir: 'DESC',
    });

  it('patches the board in every cached list and rolls back to the prior lists', () => {
    const client = createClient();
    const key = getBoardsKey();
    const untouched = createBoard('board-2');

    client.setQueryData(key, [createBoard('board-1'), untouched]);
    const before = client.getQueryData<GalleryBoard[]>(key);

    const rollback = patchGalleryBoardCaches(client, 'board-1', { archived: true, name: 'Renamed' });
    const after = client.getQueryData<GalleryBoard[]>(key);

    expect(after?.[0]).toEqual({ ...createBoard('board-1'), archived: true, name: 'Renamed' });
    expect(after?.[1]).toBe(untouched);

    rollback();

    // Structural sharing rebuilds the array, so compare by value and keep the untouched board's identity.
    expect(client.getQueryData(key)).toEqual(before);
    expect(client.getQueryData<GalleryBoard[]>(key)?.[1]).toBe(untouched);
  });

  it('answers from the newest listing: a view-only or vanished board takes nothing, and with no listing nothing is', () => {
    const client = createClient();
    const viewOnly = {
      ...createBoard('viewed'),
      access: {
        actions: ['read' as const],
        adminOverride: false,
        generalAccess: 'viewer' as const,
        role: 'viewer' as const,
      },
    };

    // Nothing is known about any board: none is a destination, so nothing is sent where it may be refused.
    expect(canAddToCachedGalleryBoard('anything', client)).toBe(false);

    // An older listing still holds a board the newer one no longer does: revoked since.
    vi.useFakeTimers();
    try {
      vi.setSystemTime(1_000);
      client.setQueryData(
        galleryKeys.boards(captureAccountScope(), {
          includeArchived: true,
          includeDateBoards: false,
          orderBy: 'created_at',
          orderDir: 'DESC',
        }),
        [createBoard('revoked'), viewOnly]
      );
      vi.setSystemTime(2_000);
      client.setQueryData(getBoardsKey(), [viewOnly, createBoard('own')]);
    } finally {
      vi.useRealTimers();
    }

    expect(canAddToCachedGalleryBoard('own', client)).toBe(true);
    expect(canAddToCachedGalleryBoard('viewed', client)).toBe(false);
    expect(canAddToCachedGalleryBoard('revoked', client)).toBe(false);
    expect(canAddToCachedGalleryBoard('never-listed', client)).toBe(false);
    expect(getGalleryBoardDestinationIds(client)).toEqual(['own']);
  });

  const getArchivedBoardsKey = (orderBy: 'board_name' | 'created_at' = 'created_at') =>
    galleryKeys.boards(captureAccountScope(), {
      includeArchived: true,
      includeDateBoards: true,
      orderBy,
      orderDir: orderBy === 'created_at' ? 'DESC' : 'ASC',
    });
  const settle = () =>
    new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });

  it('awaits one read of every readable board, archived included, only while no such listing is cached', async () => {
    const client = createClient();
    vi.mocked(listGalleryBoards).mockResolvedValue([
      { ...createBoard('archived'), archived: true },
      createBoard('own'),
    ]);

    // A listing of live boards only cannot settle an archived board: the read is awaited.
    client.setQueryData(getBoardsKey(), [createBoard('own')]);
    const pending = ensureGalleryBoardDestinations(client);

    expect(pending).not.toBeNull();
    await pending;
    expect(vi.mocked(listGalleryBoards)).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ includeArchived: true, orderBy: 'created_at', orderDir: 'DESC' })
    );
    expect(canAddToCachedGalleryBoard('archived', client)).toBe(true);
    expect(canAddToCachedGalleryBoard('revoked', client)).toBe(false);

    // Read within the minute: nothing to await and nothing read.
    expect(ensureGalleryBoardDestinations(client)).toBeNull();
    await settle();
    expect(vi.mocked(listGalleryBoards)).toHaveBeenCalledTimes(1);
  });

  it('reads the listing of every readable board again at once after a refusal, with nothing observing it', async () => {
    const client = createClient();
    client.setQueryData(getArchivedBoardsKey(), [createBoard('refusing'), createBoard('own')]);
    let land!: (boards: GalleryBoard[]) => void;
    vi.mocked(listGalleryBoards).mockReturnValue(
      new Promise<GalleryBoard[]>((resolve) => {
        land = resolve;
      })
    );

    // Items are left alone: nothing landed.
    const itemsKey = [...galleryKeys.itemListsForAccount(captureAccountScope()), 'probe'];
    client.setQueryData(itemsKey, { pageParams: [], pages: [] });
    const refusal = invalidateGalleryBoardDestinations(client);

    // A resolution before the read lands decides from the old listing and shares the read.
    await vi.waitFor(() => expect(vi.mocked(listGalleryBoards)).toHaveBeenCalledTimes(1));
    expect(ensureGalleryBoardDestinations(client)).toBeNull();
    expect(canAddToCachedGalleryBoard('refusing', client)).toBe(true);

    land([createBoard('own')]);
    await refusal;
    expect(client.getQueryCache().find({ queryKey: itemsKey })?.state.isInvalidated).toBe(false);
    expect(canAddToCachedGalleryBoard('refusing', client)).toBe(false);
    // The older listing still describes the refused board, so a refusal can name it; an unlisted board has none.
    expect(getCachedGalleryBoard('refusing', client)).toEqual(createBoard('refusing'));
    expect(getCachedGalleryBoard('own', client)).toEqual(createBoard('own'));
    expect(getCachedGalleryBoard('never', client)).toBeUndefined();
    expect(canAddToCachedGalleryBoard('own', client)).toBe(true);
    expect(ensureGalleryBoardDestinations(client)).toBeNull();
    await settle();
    expect(vi.mocked(listGalleryBoards)).toHaveBeenCalledTimes(1);
  });

  it('cancels a listing read that began before the board was created, so it cannot land over it', async () => {
    const client = createClient();
    const key = getBoardsKey();
    client.setQueryData(key, [createBoard('own')]);
    let settleStaleRead: (boards: GalleryBoard[]) => void = () => undefined;
    const staleRead = client
      .fetchQuery({
        queryFn: () =>
          new Promise<GalleryBoard[]>((resolve) => {
            settleStaleRead = resolve;
          }),
        queryKey: key,
        staleTime: 0,
      })
      .catch(() => undefined);

    await addGalleryBoardToCaches(client, createBoard('just-created'));
    settleStaleRead([createBoard('own')]);
    await staleRead;

    expect(client.getQueryData<GalleryBoard[]>(key)?.map((board) => board.id)).toEqual(['own', 'just-created']);
    expect(canAddToCachedGalleryBoard('just-created', client)).toBe(true);
  });

  it('does not let rollback clobber a later concurrent boards update', () => {
    const client = createClient();
    const key = getBoardsKey();

    client.setQueryData(key, [createBoard('board-1')]);
    const rollback = patchGalleryBoardCaches(client, 'board-1', { name: 'Renamed' });
    const concurrent = client.setQueryData<GalleryBoard[]>(key, (boards) =>
      boards?.map((board) => ({ ...board, imageCount: 9 }))
    );

    rollback();

    expect(client.getQueryData(key)).toBe(concurrent);
  });
});

describe('Gallery window rebuild', () => {
  const listFilter: GalleryItemsFilter = {
    boardId: 'board-1',
    galleryView: 'images',
    searchTerm: '',
  };

  const createPageItems = (prefix: string, count: number): GalleryItem[] =>
    Array.from({ length: count }, (_, index) => createItem(`${prefix}-${index}.png`));

  const observeItems = (client: QueryClient, filter: GalleryItemsFilter): (() => void) =>
    new InfiniteQueryObserver(client, galleryItemsInfiniteOptions(filter)).subscribe(() => undefined);

  /** A two-page stale window under `filter`, ready for an invalidation pass. */
  const setUpStaleWindow = (filter: GalleryItemsFilter = listFilter, pages?: GalleryItem[][]) => {
    const client = createClient();
    const key = galleryKeys.items(captureAccountScope(), canonicalizeGalleryItemsFilter(filter));
    const windowPages = pages ?? [createPageItems('stale-a', 60), createPageItems('stale-b', 60)];

    client.setQueryData(key, createData(windowPages));

    return { client, key, pages: windowPages };
  };

  beforeEach(() => {
    accountLifecycle.activate('gallery-window-rebuild-test');
    vi.mocked(hydrateGalleryDateBoardItemPage).mockReset();
    vi.mocked(listGalleryDateBoardItemNames).mockReset();
    vi.mocked(listGalleryItems).mockReset();
  });

  it('refreshes an observed multi-page window with one span request, leaving it fresh and in place', async () => {
    const { client, key } = setUpStaleWindow();
    const unsubscribe = observeItems(client, listFilter);

    vi.mocked(listGalleryItems).mockResolvedValue({ items: createPageItems('fresh', 100), total: 100 });

    await invalidateGalleryItems(client);

    expect(vi.mocked(listGalleryItems)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(listGalleryItems)).toHaveBeenCalledWith(expect.objectContaining({ limit: 120, offset: 0 }));

    const data = getData(client, key);

    expect(data.pageParams).toEqual([0, 60]);
    expect(data.pages[0]?.items).toHaveLength(60);
    expect(data.pages[1]?.items).toHaveLength(40);
    expect(data.pages[0]?.items[0]?.name).toBe('fresh-0.png');
    expect(data.pages.every((page) => page.total === 100)).toBe(true);
    expect(data.pages.map((page) => Object.keys(page).sort())).toEqual([
      ['itemIndices', 'items', 'offset', 'total'],
      ['itemIndices', 'items', 'offset', 'total'],
    ]);
    expect(client.getQueryState(key)?.isInvalidated).toBe(false);
    unsubscribe();
  });

  it('discards a rebuild that lost to a concurrent cache write', async () => {
    const { client, key } = setUpStaleWindow();
    const concurrentPage = [createItem('concurrent.png')];
    let unsubscribe: (() => void) | undefined;

    vi.mocked(listGalleryItems).mockImplementation(() => {
      client.setQueryData(key, createData([concurrentPage, [createItem('concurrent-b.png')]]));
      // Deactivate so the trailing invalidation cannot refetch through this mock.
      unsubscribe?.();

      return Promise.resolve({ items: createPageItems('fresh', 120), total: 120 });
    });
    unsubscribe = observeItems(client, listFilter);

    await invalidateGalleryItems(client);

    const data = getData(client, key);

    expect(data.pageParams).toEqual([0]);
    expect(data.pages[0]?.items.map((item) => item.name)).toEqual(['concurrent.png']);
    expect(client.getQueryState(key)?.isInvalidated).toBe(true);
  });

  it('bails to the collapse when a page fetch starts during the span read', async () => {
    const { client, key } = setUpStaleWindow();
    const observer = new InfiniteQueryObserver(client, galleryItemsInfiniteOptions(listFilter));
    const unsubscribe = observer.subscribe(() => undefined);

    vi.mocked(listGalleryItems).mockImplementation(({ limit, offset }) => {
      if (limit === 120) {
        // A scroll mid-span-read snapshotted the old pages and would clobber a swap, so the rebuild stands down.
        void observer.fetchNextPage();

        return Promise.resolve({ items: createPageItems('fresh', 120), total: 200 });
      }

      if (offset === 120) {
        return new Promise(() => {
          // The mid-rebuild scroll's page never lands.
        });
      }

      return Promise.resolve({ items: createPageItems(`page-${offset}`, 60), total: 200 });
    });

    await invalidateGalleryItems(client);

    const data = getData(client, key);

    expect(data.pages.flatMap((page) => page.items.map((item) => item.name))).not.toContain('fresh-0.png');
    expect(data.pageParams).toEqual([0]);
    unsubscribe();
  });
});

describe('Gallery item marks writes', () => {
  const listFilter: GalleryItemsFilter = { boardId: 'board-1', galleryView: 'images', searchTerm: '' };
  const ref = { kind: 'image' as const, name: 'a.png' };

  const setUp = () => {
    const client = createClient();
    const key = galleryKeys.items(captureAccountScope(), canonicalizeGalleryItemsFilter(listFilter));
    const item = createItem('a.png');

    client.setQueryData(key, createData([[item]]));

    return {
      client,
      item,
      key,
      rating: () => getData(client, key).pages[0]?.items[0]?.rating,
      unsubscribe: new InfiniteQueryObserver(client, galleryItemsInfiniteOptions(listFilter)).subscribe(
        () => undefined
      ),
    };
  };

  beforeEach(() => {
    accountLifecycle.activate('gallery-marks-write-test');
    vi.mocked(listGalleryItems).mockReset();
  });

  it('keeps a mark over a listing read that started before it, until a refetch after the server has it', async () => {
    const { client, item, key, rating, unsubscribe } = setUp();
    let answerRead = (_page: GalleryItemsPage) => {};

    vi.mocked(listGalleryItems).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          answerRead = resolve;
        })
    );
    const read = client.refetchQueries({ queryKey: key });
    const write = writeGalleryItemMarks(client, [ref], { rating: 4 }, { boards: false, refetchDelayMs: 0 });

    answerRead({ items: [item], total: 1 });
    await read;

    expect(rating()).toBe(4);

    // Once the server has the mark, its trailing refetch is the newer word, even from another client.
    vi.mocked(listGalleryItems).mockResolvedValue({ items: [{ ...item, rating: 5 }], total: 1 });
    write.settle();

    await vi.waitFor(() => expect(rating()).toBe(5));
    unsubscribe();
  });

  it('refetches the listings once after a burst, never while a write is pending', async () => {
    const { client, item, unsubscribe } = setUp();

    vi.mocked(listGalleryItems).mockResolvedValue({ items: [item], total: 1 });
    const first = writeGalleryItemMarks(client, [ref], { rating: 1 }, { boards: false, refetchDelayMs: 10 });
    const second = writeGalleryItemMarks(client, [ref], { rating: 2 }, { boards: false, refetchDelayMs: 10 });

    first.settle();
    await new Promise((resolve) => {
      setTimeout(resolve, 40);
    });

    expect(vi.mocked(listGalleryItems)).not.toHaveBeenCalled();

    second.settle();

    await vi.waitFor(() => expect(vi.mocked(listGalleryItems)).toHaveBeenCalledOnce());
    unsubscribe();
  });

  it('moves an item from the Bookmarked strip into the grid at its place, and rolls both listings back', () => {
    const client = createClient();
    const owner = captureAccountScope();
    const strip = createItem('s.png', 'board-1', true);
    const marked = createItem('b.png', 'board-1', true);
    const [newest, oldest, later] = ['c.png', 'a.png', '0.png'].map((name) => createItem(name));
    const stripKey = galleryKeys.starredStrip(owner, canonicalizeGalleryItemsFilter({ ...listFilter, starred: true }));
    const gridFilter = canonicalizeGalleryItemsFilter({ ...listFilter, starred: false });
    const firstPageKey = galleryKeys.itemPage(owner, gridFilter, 0);
    const laterPageKey = galleryKeys.itemPage(owner, gridFilter, 60);
    const stripBefore: GalleryItemsPage = { items: [strip, marked], total: 2 };
    const firstPageBefore: GalleryItemsPage = { items: [newest!, oldest!], itemIndices: [0, 1], offset: 0, total: 61 };
    const laterPageBefore: GalleryItemsPage = { items: [later!], itemIndices: [60], offset: 60, total: 61 };

    client.setQueryData(stripKey, stripBefore);
    client.setQueryData(firstPageKey, firstPageBefore);
    client.setQueryData(laterPageKey, laterPageBefore);
    const write = writeGalleryItemMarks(client, [toRef(marked)], { flag: 'none' }, { boards: true, refetchDelayMs: 0 });
    const unmarked = { ...marked, starred: false };

    expect(client.getQueryData(stripKey)).toEqual({ items: [strip], total: 1 });
    // Between its neighbours, with no index left between them: every index from its own moves up one.
    expect(client.getQueryData(firstPageKey)).toEqual({
      items: [newest, unmarked, oldest],
      itemIndices: [0, 1, 2],
      offset: 0,
      total: 62,
    });
    expect(client.getQueryData(laterPageKey)).toEqual({ items: [later], itemIndices: [61], offset: 60, total: 62 });

    write.rollback();

    expect(client.getQueryData(stripKey)).toEqual(stripBefore);
    expect(client.getQueryData(firstPageKey)).toEqual(firstPageBefore);
    expect(client.getQueryData(laterPageKey)).toEqual(laterPageBefore);
  });

  it('narrows a partly failed write in place, so a later pending write on a succeeded item stays the newer word', async () => {
    const client = createClient();
    const key = galleryKeys.items(captureAccountScope(), canonicalizeGalleryItemsFilter(listFilter));
    const [kept, failed] = ['x.png', 'y.png'].map((name) => createItem(name));
    const ratings = () => getData(client, key).pages[0]?.items.map((item) => item.rating);
    const unsubscribe = new InfiniteQueryObserver(client, galleryItemsInfiniteOptions(listFilter)).subscribe(
      () => undefined
    );

    client.setQueryData(key, createData([[kept!, failed!]]));
    const first = writeGalleryItemMarks(
      client,
      [toRef(kept!), toRef(failed!)],
      { rating: 3 },
      { boards: false, refetchDelayMs: 0 }
    );
    const second = writeGalleryItemMarks(client, [toRef(kept!)], { rating: 5 }, { boards: false, refetchDelayMs: 0 });

    first.settle([{ patch: { rating: 0 }, refs: [toRef(failed!)] }]);
    expect(ratings()).toEqual([5, 0]);

    // A read the server answered before either write lands while the second is still pending.
    vi.mocked(listGalleryItems).mockResolvedValue({ items: [kept!, failed!], total: 2 });
    await client.refetchQueries({ queryKey: key });

    expect(ratings()).toEqual([5, 0]);
    second.rollback();
    unsubscribe();
  });

  const setUpBookmark = () => {
    const client = createClient();
    const owner = captureAccountScope();
    const item = createItem('x.png');
    const stripKey = (boardId: string) =>
      galleryKeys.starredStrip(owner, canonicalizeGalleryItemsFilter({ ...listFilter, boardId, starred: true }));
    const strip: GalleryItemsPage = { items: [createItem('s.png', 'board-1', true)], total: 1 };
    const land = (boardId: string, page: GalleryItemsPage) =>
      client.fetchQuery({ queryFn: () => page, queryKey: stripKey(boardId), staleTime: 0 });

    client.setQueryData(stripKey('board-1'), strip);
    client.setQueryData(
      galleryKeys.itemPage(owner, canonicalizeGalleryItemsFilter({ ...listFilter, starred: false }), 0),
      { items: [item], itemIndices: [0], offset: 0, total: 1 }
    );
    writeGalleryItemMarks(client, [toRef(item)], { flag: 'pick' }, { boards: true, refetchDelayMs: 0 });
    expect(client.getQueryData<GalleryItemsPage>(stripKey('board-1'))?.total).toBe(2);

    return { client, item, land, strip, stripKey };
  };

  it('never puts an item deleted while its mark is pending back into a listing, unless the deletion rolls back', async () => {
    const { client, item, land, strip, stripKey } = setUpBookmark();
    const rollback = patchGalleryItemCaches(client, { kind: 'delete', result: getResult([toRef(item)]) });

    await land('board-1', strip);
    expect(client.getQueryData(stripKey('board-1'))).toEqual(strip);

    rollback();
    await land('board-1', strip);
    expect(client.getQueryData<GalleryItemsPage>(stripKey('board-1'))?.total).toBe(2);
  });

  it('puts an item moved while its mark is pending only into its new board’s listings', async () => {
    const { client, item, land, strip, stripKey } = setUpBookmark();

    patchGalleryItemCaches(client, { boardId: 'board-2', kind: 'move', result: getResult([toRef(item)]) });
    await land('board-1', strip);
    await land('board-2', { items: [], total: 0 });

    expect(client.getQueryData(stripKey('board-1'))).toEqual(strip);
    expect(client.getQueryData(stripKey('board-2'))).toEqual({
      items: [{ ...item, boardId: 'board-2', starred: true }],
      total: 1,
    });
  });
});

describe('Gallery cache invalidation', () => {
  beforeEach(() => {
    accountLifecycle.activate('gallery-query-cache-invalidation-test');
  });

  it('invalidates only current-account Gallery domains and never grows query cardinality', async () => {
    const client = createClient();
    const oldOwner = captureAccountScope();
    const oldItemsKey = getItemsKey('board-1', oldOwner);
    const oldFilter = canonicalizeGalleryItemsFilter({
      boardId: 'by_date:2026-07-25',
      galleryView: 'images',
      searchTerm: '',
    });
    const oldDateNamesKey = galleryKeys.itemNames(oldOwner, oldFilter);
    const oldBoardsKey = galleryKeys.boards(oldOwner, {
      includeArchived: false,
      includeDateBoards: true,
      orderBy: 'created_at',
      orderDir: 'DESC',
    });

    client.setQueryData(oldItemsKey, createData([[createItem('old.png')]]));
    client.setQueryData(oldDateNamesKey, { items: [{ kind: 'image', name: 'old.png' }], total: 1 });
    client.setQueryData(oldBoardsKey, []);

    accountLifecycle.activate('gallery-query-cache-current-account');
    const currentOwner = captureAccountScope();
    const currentItemsKey = getItemsKey('board-1', currentOwner);
    const currentFilter = canonicalizeGalleryItemsFilter({
      boardId: 'by_date:2026-07-26',
      galleryView: 'images',
      searchTerm: '',
    });
    const currentDateNamesKey = galleryKeys.itemNames(currentOwner, currentFilter);
    const currentBoardsKey = galleryKeys.boards(currentOwner, {
      includeArchived: false,
      includeDateBoards: true,
      orderBy: 'created_at',
      orderDir: 'DESC',
    });

    client.setQueryData(currentItemsKey, createData([[createItem('current.png')]]));
    client.setQueryData(currentDateNamesKey, { items: [{ kind: 'image', name: 'current.png' }], total: 1 });
    client.setQueryData(currentBoardsKey, []);
    const initialCardinality = client.getQueryCache().getAll().length;

    for (let index = 0; index < 100; index += 1) {
      await invalidateGalleryItems(client);
    }

    expect(client.getQueryCache().getAll()).toHaveLength(initialCardinality);
    expect(client.getQueryState(currentItemsKey)?.isInvalidated).toBe(true);
    expect(client.getQueryState(currentDateNamesKey)?.isInvalidated).toBe(true);
    expect(client.getQueryState(currentBoardsKey)?.isInvalidated).toBe(false);
    expect(client.getQueryState(oldItemsKey)?.isInvalidated).toBe(false);
    expect(client.getQueryState(oldDateNamesKey)?.isInvalidated).toBe(false);

    await invalidateGallery(client);

    expect(client.getQueryCache().getAll()).toHaveLength(initialCardinality);
    expect(client.getQueryState(currentBoardsKey)?.isInvalidated).toBe(true);
    expect(client.getQueryState(oldBoardsKey)?.isInvalidated).toBe(false);
  });
});
