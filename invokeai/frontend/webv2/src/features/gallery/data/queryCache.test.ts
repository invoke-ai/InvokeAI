import type { GalleryItem, GalleryItemMutationResult, GalleryItemsPage } from '@features/gallery/core/items';
import type { GalleryBoard } from '@features/gallery/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { QueryClient, type InfiniteData } from '@tanstack/react-query';
import { beforeEach, describe, expect, it } from 'vitest';

import { ALL_READABLE_BOARDS_ID } from './backend';
import { canonicalizeGalleryItemsFilter, galleryKeys } from './queries';
import {
  getGalleryItemBoardIdsFromCaches,
  invalidateGallery,
  invalidateGalleryItems,
  patchGalleryBoardCaches,
  patchGalleryItemCaches,
} from './queryCache';

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
        starred,
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

const getItemsKey = (boardId: string, owner: AccountScope = captureAccountScope(), starredFirst = false) =>
  galleryKeys.items(
    owner,
    canonicalizeGalleryItemsFilter({
      boardId,
      galleryView: 'images',
      searchTerm: '',
      starredFirst,
    })
  );

const getData = (client: QueryClient, queryKey: ReturnType<typeof getItemsKey>): GalleryItemsData => {
  const data = client.getQueryData<GalleryItemsData>(queryKey);

  expect(data).toBeDefined();

  return data as GalleryItemsData;
};

const getResult = (
  succeeded: GalleryItemMutationResult['succeeded'],
  failed: GalleryItemMutationResult['failed'] = []
): GalleryItemMutationResult => ({ failed, succeeded });

describe('Gallery item cache patches', () => {
  beforeEach(() => {
    accountLifecycle.activate('gallery-query-cache-test');
  });

  it('patches a qualified star in place and rolls back without rebuilding untouched pages or page params', () => {
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
      kind: 'star',
      result: getResult([{ kind: 'image', name: target.name }]),
      starred: true,
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

  it('patches star state immediately when the active list sorts starred items first', () => {
    const client = createClient();
    const target = createItem('target.png');
    const key = getItemsKey('board-1', captureAccountScope(), true);

    client.setQueryData(key, createData([[target]]));

    patchGalleryItemCaches(client, {
      kind: 'star',
      result: getResult([{ kind: 'image', name: target.name }]),
      starred: true,
    });

    expect(getData(client, key).pages[0]?.items[0]).toEqual({ ...target, starred: true });
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

  it('keeps moved items in ranked similarity windows regardless of the selected board', () => {
    // A semantic window inherits whatever board happens to be selected, but
    // its membership is similarity, not board: moving an image must update
    // its board in place, never evict it from the ranked list.
    const client = createClient();
    const target = createItem('ranked.png', 'board-1');
    const untouched = createItem('also-ranked.png', 'board-1');
    const semanticKey = galleryKeys.items(
      captureAccountScope(),
      canonicalizeGalleryItemsFilter({
        boardId: 'board-1',
        galleryView: 'images',
        searchTerm: '',
        semanticQuery: { imageName: 'ref.png', kind: 'image' },
      })
    );

    client.setQueryData(semanticKey, createData([[target, untouched]]));

    patchGalleryItemCaches(client, {
      boardId: 'board-2',
      kind: 'move',
      result: getResult([{ kind: 'image', name: target.name }]),
    });

    const after = getData(client, semanticKey);

    expect(after.pages.flatMap((page) => page.items)).toEqual([{ ...target, boardId: 'board-2' }, untouched]);
    expect(after.pages.map((page) => page.total)).toEqual([2]);
  });

  it('does not let rollback clobber a later concurrent cache update', () => {
    const client = createClient();
    const target = createItem('target.png');
    const key = getItemsKey('board-1');

    client.setQueryData(key, createData([[target]]));
    const rollback = patchGalleryItemCaches(client, {
      kind: 'star',
      result: getResult([{ kind: 'image', name: target.name }]),
      starred: true,
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
  id,
  imageCount: 1,
  kind: 'board',
  name: `Board ${id}`,
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

    // Structural sharing rebuilds the array, so compare by value and keep the
    // untouched board's identity.
    expect(client.getQueryData(key)).toEqual(before);
    expect(client.getQueryData<GalleryBoard[]>(key)?.[1]).toBe(untouched);
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

  it('leaves lists without the board untouched', () => {
    const client = createClient();
    const key = getBoardsKey();
    const boards = [createBoard('board-2')];

    client.setQueryData(key, boards);
    patchGalleryBoardCaches(client, 'board-1', { archived: true });

    expect(client.getQueryData(key)).toBe(boards);
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
