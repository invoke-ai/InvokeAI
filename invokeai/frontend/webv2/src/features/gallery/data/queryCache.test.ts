import type { GalleryImage, GalleryImagesPage } from '@features/gallery/core/types';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { QueryClient, type InfiniteData } from '@tanstack/react-query';
import { beforeEach, describe, expect, it } from 'vitest';

import { ALL_READABLE_BOARDS_ID } from './backend';
import { canonicalizeGalleryImagesFilter, galleryKeys } from './queries';
import { invalidateGallery, invalidateGalleryImages, patchGalleryImageCaches } from './queryCache';

type GalleryImagesData = InfiniteData<GalleryImagesPage, number>;

const createClient = (): QueryClient =>
  new QueryClient({
    defaultOptions: {
      queries: {
        gcTime: Infinity,
        retry: false,
      },
    },
  });

const createImage = (imageName: string, boardId = 'board-1', starred = false): GalleryImage => ({
  boardId,
  height: 768,
  imageCategory: 'general',
  imageName,
  imageUrl: `/api/v1/images/i/${imageName}/full`,
  queuedAt: '2026-07-26T00:00:00.000Z',
  sourceQueueItemId: 'backend-gallery',
  starred,
  thumbnailUrl: `/api/v1/images/i/${imageName}/thumbnail`,
  width: 512,
});

const createData = (pages: GalleryImage[][]): GalleryImagesData => {
  const total = pages.reduce((count, images) => count + images.length, 0);

  return {
    pageParams: pages.map((_, index) => index * 60),
    pages: pages.map((images) => ({ images, total })),
  };
};

const getImagesKey = (boardId: string, owner: AccountScope = captureAccountScope()) =>
  galleryKeys.images(
    owner,
    canonicalizeGalleryImagesFilter({
      boardId,
      galleryView: 'images',
      searchTerm: '',
    })
  );

const getData = (client: QueryClient, queryKey: ReturnType<typeof getImagesKey>): GalleryImagesData => {
  const data = client.getQueryData<GalleryImagesData>(queryKey);

  expect(data).toBeDefined();

  return data as GalleryImagesData;
};

describe('Gallery image cache patches', () => {
  beforeEach(() => {
    accountLifecycle.activate('gallery-query-cache-test');
  });

  it('patches a star in place and rolls back without rebuilding untouched pages or page params', () => {
    const client = createClient();
    const target = createImage('target.png');
    const samePageUntouched = createImage('same-page-untouched.png');
    const untouchedPageImage = createImage('untouched-page.png');
    const key = getImagesKey('board-1');

    client.setQueryData(key, createData([[target, samePageUntouched], [untouchedPageImage]]));
    const before = getData(client, key);
    const firstPageBefore = before.pages[0];
    const untouchedPageBefore = before.pages[1];
    const pageParamsBefore = before.pageParams;

    const rollback = patchGalleryImageCaches(client, {
      imageNames: [target.imageName],
      kind: 'star',
      starred: true,
    });
    const after = getData(client, key);

    expect(after).not.toBe(before);
    expect(after.pageParams).toBe(pageParamsBefore);
    expect(after.pages[0]).not.toBe(firstPageBefore);
    expect(after.pages[1]).toBe(untouchedPageBefore);
    expect(after.pages[0]?.images[0]).toEqual({ ...target, starred: true });
    expect(after.pages[0]?.images[1]).toBe(samePageUntouched);

    rollback();

    const rolledBack = getData(client, key);

    expect(rolledBack).toEqual(before);
    expect(rolledBack.pageParams).toBe(pageParamsBefore);
    expect(rolledBack.pages[1]).toBe(untouchedPageBefore);
    expect(rolledBack.pages[0]?.images[1]).toBe(samePageUntouched);
  });

  it('deletes matching images across pages and keeps each page total consistent', () => {
    const client = createClient();
    const firstTarget = createImage('first-target.png');
    const firstUntouched = createImage('first-untouched.png');
    const secondTarget = createImage('second-target.png');
    const secondUntouched = createImage('second-untouched.png');
    const key = getImagesKey('board-1');

    client.setQueryData(
      key,
      createData([
        [firstTarget, firstUntouched],
        [secondTarget, secondUntouched],
      ])
    );
    const before = getData(client, key);

    patchGalleryImageCaches(client, {
      imageNames: [firstTarget.imageName, secondTarget.imageName],
      kind: 'delete',
    });
    const after = getData(client, key);

    expect(after.pageParams).toBe(before.pageParams);
    expect(after.pages.map((page) => page.images)).toEqual([[firstUntouched], [secondUntouched]]);
    expect(after.pages.map((page) => page.total)).toEqual([2, 2]);
  });

  it('removes moved images from source boards while all-images and date views retain the updated image', () => {
    const client = createClient();
    const target = createImage('moved.png', 'board-1');
    const untouched = createImage('untouched.png', 'board-1');
    const sourceKey = getImagesKey('board-1');
    const allKey = getImagesKey(ALL_READABLE_BOARDS_ID);
    const dateKey = getImagesKey('by_date:2026-07-26');

    client.setQueryData(sourceKey, createData([[target], [untouched]]));
    client.setQueryData(allKey, createData([[target], [untouched]]));
    client.setQueryData(dateKey, createData([[target], [untouched]]));

    patchGalleryImageCaches(client, {
      boardId: 'board-2',
      imageNames: [target.imageName],
      kind: 'move',
    });

    const source = getData(client, sourceKey);
    const all = getData(client, allKey);
    const date = getData(client, dateKey);

    expect(source.pages.flatMap((page) => page.images)).toEqual([untouched]);
    expect(source.pages.map((page) => page.total)).toEqual([1, 1]);
    for (const retained of [all, date]) {
      expect(retained.pages.flatMap((page) => page.images)).toEqual([{ ...target, boardId: 'board-2' }, untouched]);
      expect(retained.pages.map((page) => page.total)).toEqual([2, 2]);
    }
  });

  it('does not let rollback clobber a later concurrent cache update', () => {
    const client = createClient();
    const target = createImage('target.png');
    const key = getImagesKey('board-1');

    client.setQueryData(key, createData([[target]]));
    const rollback = patchGalleryImageCaches(client, {
      imageNames: [target.imageName],
      kind: 'star',
      starred: true,
    });
    const concurrent = client.setQueryData<GalleryImagesData>(key, (current) => {
      if (!current) {
        return current;
      }

      return {
        ...current,
        pages: current.pages.map((page) => ({
          ...page,
          images: page.images.map((image) =>
            image.imageName === target.imageName ? { ...image, boardId: 'board-concurrent' } : image
          ),
        })),
      };
    });

    rollback();

    expect(getData(client, key)).toBe(concurrent);
    expect(getData(client, key).pages[0]?.images[0]).toMatchObject({
      boardId: 'board-concurrent',
      imageName: target.imageName,
      starred: true,
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
    const oldImagesKey = getImagesKey('board-1', oldOwner);
    const oldFilter = canonicalizeGalleryImagesFilter({
      boardId: 'by_date:2026-07-25',
      galleryView: 'images',
      searchTerm: '',
    });
    const oldDateNamesKey = galleryKeys.dateBoardNames(oldOwner, oldFilter);
    const oldBoardsKey = galleryKeys.boards(oldOwner, {
      includeArchived: false,
      includeDateBoards: true,
      orderBy: 'created_at',
      orderDir: 'DESC',
    });

    client.setQueryData(oldImagesKey, createData([[createImage('old.png')]]));
    client.setQueryData(oldDateNamesKey, { imageNames: ['old.png'], total: 1 });
    client.setQueryData(oldBoardsKey, []);

    accountLifecycle.activate('gallery-query-cache-current-account');
    const currentOwner = captureAccountScope();
    const currentImagesKey = getImagesKey('board-1', currentOwner);
    const currentFilter = canonicalizeGalleryImagesFilter({
      boardId: 'by_date:2026-07-26',
      galleryView: 'images',
      searchTerm: '',
    });
    const currentDateNamesKey = galleryKeys.dateBoardNames(currentOwner, currentFilter);
    const currentBoardsKey = galleryKeys.boards(currentOwner, {
      includeArchived: false,
      includeDateBoards: true,
      orderBy: 'created_at',
      orderDir: 'DESC',
    });

    client.setQueryData(currentImagesKey, createData([[createImage('current.png')]]));
    client.setQueryData(currentDateNamesKey, { imageNames: ['current.png'], total: 1 });
    client.setQueryData(currentBoardsKey, []);
    const initialCardinality = client.getQueryCache().getAll().length;

    for (let index = 0; index < 100; index += 1) {
      await invalidateGalleryImages(client);
    }

    expect(client.getQueryCache().getAll()).toHaveLength(initialCardinality);
    expect(client.getQueryState(currentImagesKey)?.isInvalidated).toBe(true);
    expect(client.getQueryState(currentDateNamesKey)?.isInvalidated).toBe(true);
    expect(client.getQueryState(currentBoardsKey)?.isInvalidated).toBe(false);
    expect(client.getQueryState(oldImagesKey)?.isInvalidated).toBe(false);
    expect(client.getQueryState(oldDateNamesKey)?.isInvalidated).toBe(false);

    await invalidateGallery(client);

    expect(client.getQueryCache().getAll()).toHaveLength(initialCardinality);
    expect(client.getQueryState(currentBoardsKey)?.isInvalidated).toBe(true);
    expect(client.getQueryState(oldBoardsKey)?.isInvalidated).toBe(false);
  });
});
