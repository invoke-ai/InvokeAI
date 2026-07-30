import type { GalleryItem, GalleryItemMutationResult, GalleryItemsPage } from '@features/gallery/core/items';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { QueryClient, type InfiniteData } from '@tanstack/react-query';
import { beforeEach, describe, expect, it } from 'vitest';

import { canonicalizeGalleryItemsFilter, galleryKeys } from './queries';
import { invalidateGallery, patchGalleryItemCaches } from './queryCache';

const createItem = (kind: 'image' | 'video', name: string, boardId = 'board-1'): GalleryItem =>
  kind === 'image'
    ? {
        boardId,
        category: 'general',
        createdAt: '2026-07-30T12:00:00.000Z',
        fullUrl: `/full/${name}`,
        height: 64,
        isIntermediate: false,
        kind,
        name,
        starred: false,
        thumbnailUrl: `/thumbnail/${name}`,
        width: 64,
      }
    : {
        boardId,
        category: 'general',
        createdAt: '2026-07-30T12:00:00.000Z',
        durationSeconds: 1,
        fullUrl: `/full/${name}`,
        height: 64,
        isIntermediate: false,
        kind,
        name,
        starred: false,
        thumbnailUrl: `/thumbnail/${name}`,
        width: 64,
      };

const createClient = () => new QueryClient({ defaultOptions: { queries: { gcTime: Infinity, retry: false } } });

const getItemsKey = (owner: AccountScope = captureAccountScope()) =>
  galleryKeys.items(
    owner,
    canonicalizeGalleryItemsFilter({ boardId: 'board-1', galleryView: 'images', searchTerm: '' })
  );

const createData = (items: GalleryItem[]): InfiniteData<GalleryItemsPage, number> => ({
  pageParams: [0],
  pages: [{ items, total: items.length }],
});

describe('Qualified gallery cache patches', () => {
  beforeEach(() => {
    accountLifecycle.activate('task-4-cache-test');
  });

  it('patches only confirmed successful qualified refs', () => {
    const client = createClient();
    const key = getItemsKey();
    const image = createItem('image', 'shared-name');
    const video = createItem('video', 'shared-name');
    const result: GalleryItemMutationResult = {
      failed: [{ kind: 'video', name: 'shared-name' }],
      succeeded: [{ kind: 'image', name: 'shared-name' }],
    };

    client.setQueryData(key, createData([image, video]));
    patchGalleryItemCaches(client, { kind: 'delete', result });

    expect(
      client.getQueryData<InfiniteData<GalleryItemsPage, number>>(key)?.pages.flatMap((page) => page.items)
    ).toEqual([video]);
  });

  it('invalidates item pages, item-name lists, and boards once for a same-tick burst', async () => {
    const client = createClient();
    const owner = captureAccountScope();
    const filter = canonicalizeGalleryItemsFilter({ boardId: 'board-1', galleryView: 'images', searchTerm: '' });
    const itemsKey = galleryKeys.items(owner, filter);
    const namesKey = galleryKeys.itemNames(owner, filter);
    const boardsKey = galleryKeys.boards(owner, {
      includeArchived: false,
      includeDateBoards: true,
      orderBy: 'created_at',
      orderDir: 'DESC',
    });

    client.setQueryData(itemsKey, createData([createItem('image', 'one')]));
    client.setQueryData(namesKey, { items: [{ kind: 'image', name: 'one' }], starredCount: 0, total: 1 });
    client.setQueryData(boardsKey, []);

    await Promise.all([invalidateGallery(client), invalidateGallery(client), invalidateGallery(client)]);

    expect(client.getQueryState(itemsKey)?.isInvalidated).toBe(true);
    expect(client.getQueryState(namesKey)?.isInvalidated).toBe(true);
    expect(client.getQueryState(boardsKey)?.isInvalidated).toBe(true);
  });
});
