import type { GalleryItem, GalleryItemsPage } from '@features/gallery/core/items';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, type InfiniteData } from '@tanstack/react-query';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const backend = vi.hoisted(() => ({
  hydrateGalleryDateBoardItemPage: vi.fn(),
  isDateBoardId: vi.fn((boardId: string) => boardId.startsWith('by_date:')),
  listGalleryBoards: vi.fn(),
  listGalleryDateBoardItemNames: vi.fn(),
  listGalleryDateBoards: vi.fn(),
  listGalleryItemNames: vi.fn(),
  listGalleryItems: vi.fn(),
  listPaletteImages: vi.fn(),
}));

vi.mock('./backend', () => backend);

import {
  canonicalizeGalleryItemsFilter,
  flattenGalleryItemsData,
  galleryItemNamesOptions,
  galleryItemsInfiniteOptions,
  galleryKeys,
  getGalleryItemsFilterFromKey,
  type GalleryItemsFilter,
} from './queries';

const baseFilter: GalleryItemsFilter = {
  boardId: 'board-1',
  galleryView: 'images',
  orderDir: 'DESC',
  searchTerm: ' portrait ',
  starredFirst: true,
};

const createItem = (kind: 'image' | 'video', name: string): GalleryItem =>
  kind === 'image'
    ? {
        boardId: 'board-1',
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
        boardId: 'board-1',
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

describe('Mixed gallery queries', () => {
  beforeEach(() => {
    accountLifecycle.activate('task-4-query-test');
    vi.clearAllMocks();
    backend.listGalleryItems.mockResolvedValue({ items: [], total: 0 });
    backend.listGalleryItemNames.mockResolvedValue({ items: [], starredCount: 0, total: 0 });
  });

  afterEach(() => {
    accountLifecycle.invalidate();
  });

  it('uses the items key segment and reads the canonical filter only from its guarded position', () => {
    const filter = canonicalizeGalleryItemsFilter(baseFilter);
    const key = galleryItemsInfiniteOptions(baseFilter).queryKey;

    expect(key.slice(0, 3)).toEqual(['gallery', 'items', 'list']);
    expect(getGalleryItemsFilterFromKey(key)).toEqual(filter);
    expect(getGalleryItemsFilterFromKey(['gallery', 'images', 'list', key[3], filter])).toBeNull();
    expect(getGalleryItemsFilterFromKey(['gallery', 'items', 'list', key[3], null])).toBeNull();
    expect(getGalleryItemsFilterFromKey(['gallery', 'items', filter, key[3]])).toBeNull();
  });

  it('deduplicates by qualified item key so an image and video may share a name', () => {
    const image = createItem('image', 'shared-name');
    const video = createItem('video', 'shared-name');
    const data: InfiniteData<GalleryItemsPage, number> = {
      pageParams: [0, 60],
      pages: [
        { items: [image, video], total: 3 },
        { items: [image], total: 3 },
      ],
    };

    expect(flattenGalleryItemsData(data)).toEqual([image, video]);
  });

  it('builds lazy normal-board item-name options with created ranges', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const options = galleryItemNamesOptions({
      ...baseFilter,
      createdFrom: '2026-07-01',
      createdTo: '2026-07-30',
    });

    expect(options.queryKey.slice(0, 3)).toEqual(galleryKeys.itemNamesRoot());
    expect(backend.listGalleryItemNames).not.toHaveBeenCalled();

    await client.fetchQuery(options);

    expect(backend.listGalleryItemNames).toHaveBeenCalledWith(
      expect.objectContaining({
        boardId: 'board-1',
        createdFrom: '2026-07-01',
        createdTo: '2026-07-30',
        searchTerm: 'portrait',
      })
    );
    expect(backend.listGalleryDateBoardItemNames).not.toHaveBeenCalled();
  });

  it('loads mixed pages through listGalleryItems without returning a legacy image page', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const image = createItem('image', 'same-name');
    const video = createItem('video', 'same-name');

    backend.listGalleryItems.mockResolvedValue({ items: [image, video], total: 2 });

    const data = await client.fetchInfiniteQuery(galleryItemsInfiniteOptions(baseFilter));

    expect(data.pages[0]).toEqual({ items: [image, video], total: 2 });
    expect(backend.listGalleryItems).toHaveBeenCalledOnce();
    expect(backend.listPaletteImages).not.toHaveBeenCalled();
  });
});
