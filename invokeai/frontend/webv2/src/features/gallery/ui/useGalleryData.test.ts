import type { GalleryItem } from '@features/gallery/core/items';
import type { GalleryImage, GeneratedImageContract } from '@features/gallery/core/types';

import { getBoundedRecentImages } from '@features/gallery/core/recentImages';
import { describe, expect, it } from 'vitest';

import { isGalleryWindowTruncated, mergeGalleryItemWindow } from './useGalleryData';

const createImage = (index: number, overrides: Partial<GalleryImage> = {}): GalleryImage => ({
  boardId: 'none',
  height: 512,
  imageCategory: 'general',
  imageName: `image-${String(index).padStart(4, '0')}.png`,
  imageUrl: `/images/${index}`,
  queuedAt: new Date(Date.UTC(2026, 0, 1, 0, 0, index)).toISOString(),
  sourceQueueItemId: `queue-${index}`,
  starred: false,
  thumbnailUrl: `/thumbnails/${index}`,
  width: 512,
  ...overrides,
});

const asGenerated = (image: GalleryImage): GeneratedImageContract => image;

const filter = {
  boardId: 'none',
  galleryView: 'images' as const,
  orderDir: 'DESC' as const,
  searchTerm: '',
  starredFirst: false,
};

describe('mergeGalleryItemWindow', () => {
  it('deduplicates by qualified key and mirrors server starred/time/kind/name ordering', () => {
    const image = {
      boardId: 'none',
      category: 'general',
      createdAt: '2026-07-30T12:00:00.000Z',
      fullUrl: '/images/shared',
      height: 64,
      isIntermediate: false,
      kind: 'image',
      name: 'shared',
      starred: false,
      thumbnailUrl: '/thumbnails/shared',
      width: 64,
    } satisfies GalleryItem;
    const video = {
      ...image,
      durationSeconds: 2,
      fullUrl: '/videos/shared',
      kind: 'video',
    } satisfies GalleryItem;
    const recent = asGenerated(
      createImage(99, {
        imageName: 'recent',
        queuedAt: image.createdAt,
        starred: true,
      })
    );

    expect(
      mergeGalleryItemWindow({
        backendItems: [image, video, image],
        filter: { ...filter, starredFirst: true },
        maxRows: 60,
        recentImages: [recent],
      }).map(({ kind, name }) => `${kind}:${name}`)
    ).toEqual(['image:recent', 'video:shared', 'image:shared']);

    expect(
      mergeGalleryItemWindow({
        backendItems: [image, video],
        filter: { ...filter, orderDir: 'ASC' },
        maxRows: 60,
        recentImages: [],
      }).map(({ kind, name }) => `${kind}:${name}`)
    ).toEqual(['image:shared', 'video:shared']);
  });

  it('bounds the optimistic image overlay and the mixed rendered window', () => {
    const createItem = (index: number): GalleryItem => ({
      boardId: 'none',
      category: 'general',
      createdAt: new Date(Date.UTC(2026, 0, 1, 0, 0, index)).toISOString(),
      fullUrl: `/images/${index}`,
      height: 512,
      isIntermediate: false,
      kind: 'image',
      name: `image-${index}.png`,
      starred: false,
      thumbnailUrl: `/thumbnails/${index}`,
      width: 512,
    });
    const backendItems = Array.from({ length: 600 }, (_, index) => createItem(index));
    const recentImages = getBoundedRecentImages(
      Array.from({ length: 1_000 }, (_, index) => asGenerated(createImage(1_000 + index)))
    );
    const items = mergeGalleryItemWindow({ backendItems, filter, maxRows: 600, recentImages });

    expect(recentImages).toHaveLength(60);
    expect(items).toHaveLength(600);
    expect(items.slice(0, 60).map((item) => item.name)).toEqual(recentImages.map((image) => image.imageName).reverse());
  });

  it('preserves backend relevance order and skips the recent overlay while a semantic query is active', () => {
    const createRankedItem = (name: string, createdAt: string): GalleryItem => ({
      boardId: 'none',
      category: 'general',
      createdAt,
      fullUrl: `/images/${name}`,
      height: 64,
      isIntermediate: false,
      kind: 'image',
      name,
      starred: false,
      thumbnailUrl: `/thumbnails/${name}`,
      width: 64,
    });
    // Deliberately out of chronological order: relevance is the order.
    const rankedItems = [
      createRankedItem('oldest.png', '2026-01-01T00:00:00.000Z'),
      createRankedItem('newest.png', '2026-07-01T00:00:00.000Z'),
      createRankedItem('middle.png', '2026-03-01T00:00:00.000Z'),
    ];

    expect(
      mergeGalleryItemWindow({
        backendItems: rankedItems,
        filter: { ...filter, semanticQuery: { imageName: 'ref.png', kind: 'image' } },
        maxRows: 60,
        recentImages: [asGenerated(createImage(1))],
      }).map((item) => item.name)
    ).toEqual(['oldest.png', 'newest.png', 'middle.png']);
  });

  it('uses SQLite binary ordering for mixed-case and punctuation name ties in both directions', () => {
    const createTiedItem = (name: string): GalleryItem => ({
      boardId: 'none',
      category: 'general',
      createdAt: '2026-07-30T12:00:00.000Z',
      fullUrl: `/images/${name}`,
      height: 64,
      isIntermediate: false,
      kind: 'image',
      name,
      starred: false,
      thumbnailUrl: `/thumbnails/${name}`,
      width: 64,
    });
    const items = ['a.png', 'Z.png', '_draft.png', 'A.png', '!bang.png'].map(createTiedItem);

    expect(
      mergeGalleryItemWindow({
        backendItems: items,
        filter: { ...filter, orderDir: 'ASC' },
        maxRows: 60,
        recentImages: [],
      }).map((item) => item.name)
    ).toEqual(['!bang.png', 'A.png', 'Z.png', '_draft.png', 'a.png']);

    expect(
      mergeGalleryItemWindow({
        backendItems: items,
        filter: { ...filter, orderDir: 'DESC' },
        maxRows: 60,
        recentImages: [],
      }).map((item) => item.name)
    ).toEqual(['a.png', '_draft.png', 'Z.png', 'A.png', '!bang.png']);
  });
});

describe('isGalleryWindowTruncated', () => {
  const atCap = { hasNextPage: false, isPaginated: false, loadedRowCount: 600, maxRows: 600, total: 1_000 };

  it('reports truncation only when the full window hides reachable images', () => {
    expect(isGalleryWindowTruncated(atCap)).toBe(true);
  });

  it('does not report truncation at the true end of a board', () => {
    expect(isGalleryWindowTruncated({ ...atCap, total: 600 })).toBe(false);
    expect(isGalleryWindowTruncated({ ...atCap, loadedRowCount: 240, total: 240 })).toBe(false);
  });

  it('does not report truncation while more pages can still be loaded', () => {
    expect(isGalleryWindowTruncated({ ...atCap, hasNextPage: true })).toBe(false);
  });

  it('never reports truncation in paginated mode, where every page is reachable', () => {
    expect(isGalleryWindowTruncated({ ...atCap, isPaginated: true, loadedRowCount: 60, maxRows: 60 })).toBe(false);
  });

  it('does not report truncation before the backend total is known', () => {
    expect(isGalleryWindowTruncated({ ...atCap, total: null })).toBe(false);
  });
});
