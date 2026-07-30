import type { GalleryItem } from '@features/gallery/core/items';
import type { GalleryImage, GeneratedImageContract } from '@features/gallery/core/types';

import { getBoundedRecentImages } from '@features/gallery/core/recentImages';
import { describe, expect, it } from 'vitest';

import { isGalleryWindowTruncated, mergeGalleryImageWindow, mergeGalleryItemWindow } from './useGalleryData';

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

describe('mergeGalleryImageWindow', () => {
  it('deduplicates the recent overlay and keeps newest-first ordering', () => {
    const backend = [createImage(1), createImage(2)];
    const recent = [asGenerated(createImage(3)), asGenerated(createImage(2))];

    expect(
      mergeGalleryImageWindow({ backendImages: backend, filter, maxRows: 600, recentImages: recent }).map(
        (image) => image.imageName
      )
    ).toEqual(['image-0003.png', 'image-0002.png', 'image-0001.png']);
  });

  it('bounds both the optimistic overlay and the rendered infinite window', () => {
    const backend = Array.from({ length: 600 }, (_, index) => createImage(index));
    const rawRecent = Array.from({ length: 1_000 }, (_, index) => asGenerated(createImage(1_000 + index)));
    const recent = getBoundedRecentImages(rawRecent);
    const images = mergeGalleryImageWindow({ backendImages: backend, filter, maxRows: 600, recentImages: recent });

    expect(recent).toHaveLength(60);
    expect(images).toHaveLength(600);
    expect(images.slice(0, 60).map((image) => image.imageName)).toEqual(
      recent.map((image) => image.imageName).reverse()
    );
  });

  it('does not overlay recent images into incompatible board, asset, search, or date filters', () => {
    const recent = [asGenerated(createImage(2))];
    const filters = [
      { ...filter, boardId: 'board-2' },
      { ...filter, galleryView: 'assets' as const },
      { ...filter, searchTerm: 'portrait' },
      { ...filter, createdFrom: '2026-01-01' },
      { ...filter, boardId: 'by_date:2026-01-01' },
    ];

    for (const candidate of filters) {
      expect(
        mergeGalleryImageWindow({
          backendImages: [createImage(1)],
          filter: candidate,
          maxRows: 600,
          recentImages: recent,
        })
      ).toHaveLength(1);
    }
  });

  it('preserves starred-first ordering and the paginated row cap', () => {
    const images = mergeGalleryImageWindow({
      backendImages: Array.from({ length: 60 }, (_, index) => createImage(index)),
      filter: { ...filter, starredFirst: true },
      maxRows: 60,
      recentImages: [asGenerated(createImage(100, { starred: true }))],
    });

    expect(images).toHaveLength(60);
    expect(images[0]).toMatchObject({ imageName: 'image-0100.png', starred: true });
  });
});

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
