import { describe, expect, it } from 'vitest';

import type { GalleryImage, GeneratedImageContract } from './types';

import {
  assertNeverGalleryItem,
  formatGalleryVideoDuration,
  galleryImageItemToGalleryImage,
  isGalleryImageItem,
  legacyGeneratedImageToGalleryItem,
  parseGalleryItemKey,
  shouldStarSelection,
  toGalleryItemKey,
  toGalleryItemRef,
  type GalleryImageItem,
} from './items';

const galleryItem = (name: string, starred: boolean): GalleryImageItem => ({
  boardId: 'none',
  category: 'general',
  createdAt: '2026-07-30T12:00:00.000Z',
  fullUrl: `/images/${name}/full`,
  height: 512,
  isIntermediate: false,
  kind: 'image',
  name,
  starred,
  thumbnailUrl: `/images/${name}/thumbnail`,
  width: 768,
});

const generatedImage: GeneratedImageContract = {
  height: 512,
  imageName: 'legacy.png',
  imageUrl: '/images/legacy.png/full',
  queuedAt: '2026-07-30T12:00:00.000Z',
  sourceQueueItemId: 'queue-123',
  thumbnailUrl: '/images/legacy.png/thumbnail',
  width: 768,
};

describe('gallery item keys', () => {
  it('creates an unambiguous key and reference for every media kind', () => {
    const item: GalleryImageItem = {
      boardId: 'none',
      category: 'general',
      createdAt: '2026-07-30T12:00:00.000Z',
      fullUrl: '/images/example.png/full',
      height: 512,
      isIntermediate: false,
      kind: 'image',
      name: 'folder:example.png',
      starred: false,
      thumbnailUrl: '/images/example.png/thumbnail',
      width: 768,
    };

    expect(toGalleryItemKey(item)).toBe('image:folder:example.png');
    expect(toGalleryItemRef(item)).toEqual({ kind: 'image', name: 'folder:example.png' });
  });

  it('parses only non-empty exact media prefixes while retaining legacy names', () => {
    expect(parseGalleryItemKey('image:folder:example.png')).toEqual({ kind: 'image', name: 'folder:example.png' });
    expect(parseGalleryItemKey('video:clip:take-1.mp4')).toEqual({ kind: 'video', name: 'clip:take-1.mp4' });
    expect(parseGalleryItemKey('legacy:folder:example.png')).toEqual({
      kind: 'image',
      name: 'legacy:folder:example.png',
    });
    expect(parseGalleryItemKey('image:')).toEqual({ kind: 'image', name: 'image:' });
    expect(parseGalleryItemKey('video:')).toEqual({ kind: 'image', name: 'video:' });
    expect(parseGalleryItemKey('bare.png')).toEqual({ kind: 'image', name: 'bare.png' });
  });
});

describe('selection starring policy', () => {
  const starred = galleryItem('starred.png', true);
  const unstarred = galleryItem('unstarred.png', false);

  it('does not star an empty selection', () => {
    expect(shouldStarSelection([starred], [])).toBe(false);
  });

  it('unstars only when every selected item is loaded and starred', () => {
    expect(shouldStarSelection([starred], [{ kind: 'image', name: starred.name }])).toBe(false);
  });

  it('stars when any selected item is loaded and unstarred', () => {
    expect(
      shouldStarSelection(
        [starred, unstarred],
        [
          { kind: 'image', name: starred.name },
          { kind: 'image', name: unstarred.name },
        ]
      )
    ).toBe(true);
  });

  it('stars when any selected item has not been loaded', () => {
    expect(
      shouldStarSelection(
        [starred],
        [
          { kind: 'image', name: starred.name },
          { kind: 'video', name: 'not-loaded.mp4' },
        ]
      )
    ).toBe(true);
  });
});

describe('gallery item compatibility', () => {
  it('normalizes missing legacy gallery context when adapting a generated image', () => {
    expect(legacyGeneratedImageToGalleryItem(generatedImage)).toEqual({
      boardId: 'none',
      category: 'general',
      createdAt: '2026-07-30T12:00:00.000Z',
      fullUrl: '/images/legacy.png/full',
      height: 512,
      isIntermediate: false,
      kind: 'image',
      name: 'legacy.png',
      sourceQueueItemId: 'queue-123',
      starred: false,
      thumbnailUrl: '/images/legacy.png/thumbnail',
      width: 768,
    });
  });

  it('round-trips an image item into the legacy gallery contract with a backend source fallback', () => {
    const item: GalleryImageItem = {
      boardId: 'board-1',
      category: 'user',
      createdAt: '2026-07-30T12:00:00.000Z',
      fullUrl: '/images/upload.png/full',
      height: 512,
      isIntermediate: false,
      kind: 'image',
      name: 'upload.png',
      starred: true,
      thumbnailUrl: '/images/upload.png/thumbnail',
      width: 768,
    };

    const expected: GalleryImage = {
      boardId: 'board-1',
      height: 512,
      imageCategory: 'user',
      imageName: 'upload.png',
      imageUrl: '/images/upload.png/full',
      queuedAt: '2026-07-30T12:00:00.000Z',
      sourceQueueItemId: 'backend-gallery',
      starred: true,
      thumbnailUrl: '/images/upload.png/thumbnail',
      width: 768,
    };

    expect(galleryImageItemToGalleryImage(item)).toEqual(expected);
  });
});

describe('gallery item discriminators', () => {
  it('recognizes images and rejects videos', () => {
    const image = legacyGeneratedImageToGalleryItem(generatedImage);
    const video = { ...image, durationSeconds: 12, kind: 'video' as const };

    expect(isGalleryImageItem(image)).toBe(true);
    expect(isGalleryImageItem(video)).toBe(false);
  });

  it('throws when an exhaustive gallery-item branch receives an impossible value', () => {
    expect(() => assertNeverGalleryItem('audio' as never)).toThrow('Unexpected gallery item');
  });
});

describe('gallery video duration formatting', () => {
  it.each([
    [Number.NaN, '0:00'],
    [-1, '0:00'],
    [0, '0:00'],
    [0.1, '0:01'],
    [59.01, '1:00'],
    [65, '1:05'],
    [3_601, '1:00:01'],
  ])('formats %s seconds as %s', (duration, expected) => {
    expect(formatGalleryVideoDuration(duration)).toBe(expected);
  });
});
