import { describe, expect, it } from 'vitest';

import type { GalleryImage, GeneratedImageContract } from './types';

import {
  assertNeverGalleryItem,
  classifyGalleryUpload,
  compareGalleryItems,
  getGalleryUploadAccept,
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

  it('prefers the backend creation timestamp over the submission instant when both are present', () => {
    const item = legacyGeneratedImageToGalleryItem({
      ...generatedImage,
      createdAt: '2026-07-30T12:45:00.000Z',
      queuedAt: '2026-07-30T12:00:00.000Z',
    });

    expect(item.createdAt).toBe('2026-07-30T12:45:00.000Z');
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
      createdAt: '2026-07-30T12:00:00.000Z',
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
    expect(legacyGeneratedImageToGalleryItem(galleryImageItemToGalleryImage(item))).toEqual({
      ...item,
      sourceQueueItemId: 'backend-gallery',
    });
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

describe('gallery item ordering', () => {
  // The two timestamp shapes the grid mixes: backend rows carry SQLite's
  // `created_at`; overlaid recents carry the queue's ISO `submittedAt`.
  const sqliteItem = galleryItem('backend.png', false);
  const overlayItem = legacyGeneratedImageToGalleryItem(generatedImage);

  it('sorts a newer backend item before an older overlaid recent in DESC order', () => {
    const olderOverlay = { ...overlayItem, createdAt: '2026-08-29T02:28:40.566Z' };
    const newerBackend = { ...sqliteItem, createdAt: '2026-08-29 13:01:20.649' };

    const sorted = [olderOverlay, newerBackend].sort((a, b) => compareGalleryItems(a, b, { orderDir: 'DESC' }));

    expect(sorted.map((item) => item.name)).toEqual(['backend.png', 'legacy.png']);
  });

  it('sorts an older overlaid recent before a newer backend item in ASC order', () => {
    const olderOverlay = { ...overlayItem, createdAt: '2026-08-29T02:28:40.566Z' };
    const newerBackend = { ...sqliteItem, createdAt: '2026-08-29 13:01:20.649' };

    const sorted = [newerBackend, olderOverlay].sort((a, b) => compareGalleryItems(a, b, { orderDir: 'ASC' }));

    expect(sorted.map((item) => item.name)).toEqual(['legacy.png', 'backend.png']);
  });

  it('falls through to the kind tie-breaker when the two shapes name the same instant', () => {
    const image = { ...sqliteItem, createdAt: '2026-08-29 13:01:20.649' };
    const video = { ...image, durationSeconds: 2, kind: 'video' as const, name: 'backend.png' };

    // The same instant written in both shapes is a chronological tie. (DESC
    // ties come back as -0 — direction * 0 — so compare with `===`.)
    expect(
      compareGalleryItems({ ...image, createdAt: '2026-08-29T13:01:20.649Z' }, image, { orderDir: 'DESC' }) === 0
    ).toBe(true);

    const sorted = [image, video].sort((a, b) => compareGalleryItems(a, b, { orderDir: 'DESC' }));

    expect(sorted.map((item) => item.kind)).toEqual(['video', 'image']);
  });
});

describe('classifyGalleryUpload', () => {
  it.each([
    ['image/png', 'photo.bin', 'image'],
    ['image/jpeg', 'photo.bin', 'image'],
    ['image/jpg', 'photo.bin', 'image'],
    ['image/webp', 'photo.bin', 'image'],
    ['video/mp4', 'clip.bin', 'video'],
    ['video/quicktime', 'clip.bin', 'video'],
    ['video/webm', 'clip.webm', 'video'],
    ['audio/mpeg', 'song.bin', 'video'],
    ['audio/wav', 'memo.bin', 'video'],
    ['', 'photo.PNG', 'image'],
    ['', 'clip.MOV', 'video'],
    ['', 'memo.m4a', 'video'],
    ['application/octet-stream', 'photo.jpeg', 'image'],
    ['application/octet-stream', 'song.mp3', 'video'],
    ['binary/octet-stream', 'clip.MP4', 'video'],
    ['application/octet-stream', 'clip.wmv', 'video'],
    ['application/octet-stream', 'song.WMA', 'video'],
    ['application/pdf', 'photo.png', 'image'],
  ] as const)('classifies MIME %s and name %s as %s', (type, name, kind) => {
    expect(classifyGalleryUpload(new File(['media'], name, { type }))).toEqual({ kind });
  });

  it.each([
    ['application/pdf', 'document.pdf'],
    ['', 'archive.zip'],
    ['image/gif', 'animation.gif'],
  ] as const)('rejects unsupported MIME %s and name %s', (type, name) => {
    expect(classifyGalleryUpload(new File(['media'], name, { type }))).toBeNull();
  });

  it('uses a supported MIME before a conflicting extension', () => {
    expect(classifyGalleryUpload(new File(['media'], 'looks-like-video.mp4', { type: 'image/png' }))).toEqual({
      kind: 'image',
    });
  });
});

describe('getGalleryUploadAccept', () => {
  // Assert every accepted video/audio extension from videos.py; sampled expectations would miss picker
  // restrictions.
  it('offers every container and audio format the video upload route ingests', () => {
    expect(getGalleryUploadAccept(['video']).split(',')).toEqual([
      'video/*',
      'audio/*',
      '.mp4',
      '.mov',
      '.m4v',
      '.webm',
      '.mkv',
      '.avi',
      '.mpg',
      '.mpeg',
      '.3gp',
      '.wmv',
      '.asf',
      '.mp3',
      '.m4a',
      '.aac',
      '.wav',
      '.flac',
      '.ogg',
      '.oga',
      '.opus',
      '.aiff',
      '.aif',
      '.wma',
    ]);
  });

  it('keeps images to the three formats the app stores without re-encoding', () => {
    expect(getGalleryUploadAccept(['image'])).toBe('image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp');
  });

  it('concatenates the kinds it is given, so a mixed picker offers both', () => {
    expect(getGalleryUploadAccept(['image', 'video'])).toBe(
      `${getGalleryUploadAccept(['image'])},${getGalleryUploadAccept(['video'])}`
    );
  });
});
