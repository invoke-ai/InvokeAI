import { describe, expect, it } from 'vitest';

import type { GalleryImageItem, GalleryItem, GalleryVideoItem } from './items';
import type { GeneratedImageContract } from './types';

import * as selection from './selection';

const legacyImage: GeneratedImageContract = {
  height: 768,
  imageName: 'legacy.png',
  imageUrl: '/images/legacy.png',
  queuedAt: '2026-07-30T00:00:00.000Z',
  sourceQueueItemId: 'queue-item',
  thumbnailUrl: '/thumbnails/legacy.png',
  width: 512,
};

const imageItem: GalleryImageItem = {
  boardId: 'board-1',
  category: 'general',
  createdAt: '2026-07-30T01:00:00.000Z',
  fullUrl: '/images/shared',
  height: 768,
  isIntermediate: false,
  kind: 'image',
  name: 'shared',
  sourceQueueItemId: 'backend-gallery',
  starred: true,
  thumbnailUrl: '/thumbnails/shared',
  width: 512,
};

const videoItem: GalleryVideoItem = {
  boardId: 'board-1',
  category: 'general',
  createdAt: '2026-07-30T02:00:00.000Z',
  durationSeconds: 4,
  fullUrl: '/videos/shared',
  height: 768,
  isIntermediate: false,
  kind: 'video',
  name: 'shared',
  starred: false,
  thumbnailUrl: '/video-thumbnails/shared',
  width: 512,
};

type SelectionModule = typeof selection & {
  getSelectedGalleryItemFromValues?: (values: Record<string, unknown>) => GalleryItem | null;
};

const getSelectedGalleryItemFromValues = (values: Record<string, unknown>): GalleryItem | null | undefined =>
  (selection as SelectionModule).getSelectedGalleryItemFromValues?.(values);

describe('persisted Gallery selection readers', () => {
  it('returns a canonical selected image item unchanged', () => {
    expect(getSelectedGalleryItemFromValues({ selectedImage: imageItem })).toBe(imageItem);
  });

  it('returns a canonical selected video while the image-only reader rejects it', () => {
    const values = { selectedImage: videoItem, selectedImageName: 'video:shared' };

    expect(getSelectedGalleryItemFromValues(values)).toBe(videoItem);
    expect(selection.getSelectedGalleryImageFromValues(values)).toBeNull();
  });

  it('adapts a legacy generated-image object into a canonical image item', () => {
    expect(
      getSelectedGalleryItemFromValues({
        selectedBoardId: 'board-1',
        selectedImage: legacyImage,
      })
    ).toEqual({
      boardId: 'board-1',
      category: 'general',
      createdAt: legacyImage.queuedAt,
      fullUrl: legacyImage.imageUrl,
      height: legacyImage.height,
      isIntermediate: false,
      kind: 'image',
      name: legacyImage.imageName,
      sourceQueueItemId: legacyImage.sourceQueueItemId,
      starred: false,
      thumbnailUrl: legacyImage.thumbnailUrl,
      width: legacyImage.width,
    });
  });

  it.each(['legacy.png', 'image:legacy.png'])(
    'adapts the legacy recent-image fallback selected by %s',
    (selectedImageName) => {
      const values = {
        recentImages: [legacyImage],
        selectedBoardId: 'board-1',
        selectedImageName,
      };

      expect(getSelectedGalleryItemFromValues(values)).toMatchObject({
        boardId: 'board-1',
        kind: 'image',
        name: legacyImage.imageName,
      });
      expect(selection.getSelectedGalleryImageFromValues(values)).toMatchObject({
        boardId: 'board-1',
        imageName: legacyImage.imageName,
      });
    }
  );

  it('converts a canonical image item for legacy image-only consumers', () => {
    expect(selection.getSelectedGalleryImageFromValues({ selectedImage: imageItem })).toEqual({
      boardId: imageItem.boardId,
      height: imageItem.height,
      imageCategory: imageItem.category,
      imageName: imageItem.name,
      imageUrl: imageItem.fullUrl,
      queuedAt: imageItem.createdAt,
      sourceQueueItemId: imageItem.sourceQueueItemId,
      starred: imageItem.starred,
      thumbnailUrl: imageItem.thumbnailUrl,
      width: imageItem.width,
    });
  });
});
