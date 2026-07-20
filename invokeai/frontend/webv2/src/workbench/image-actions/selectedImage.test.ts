import type { GeneratedImageContract } from '@features/gallery';

import { describe, expect, it } from 'vitest';

import { getSelectedGalleryImageFromValues } from './selectedImage';

const image: GeneratedImageContract = {
  height: 768,
  imageName: 'selected.png',
  imageUrl: '/selected.png',
  queuedAt: '2026-06-19T00:00:00.000Z',
  sourceQueueItemId: 'queue-item',
  thumbnailUrl: '/selected-thumb.png',
  width: 512,
};

describe('getSelectedGalleryImageFromValues', () => {
  it('normalizes the stored selected image for shared image actions', () => {
    expect(getSelectedGalleryImageFromValues({ selectedBoardId: 'board-1', selectedImage: image })).toMatchObject({
      boardId: 'board-1',
      imageCategory: 'general',
      imageName: 'selected.png',
      starred: false,
    });
  });

  it('falls back to recent images by selected name', () => {
    expect(
      getSelectedGalleryImageFromValues({
        recentImages: [image],
        selectedBoardId: 'board-1',
        selectedImageName: image.imageName,
      })
    )?.toMatchObject({ imageName: 'selected.png' });
  });
});
