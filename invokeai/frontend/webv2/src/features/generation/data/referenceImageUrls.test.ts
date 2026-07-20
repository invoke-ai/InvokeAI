import { describe, expect, it, vi } from 'vitest';

vi.mock('@features/gallery/utility', () => ({
  galleryImageUrls: {
    full: (imageName: string) => `/full/${imageName}`,
    thumbnail: (imageName: string) => `/thumbnail/${imageName}`,
  },
}));

import { getReferenceImageUrls } from './referenceImageUrls';

describe('reference image URLs', () => {
  it('resolves URLs for the effective cropped image', () => {
    expect(
      getReferenceImageUrls({
        crop: {
          box: { height: 10, width: 10, x: 0, y: 0 },
          image: { height: 10, image_name: 'crop.png', width: 10 },
          ratio: 1,
        },
        original: { image: { height: 20, image_name: 'original.png', width: 20 } },
      })
    ).toEqual({ imageUrl: '/full/crop.png', thumbnailUrl: '/thumbnail/crop.png' });
  });
});
