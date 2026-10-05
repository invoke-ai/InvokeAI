import { describe, expect, it, vi } from 'vitest';

vi.mock('@platform/transport/http', () => ({
  absolutizeApiUrl: (path: string) => `https://api.test${path}`,
}));

import { getGalleryImageFullUrl, getGalleryImageThumbnailUrl } from './imageUrls';

describe('gallery image URLs', () => {
  it('absolutizes pure gallery image paths at the Data boundary', () => {
    expect(getGalleryImageFullUrl('image.png')).toBe('https://api.test/api/v1/images/i/image.png/full');
    expect(getGalleryImageThumbnailUrl('image.png')).toBe('https://api.test/api/v1/images/i/image.png/thumbnail');
  });
});
