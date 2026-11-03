import { describe, expect, it } from 'vitest';

import { getGalleryImageFullPath, getGalleryImageThumbnailPath } from './imagePaths';

describe('gallery image paths', () => {
  it('encodes image names without depending on transport URL resolution', () => {
    expect(getGalleryImageFullPath('folder/image one.png')).toBe('/api/v1/images/i/folder%2Fimage%20one.png/full');
    expect(getGalleryImageThumbnailPath('folder/image one.png')).toBe(
      '/api/v1/images/i/folder%2Fimage%20one.png/thumbnail'
    );
  });
});
