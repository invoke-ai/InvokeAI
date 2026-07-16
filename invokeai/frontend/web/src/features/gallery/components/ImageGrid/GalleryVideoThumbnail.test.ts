import { describe, expect, it } from 'vitest';

import { getVideoThumbnailMode } from './GalleryVideoThumbnail';

describe('getVideoThumbnailMode', () => {
  it('uses the video after the current thumbnail fails', () => {
    expect(getVideoThumbnailMode('thumb.webp', 'thumb.webp')).toBe('video');
  });

  it('retries the image when the thumbnail URL changes', () => {
    expect(getVideoThumbnailMode('new-thumb.webp', 'old-thumb.webp')).toBe('image');
  });
});
