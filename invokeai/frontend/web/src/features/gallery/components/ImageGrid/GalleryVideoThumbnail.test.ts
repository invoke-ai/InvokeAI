import { describe, expect, it } from 'vitest';

import { getVideoThumbnailKey, getVideoThumbnailMode } from './GalleryVideoThumbnail';

describe('getVideoThumbnailMode', () => {
  it('uses the video after the current thumbnail fails', () => {
    expect(getVideoThumbnailMode('thumb.webp', { url: 'thumb.webp', mediaCookieVersion: 1 }, 1)).toBe('video');
  });

  it('retries the image when the thumbnail URL changes', () => {
    expect(getVideoThumbnailMode('new-thumb.webp', { url: 'old-thumb.webp', mediaCookieVersion: 1 }, 1)).toBe('image');
  });

  it('retries the image after the media cookie refreshes', () => {
    expect(getVideoThumbnailMode('thumb.webp', { url: 'thumb.webp', mediaCookieVersion: 1 }, 2)).toBe('image');
  });

  it('returns to the video when the post-refresh image retry fails', () => {
    expect(getVideoThumbnailMode('thumb.webp', { url: 'thumb.webp', mediaCookieVersion: 2 }, 2)).toBe('video');
  });
});

describe('getVideoThumbnailKey', () => {
  it('changes after the media cookie refreshes', () => {
    expect(getVideoThumbnailKey('thumb.webp', 2)).not.toBe(getVideoThumbnailKey('thumb.webp', 1));
  });

  it('is stable when neither the URL nor cookie version changes', () => {
    expect(getVideoThumbnailKey('thumb.webp', 1)).toBe(getVideoThumbnailKey('thumb.webp', 1));
  });
});
