import { describe, expect, it } from 'vitest';

import { getMediaUrl } from './mediaCookieRefresh';

describe('getMediaUrl', () => {
  it('leaves the URL unchanged before the media cookie is refreshed', () => {
    expect(getMediaUrl('/api/v1/images/i/image.png/full', 0)).toBe('/api/v1/images/i/image.png/full');
  });

  it('changes the URL after the media cookie is refreshed', () => {
    expect(getMediaUrl('/api/v1/images/i/image.png/full', 1)).not.toBe(
      getMediaUrl('/api/v1/images/i/image.png/full', 2)
    );
  });

  it('preserves existing query parameters and fragments', () => {
    expect(getMediaUrl('/media/video.mp4?download=false#preview', 3)).toBe(
      '/media/video.mp4?download=false&media_cookie_version=3#preview'
    );
  });

  it('preserves an empty URL', () => {
    expect(getMediaUrl(undefined, 1)).toBeUndefined();
  });
});
