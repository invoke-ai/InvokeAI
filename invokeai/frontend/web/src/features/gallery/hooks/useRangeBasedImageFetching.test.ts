import { describe, expect, it } from 'vitest';

import { getVideoPrefetchOptions, hasCachedVideoDTO } from './useRangeBasedImageFetching';

describe('video range prefetch', () => {
  it('does not retain an RTK Query subscription', () => {
    expect(getVideoPrefetchOptions()).toEqual({ subscribe: false, forceRefetch: true });
  });

  it('only treats fulfilled DTO queries as cached', () => {
    expect(hasCachedVideoDTO({ data: { video_name: 'video.mp4' } })).toBe(true);
    expect(hasCachedVideoDTO({ isError: true })).toBe(false);
  });
});
