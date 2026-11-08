/**
 * Regression test for the gallery-has-content decision (PR #9163 review).
 *
 * The bug: `useHasImages` decided emptiness from boards + the uncategorized *image* total
 * only. A gallery whose sole content was one uncategorized video was treated as empty, so
 * `NoContentForViewer` showed the new-user/get-started view instead of the normal
 * no-selection state.
 */
import { describe, expect, it } from 'vitest';

import { getHasGalleryContent } from './useHasImages';

describe('getHasGalleryContent', () => {
  it('counts a single uncategorized video as content (zero boards, zero uncategorized images)', () => {
    expect(getHasGalleryContent([], 0, 1)).toBe(true);
  });

  it('is empty with zero boards, zero uncategorized images and zero uncategorized videos', () => {
    expect(getHasGalleryContent([], 0, 0)).toBe(false);
  });

  it('counts a video-only board as content', () => {
    expect(getHasGalleryContent([{ image_count: 0, video_count: 2 }], 0, 0)).toBe(true);
  });

  it('counts uncategorized images as content', () => {
    expect(getHasGalleryContent([], 3, 0)).toBe(true);
  });

  it('defaults to content when no totals have arrived yet', () => {
    expect(getHasGalleryContent(undefined, undefined, undefined)).toBe(true);
  });
});
