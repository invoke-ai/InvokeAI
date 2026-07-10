/**
 * Pins the polymorphic name-flattening used by `useGalleryImageNames` for both regular boards
 * and date-based virtual boards.
 *
 * The bug (PR #9163 review): virtual boards were image-only — selecting a virtual date fetched
 * from the legacy image_names endpoint, so videos created on that date never appeared. The hook
 * now consumes the by-date item_names endpoint, which returns the same (kind, name) refs as the
 * regular gallery names endpoint, and this shared mapper must keep video refs in the flat list.
 * (The server-side guarantee that a date query returns video refs is pinned by
 * tests/app/routers/test_virtual_boards.py.)
 */
import type { GalleryItemRef } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import { itemRefsToNames } from './use-gallery-image-names';

describe('itemRefsToNames', () => {
  it('keeps video refs interleaved with images, preserving order', () => {
    const items: GalleryItemRef[] = [
      { kind: 'image', name: 'newest.png' },
      { kind: 'video', name: 'middle.mp4' },
      { kind: 'image', name: 'oldest.png' },
    ];
    expect(itemRefsToNames(items)).toEqual(['newest.png', 'middle.mp4', 'oldest.png']);
  });

  it('handles a video-only list (video-only virtual date)', () => {
    const items: GalleryItemRef[] = [
      { kind: 'video', name: 'a.mp4' },
      { kind: 'video', name: 'b.mp4' },
    ];
    expect(itemRefsToNames(items)).toEqual(['a.mp4', 'b.mp4']);
  });
});
