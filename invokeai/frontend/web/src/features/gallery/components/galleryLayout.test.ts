import { describe, expect, test } from 'vitest';

import { getEffectiveGalleryLayout } from './galleryLayout';

describe('gallery layout', () => {
  test('uses selected layout when paged gallery is disabled', () => {
    expect(getEffectiveGalleryLayout('grid', false)).toBe('grid');
    expect(getEffectiveGalleryLayout('masonry', false)).toBe('masonry');
  });

  test('forces grid layout when paged gallery is enabled', () => {
    expect(getEffectiveGalleryLayout('masonry', true)).toBe('grid');
  });
});
