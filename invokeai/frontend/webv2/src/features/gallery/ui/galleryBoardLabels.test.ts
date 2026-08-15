import type { GalleryBoard } from '@features/gallery/core/types';

import { describe, expect, it } from 'vitest';

import { getGalleryCountForView } from './galleryBoardLabels';

const createBoard = (overrides: Partial<GalleryBoard> = {}): GalleryBoard => ({
  archived: false,
  assetCount: 3,
  id: 'dogs',
  imageCount: 50,
  kind: 'board',
  name: 'dogs',
  projectId: null,
  videoCount: 4,
  ...overrides,
});

describe('getGalleryCountForView', () => {
  it('counts images and videos together for the media view', () => {
    expect(getGalleryCountForView(createBoard(), 'images')).toBe(54);
  });

  it('counts assets alone for the assets view', () => {
    expect(getGalleryCountForView(createBoard(), 'assets')).toBe(3);
  });
});
