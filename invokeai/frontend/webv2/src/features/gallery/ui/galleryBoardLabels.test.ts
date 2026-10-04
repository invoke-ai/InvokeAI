import type { GalleryBoard } from '@features/gallery/core/types';

import { describe, expect, it } from 'vitest';

import { getGalleryCountForView } from './galleryBoardLabels';

const createBoard = (overrides: Partial<GalleryBoard> = {}): GalleryBoard => ({
  archived: false,
  assetCount: 3,
  assetVideoCount: 0,
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

  it('splits asset-category videos out of media and into assets', () => {
    // 4 videos total, 1 uploaded (asset): media shows 50 images + 3 media videos,
    // assets shows 3 asset images + 1 asset video.
    const board = createBoard({ assetVideoCount: 1 });

    expect(getGalleryCountForView(board, 'images')).toBe(53);
    expect(getGalleryCountForView(board, 'assets')).toBe(4);
  });
});
