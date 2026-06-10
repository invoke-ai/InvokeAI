import { describe, expect, it } from 'vitest';

import type { GalleryBoard } from '../../gallery/api';
import type { GeneratedImageContract } from '../../types';
import { getBoardCounts, getGallerySelectedBoardId, getGalleryStateView } from './galleryStateView';

const boards: GalleryBoard[] = [
  { assetCount: 0, id: 'none', imageCount: 1, isVirtual: true, name: 'Uncategorized' },
  { assetCount: 0, id: 'board-1', imageCount: 2, name: 'Board 1' },
];

const createImage = (imageName: string): GeneratedImageContract => ({
  height: 768,
  imageName,
  imageUrl: `/api/v1/images/i/${imageName}/full`,
  queuedAt: '2026-06-09T00:00:00.000Z',
  sourceQueueItemId: 'queue-item-1',
  thumbnailUrl: `/api/v1/images/i/${imageName}/thumbnail`,
  width: 512,
});

describe('gallery state view', () => {
  it('preserves a selected backend board id while boards are still loading', () => {
    const values = { selectedBoardId: 'board-1' };

    expect(getGallerySelectedBoardId(values, [])).toBe('board-1');
    expect(getGalleryStateView(values, [], null, false).selectedBoardId).toBe('board-1');
  });

  it('falls back to uncategorized after loaded boards do not contain the selected board', () => {
    const values = { selectedBoardId: 'missing-board' };

    expect(getGallerySelectedBoardId(values, boards)).toBe('none');
    expect(getGalleryStateView(values, boards, [], false).selectedBoardId).toBe('none');
  });

  it('does not render local fallback images while backend images are loading', () => {
    const values = { recentImages: [createImage('local-fallback.png')], selectedBoardId: 'none' };
    const gallery = getGalleryStateView(values, boards, null, true);

    expect(gallery.images).toEqual([]);
    expect(gallery.isLoading).toBe(true);
  });

  it('exposes both image and asset counts for board labels', () => {
    expect(getBoardCounts(boards[1])).toEqual({ assetCount: 0, imageCount: 2 });
  });
});
