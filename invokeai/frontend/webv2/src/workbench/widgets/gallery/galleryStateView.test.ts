import { describe, expect, it } from 'vitest';

import type { GalleryBoard } from '@workbench/gallery/api';
import type { GeneratedImageContract, QueueItem, QueueItemStatus } from '@workbench/types';
import {
  getBoardCounts,
  getGalleryQueuePlaceholders,
  getGallerySelectedBoardId,
  getGalleryStateView,
} from './galleryStateView';

const boards: GalleryBoard[] = [
  { archived: false, assetCount: 0, id: 'none', imageCount: 1, kind: 'uncategorized', name: 'Uncategorized' },
  { archived: false, assetCount: 0, id: 'board-1', imageCount: 2, kind: 'board', name: 'Board 1' },
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

const createQueueItem = ({
  backendItemIds,
  batchCount = 1,
  boardId = 'none',
  destination = 'gallery',
  status = 'pending',
}: {
  backendItemIds?: number[];
  batchCount?: number;
  boardId?: string;
  destination?: 'gallery' | 'canvas';
  status?: QueueItemStatus;
}): QueueItem =>
  ({
    backendItemIds,
    cancellable: true,
    id: `queue-item-${status}-${boardId}`,
    snapshot: {
      destination,
      widgetStates: {
        gallery: { id: 'gallery', label: 'Gallery', values: { selectedBoardId: boardId }, version: 1 },
        generate: { id: 'generate', label: 'Generate', values: { batchCount, height: 768, width: 512 }, version: 1 },
      },
    },
    status,
  }) as unknown as QueueItem;

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

  it('parses persisted gallery settings with safe defaults', () => {
    const gallery = getGalleryStateView({ boardOrderBy: 'board_name', starredFirst: false }, boards, [], false);

    expect(gallery.settings).toEqual({
      boardOrderBy: 'board_name',
      boardOrderDir: 'DESC',
      imageDensityPercent: 50,
      imageOrderDir: 'DESC',
      paginationMode: 'infinite',
      showArchivedBoards: false,
      showDateBoards: false,
      showImageDimensions: false,
      starredFirst: false,
      thumbnailFit: 'square',
    });
  });

  it('derives the multi-selection from persisted values with a single-selection fallback', () => {
    expect(
      getGalleryStateView({ selectedImageNames: ['a.png', 'b.png', 7] }, boards, [], false).selectedImageNames
    ).toEqual(['a.png', 'b.png']);
    expect(getGalleryStateView({ selectedImageName: 'a.png' }, boards, [], false).selectedImageNames).toEqual([
      'a.png',
    ]);
  });

  it('creates placeholders for in-flight gallery queue items on the viewed board', () => {
    const queueItems = [
      createQueueItem({ batchCount: 2, boardId: 'board-1', status: 'pending' }),
      createQueueItem({ backendItemIds: [11, 12, 13], boardId: 'board-1', status: 'running' }),
    ];
    const placeholders = getGalleryQueuePlaceholders(queueItems, {
      galleryView: 'images',
      searchTerm: '',
      selectedBoardId: 'board-1',
    });

    expect(placeholders).toHaveLength(5);
    expect(placeholders[0]).toMatchObject({ height: 768, width: 512 });
  });

  it('does not create placeholders for other boards, finished items, or canvas runs', () => {
    const queueItems = [
      createQueueItem({ boardId: 'board-2', status: 'pending' }),
      createQueueItem({ boardId: 'board-1', status: 'completed' }),
      createQueueItem({ boardId: 'board-1', destination: 'canvas', status: 'pending' }),
    ];

    expect(
      getGalleryQueuePlaceholders(queueItems, { galleryView: 'images', searchTerm: '', selectedBoardId: 'board-1' })
    ).toEqual([]);
  });

  it('hides placeholders while searching or browsing assets', () => {
    const queueItems = [createQueueItem({ boardId: 'none', status: 'pending' })];

    expect(
      getGalleryQueuePlaceholders(queueItems, { galleryView: 'images', searchTerm: 'cat', selectedBoardId: 'none' })
    ).toEqual([]);
    expect(
      getGalleryQueuePlaceholders(queueItems, { galleryView: 'assets', searchTerm: '', selectedBoardId: 'none' })
    ).toEqual([]);
    expect(
      getGalleryStateView({ selectedBoardId: 'none' }, boards, [], false, queueItems).pendingPlaceholders
    ).toHaveLength(1);
  });
});
