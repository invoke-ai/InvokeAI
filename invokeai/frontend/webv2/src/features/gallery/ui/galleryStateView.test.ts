import type { GalleryBoard, GeneratedImageContract } from '@features/gallery/core/types';
import type { QueueHistoryItemStatus, QueueItem } from '@features/queue/contracts';

import { describe, expect, it } from 'vitest';

import {
  getBoardCounts,
  getGalleryCurrentItem,
  getGalleryGenerationSequence,
  getGalleryImagesRefreshToken,
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
  completedBackendItemIds,
  cancelledBackendItemIds,
  destination = 'gallery',
  status = 'pending',
  localId,
}: {
  backendItemIds?: number[];
  batchCount?: number;
  boardId?: string;
  completedBackendItemIds?: number[];
  cancelledBackendItemIds?: number[];
  destination?: 'gallery' | 'canvas';
  status?: QueueHistoryItemStatus;
  localId?: string;
}): QueueItem => ({
  backendItemIds,
  cancellable: true,
  cancelledBackendItemIds,
  completedBackendItemIds,
  id: localId ?? `queue-item-${status}-${boardId}`,
  snapshot: {
    backendSubmission: {
      batchCount,
      graph: { edges: [], id: 'gallery-placeholder-graph', nodes: {} },
      kind: 'workflow',
    },
    destination,
    filterIntermediateResults: false,
    galleryBoardId: boardId,
    graph: { id: 'gallery-placeholder-graph', label: 'Gallery placeholder' },
    presentation: { batchCount, height: 768, width: 512 },
    sourceId: 'workflow',
    submittedAt: '2026-06-09T00:00:00.000Z',
  },
  status,
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
      showPendingItems: true,
      starredFirst: false,
      thumbnailFit: 'square',
    });
  });

  it('falls back to the full gallery refresh token when image refresh token is absent', () => {
    expect(getGalleryImagesRefreshToken({ galleryRefreshToken: 'full-refresh' })).toBe('full-refresh');
    expect(
      getGalleryImagesRefreshToken({ galleryImagesRefreshToken: 'image-refresh', galleryRefreshToken: 'full-refresh' })
    ).toBe('image-refresh');
  });

  it('derives the multi-selection from persisted values with a single-selection fallback', () => {
    expect(
      getGalleryStateView({ selectedImageNames: ['a.png', 'b.png', 7] }, boards, [], false).selectedImageNames
    ).toEqual(['a.png', 'b.png']);
    expect(getGalleryStateView({ selectedImageName: 'a.png' }, boards, [], false).selectedImageNames).toEqual([
      'a.png',
    ]);
  });

  it('restores the selection set from a visible primary image after tab switches clear it', () => {
    const image = createImage('selected.png');
    const gallery = getGalleryStateView(
      { selectedImageName: image.imageName, selectedImageNames: [] },
      boards,
      [{ ...image, boardId: 'none', imageCategory: 'general', starred: false }],
      false
    );

    expect(gallery.selectedImageName).toBe(image.imageName);
    expect(gallery.selectedImageNames).toEqual([image.imageName]);
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
    expect(placeholders[0]).toMatchObject({
      height: 768,
      itemIndex: 1,
      queueItemId: 'queue-item-running-board-1',
      width: 512,
    });
  });

  it('skips completed backend item slots when creating placeholders', () => {
    const queueItems = [
      createQueueItem({
        backendItemIds: [11, 12, 13],
        boardId: 'board-1',
        completedBackendItemIds: [11],
        status: 'running',
      }),
    ];

    const placeholders = getGalleryQueuePlaceholders(queueItems, {
      galleryView: 'images',
      searchTerm: '',
      selectedBoardId: 'board-1',
    });

    expect(placeholders.map((placeholder) => placeholder.itemIndex)).toEqual([2, 3]);
  });

  it('resolves the live slot from the exact denoising target instead of the newest submitted batch', () => {
    const newerBatch = createQueueItem({
      backendItemIds: [21, 22],
      boardId: 'board-1',
      localId: 'newer-batch',
      status: 'running',
    });
    const activeEarlierBatch = createQueueItem({
      backendItemIds: [11, 12],
      boardId: 'board-1',
      completedBackendItemIds: [11],
      localId: 'active-earlier-batch',
      status: 'running',
    });

    expect(
      getGalleryGenerationSequence([newerBatch, activeEarlierBatch], {
        itemIndex: 2,
        queueItemId: activeEarlierBatch.id,
      }).liveSlot
    ).toMatchObject({
      boardId: 'board-1',
      itemIndex: 2,
      queueItemId: activeEarlierBatch.id,
    });
  });

  it('flattens pending slots from multiple batches into one global execution order', () => {
    const newerBatch = createQueueItem({
      backendItemIds: [21, 22],
      boardId: 'board-1',
      localId: 'newer-batch',
      status: 'running',
    });
    const earlierBatch = createQueueItem({
      backendItemIds: [11, 12],
      boardId: 'board-1',
      localId: 'earlier-batch',
      status: 'running',
    });

    const sequence = getGalleryGenerationSequence([newerBatch, earlierBatch], null);

    expect(sequence.chronologicalSlots.map((slot) => slot.id)).toEqual([
      'earlier-batch:0',
      'earlier-batch:1',
      'newer-batch:0',
      'newer-batch:1',
    ]);
    expect(sequence.liveSlot).toBeNull();
  });

  it('keeps unsubmitted batches grouped after backend-ranked slots', () => {
    const submittedBatch = createQueueItem({
      backendItemIds: [11],
      boardId: 'board-1',
      localId: 'submitted-batch',
      status: 'running',
    });
    const pendingBatchB = createQueueItem({ batchCount: 2, boardId: 'board-1', localId: 'batch-b' });
    const pendingBatchA = createQueueItem({ batchCount: 2, boardId: 'board-1', localId: 'batch-a' });

    expect(
      getGalleryGenerationSequence([pendingBatchB, submittedBatch, pendingBatchA], null).chronologicalSlots.map(
        (slot) => slot.id
      )
    ).toEqual(['submitted-batch:0', 'batch-a:0', 'batch-a:1', 'batch-b:0', 'batch-b:1']);
  });

  it('reverses the complete flattened sequence for newest-first galleries', () => {
    const newerBatch = createQueueItem({
      backendItemIds: [21, 22],
      boardId: 'board-1',
      localId: 'newer-batch',
      status: 'running',
    });
    const earlierBatch = createQueueItem({
      backendItemIds: [11, 12],
      boardId: 'board-1',
      localId: 'earlier-batch',
      status: 'running',
    });

    const placeholders = getGalleryQueuePlaceholders([newerBatch, earlierBatch], {
      galleryView: 'images',
      imageOrderDir: 'DESC',
      searchTerm: '',
      selectedBoardId: 'board-1',
    });

    expect(placeholders.map((slot) => slot.id)).toEqual([
      'newer-batch:1',
      'newer-batch:0',
      'earlier-batch:1',
      'earlier-batch:0',
    ]);
  });

  it('uses one derived current item while live-follow is active', () => {
    const placeholder = {
      boardId: 'board-1',
      height: 768,
      id: 'queue-item:0',
      itemIndex: 1,
      queueItemId: 'queue-item',
      width: 512,
    };

    expect(
      getGalleryCurrentItem({
        activePlaceholder: placeholder,
        isComparisonActive: false,
        liveFollowEnabled: true,
        selectedImageName: 'saved.png',
      })
    ).toEqual({ kind: 'placeholder', placeholder });
    expect(
      getGalleryCurrentItem({
        activePlaceholder: placeholder,
        isComparisonActive: false,
        liveFollowEnabled: false,
        selectedImageName: 'saved.png',
      })
    ).toEqual({ imageName: 'saved.png', kind: 'image' });
  });

  it('keeps the visible image selected when the active placeholder belongs to another board', () => {
    const image = {
      ...createImage('selected.png'),
      boardId: 'board-1',
      imageCategory: 'general',
      starred: false,
    } as const;
    const activeItem = createQueueItem({ boardId: 'board-2', localId: 'active-other-board', status: 'running' });
    const gallery = getGalleryStateView(
      { selectedBoardId: 'board-1', selectedImageName: image.imageName },
      boards,
      [image],
      false,
      [activeItem],
      true,
      { itemIndex: 1, queueItemId: activeItem.id }
    );

    expect(gallery.currentItem).toEqual({ imageName: image.imageName, kind: 'image' });
  });

  it('skips cancelled backend item slots when creating placeholders', () => {
    const queueItems = [
      createQueueItem({
        backendItemIds: [11, 12, 13],
        boardId: 'board-1',
        cancelledBackendItemIds: [12],
        completedBackendItemIds: [11],
        status: 'running',
      }),
    ];

    const placeholders = getGalleryQueuePlaceholders(queueItems, {
      galleryView: 'images',
      searchTerm: '',
      selectedBoardId: 'board-1',
    });

    expect(placeholders.map((placeholder) => placeholder.itemIndex)).toEqual([3]);
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

  it('hides only pending placeholders when the persisted setting is disabled', () => {
    const queueItems = [createQueueItem({ boardId: 'none', status: 'pending' })];
    const image = {
      ...createImage('completed.png'),
      boardId: 'none',
      imageCategory: 'general',
      starred: false,
    } as const;
    const gallery = getGalleryStateView(
      { selectedBoardId: 'none', showPendingItems: false },
      boards,
      [image],
      false,
      queueItems
    );

    expect(gallery.pendingPlaceholders).toEqual([]);
    expect(gallery.images).toEqual([image]);
    expect(gallery.settings.showPendingItems).toBe(false);
  });
});
