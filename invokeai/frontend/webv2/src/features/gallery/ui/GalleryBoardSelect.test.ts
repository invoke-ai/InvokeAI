import type { GalleryItem } from '@features/gallery/contracts';

import { describe, expect, it, vi } from 'vitest';

import { forwardGalleryBoardDrop } from './GalleryBoardSelect';
import { getGalleryBoardDropData, getGalleryItemDragData } from './galleryDnd';

const createItem = (kind: GalleryItem['kind'], name: string, boardId: string): GalleryItem => {
  const base = {
    boardId,
    category: 'general' as const,
    createdAt: '2026-07-30T00:00:00.000Z',
    fullUrl: `/full/${name}`,
    height: 64,
    isIntermediate: false,
    name,
    starred: false,
    thumbnailUrl: `/thumbnail/${name}`,
    width: 64,
  };

  return kind === 'video' ? { ...base, durationSeconds: 8, kind } : { ...base, kind };
};

describe('forwardGalleryBoardDrop', () => {
  it('forwards the full ordered mixed resolution without stripping video refs', () => {
    const moveItemsToBoard = vi.fn();
    const dragData = getGalleryItemDragData([
      { kind: 'video', name: 'shared' },
      { kind: 'image', name: 'image.png' },
      { kind: 'image', name: 'shared' },
    ]);
    const loadedItems = [
      createItem('image', 'shared', 'board-b'),
      createItem('video', 'shared', 'board-a'),
      createItem('image', 'image.png', 'none'),
    ];

    expect(
      forwardGalleryBoardDrop({
        activeData: dragData,
        loadedItems,
        moveItemsToBoard,
        overData: getGalleryBoardDropData('board-b', 'board'),
      })
    ).toBe(true);
    expect(moveItemsToBoard).toHaveBeenCalledOnce();
    expect(moveItemsToBoard).toHaveBeenCalledWith(
      [
        { kind: 'video', name: 'shared' },
        { kind: 'image', name: 'image.png' },
      ],
      'board-b'
    );
  });
});
