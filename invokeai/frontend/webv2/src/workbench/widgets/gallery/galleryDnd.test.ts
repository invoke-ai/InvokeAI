import { describe, expect, it } from 'vitest';

import {
  getGalleryBoardDropData,
  getGalleryImageDragData,
  getGalleryImageNamesOutsideBoard,
  isGalleryBoardDropData,
  isGalleryImageDragData,
} from './galleryDnd';

describe('galleryDnd', () => {
  it('keeps only images outside the target board', () => {
    const dragData = getGalleryImageDragData([
      { boardId: 'board-a', imageName: 'own.png' },
      { boardId: 'board-b', imageName: 'other.png' },
      { boardId: 'none', imageName: 'uncategorized.png' },
    ]);

    expect(getGalleryImageNamesOutsideBoard(dragData, 'board-a')).toEqual(['other.png', 'uncategorized.png']);
  });

  it('recognizes gallery image drag data', () => {
    expect(isGalleryImageDragData(getGalleryImageDragData([{ boardId: 'none', imageName: 'image.png' }]))).toBe(true);
    expect(isGalleryImageDragData({ images: [{ boardId: 'none' }], kind: 'gallery-image' })).toBe(false);
  });

  it('recognizes gallery board drop data', () => {
    expect(isGalleryBoardDropData(getGalleryBoardDropData('board-a', 'board'))).toBe(true);
    expect(isGalleryBoardDropData({ boardId: 'board-a', boardKind: 'generated', kind: 'gallery-board' })).toBe(false);
  });
});
