import type { GalleryItem } from '@features/gallery/contracts';

import { describe, expect, it } from 'vitest';

import {
  getGalleryBoardDropData,
  getGalleryItemDragData,
  getGalleryItemDragId,
  isGalleryBoardDropData,
  isGalleryImageDragData,
  isGalleryItemDragData,
  resolveGalleryBoardDrop,
} from './galleryDnd';

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

  return kind === 'video' ? { ...base, durationSeconds: 12.4, kind } : { ...base, kind };
};

describe('galleryDnd', () => {
  it('constructs the shared item payload and namespaces qualified draggable ids by source', () => {
    const refs = [
      { kind: 'image', name: 'shared' },
      { kind: 'video', name: 'shared' },
    ] as const;

    expect(getGalleryItemDragData(refs)).toEqual({
      items: refs,
      kind: 'gallery-item',
    });
    expect(getGalleryItemDragId(refs[0], 'gallery-grid')).toBe('gallery-grid:image:shared');
    expect(getGalleryItemDragId(refs[0], 'preview-frame')).toBe('preview-frame:image:shared');
    expect(getGalleryItemDragId(refs[0], 'preview-filmstrip')).toBe('preview-filmstrip:image:shared');
    expect(getGalleryItemDragId(refs[1], 'gallery-grid')).toBe('gallery-grid:video:shared');
  });

  it('recognizes only non-empty, well-formed gallery item payloads', () => {
    expect(isGalleryItemDragData(getGalleryItemDragData([{ kind: 'video', name: 'clip.mp4' }]))).toBe(true);
    expect(isGalleryItemDragData({ items: [], kind: 'gallery-item' })).toBe(false);
    expect(isGalleryItemDragData({ items: [{ kind: 'video' }], kind: 'gallery-item' })).toBe(false);
    expect(
      isGalleryItemDragData({ images: [{ boardId: 'none', imageName: 'legacy.png' }], kind: 'gallery-image' })
    ).toBe(false);
  });

  it('refines a non-empty all-image payload and exposes image names without a video cast', () => {
    const getImageNames = (value: unknown): string[] =>
      isGalleryImageDragData(value) ? value.items.map((item) => item.name) : [];
    const imagePayload = getGalleryItemDragData([
      { kind: 'image', name: 'first.png' },
      { kind: 'image', name: 'second.png' },
    ]);

    expect(getImageNames(imagePayload)).toEqual(['first.png', 'second.png']);
    expect(isGalleryImageDragData(getGalleryItemDragData([{ kind: 'video', name: 'clip.mp4' }]))).toBe(false);
    expect(
      isGalleryImageDragData(
        getGalleryItemDragData([
          { kind: 'image', name: 'still.png' },
          { kind: 'video', name: 'clip.mp4' },
        ])
      )
    ).toBe(false);
    expect(isGalleryImageDragData({ items: [], kind: 'gallery-item' })).toBe(false);
  });

  it('forwards ordered image, video, and mixed refs to real board drops', () => {
    const imageRef = { kind: 'image', name: 'image.png' } as const;
    const videoRef = { kind: 'video', name: 'video.mp4' } as const;
    const boardDrop = getGalleryBoardDropData('board-b', 'board');

    expect(resolveGalleryBoardDrop(getGalleryItemDragData([imageRef]), boardDrop, [])).toEqual({
      boardId: 'board-b',
      items: [imageRef],
    });
    expect(resolveGalleryBoardDrop(getGalleryItemDragData([videoRef]), boardDrop, [])).toEqual({
      boardId: 'board-b',
      items: [videoRef],
    });
    expect(resolveGalleryBoardDrop(getGalleryItemDragData([videoRef, imageRef]), boardDrop, [])).toEqual({
      boardId: 'board-b',
      items: [videoRef, imageRef],
    });
  });

  it('uses canonical loaded item identity to exclude only refs already on the destination board', () => {
    const dragData = getGalleryItemDragData([
      { kind: 'image', name: 'shared' },
      { kind: 'video', name: 'shared' },
      { kind: 'image', name: 'other.png' },
    ]);
    const loadedItems = [
      createItem('image', 'shared', 'board-a'),
      createItem('video', 'shared', 'board-b'),
      createItem('image', 'other.png', 'none'),
    ];

    expect(resolveGalleryBoardDrop(dragData, getGalleryBoardDropData('board-a', 'board'), loadedItems)).toEqual({
      boardId: 'board-a',
      items: [
        { kind: 'video', name: 'shared' },
        { kind: 'image', name: 'other.png' },
      ],
    });
  });

  it('rejects non-item drags, virtual-board drops, and drops with no item left to move', () => {
    const image = createItem('image', 'image.png', 'board-a');
    const dragData = getGalleryItemDragData([{ kind: 'image', name: image.name }]);

    expect(
      resolveGalleryBoardDrop({ kind: 'widget-instance' }, getGalleryBoardDropData('board-b', 'board'), [image])
    ).toBe(null);
    expect(
      resolveGalleryBoardDrop(dragData, getGalleryBoardDropData('by_date:2026-07-30', 'date'), [image])
    ).toBeNull();
    expect(resolveGalleryBoardDrop(dragData, getGalleryBoardDropData('board-a', 'board'), [image])).toBeNull();
  });

  it('recognizes gallery board drop data', () => {
    expect(isGalleryBoardDropData(getGalleryBoardDropData('board-a', 'board'))).toBe(true);
    expect(isGalleryBoardDropData({ boardId: 'board-a', boardKind: 'generated', kind: 'gallery-board' })).toBe(false);
  });
});
