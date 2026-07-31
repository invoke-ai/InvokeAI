import type { GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery/core/items';
import type { GalleryBoardKind } from '@features/gallery/core/types';

import { useDndContext, useDroppable, type UseDroppableArguments } from '@dnd-kit/core';
import { toGalleryItemKey } from '@features/gallery/core/items';

export interface GalleryItemDragData {
  kind: 'gallery-item';
  items: GalleryItemRef[];
}

export interface GalleryImageDragItem extends GalleryItemRef {
  kind: 'image';
}

export interface GalleryImageDragData extends GalleryItemDragData {
  items: [GalleryImageDragItem, ...GalleryImageDragItem[]];
}

export interface GalleryBoardDropData {
  boardId: string;
  boardKind: GalleryBoardKind;
  kind: 'gallery-board';
}

export interface GalleryBoardDropResolution {
  boardId: string;
  items: GalleryItemRef[];
}

export type GalleryItemDragSource = 'gallery-grid' | 'preview-filmstrip' | 'preview-frame';
export type GalleryItemDragId = `${GalleryItemDragSource}:${GalleryItemKey}`;

export const getGalleryItemDragId = (item: GalleryItemRef, source: GalleryItemDragSource): GalleryItemDragId =>
  `${source}:${toGalleryItemKey(item)}`;

export const getGalleryBoardDropId = (boardId: string): string => `gallery-board:${boardId}`;

export const getGalleryItemDragData = (items: readonly GalleryItemRef[]): GalleryItemDragData => ({
  items: [...items],
  kind: 'gallery-item',
});

export const getGalleryBoardDropData = (boardId: string, boardKind: GalleryBoardKind): GalleryBoardDropData => ({
  boardId,
  boardKind,
  kind: 'gallery-board',
});

export const isGalleryItemDragData = (value: unknown): value is GalleryItemDragData =>
  isRecord(value) &&
  value.kind === 'gallery-item' &&
  Array.isArray(value.items) &&
  value.items.length > 0 &&
  value.items.every(isGalleryItemRef);

export const isGalleryImageDragData = (value: unknown): value is GalleryImageDragData =>
  isGalleryItemDragData(value) && value.items.every((item): item is GalleryImageDragItem => item.kind === 'image');

export const useGalleryImageDroppable = ({ disabled = false, ...args }: UseDroppableArguments) => {
  const { active } = useDndContext();
  const acceptsActiveDrag = isGalleryImageDragData(active?.data.current);
  const droppable = useDroppable({ ...args, disabled: disabled || !acceptsActiveDrag });

  return { ...droppable, isOver: acceptsActiveDrag && droppable.isOver };
};

export const isGalleryBoardDropData = (value: unknown): value is GalleryBoardDropData =>
  isRecord(value) &&
  value.kind === 'gallery-board' &&
  typeof value.boardId === 'string' &&
  isGalleryBoardKind(value.boardKind);

export const getGalleryItemRefsOutsideBoard = (
  dragData: GalleryItemDragData,
  boardId: string,
  loadedItems: readonly GalleryItem[]
): GalleryItemRef[] => {
  const loadedItemsByKey = new Map(loadedItems.map((item) => [toGalleryItemKey(item), item]));

  return dragData.items.filter((ref) => loadedItemsByKey.get(toGalleryItemKey(ref))?.boardId !== boardId);
};

export const resolveGalleryBoardDrop = (
  activeData: unknown,
  overData: unknown,
  loadedItems: readonly GalleryItem[]
): GalleryBoardDropResolution | null => {
  if (!isGalleryItemDragData(activeData) || !isGalleryBoardDropData(overData) || overData.boardKind !== 'board') {
    return null;
  }

  const items = getGalleryItemRefsOutsideBoard(activeData, overData.boardId, loadedItems);

  return items.length > 0 ? { boardId: overData.boardId, items } : null;
};

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

const isGalleryItemRef = (value: unknown): value is GalleryItemRef =>
  isRecord(value) &&
  (value.kind === 'image' || value.kind === 'video') &&
  typeof value.name === 'string' &&
  value.name.length > 0;

const isGalleryBoardKind = (value: unknown): value is GalleryBoardKind =>
  value === 'board' || value === 'date' || value === 'uncategorized';
