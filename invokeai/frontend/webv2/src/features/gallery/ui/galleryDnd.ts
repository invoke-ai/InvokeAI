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
export type GalleryItemDragId = `${GalleryItemDragSource}${string}:${GalleryItemKey}`;

/**
 * `scope` separates surfaces showing the same item: dnd-kit keys drag state by
 * id, so two galleries sharing one both report as dragging.
 */
export const getGalleryItemDragId = (
  item: GalleryItemRef,
  source: GalleryItemDragSource,
  scope?: string
): GalleryItemDragId => `${source}${scope ? `#${scope}` : ''}:${toGalleryItemKey(item)}`;

export const getGalleryBoardDropId = (boardId: string): string => `gallery-board:${boardId}`;

/** Droppable id for the search field: dropping a gallery image there searches by similarity. */
export const GALLERY_SEMANTIC_SEARCH_DROP_ID = 'gallery-semantic-search-drop';

/**
 * Resolves an item drag released over the search field to an image-similarity
 * reference. Semantic search is image-only, so a video-only drag resolves to
 * nothing rather than to a broken query.
 */
export const resolveGallerySemanticSearchDrop = (
  activeData: unknown,
  overId: unknown
): { imageName: string } | null => {
  if (overId !== GALLERY_SEMANTIC_SEARCH_DROP_ID || !isGalleryItemDragData(activeData)) {
    return null;
  }

  const image = activeData.items.find((item) => item.kind === 'image');

  return image ? { imageName: image.name } : null;
};

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

/** For targets that consume exactly one image (keyframes, the upscale input). */
export const isSingleGalleryImageDragData = (value: unknown): value is GalleryImageDragData =>
  isGalleryImageDragData(value) && value.items.length === 1;

/** For targets that consume exactly one video (the initial-video clip). */
export const isSingleGalleryVideoDragData = (value: unknown): value is GalleryItemDragData =>
  isGalleryItemDragData(value) && value.items.length === 1 && value.items[0]?.kind === 'video';

/**
 * A droppable that only participates while a drag matching `shields` is in
 * flight. `acceptsActiveDrag` is the "potential target" bit the drop
 * affordances render from (a drag the field's handler would consume exists
 * ANYWHERE — legacy dnd's `potential` state); `isOver` narrows it to
 * "hovering this target". `accepts` must match what the drop handler
 * actually consumes, or the affordance advertises a drop that would no-op.
 *
 * `shields` (default: `accepts`) keeps the droppable armed for a WIDER
 * payload family the handler merely ignores: a single-image field stays a
 * dead drop for multi-image drags, exactly as before the affordances —
 * disabling it instead would hand the release to whatever droppable sits
 * underneath (the collision pipeline is z-blind), e.g. a board row hidden
 * under a floated widget.
 */
export const useGalleryItemDroppable = (
  accepts: (data: unknown) => boolean,
  { disabled = false, ...args }: UseDroppableArguments,
  shields: (data: unknown) => boolean = accepts
) => {
  const { active } = useDndContext();
  const activeData = active?.data.current;
  const acceptsActiveDrag = !disabled && accepts(activeData);
  const isShielding = !disabled && shields(activeData);
  const droppable = useDroppable({ ...args, disabled: !isShielding });

  return { ...droppable, acceptsActiveDrag, isOver: acceptsActiveDrag && droppable.isOver };
};

export const useGalleryImageDroppable = (args: UseDroppableArguments) =>
  useGalleryItemDroppable(isGalleryImageDragData, args);

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

/** Applies a board drop if the drag resolves to one, reporting whether it did. */
export const forwardGalleryBoardDrop = ({
  activeData,
  loadedItems,
  moveItemsToBoard,
  overData,
}: {
  activeData: unknown;
  loadedItems: readonly GalleryItem[];
  moveItemsToBoard: (items: GalleryItemRef[], boardId: string) => void;
  overData: unknown;
}): boolean => {
  const resolution = resolveGalleryBoardDrop(activeData, overData, loadedItems);

  if (!resolution) {
    return false;
  }

  moveItemsToBoard(resolution.items, resolution.boardId);
  return true;
};

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

const isGalleryItemRef = (value: unknown): value is GalleryItemRef =>
  isRecord(value) &&
  (value.kind === 'image' || value.kind === 'video') &&
  typeof value.name === 'string' &&
  value.name.length > 0;

const isGalleryBoardKind = (value: unknown): value is GalleryBoardKind =>
  value === 'board' || value === 'date' || value === 'uncategorized';
