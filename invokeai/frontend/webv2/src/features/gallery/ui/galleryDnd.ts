import type { GalleryItem, GalleryItemKey, GalleryItemRef } from '@features/gallery/core/items';
import type { GalleryBoard, GalleryBoardKind } from '@features/gallery/core/types';

import { useDndContext, useDroppable, type UseDroppableArguments } from '@dnd-kit/core';
import { toGalleryItemKey } from '@features/gallery/core/items';
import { createContext } from 'react';

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

/**
 * True inside the drag context that gallery items are dragged in (the workbench shell). dnd-kit cannot say whether a
 * DndContext is mounted, and surfaces outside it (widget dialog hosts, the Launchpad) still render gallery drop targets.
 */
export const GalleryDragScope = createContext(false);

export const getGalleryBoardDropId = (boardId: string): string => `gallery-board:${boardId}`;

/**
 * A board row on its way to another project or the Library. Each gallery resolves only its own boards' drags onto its
 * own tiers: the shell's one drag context reports every drop to every mounted gallery.
 */
export interface GalleryBoardDragData {
  boardId: string;
  kind: 'gallery-board-drag';
  /** Where it is now; null is the Library. */
  projectId: string | null;
  /** The gallery it was picked up in. */
  scope: string;
}

/** A project's boards, or the Library's (null): dropping a board anywhere on them moves it there. */
export interface GalleryBoardTierDropData {
  kind: 'gallery-board-tier';
  label: string;
  projectId: string | null;
  scope: string;
}

export interface GalleryBoardMove {
  boardId: string;
  label: string;
  projectId: string | null;
}

export const getGalleryBoardDragId = (boardId: string, scope: string): string =>
  `gallery-board-drag#${scope}:${boardId}`;

export const getGalleryBoardDragData = (
  board: Pick<GalleryBoard, 'id' | 'projectId'>,
  scope: string
): GalleryBoardDragData => ({ boardId: board.id, kind: 'gallery-board-drag', projectId: board.projectId, scope });

export const getGalleryBoardTierDropId = (projectId: string | null, scope: string): string =>
  `gallery-board-tier#${scope}:${projectId ?? 'library'}`;

export const getGalleryBoardTierDropData = (
  projectId: string | null,
  label: string,
  scope: string
): GalleryBoardTierDropData => ({ kind: 'gallery-board-tier', label, projectId, scope });

export const isGalleryBoardDragData = (value: unknown): value is GalleryBoardDragData =>
  isRecord(value) &&
  value.kind === 'gallery-board-drag' &&
  typeof value.boardId === 'string' &&
  typeof value.scope === 'string' &&
  (value.projectId === null || typeof value.projectId === 'string');

const isGalleryBoardTierDropData = (value: unknown): value is GalleryBoardTierDropData =>
  isRecord(value) &&
  value.kind === 'gallery-board-tier' &&
  typeof value.label === 'string' &&
  typeof value.scope === 'string' &&
  (value.projectId === null || typeof value.projectId === 'string');

/** Whether a dragged board would move by landing on the tier: it came from this gallery and is not there already. */
export const acceptsGalleryBoardMove = (activeData: unknown, projectId: string | null, scope: string): boolean =>
  isGalleryBoardDragData(activeData) && activeData.scope === scope && activeData.projectId !== projectId;

/** The move a board drop in this gallery asks for, or null when it was no board of its own or stayed in its tier. */
export const resolveGalleryBoardMove = (
  activeData: unknown,
  overData: unknown,
  scope: string
): GalleryBoardMove | null =>
  isGalleryBoardTierDropData(overData) &&
  overData.scope === scope &&
  acceptsGalleryBoardMove(activeData, overData.projectId, scope) &&
  isGalleryBoardDragData(activeData)
    ? { boardId: activeData.boardId, label: overData.label, projectId: overData.projectId }
    : null;

/** Real boards and Uncategorized hold items; date boards are views and accept no moves. */
export const acceptsGalleryItemMoves = (boardKind: GalleryBoardKind): boolean =>
  boardKind === 'board' || boardKind === 'uncategorized';

/** Droppable id for the search field: dropping a gallery image there searches by similarity. */
export const GALLERY_SEMANTIC_SEARCH_DROP_ID = 'gallery-semantic-search-drop';

/** Only image refs support similarity queries; video-only drags yield no query. */
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
 * accepts controls affordances; shields keeps wider ignored payloads as dead drops, preventing z-blind collision
 * handling from dropping onto hidden targets underneath.
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
  if (
    !isGalleryItemDragData(activeData) ||
    !isGalleryBoardDropData(overData) ||
    !acceptsGalleryItemMoves(overData.boardKind)
  ) {
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
