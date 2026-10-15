import type { GalleryItemKey } from '@features/gallery/core/items';
import type { GalleryBoard } from '@features/gallery/core/types';

import { Badge, Box, HStack } from '@chakra-ui/react';
import { useDndContext, useDraggable, useDroppable } from '@dnd-kit/core';
import { CSS } from '@dnd-kit/utilities';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { toGalleryItemKey } from '@features/gallery/core/items';
import { IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Tooltip } from '@platform/ui/Tooltip';
import { MoreVerticalIcon } from 'lucide-react';
import { useCallback, useMemo, type MouseEvent, type PointerEvent } from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';

import { BoardCover } from './GalleryBoardCover';
import { GalleryBoardRowShell } from './GalleryBoardRowShell';
import {
  acceptsGalleryItemMoves,
  getGalleryBoardDragData,
  getGalleryBoardDragId,
  getGalleryBoardDropData,
  getGalleryBoardDropId,
  isGalleryItemDragData,
} from './galleryDnd';
import { getBoardCounts } from './galleryStateView';

export const GalleryBoardRow = ({
  accessibleName,
  board,
  dragScope,
  isAutoAddTarget = false,
  isMenuOpen,
  isSelected,
  loadedItemBoardIds,
  onFocusLost,
  onOpenMenu,
  onSelectBoard,
}: {
  /** What assistive tech calls the row when its visible label alone is ambiguous, such as another project's Inbox. */
  accessibleName?: string;
  board: GalleryBoard;
  /**
   * The gallery the row is in, given only where the row can be dragged to another tier: never where the board is
   * managed through its project, and never outside the workbench's drag context.
   */
  dragScope?: string;
  /** Results without a board of their own land here (the gallery is not following its selection). */
  isAutoAddTarget?: boolean;
  /** Its own menu is showing, so the trigger must not fade out from under it. */
  isMenuOpen?: boolean;
  isSelected: boolean;
  loadedItemBoardIds: ReadonlyMap<GalleryItemKey, string>;
  /** The row left the list holding focus — moved to another tier, archived or deleted — so focus needs a new home. */
  onFocusLost?: (boardId: string) => void;
  /** Omitted for date rows, which have no board actions. */
  onOpenMenu?: (board: GalleryBoard, x: number, y: number) => void;
  onSelectBoard: (boardId: string) => void;
}) => {
  const { t } = useTranslation();
  const { active } = useDndContext();
  const dragData = active?.data.current;
  const boardLabel = getGalleryBoardLabel(board, t);
  const spokenLabel = accessibleName ?? boardLabel;

  const canDropItems =
    acceptsGalleryItemMoves(board.kind) &&
    isGalleryItemDragData(dragData) &&
    dragData.items.some((ref) => loadedItemBoardIds.get(toGalleryItemKey(ref)) !== board.id);

  const { isOver, setNodeRef: setDropNodeRef } = useDroppable({
    data: getGalleryBoardDropData(board.id, board.kind),
    disabled: !canDropItems,
    id: getGalleryBoardDropId(board.id),
  });
  const {
    isDragging,
    listeners,
    setNodeRef: setDragNodeRef,
    transform,
  } = useDraggable({
    data: getGalleryBoardDragData(board, dragScope ?? ''),
    disabled: dragScope === undefined,
    id: getGalleryBoardDragId(board.id, dragScope ?? ''),
  });
  const setNodeRef = useCallback(
    (node: HTMLDivElement | null) => {
      setDropNodeRef(node);
      setDragNodeRef(node);
    },
    [setDragNodeRef, setDropNodeRef]
  );
  // Pointer drags only: Enter and Space select the row, and the board menu's Move is the keyboard's way to move it.
  const pointerDragListeners = useMemo(() => {
    if (!listeners) {
      return undefined;
    }

    const { onKeyDown: _keyboardDrag, ...pointer } = listeners;

    return pointer;
  }, [listeners]);
  // Drawn above everything, from where the row was when the drag began: the list's scroll area would clip the row.
  const dragOrigin = isDragging ? (active?.rect.current.initial ?? null) : null;

  const counts = getBoardCounts(board);
  const mediaCount = Math.max(0, counts.imageCount + counts.videoCount - counts.assetVideoCount);
  const countsBreakdown = t('widgets.gallery.boardCountsBreakdown', {
    assets: counts.assetCount + counts.assetVideoCount,
    media: mediaCount,
  });

  const handleSelect = useCallback(() => onSelectBoard(board.id), [board.id, onSelectBoard]);

  const handleContextMenu = useCallback(
    (event: MouseEvent) => {
      if (!onOpenMenu) {
        return;
      }

      // Not stopped: the touch drag sensor listens for this on the window, to drop a hold the menu interrupts.
      event.preventDefault();

      // A touch long-press that started a drag means to move the board, not to open its menu under the finger.
      if (!isDragging) {
        onOpenMenu(board, event.clientX, event.clientY);
      }
    },
    [board, isDragging, onOpenMenu]
  );

  const handleActionsClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      if (!onOpenMenu) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      const rect = event.currentTarget.getBoundingClientRect();

      onOpenMenu(board, rect.left, rect.bottom);
    },
    [board, onOpenMenu]
  );

  const stopPropagation = useCallback((event: MouseEvent | PointerEvent) => event.stopPropagation(), []);

  const cover = useMemo(() => <BoardCover board={board} />, [board]);
  const subtitle = useMemo(
    () =>
      board.ownerName ? (
        <MiddleTruncate
          color={isSelected ? 'inherit' : 'fg.muted'}
          fontSize="xs"
          lineHeight="shorter"
          minW="0"
          text={board.ownerName}
        />
      ) : null,
    [board.ownerName, isSelected]
  );
  const actions = useMemo(
    () =>
      onOpenMenu ? (
        <IconButton
          aria-label={t('widgets.gallery.boardActionsForBoard', { name: spokenLabel })}
          className="board-row-actions"
          flexShrink={0}
          // Its menu anchors to this button, so it must not fade out beneath it.
          opacity={isMenuOpen ? 1 : 0}
          // 24px: the target-size floor, and short enough for the 28px row.
          size="sm"
          transition="opacity var(--wb-motion-duration-medium) ease"
          variant="ghost"
          onClick={handleActionsClick}
          // Pressing the menu button must not start dragging the row it sits on.
          onMouseDown={stopPropagation}
          onPointerDown={stopPropagation}
          onPointerUp={stopPropagation}
        >
          <MoreVerticalIcon />
        </IconButton>
      ) : null,
    [handleActionsClick, isMenuOpen, onOpenMenu, spokenLabel, stopPropagation, t]
  );

  // The container, which outlives neither a move to another tier (the row remounts there) nor an archive or delete.
  const setRowRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (!node || !onFocusLost) {
        return;
      }

      return () => {
        if (node.contains(document.activeElement)) {
          // After the commit, when the row's replacement (if any) is in the document.
          queueMicrotask(() => onFocusLost(board.id));
        }
      };
    },
    [board.id, onFocusLost]
  );

  const dragPreviewStyle = useMemo(
    () =>
      dragOrigin
        ? ({
            height: `${String(dragOrigin.height)}px`,
            left: `${String(dragOrigin.left)}px`,
            top: `${String(dragOrigin.top)}px`,
            // Translate only, as for a dragged item: the full transform would scale it to what it hovers.
            transform: CSS.Translate.toString(transform),
            width: `${String(dragOrigin.width)}px`,
          } as const)
        : null,
    [dragOrigin, transform]
  );

  return (
    <Box
      ref={setRowRef}
      data-board-row={board.id}
      // Use outlines to avoid reflow during drag highlighting; the active row gets an inset ring.
      bg={isOver ? 'accent.muted' : undefined}
      opacity={isDragging ? 0.4 : undefined}
      outline={isOver ? '2px solid' : canDropItems ? '1px dashed' : undefined}
      outlineColor={canDropItems ? 'accent.solid' : undefined}
      // Inset keeps the ring inside the row; the selected row's opaque accent
      // fill would cover it, so its ring stays outside where it reads.
      outlineOffset={isOver && !isSelected ? '-2px' : undefined}
      rounded="sm"
      transition="background var(--wb-motion-duration-fast) ease"
      w="full"
    >
      <GalleryBoardRowShell
        ref={setNodeRef}
        actions={actions}
        ariaLabel={accessibleName}
        cover={cover}
        dragListeners={pointerDragListeners}
        isDropTarget={canDropItems}
        isSelected={isSelected}
        label={boardLabel}
        subtitle={subtitle}
        onContextMenu={onOpenMenu ? handleContextMenu : undefined}
        onSelect={handleSelect}
      >
        {isAutoAddTarget ? (
          <Tooltip content={t('widgets.gallery.autoAddBadgeTooltip')}>
            <Badge colorPalette={isSelected ? undefined : 'accent'} flexShrink={0} variant="subtle">
              {t('widgets.gallery.autoAddBadge')}
            </Badge>
          </Tooltip>
        ) : null}
        <Tooltip content={countsBreakdown}>
          <Badge aria-label={countsBreakdown} flexShrink={0} fontVariantNumeric="tabular-nums" variant="subtle">
            {mediaCount} | {counts.assetCount}
          </Badge>
        </Tooltip>
      </GalleryBoardRowShell>
      {dragPreviewStyle
        ? createPortal(
            <HStack
              aria-hidden="true"
              bg="bg.panel"
              boxShadow="lg"
              gap="2"
              pointerEvents="none"
              position="fixed"
              px="1"
              rounded="sm"
              style={dragPreviewStyle}
              zIndex="1500"
            >
              {cover}
              <MiddleTruncate fontWeight="500" minW="0" text={boardLabel} />
            </HStack>,
            document.body
          )
        : null}
    </Box>
  );
};
