import type { GalleryItemKey } from '@features/gallery/core/items';
import type { GalleryBoard } from '@features/gallery/core/types';

import { Badge, Box } from '@chakra-ui/react';
import { useDndContext, useDroppable } from '@dnd-kit/core';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { toGalleryItemKey } from '@features/gallery/core/items';
import { IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Tooltip } from '@platform/ui/Tooltip';
import { MoreVerticalIcon } from 'lucide-react';
import { useCallback, useMemo, type MouseEvent, type PointerEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { BoardCover } from './GalleryBoardCover';
import { GalleryBoardRowShell } from './GalleryBoardRowShell';
import {
  acceptsGalleryItemMoves,
  getGalleryBoardDropData,
  getGalleryBoardDropId,
  isGalleryItemDragData,
} from './galleryDnd';
import { getBoardCounts } from './galleryStateView';

export const GalleryBoardRow = ({
  board,
  isAutoAddTarget = false,
  isMenuOpen,
  isSelected,
  loadedItemBoardIds,
  onOpenMenu,
  onSelectBoard,
}: {
  board: GalleryBoard;
  /** Results without a board of their own land here (the gallery is not following its selection). */
  isAutoAddTarget?: boolean;
  /** Its own menu is showing, so the trigger must not fade out from under it. */
  isMenuOpen?: boolean;
  isSelected: boolean;
  loadedItemBoardIds: ReadonlyMap<GalleryItemKey, string>;
  /** Omitted for date rows, which have no board actions. */
  onOpenMenu?: (board: GalleryBoard, x: number, y: number) => void;
  onSelectBoard: (boardId: string) => void;
}) => {
  const { t } = useTranslation();
  const { active } = useDndContext();
  const dragData = active?.data.current;
  const boardLabel = getGalleryBoardLabel(board, t);

  const canDropItems =
    acceptsGalleryItemMoves(board.kind) &&
    isGalleryItemDragData(dragData) &&
    dragData.items.some((ref) => loadedItemBoardIds.get(toGalleryItemKey(ref)) !== board.id);

  const { isOver, setNodeRef } = useDroppable({
    data: getGalleryBoardDropData(board.id, board.kind),
    disabled: !canDropItems,
    id: getGalleryBoardDropId(board.id),
  });

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

      event.preventDefault();
      event.stopPropagation();
      onOpenMenu(board, event.clientX, event.clientY);
    },
    [board, onOpenMenu]
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
          aria-label={t('widgets.gallery.boardActionsForBoard', { name: boardLabel })}
          className="board-row-actions"
          flexShrink={0}
          // Its menu anchors to this button, so it must not fade out beneath it.
          opacity={isMenuOpen ? 1 : 0}
          // 24px: the target-size floor, and short enough for the 28px row.
          size="sm"
          transition="opacity var(--wb-motion-duration-medium) ease"
          variant="ghost"
          onClick={handleActionsClick}
          onPointerDown={stopPropagation}
          onPointerUp={stopPropagation}
        >
          <MoreVerticalIcon />
        </IconButton>
      ) : null,
    [boardLabel, handleActionsClick, isMenuOpen, onOpenMenu, stopPropagation, t]
  );

  return (
    <Box
      // Use outlines to avoid reflow during drag highlighting; the active row gets an inset ring.
      bg={isOver ? 'accent.muted' : undefined}
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
        cover={cover}
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
        {board.projectId !== null ? (
          <Badge colorPalette={isSelected ? undefined : 'accent'} flexShrink={0} variant="subtle">
            {t('common.project')}
          </Badge>
        ) : null}
        <Tooltip content={countsBreakdown}>
          <Badge aria-label={countsBreakdown} flexShrink={0} fontVariantNumeric="tabular-nums" variant="subtle">
            {mediaCount} | {counts.assetCount}
          </Badge>
        </Tooltip>
      </GalleryBoardRowShell>
    </Box>
  );
};
