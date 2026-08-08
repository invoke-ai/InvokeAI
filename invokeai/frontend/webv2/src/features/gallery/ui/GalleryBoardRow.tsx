import type { GalleryItemKey } from '@features/gallery/core/items';
import type { GalleryBoard } from '@features/gallery/core/types';

import { Badge, Box, Text } from '@chakra-ui/react';
import { useDndContext, useDroppable } from '@dnd-kit/core';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { toGalleryItemKey } from '@features/gallery/core/items';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { MoreVerticalIcon } from 'lucide-react';
import { useCallback, useMemo, type MouseEvent, type PointerEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { BoardCover } from './GalleryBoardCover';
import { GalleryBoardRowShell } from './GalleryBoardRowShell';
import { getGalleryBoardDropData, getGalleryBoardDropId, isGalleryItemDragData } from './galleryDnd';
import { getBoardCounts } from './galleryStateView';

export const GalleryBoardRow = ({
  board,
  isMenuOpen,
  isSelected,
  loadedItemBoardIds,
  onOpenMenu,
  onSelectBoard,
}: {
  board: GalleryBoard;
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
    board.kind === 'board' &&
    isGalleryItemDragData(dragData) &&
    dragData.items.some((ref) => loadedItemBoardIds.get(toGalleryItemKey(ref)) !== board.id);

  const { isOver, setNodeRef } = useDroppable({
    data: getGalleryBoardDropData(board.id, board.kind),
    disabled: !canDropItems,
    id: getGalleryBoardDropId(board.id),
  });

  const counts = getBoardCounts(board);
  const mediaCount = counts.imageCount + counts.videoCount;
  const countsBreakdown = t('widgets.gallery.boardCountsBreakdown', {
    assets: counts.assetCount,
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
        <Text color={isSelected ? 'inherit' : 'fg.muted'} fontSize="2xs" lineHeight="shorter" minW="0" truncate>
          {board.ownerName}
        </Text>
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
          size="2xs"
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
      // An outline rather than a border, so the drop affordance never reflows the list.
      bg={isOver ? 'accent.muted' : undefined}
      outline={canDropItems ? '1px dashed' : undefined}
      outlineColor={canDropItems ? 'accent.solid' : undefined}
      rounded="sm"
      w="full"
    >
      <GalleryBoardRowShell
        ref={setNodeRef}
        actions={actions}
        cover={cover}
        isSelected={isSelected}
        label={boardLabel}
        subtitle={subtitle}
        onContextMenu={onOpenMenu ? handleContextMenu : undefined}
        onSelect={handleSelect}
      >
        {board.projectId !== null ? (
          <Badge colorPalette={isSelected ? undefined : 'accent'} flexShrink={0} size="xs" variant="subtle">
            {t('common.project')}
          </Badge>
        ) : null}
        <Tooltip content={countsBreakdown}>
          <Badge
            aria-label={countsBreakdown}
            flexShrink={0}
            fontVariantNumeric="tabular-nums"
            size="xs"
            variant="subtle"
          >
            {mediaCount} | {counts.assetCount}
          </Badge>
        </Tooltip>
      </GalleryBoardRowShell>
    </Box>
  );
};
