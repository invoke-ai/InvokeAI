import type { GalleryBoard, GalleryView } from '@features/gallery/core/types';

import { Badge, Box, Text } from '@chakra-ui/react';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { BoardCover } from '@features/gallery/ui/GalleryBoardCover';
import { getGalleryCountForView } from '@features/gallery/ui/galleryBoardLabels';
import { GalleryBoardRowShell } from '@features/gallery/ui/GalleryBoardRowShell';
import { Scrollable } from '@platform/ui/Scrollable';
import { dropdownGroupLabel } from '@theme/recipes';
import { memo, useCallback, useMemo, type KeyboardEvent } from 'react';
import { useTranslation } from 'react-i18next';

export interface GalleryPickerBoardGroup {
  boards: GalleryBoard[];
  id: string;
  label: string;
}

const GalleryPickerBoardRow = memo(function GalleryPickerBoardRow({
  board,
  galleryView,
  isSelected,
  onSelect,
}: {
  board: GalleryBoard;
  galleryView: GalleryView;
  isSelected: boolean;
  onSelect: (boardId: string) => void;
}) {
  const { t } = useTranslation();
  const handleSelect = useCallback(() => onSelect(board.id), [board.id, onSelect]);
  const cover = useMemo(() => <BoardCover board={board} />, [board]);

  return (
    <GalleryBoardRowShell
      cover={cover}
      isSelected={isSelected}
      label={getGalleryBoardLabel(board, t)}
      onSelect={handleSelect}
    >
      <Badge flexShrink={0} fontVariantNumeric="tabular-nums" size="xs" variant="subtle">
        {getGalleryCountForView(board, galleryView)}
      </Badge>
    </GalleryBoardRowShell>
  );
});

/** Arrow navigation leaving the top board row returns focus to search. */
export const GalleryPickerBoards = ({
  emptyMessage,
  galleryView,
  groups,
  id,
  label,
  selectedBoardId,
  onExitTop,
  onSelect,
}: {
  emptyMessage: string;
  galleryView: GalleryView;
  groups: GalleryPickerBoardGroup[];
  id: string;
  label: string;
  selectedBoardId: string;
  onExitTop: () => void;
  onSelect: (boardId: string) => void;
}) => {
  const viewportProps = useMemo(() => ({ id }), [id]);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') {
        return;
      }

      const rows = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('button')];
      const index = rows.indexOf(document.activeElement as HTMLButtonElement);

      if (index === -1) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      if (event.key === 'ArrowUp' && index === 0) {
        onExitTop();
        return;
      }

      rows[Math.min(rows.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1))]?.focus();
    },
    [onExitTop]
  );

  return (
    <Scrollable flex="1" label={label} minH="0" viewportProps={viewportProps}>
      <Box px="1" py="1" onKeyDown={handleKeyDown}>
        {groups.length === 0 ? (
          <Text color="fg.muted" fontSize="2xs" px="2" py="3" textAlign="center">
            {emptyMessage}
          </Text>
        ) : (
          groups.map((group) => (
            <Box key={group.id}>
              <Text css={dropdownGroupLabel} px="2" py="1">
                {group.label}
              </Text>
              {group.boards.map((board) => (
                <GalleryPickerBoardRow
                  key={board.id}
                  board={board}
                  galleryView={galleryView}
                  isSelected={board.id === selectedBoardId}
                  onSelect={onSelect}
                />
              ))}
            </Box>
          ))
        )}
      </Box>
    </Scrollable>
  );
};
