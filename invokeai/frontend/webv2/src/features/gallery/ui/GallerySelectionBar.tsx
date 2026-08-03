import type { GalleryBoard } from '@features/gallery/core/types';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { parseGalleryItemKey, shouldStarSelection } from '@features/gallery/core/items';
import { IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Tooltip } from '@platform/ui/Tooltip';
import { DownloadIcon, FolderInputIcon, StarIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { BoardCover } from './GalleryBoardCover';
import { useMenuTriggerIds } from './galleryMenuIds';
import { useGalleryWidget } from './GalleryWidgetContext';

const MOVE_MENU_POSITIONING = { placement: 'top-end' } as const;

/**
 * Bulk actions for the current multi-selection. Until now these lived only in
 * the right-click menu, which made them invisible to anyone who did not think
 * to look for them.
 */
export const GallerySelectionBar = () => {
  const { t } = useTranslation();
  const { gallery, itemActions } = useGalleryWidget();
  const moveTriggerIds = useMenuTriggerIds();
  const selectedItemKeys = gallery.selectedItemKeys;
  const selectionCount = selectedItemKeys.length;

  const selectedItemRefs = useMemo(() => selectedItemKeys.map(parseGalleryItemKey), [selectedItemKeys]);

  // Star acts on the whole selection: only "unstar all" when every selected
  // item is already starred, matching the `.` hotkey.
  const shouldStar = useMemo(
    () => shouldStarSelection(gallery.items, selectedItemRefs),
    [gallery.items, selectedItemRefs]
  );

  const moveTargets = useMemo(
    () =>
      gallery.boards.filter(
        (board) => board.kind !== 'date' && !board.archived && board.id !== gallery.selectedBoardId
      ),
    [gallery.boards, gallery.selectedBoardId]
  );

  const handleToggleStarred = useCallback(
    () => void itemActions.setItemsStarred(selectedItemRefs, shouldStar),
    [itemActions, selectedItemRefs, shouldStar]
  );

  const handleDownload = useCallback(
    () => void itemActions.downloadItems(selectedItemRefs, gallery.items),
    [gallery.items, itemActions, selectedItemRefs]
  );

  const handleDelete = useCallback(
    () => void itemActions.deleteItems(selectedItemRefs),
    [itemActions, selectedItemRefs]
  );

  const handleMoveToBoard = useCallback(
    (boardId: string) => void itemActions.moveItemsToBoard(selectedItemRefs, boardId),
    [itemActions, selectedItemRefs]
  );

  if (selectionCount === 0) {
    return null;
  }

  const starLabel = shouldStar ? t('widgets.gallery.starSelection') : t('widgets.gallery.unstarSelection');

  return (
    <HStack
      bg="bg.subtle"
      borderColor="border.subtle"
      borderTopWidth="1px"
      gap="1"
      flexShrink={0}
      minH="8"
      px="2"
      py="1"
      role="toolbar"
      aria-label={t('widgets.gallery.selectionActions')}
      w="full"
    >
      <Text color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums" me="auto">
        {t('widgets.gallery.selectionCount', { count: selectionCount })}
      </Text>
      <Tooltip content={starLabel}>
        <IconButton aria-label={starLabel} size="2xs" variant="ghost" onClick={handleToggleStarred}>
          <Icon as={StarIcon} boxSize="3.5" fill={shouldStar ? 'none' : 'currentColor'} />
        </IconButton>
      </Tooltip>
      <Menu.Root ids={moveTriggerIds} positioning={MOVE_MENU_POSITIONING}>
        <Tooltip content={t('widgets.gallery.moveSelectionToBoard')} ids={moveTriggerIds}>
          <Menu.Trigger asChild>
            <IconButton
              aria-label={t('widgets.gallery.moveSelectionToBoard')}
              disabled={moveTargets.length === 0}
              size="2xs"
              variant="ghost"
            >
              <Icon as={FolderInputIcon} boxSize="3.5" />
            </IconButton>
          </Menu.Trigger>
        </Tooltip>
        <Portal>
          <Menu.Positioner>
            <MenuContent maxH="18rem" minW="12rem" overflowY="auto">
              {moveTargets.map((board) => (
                <MoveTargetItem key={board.id} board={board} onSelect={handleMoveToBoard} />
              ))}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <Tooltip content={t('widgets.gallery.downloadSelection')}>
        <IconButton
          aria-label={t('widgets.gallery.downloadSelection')}
          size="2xs"
          variant="ghost"
          onClick={handleDownload}
        >
          <Icon as={DownloadIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
      <Tooltip content={t('widgets.gallery.deleteSelection')}>
        <IconButton
          aria-label={t('widgets.gallery.deleteSelection')}
          colorPalette="red"
          size="2xs"
          variant="ghost"
          onClick={handleDelete}
        >
          <Icon as={Trash2Icon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
    </HStack>
  );
};

const MoveTargetItem = ({ board, onSelect }: { board: GalleryBoard; onSelect: (boardId: string) => void }) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => onSelect(board.id), [board.id, onSelect]);

  return (
    <Menu.Item value={board.id} onClick={handleClick}>
      <BoardCover board={board} />
      <Menu.ItemText>{getGalleryBoardLabel(board, t)}</Menu.ItemText>
    </Menu.Item>
  );
};
