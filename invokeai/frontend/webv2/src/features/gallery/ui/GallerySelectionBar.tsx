import type { GalleryBoard } from '@features/gallery/core/types';

import { HStack, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import {
  getGalleryBoardDestinationGroups,
  getGalleryBoardLabel,
  getGalleryMoveTargets,
} from '@features/gallery/core/boardLabels';
import { parseGalleryItemKey, toGalleryItemKey } from '@features/gallery/core/items';
import { IconButton } from '@platform/ui/Button';
import { ScrollableMenuContent } from '@platform/ui/ScrollableMenuContent';
import { Tooltip } from '@platform/ui/Tooltip';
import { DownloadIcon, FolderInputIcon, FolderMinusIcon, StarIcon, Trash2Icon, XIcon } from 'lucide-react';
import { useCallback, useMemo, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { BoardCover, BoardCoverIcon } from './GalleryBoardCover';
import { focusVisibleOperable } from './GalleryLoadError';
import { useMenuTriggerIds } from './galleryMenuIds';
import { GALLERY_TAB_STOP_SELECTOR } from './GalleryThumbnail';
import { useGallerySelectionStarred, useGalleryWidget } from './GalleryWidgetContext';

const MOVE_MENU_POSITIONING = { placement: 'top-end' } as const;
const GALLERY_TILE_SELECTOR = 'button[data-gallery-item-key]';

export const GallerySelectionBar = () => {
  const { t } = useTranslation();
  const { actions, gallery, itemActions, loadedItems, projectId, projectName, projectNames } = useGalleryWidget();
  const moveTriggerIds = useMenuTriggerIds();
  const selectedItemKeys = gallery.selectedItemKeys;
  const selectionCount = selectedItemKeys.length;

  const selectedItemRefs = useMemo(() => selectedItemKeys.map(parseGalleryItemKey), [selectedItemKeys]);
  // The selection is this listing's; a member no page holds is still off screen, which deletion must not skip past.
  const loadedSelection = useMemo(() => {
    const selected = new Set(selectedItemKeys);

    return loadedItems.filter((item) => selected.has(toGalleryItemKey(item)));
  }, [loadedItems, selectedItemKeys]);
  const hasUnloadedMembers = loadedSelection.length < selectionCount;

  // Star acts on the whole selection: only "unstar all" when every selected
  // item is already starred, matching the `.` hotkey.
  const shouldStar = useGallerySelectionStarred(selectedItemRefs, loadedItems);

  const moveTargets = useMemo(
    () =>
      getGalleryMoveTargets(gallery.boards, {
        itemBoardIds: loadedSelection.map((item) => item.boardId),
        listingBoardId: gallery.listingBoardId,
        unloadedCount: selectionCount - loadedSelection.length,
      }),
    [gallery.boards, gallery.listingBoardId, loadedSelection, selectionCount]
  );
  const moveGroups = useMemo(
    () => getGalleryBoardDestinationGroups({ boards: moveTargets.boards, projectId, projectName, projectNames, t }),
    [moveTargets.boards, projectId, projectName, projectNames, t]
  );

  const handleToggleStarred = useCallback(
    () => void itemActions.setItemsStarred(selectedItemRefs, shouldStar),
    [itemActions, selectedItemRefs, shouldStar]
  );

  const handleDownload = useCallback(
    () => void itemActions.downloadItems(selectedItemRefs, loadedItems),
    [itemActions, loadedItems, selectedItemRefs]
  );

  const handleDelete = useCallback(
    () => void itemActions.deleteItems(selectedItemRefs, { targetsOffScreen: hasUnloadedMembers }),
    [hasUnloadedMembers, itemActions, selectedItemRefs]
  );

  const handleMoveToBoard = useCallback(
    (boardId: string) => void itemActions.moveItemsToBoard(selectedItemRefs, boardId),
    [itemActions, selectedItemRefs]
  );
  const handleRemoveFromBoard = useCallback(() => handleMoveToBoard('none'), [handleMoveToBoard]);
  const primaryItemKey = gallery.primarySelectedItemKey;
  // The bar leaves with the selection, taking the focused Clear button along: focus goes to the grid first, to the
  // tile that was primary where it is listed, else the grid's tab stop, else its first tile in view. Not the click
  // handler itself: the command takes an optional project, which the click event would fill.
  const handleClear = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      let scope = event.currentTarget.parentElement;

      while (scope && !scope.querySelector(GALLERY_TILE_SELECTOR)) {
        scope = scope.parentElement;
      }

      const tile =
        (primaryItemKey
          ? scope?.querySelector<HTMLElement>(`button[data-gallery-item-key="${CSS.escape(primaryItemKey)}"]`)
          : null) ?? scope?.querySelector<HTMLElement>(GALLERY_TAB_STOP_SELECTOR);

      actions.clearSelection();

      if (tile) {
        tile.focus({ preventScroll: true });
      } else {
        focusVisibleOperable(scope ?? null, { selector: GALLERY_TILE_SELECTOR });
      }
    },
    [actions, primaryItemKey]
  );

  if (selectionCount === 0) {
    return null;
  }

  const starLabel = shouldStar ? t('widgets.gallery.starSelection') : t('widgets.gallery.unstarSelection');
  const canMove = moveTargets.canRemoveFromBoard || moveGroups.length > 0;

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
      <Tooltip content={t('widgets.gallery.commands.clearSelection')}>
        <IconButton
          aria-label={t('widgets.gallery.commands.clearSelection')}
          size="sm"
          variant="ghost"
          onClick={handleClear}
        >
          <Icon as={XIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
      <Text color="fg.muted" fontSize="md" fontVariantNumeric="tabular-nums" me="auto">
        {t('widgets.gallery.selectionCount', { count: selectionCount })}
      </Text>
      <Tooltip content={starLabel}>
        <IconButton aria-label={starLabel} size="sm" variant="ghost" onClick={handleToggleStarred}>
          <Icon as={StarIcon} boxSize="3.5" fill={shouldStar ? 'none' : 'currentColor'} />
        </IconButton>
      </Tooltip>
      <Menu.Root ids={moveTriggerIds} positioning={MOVE_MENU_POSITIONING}>
        <Tooltip content={t('widgets.gallery.moveSelectionToBoard')} ids={moveTriggerIds}>
          <Menu.Trigger asChild>
            <IconButton
              aria-label={t('widgets.gallery.moveSelectionToBoard')}
              disabled={!canMove}
              size="sm"
              variant="ghost"
            >
              <Icon as={FolderInputIcon} boxSize="3.5" />
            </IconButton>
          </Menu.Trigger>
        </Tooltip>
        <Portal>
          <Menu.Positioner>
            <ScrollableMenuContent minW="12rem">
              {moveTargets.canRemoveFromBoard ? (
                <>
                  <Menu.Item value="remove-from-board" onClick={handleRemoveFromBoard}>
                    <BoardCoverIcon icon={FolderMinusIcon} />
                    <Menu.ItemText>{t('widgets.gallery.removeFromBoard')}</Menu.ItemText>
                  </Menu.Item>
                  {moveGroups.length > 0 ? <Menu.Separator /> : null}
                </>
              ) : null}
              {moveGroups.map((group) => (
                <Menu.ItemGroup key={group.id}>
                  <Menu.ItemGroupLabel>{group.label}</Menu.ItemGroupLabel>
                  {group.boards.map((board) => (
                    <MoveTargetItem key={board.id} board={board} onSelect={handleMoveToBoard} />
                  ))}
                </Menu.ItemGroup>
              ))}
            </ScrollableMenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <Tooltip content={t('widgets.gallery.downloadSelection')}>
        <IconButton
          aria-label={t('widgets.gallery.downloadSelection')}
          size="sm"
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
          size="sm"
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
