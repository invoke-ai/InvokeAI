/* eslint-disable react/react-compiler */
import type { GalleryBoard } from '@workbench/gallery/api';

import { Dialog, HStack, Icon, Input, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { ArchiveIcon, DownloadIcon, PencilIcon, Trash2Icon, type LucideIcon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useGalleryWidget } from './GalleryWidgetContext';

const MENU_CONTENT_PADDING_X = '0';

export interface GalleryBoardMenuTarget {
  board: GalleryBoard;
  x: number;
  y: number;
}

/**
 * Cursor-anchored actions menu for a single board row (download, rename,
 * archive, delete) so boards can be managed without selecting them first.
 * Opened from the board dropdown via right-click or the row's hover actions
 * button; the dropdown stays open underneath (see onActiveChange).
 */
export const GalleryBoardMenu = ({
  target,
  onActiveChange,
  onClose,
}: {
  target: GalleryBoardMenuTarget | null;
  /** Reports whether the menu or one of its dialogs is showing, so the host dropdown can stay open. */
  onActiveChange?: (isActive: boolean) => void;
  onClose: () => void;
}) => {
  const { t } = useTranslation();
  const { actions, gallery } = useGalleryWidget();
  const [renameTarget, setRenameTarget] = useState<GalleryBoard | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<GalleryBoard | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const targetRef = useRef(target);

  targetRef.current = target;

  const board = target?.board ?? null;
  // Project boards can be downloaded but never renamed, archived, or deleted.
  const isManagedBoard = board !== null && board.kind === 'board' && board.id !== gallery.projectBoardId;
  const isActive = target !== null || renameTarget !== null || deleteTarget !== null;

  useEffect(() => {
    onActiveChange?.(isActive);
  }, [isActive, onActiveChange]);

  const positioning = useMemo(
    () => ({
      getAnchorRect: () => {
        const currentTarget = targetRef.current;

        return currentTarget ? { height: 1, width: 1, x: currentTarget.x, y: currentTarget.y } : null;
      },
      placement: 'bottom-start' as const,
    }),
    []
  );
  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  const submitRename = useCallback(() => {
    const trimmedName = renameValue.trim();

    if (renameTarget && trimmedName && trimmedName !== renameTarget.name) {
      void actions.renameBoard(renameTarget.id, trimmedName);
    }

    setRenameTarget(null);
  }, [actions, renameTarget, renameValue]);

  const handleRenameDialogOpenChange = useCallback((event: { open: boolean }) => {
    if (!event.open) {
      setRenameTarget(null);
    }
  }, []);

  const handleRenameValueChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setRenameValue(event.currentTarget.value);
  }, []);

  const handleRenameKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        submitRename();
      }
    },
    [submitRename]
  );

  const handleCancelRename = useCallback(() => setRenameTarget(null), []);

  const handleDeleteDialogOpenChange = useCallback((event: { open: boolean }) => {
    if (!event.open) {
      setDeleteTarget(null);
    }
  }, []);

  const handleCancelDelete = useCallback(() => setDeleteTarget(null), []);

  const handleDeleteBoardOnly = useCallback(() => {
    if (deleteTarget) {
      void actions.deleteBoard(deleteTarget.id, false);
    }

    setDeleteTarget(null);
  }, [actions, deleteTarget]);

  const handleDeleteBoardAndImages = useCallback(() => {
    if (deleteTarget) {
      void actions.deleteBoard(deleteTarget.id, true);
    }

    setDeleteTarget(null);
  }, [actions, deleteTarget]);

  return (
    <>
      <Menu.Root
        key={board ? board.id : 'closed'}
        lazyMount
        open={target !== null}
        positioning={positioning}
        unmountOnExit
        onOpenChange={handleOpenChange}
      >
        <Portal>
          <Menu.Positioner>
            {board && (
              <Menu.Content minW="12rem" py="1" px={MENU_CONTENT_PADDING_X}>
                <BoardDownloadMenuItem boardId={board.id} />
                {isManagedBoard && (
                  <>
                    <BoardRenameMenuItem board={board} onRename={setRenameTarget} onRenameValue={setRenameValue} />
                    <BoardArchiveMenuItem archived={board.archived} boardId={board.id} />
                    <Menu.Separator borderColor="border.subtle" />
                    <BoardDeleteMenuItem board={board} onDelete={setDeleteTarget} />
                  </>
                )}
              </Menu.Content>
            )}
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <Dialog.Root
        initialFocusEl={undefined}
        open={renameTarget !== null}
        onOpenChange={handleRenameDialogOpenChange}
        size="sm"
      >
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content>
              <Dialog.Header>
                <Dialog.Title fontSize="sm">{t('widgets.gallery.renameBoard')}</Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Input
                  aria-label={t('widgets.gallery.boardName')}
                  autoFocus
                  size="sm"
                  value={renameValue}
                  onChange={handleRenameValueChange}
                  onKeyDown={handleRenameKeyDown}
                />
              </Dialog.Body>
              <Dialog.Footer gap="2">
                <Button size="xs" variant="outline" onClick={handleCancelRename}>
                  {t('common.cancel')}
                </Button>
                <Button disabled={renameValue.trim().length === 0} size="xs" onClick={submitRename}>
                  {t('common.rename')}
                </Button>
              </Dialog.Footer>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
      <Dialog.Root open={deleteTarget !== null} role="alertdialog" onOpenChange={handleDeleteDialogOpenChange}>
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content>
              <Dialog.Header>
                <Dialog.Title fontSize="sm">
                  {t('widgets.gallery.deleteBoardQuestion', { name: deleteTarget?.name ?? '' })}
                </Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Stack gap="2">
                  <Text color="fg.subtle" fontSize="xs">
                    {t('widgets.gallery.deleteBoardDescription')}
                  </Text>
                  <Text color="fg.subtle" fontSize="2xs">
                    {t('widgets.gallery.boardItemCounts', {
                      assets: deleteTarget?.assetCount ?? 0,
                      images: deleteTarget?.imageCount ?? 0,
                    })}
                  </Text>
                </Stack>
              </Dialog.Body>
              <Dialog.Footer gap="2">
                <Button size="xs" variant="outline" onClick={handleCancelDelete}>
                  {t('common.cancel')}
                </Button>
                <Button colorPalette="red" size="xs" variant="outline" onClick={handleDeleteBoardOnly}>
                  {t('widgets.gallery.deleteBoardOnly')}
                </Button>
                <Button colorPalette="red" size="xs" onClick={handleDeleteBoardAndImages}>
                  {t('widgets.gallery.deleteBoardAndImages')}
                </Button>
              </Dialog.Footer>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
    </>
  );
};

const BoardDownloadMenuItem = ({ boardId }: { boardId: string }) => {
  const { t } = useTranslation();
  const { actions } = useGalleryWidget();
  const handleClick = useCallback(() => void actions.downloadBoard(boardId), [actions, boardId]);

  return (
    <BoardMenuItem
      icon={DownloadIcon}
      label={t('widgets.gallery.downloadBoard')}
      value="download-board"
      onClick={handleClick}
    />
  );
};

const BoardRenameMenuItem = ({
  board,
  onRename,
  onRenameValue,
}: {
  board: GalleryBoard;
  onRename: React.Dispatch<React.SetStateAction<GalleryBoard | null>>;
  onRenameValue: React.Dispatch<React.SetStateAction<string>>;
}) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => {
    onRenameValue(board.name);
    onRename(board);
  }, [board, onRename, onRenameValue]);

  return (
    <BoardMenuItem
      icon={PencilIcon}
      label={t('widgets.gallery.renameBoard')}
      value="rename-board"
      onClick={handleClick}
    />
  );
};

const BoardArchiveMenuItem = ({ archived, boardId }: { archived: boolean; boardId: string }) => {
  const { t } = useTranslation();
  const { actions } = useGalleryWidget();
  const handleClick = useCallback(() => void actions.archiveBoard(boardId, !archived), [actions, archived, boardId]);

  return (
    <BoardMenuItem
      icon={ArchiveIcon}
      label={archived ? t('widgets.gallery.unarchiveBoard') : t('widgets.gallery.archiveBoard')}
      value="toggle-archived"
      onClick={handleClick}
    />
  );
};

const BoardDeleteMenuItem = ({
  board,
  onDelete,
}: {
  board: GalleryBoard;
  onDelete: React.Dispatch<React.SetStateAction<GalleryBoard | null>>;
}) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => onDelete(board), [board, onDelete]);

  return (
    <BoardMenuItem
      color="fg.error"
      icon={Trash2Icon}
      label={t('widgets.gallery.deleteBoard')}
      value="delete-board"
      onClick={handleClick}
    />
  );
};

const BoardMenuItem = ({
  color,
  icon,
  label,
  value,
  onClick,
}: {
  color?: string;
  icon: LucideIcon;
  label: string;
  value: string;
  onClick: () => void;
}) => (
  <Menu.Item color={color} value={value} onClick={onClick}>
    <HStack gap="2" minW="0" w="full">
      <Icon as={icon} boxSize="3.5" color={color ?? 'fg.subtle'} flexShrink={0} />
      <Text flex="1" fontSize="xs">
        {label}
      </Text>
    </HStack>
  </Menu.Item>
);
