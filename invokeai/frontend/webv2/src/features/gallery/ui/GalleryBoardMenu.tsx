import type { GalleryBoard } from '@features/gallery/core/types';

import { HStack, Icon, Input, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { GALLERY_AUTO_ADD_FOLLOW } from '@features/gallery/core/settings';
import { Button } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { MenuContent } from '@platform/ui/Menu';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import {
  ArchiveIcon,
  ChevronRightIcon,
  DownloadIcon,
  FileDownIcon,
  FolderInputIcon,
  PencilIcon,
  SquareArrowDownIcon,
  Trash2Icon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { canMoveGalleryBoard, isManagedGalleryBoard } from './galleryBoardGroups';
import { useGalleryWidget } from './GalleryWidgetContext';

export interface GalleryBoardMenuTarget {
  board: GalleryBoard;
  x: number;
  y: number;
}

/** Keep the board dropdown open beneath its cursor-anchored actions menu; actions need not select the board. */
export const GalleryBoardMenu = ({
  target,
  onBoardRelocated,
  onClose,
}: {
  target: GalleryBoardMenuTarget | null;
  /** A move or an archive was started from the menu: the board's row is leaving its place in the list. */
  onBoardRelocated?: (boardId: string) => void;
  onClose: () => void;
}) => {
  const { t } = useTranslation();
  const { actions, currentUserId, gallery } = useGalleryWidget();
  const [renameTarget, setRenameTarget] = useState<GalleryBoard | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<GalleryBoard | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const targetRef = useRef(target);

  // eslint-disable-next-line react/refs
  targetRef.current = target;

  const board = target?.board ?? null;
  const isManagedBoard = board !== null && isManagedGalleryBoard(board, gallery.projectBoardId);
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
              <MenuContent minW="12rem">
                {board.isInbox && board.projectId !== null && (
                  <>
                    <BoardExportProjectMenuItem board={board} />
                    <Menu.Separator />
                  </>
                )}
                {board.id === gallery.settings.autoAddBoardId ? (
                  <BoardAutoAddMenuItem boardId={GALLERY_AUTO_ADD_FOLLOW} label={t('widgets.gallery.stopAutoAdd')} />
                ) : board.kind !== 'date' && !board.archived ? (
                  <BoardAutoAddMenuItem boardId={board.id} label={t('widgets.gallery.autoAddToBoard')} />
                ) : null}
                <BoardDownloadMenuItem board={board} />
                {isManagedBoard && (
                  <>
                    <BoardRenameMenuItem board={board} onRename={setRenameTarget} onRenameValue={setRenameValue} />
                    {canMoveGalleryBoard(board, gallery.projectBoardId, currentUserId) && (
                      <BoardMoveSubMenu board={board} onBoardMoved={onBoardRelocated} />
                    )}
                    <BoardArchiveMenuItem archived={board.archived} boardId={board.id} onArchive={onBoardRelocated} />
                    <Menu.Separator />
                    <BoardDeleteMenuItem board={board} onDelete={setDeleteTarget} />
                  </>
                )}
              </MenuContent>
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
                <Dialog.Title>{t('widgets.gallery.renameBoard')}</Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Input
                  aria-label={t('widgets.gallery.boardName')}
                  autoFocus
                  size="lg"
                  value={renameValue}
                  onChange={handleRenameValueChange}
                  onKeyDown={handleRenameKeyDown}
                />
              </Dialog.Body>
              <Dialog.Footer>
                <Button variant="ghost" onClick={handleCancelRename}>
                  {t('common.cancel')}
                </Button>
                <Button disabled={renameValue.trim().length === 0} onClick={submitRename}>
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
                <Dialog.Title>
                  {t('widgets.gallery.deleteBoardQuestion', { name: deleteTarget?.name ?? '' })}
                </Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Stack gap="2">
                  <Text color="fg.subtle" fontSize="md">
                    {t('widgets.gallery.deleteBoardDescription')}
                  </Text>
                  <Text color="fg.subtle" fontSize="xs">
                    {t('widgets.gallery.boardItemCounts', {
                      assets: t('widgets.gallery.assetCount', { count: deleteTarget?.assetCount ?? 0 }),
                      images: t('widgets.gallery.imageCount', { count: deleteTarget?.imageCount ?? 0 }),
                      videos: t('widgets.gallery.videoCount', { count: deleteTarget?.videoCount ?? 0 }),
                    })}
                  </Text>
                </Stack>
              </Dialog.Body>
              <Dialog.Footer>
                <Button variant="ghost" onClick={handleCancelDelete}>
                  {t('common.cancel')}
                </Button>
                <Button colorPalette="red" variant="outline" onClick={handleDeleteBoardOnly}>
                  {t('widgets.gallery.deleteBoardOnly')}
                </Button>
                <Button colorPalette="red" onClick={handleDeleteBoardAndImages}>
                  {t('widgets.gallery.deleteBoardAndMedia')}
                </Button>
              </Dialog.Footer>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
    </>
  );
};

/** Project export includes the document; the separate media-only download retains its video-omission warning. */
const BoardExportProjectMenuItem = ({ board }: { board: GalleryBoard }) => {
  const { t } = useTranslation();
  const { actions } = useGalleryWidget();
  const projectId = board.projectId;
  const handleClick = useCallback(() => {
    if (projectId !== null) {
      actions.exportProject(projectId, board.name);
    }
  }, [actions, board.name, projectId]);

  return (
    <BoardMenuItem
      icon={FileDownIcon}
      label={t('widgets.gallery.exportProjectFromBoard')}
      value="export-project"
      onClick={handleClick}
    />
  );
};

const SUBMENU_POSITIONING = { gutter: 2, placement: 'right-start' } as const;

/**
 * Where a board can go: the Library, the open project, then the account's other projects by name — every tier
 * but the one it is in. Nested so the board menu keeps its one-level shape for the common actions.
 */
const BoardMoveSubMenu = ({
  board,
  onBoardMoved,
}: {
  board: GalleryBoard;
  onBoardMoved?: (boardId: string) => void;
}) => {
  const { t } = useTranslation();
  const { actions, projectId, projectName, projectNames } = useGalleryWidget();
  const handleMove = useCallback(
    (boardId: string, destinationId: string | null, label: string) => {
      onBoardMoved?.(boardId);

      return actions.moveBoard(boardId, destinationId, label);
    },
    [actions, onBoardMoved]
  );
  const destinations = useMemo(() => {
    const others = [...projectNames]
      .filter(([id]) => id !== projectId && id !== board.projectId)
      .map(([id, name]) => ({ id, label: name }))
      .sort((left, right) => left.label.localeCompare(right.label));

    return [
      ...(board.projectId !== null ? [{ id: null, label: t('widgets.gallery.boardGroups.library') }] : []),
      ...(board.projectId !== projectId ? [{ id: projectId, label: projectName }] : []),
      ...others,
    ];
  }, [board.projectId, projectId, projectName, projectNames, t]);

  if (destinations.length === 0) {
    return null;
  }

  return (
    <Menu.Root positioning={SUBMENU_POSITIONING}>
      <Menu.TriggerItem>
        <HStack gap="2" minW="0" w="full">
          <Icon as={FolderInputIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
          <Text flex="1" fontSize="md">
            {t('widgets.gallery.moveBoard')}
          </Text>
          <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
        </HStack>
      </Menu.TriggerItem>
      <Portal>
        <Menu.Positioner>
          <MenuContent maxH="18rem" minW="12rem" overflowY="auto">
            {destinations.map((destination) => (
              <BoardMoveMenuItem
                key={destination.id ?? 'library'}
                boardId={board.id}
                destinationId={destination.id}
                label={destination.label}
                onMove={handleMove}
              />
            ))}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const BoardMoveMenuItem = ({
  boardId,
  destinationId,
  label,
  onMove,
}: {
  boardId: string;
  destinationId: string | null;
  label: string;
  onMove: (boardId: string, projectId: string | null, destinationLabel: string) => Promise<void>;
}) => {
  const handleClick = useCallback(
    () => void onMove(boardId, destinationId, label),
    [boardId, destinationId, label, onMove]
  );

  return (
    <Menu.Item value={`move-board-to-${destinationId ?? 'library'}`} onClick={handleClick}>
      <MiddleTruncate fontSize="md" minW="0" text={label} />
    </Menu.Item>
  );
};

/** Sets the auto-add board; `follow` hands results back to the selected board. */
const BoardAutoAddMenuItem = ({ boardId, label }: { boardId: string; label: string }) => {
  const { actions } = useGalleryWidget();
  const handleClick = useCallback(() => actions.updateSettings({ autoAddBoardId: boardId }), [actions, boardId]);

  return <BoardMenuItem icon={SquareArrowDownIcon} label={label} value="auto-add-board" onClick={handleClick} />;
};

const BoardDownloadMenuItem = ({ board }: { board: GalleryBoard }) => {
  const { t } = useTranslation();
  const { actions } = useGalleryWidget();
  const handleClick = useCallback(() => void actions.downloadBoard(board.id), [actions, board.id]);

  return (
    <BoardMenuItem
      icon={DownloadIcon}
      label={t('widgets.gallery.downloadBoardWithOmission', { count: board.videoCount })}
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

const BoardArchiveMenuItem = ({
  archived,
  boardId,
  onArchive,
}: {
  archived: boolean;
  boardId: string;
  onArchive?: (boardId: string) => void;
}) => {
  const { t } = useTranslation();
  const { actions } = useGalleryWidget();
  const handleClick = useCallback(() => {
    onArchive?.(boardId);
    void actions.archiveBoard(boardId, !archived);
  }, [actions, archived, boardId, onArchive]);

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
      icon={Trash2Icon}
      isDanger
      label={t('widgets.gallery.deleteBoard')}
      value="delete-board"
      onClick={handleClick}
    />
  );
};

const BoardMenuItem = ({
  icon,
  isDanger,
  label,
  value,
  onClick,
}: {
  icon: LucideIcon;
  isDanger?: boolean;
  label: string;
  value: string;
  onClick: () => void;
}) => (
  <Menu.Item data-danger={isDanger ? '' : undefined} value={value} onClick={onClick}>
    <HStack gap="2" minW="0" w="full">
      <Icon as={icon} boxSize="3.5" color={isDanger ? undefined : 'fg.subtle'} flexShrink={0} />
      <Text flex="1" fontSize="md">
        {label}
      </Text>
    </HStack>
  </Menu.Item>
);
