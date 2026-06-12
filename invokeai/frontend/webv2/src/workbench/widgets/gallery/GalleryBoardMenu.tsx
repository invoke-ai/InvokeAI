import { Dialog, HStack, Icon, Input, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { useEffect, useRef, useState } from 'react';
import { ArchiveIcon, DownloadIcon, PencilIcon, Trash2Icon, type LucideIcon } from 'lucide-react';

import type { GalleryBoard } from '../../gallery/api';
import { Button } from '../../components/ui/Button';
import { useGalleryWidget } from './GalleryWidgetContext';

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

  const submitRename = () => {
    const trimmedName = renameValue.trim();

    if (renameTarget && trimmedName && trimmedName !== renameTarget.name) {
      void actions.renameBoard(renameTarget.id, trimmedName);
    }

    setRenameTarget(null);
  };

  return (
    <>
      <Menu.Root
        key={board ? board.id : 'closed'}
        lazyMount
        open={target !== null}
        positioning={{
          getAnchorRect: () => {
            const currentTarget = targetRef.current;

            return currentTarget ? { height: 1, width: 1, x: currentTarget.x, y: currentTarget.y } : null;
          },
          placement: 'bottom-start',
        }}
        unmountOnExit
        onOpenChange={(event) => {
          if (!event.open) {
            onClose();
          }
        }}
      >
        <Portal>
          <Menu.Positioner>
            {board && (
              <Menu.Content minW="12rem" py="1" px="0">
                <BoardMenuItem
                  icon={DownloadIcon}
                  label="Download Board"
                  value="download-board"
                  onClick={() => void actions.downloadBoard(board.id)}
                />
                {isManagedBoard && (
                  <>
                    <BoardMenuItem
                      icon={PencilIcon}
                      label="Rename Board"
                      value="rename-board"
                      onClick={() => {
                        setRenameValue(board.name);
                        setRenameTarget(board);
                      }}
                    />
                    <BoardMenuItem
                      icon={ArchiveIcon}
                      label={board.archived ? 'Unarchive Board' : 'Archive Board'}
                      value="toggle-archived"
                      onClick={() => void actions.archiveBoard(board.id, !board.archived)}
                    />
                    <Menu.Separator borderColor="border.subtle" />
                    <BoardMenuItem
                      color="fg.error"
                      icon={Trash2Icon}
                      label="Delete Board"
                      value="delete-board"
                      onClick={() => setDeleteTarget(board)}
                    />
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
        onOpenChange={(event) => {
          if (!event.open) {
            setRenameTarget(null);
          }
        }}
        size="sm"
      >
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content>
              <Dialog.Header>
                <Dialog.Title fontSize="sm">Rename board</Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Input
                  aria-label="Board name"
                  autoFocus
                  size="sm"
                  value={renameValue}
                  onChange={(event) => setRenameValue(event.currentTarget.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault();
                      submitRename();
                    }
                  }}
                />
              </Dialog.Body>
              <Dialog.Footer gap="2">
                <Button size="xs" variant="outline" onClick={() => setRenameTarget(null)}>
                  Cancel
                </Button>
                <Button disabled={renameValue.trim().length === 0} size="xs" onClick={submitRename}>
                  Rename
                </Button>
              </Dialog.Footer>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
      <Dialog.Root
        open={deleteTarget !== null}
        role="alertdialog"
        onOpenChange={(event) => {
          if (!event.open) {
            setDeleteTarget(null);
          }
        }}
      >
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content>
              <Dialog.Header>
                <Dialog.Title fontSize="sm">Delete board &ldquo;{deleteTarget?.name}&rdquo;?</Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Stack gap="2">
                  <Text color="fg.subtle" fontSize="xs">
                    Deleted boards cannot be restored. Choose whether the board&rsquo;s images move to Uncategorized or
                    are permanently deleted with it.
                  </Text>
                  <Text color="fg.subtle" fontSize="2xs">
                    {deleteTarget?.imageCount ?? 0} images | {deleteTarget?.assetCount ?? 0} assets
                  </Text>
                </Stack>
              </Dialog.Body>
              <Dialog.Footer gap="2">
                <Button size="xs" variant="outline" onClick={() => setDeleteTarget(null)}>
                  Cancel
                </Button>
                <Button
                  colorPalette="red"
                  size="xs"
                  variant="outline"
                  onClick={() => {
                    if (deleteTarget) {
                      void actions.deleteBoard(deleteTarget.id, false);
                    }

                    setDeleteTarget(null);
                  }}
                >
                  Delete Board Only
                </Button>
                <Button
                  colorPalette="red"
                  size="xs"
                  onClick={() => {
                    if (deleteTarget) {
                      void actions.deleteBoard(deleteTarget.id, true);
                    }

                    setDeleteTarget(null);
                  }}
                >
                  Delete Board and Images
                </Button>
              </Dialog.Footer>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
    </>
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
