import type {
  GalleryMediaUsageCheck,
  GalleryMediaUsageReport,
  GalleryMediaUsageSettledCheck,
} from '@features/gallery/core/mediaUsage';
import type { GalleryBoard, GalleryBoardAction, GalleryGeneralAccess } from '@features/gallery/core/types';

import { chakra, Checkbox, HStack, Icon, Input, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { canOnBoard, isOwnGalleryBoard } from '@features/gallery/core/boardAccess';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { formatGalleryMediaCounts } from '@features/gallery/core/mediaCounts';
import { galleryBoardMediaUsageQueryOptions } from '@features/gallery/data/queries';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { Button } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { MenuContent } from '@platform/ui/Menu';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { ScrollableMenuContent } from '@platform/ui/ScrollableMenuContent';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ArchiveIcon,
  ChevronRightIcon,
  DownloadIcon,
  FileDownIcon,
  FolderInputIcon,
  PencilIcon,
  SquareArrowDownIcon,
  Trash2Icon,
  UsersIcon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { isManagedGalleryBoard, isMovableGalleryBoard } from './galleryBoardGroups';
import { GalleryMediaUsagePanel } from './GalleryMediaUsagePanel';
import { getBoardCounts, getGalleryIntendedDestinationId } from './galleryStateView';
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
  const { actions, boardSharing = false, gallery } = useGalleryWidget();
  const [renameTarget, setRenameTarget] = useState<GalleryBoard | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<GalleryBoard | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const accessHeaderId = useId();
  const targetRef = useRef(target);

  // eslint-disable-next-line react/refs
  targetRef.current = target;

  const board = target?.board ?? null;
  // Other than an inbox, a real board is managed as far as the server allows the caller.
  const isManagedBoard = board !== null && isManagedGalleryBoard(board, gallery.projectBoardId);
  const can = (action: GalleryBoardAction) => board !== null && isManagedBoard && canOnBoard(board, action);
  // A dialog stays open only while the board is listed and still allows its action: a refetch after a revocation
  // or a deletion elsewhere closes it, for good, rather than leaving a change that would only be refused or reopening
  // it with what was typed when access returns.
  const isStillAllowed = (dialogBoard: GalleryBoard, action: GalleryBoardAction) => {
    const listed = gallery.boards.find((candidate) => candidate.id === dialogBoard.id);
    return Boolean(listed && canOnBoard(listed, action));
  };
  if (renameTarget !== null && !isStillAllowed(renameTarget, 'rename')) {
    setRenameTarget(null);
  }
  if (deleteTarget !== null && !isStillAllowed(deleteTarget, 'delete')) {
    setDeleteTarget(null);
  }
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

  const handleDelete = useCallback(
    (includeMedia: boolean) => {
      if (deleteTarget) {
        void actions.deleteBoard(deleteTarget.id, includeMedia);
      }

      setDeleteTarget(null);
    },
    [actions, deleteTarget]
  );

  return (
    <>
      <Menu.Root
        key={board ? `menu:${board.id}` : 'menu:closed'}
        lazyMount
        open={target !== null}
        positioning={positioning}
        unmountOnExit
        onOpenChange={handleOpenChange}
      >
        <Portal>
          <Menu.Positioner>
            {board && (
              <MenuContent
                // Focus moves to the items, outside the header's group: the menu itself is described by it, so
                // whose board it is and what the caller may do is announced as the menu opens.
                aria-describedby={hasAccessHeader(board) ? accessHeaderId : undefined}
                minW="12rem"
              >
                <BoardAccessHeader board={board} id={accessHeaderId} />
                {board.isInbox && board.projectId !== null && isOwnGalleryBoard(board) && (
                  <>
                    <BoardExportProjectMenuItem board={board} />
                    <Menu.Separator />
                  </>
                )}
                {!gallery.settings.followSelectedBoard && board.id === gallery.settings.autoAddBoardId ? (
                  // Back to the default destination; the default itself has nothing to stop.
                  <BoardAutoAddMenuItem boardId={null} label={t('widgets.gallery.stopAutoAdd')} />
                ) : (gallery.settings.followSelectedBoard || board.id !== getGalleryIntendedDestinationId(gallery)) &&
                  board.kind !== 'date' &&
                  !board.archived &&
                  canOnBoard(board, 'add_media') ? (
                  <BoardAutoAddMenuItem boardId={board.id} label={t('widgets.gallery.autoAddToBoard')} />
                ) : null}
                <BoardDownloadMenuItem board={board} />
                {can('rename') && (
                  <BoardRenameMenuItem board={board} onRename={setRenameTarget} onRenameValue={setRenameValue} />
                )}
                {boardSharing && can('set_general_access') && <BoardGeneralAccessSubMenu board={board} />}
                {isMovableGalleryBoard(board, gallery.projectBoardId) && (
                  <BoardMoveSubMenu board={board} onBoardMoved={onBoardRelocated} />
                )}
                {can('archive') && (
                  <BoardArchiveMenuItem archived={board.archived} boardId={board.id} onArchive={onBoardRelocated} />
                )}
                {can('delete') && (
                  <>
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
      <BoardDeleteDialog
        board={deleteTarget}
        getMediaUsage={actions.getBoardMediaUsage}
        onCancel={handleCancelDelete}
        onDelete={handleDelete}
        onOpenChange={handleDeleteDialogOpenChange}
      />
    </>
  );
};

/** From this many items, also deleting them needs the board's name typed, not only a tick. */
export const BOARD_DELETE_TYPED_CONFIRMATION_MIN_ITEMS = 50;

/**
 * Deleting a board keeps its media by default: it moves to Uncategorized in the Library, where the server puts what a
 * deleted board held. Deleting the media too is the opt-in, behind a checkbox and, for many items, the typed name.
 */
const BoardDeleteDialog = ({
  board,
  getMediaUsage,
  onCancel,
  onDelete,
  onOpenChange,
}: {
  board: GalleryBoard | null;
  /** Where the board's media is still used, checked once deleting it is ticked. */
  getMediaUsage: (boardId: string, signal: AbortSignal) => Promise<GalleryMediaUsageReport>;
  onCancel: () => void;
  onDelete: (includeMedia: boolean) => void;
  onOpenChange: (event: { open: boolean }) => void;
}) => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  // The board and the choices stay shown while the dialog animates out; the choices belong to one opening and start
  // over with the next one, for the same board too. The dialog is not remounted, which would cut the exit animation.
  const [shownBoard, setShownBoard] = useState(board);
  const [openedBoardId, setOpenedBoardId] = useState(board?.id ?? null);
  // `hasTyped`: the user has typed into the name field this opening, which then stays (see `needsTypedName`).
  const [choices, setChoices] = useState({ hasTyped: false, includeMedia: false, typedName: '' });
  // Each opening asks where the media is used afresh; ticking and unticking within one opening does not.
  const [opening, setOpening] = useState(0);
  if (board !== null && board !== shownBoard) {
    setShownBoard(board);
  }
  if ((board?.id ?? null) !== openedBoardId) {
    setOpenedBoardId(board?.id ?? null);
    if (board !== null) {
      setChoices({ hasTyped: false, includeMedia: false, typedName: '' });
      setOpening((current) => current + 1);
    }
  }
  const { hasTyped, includeMedia, typedName } = choices;
  const isOpen = board !== null;
  const checksUsage = includeMedia && shownBoard !== null;
  const boardId = shownBoard?.id ?? '';
  const owner = captureAccountScope();
  // An opening's check lives under its own key while the dialog is open. However the dialog closes (Cancel, Escape,
  // Delete, or the board revoked or deleted elsewhere), the observer leaves that key, and Query stops a check still
  // running there (its server call and project reads) and drops it.
  const usageOptions = useMemo(
    () => galleryBoardMediaUsageQueryOptions({ boardId, load: getMediaUsage, opening: isOpen ? opening : null, owner }),
    [boardId, getMediaUsage, isOpen, opening, owner]
  );
  const { queryKey: usageQueryKey } = usageOptions;
  const usageQuery = useQuery({ ...usageOptions, enabled: isOpen && checksUsage });
  const { data: usageReport, isFetching: isCheckingUsage, refetch: refetchUsage, status: usageStatus } = usageQuery;
  // The opening's last answer, kept while the box is unticked and as the dialog closes, so neither the report nor the
  // counts it gives come and go.
  const [keptAnswer, setKeptAnswer] = useState<{ check: GalleryMediaUsageSettledCheck; opening: number } | null>(null);
  const answer: GalleryMediaUsageSettledCheck | null =
    usageStatus === 'success' && usageReport
      ? { report: usageReport, status: 'ready' }
      : usageStatus === 'error'
        ? { status: 'failed' }
        : null;
  const keptCheck = keptAnswer?.opening === opening ? keptAnswer.check : null;
  if (
    isOpen &&
    answer &&
    (answer.status === 'ready'
      ? keptCheck?.status !== 'ready' || keptCheck.report !== answer.report
      : keptCheck?.status !== 'failed')
  ) {
    setKeptAnswer({ check: answer, opening });
  }
  // Unticking stops a check still running too; it reverts, so ticking again asks again. An answer stays for the opening.
  const cancelUsageCheck = useCallback(
    () => void queryClient.cancelQueries({ exact: true, queryKey: usageQueryKey }),
    [queryClient, usageQueryKey]
  );
  const usageCheck = useMemo<GalleryMediaUsageCheck | null>(() => {
    if (!checksUsage) {
      return null;
    }

    // A retry keeps the last answer on screen while it runs.
    return isCheckingUsage || !keptCheck ? { previous: keptCheck ?? undefined, status: 'checking' } : keptCheck;
  }, [checksUsage, isCheckingUsage, keptCheck]);
  // A closing dialog's Retry stays in place, disabled: running it would ask on the closed key, which nothing stops.
  const handleRetryUsage = useCallback(() => {
    if (isOpen) {
      void refetchUsage();
    }
  }, [isOpen, refetchUsage]);
  const cancelRef = useRef<HTMLButtonElement | null>(null);
  const mediaHintId = useId();
  const nameInputId = useId();
  const boardName = shownBoard ? getGalleryBoardLabel(shownBoard, t) : '';
  const counts = shownBoard ? getBoardCounts(shownBoard) : null;
  const itemCount = counts?.items ?? 0;
  const media = formatGalleryMediaCounts(t, {
    images: counts?.images ?? 0,
    uploads: counts?.uploads ?? 0,
    videos: counts?.videos ?? 0,
  });
  // Deleting the media takes what the check counted, once it has: on a shared board only the caller's own items, while
  // the board's counts include everyone's; the rest moves to Uncategorized, as when the media is kept. An administrator
  // overriding takes everything, so a lower count there is only a board count gone stale.
  const keptReport = keptCheck?.status === 'ready' ? keptCheck.report : undefined;
  const deletedCount = keptReport?.checkedCount ?? itemCount;
  const deletesOwnMedia = deletedCount < itemCount && shownBoard?.access?.adminOverride !== true;
  const deletesNoMedia = deletesOwnMedia && deletedCount === 0;
  const deletedMedia =
    deletedCount === itemCount || !keptReport
      ? media
      : t(keptReport.onlyImages ? 'widgets.gallery.imageCount' : 'widgets.gallery.deleteInUse.itemCount', {
          count: deletedCount,
        });
  const includeMediaLabel = deletesNoMedia
    ? t('widgets.gallery.deleteBoardIncludeOwnMediaNone')
    : t(deletesOwnMedia ? 'widgets.gallery.deleteBoardIncludeOwnMedia' : 'widgets.gallery.deleteBoardIncludeMedia', {
        count: deletedCount,
      });
  const withMediaDescription = deletesNoMedia
    ? t('widgets.gallery.deleteBoardWithNoOwnMediaDescription')
    : t(
        deletesOwnMedia
          ? 'widgets.gallery.deleteBoardWithOwnMediaDescription'
          : 'widgets.gallery.deleteBoardWithMediaDescription',
        { count: deletedCount, media: deletedMedia }
      );
  const deleteLabel =
    !includeMedia || deletesNoMedia
      ? t('widgets.gallery.deleteBoard')
      : t(deletesOwnMedia ? 'widgets.gallery.deleteBoardAndOwnMedia' : 'widgets.gallery.deleteBoardAndMedia', {
          count: deletedCount,
        });
  // The board's count until the check answers, then what deleting the media actually takes. Once the user has typed
  // into the field this opening it stays, with its requirement: an answer arriving mid-typing never takes their text.
  const needsTypedName = includeMedia && (hasTyped || deletedCount >= BOARD_DELETE_TYPED_CONFIRMATION_MIN_ITEMS);
  const canDelete = !needsTypedName || typedName.trim() === boardName.trim();
  const handleIncludeMediaChange = useCallback(
    (details: { checked: boolean | 'indeterminate' }) => {
      if (details.checked !== true) {
        cancelUsageCheck();
      }
      setChoices((current) => ({ ...current, includeMedia: details.checked === true }));
    },
    [cancelUsageCheck]
  );
  const handleTypedNameChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const typed = event.currentTarget.value;

    setChoices((current) => ({ ...current, hasTyped: current.hasTyped || typed !== '', typedName: typed }));
  }, []);
  const getInitialFocus = useCallback(() => cancelRef.current, []);
  // With none of the media the caller's, nothing is asked to be deleted: the board goes, and its notices say only that.
  const deletesMedia = includeMedia && !deletesNoMedia;
  const handleConfirm = useCallback(() => onDelete(deletesMedia), [deletesMedia, onDelete]);

  return (
    // Cancel takes focus, not the checkbox that would delete the media.
    // Top-anchored: the usage report arrives after the dialog opens, and grows it downward instead of moving it.
    <Dialog.Root
      initialFocusEl={getInitialFocus}
      open={board !== null}
      placement="top"
      role="alertdialog"
      onOpenChange={onOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header>
              <Dialog.Title>{t('widgets.gallery.deleteBoardQuestion', { name: boardName })}</Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Stack gap="3">
                <Dialog.Description color="fg" fontSize="md" lineHeight="inherit">
                  {itemCount === 0
                    ? t('widgets.gallery.deleteBoardEmptyDescription')
                    : includeMedia
                      ? withMediaDescription
                      : t('widgets.gallery.deleteBoardDescription', { count: itemCount, media })}
                </Dialog.Description>
                {itemCount > 0 ? (
                  <Stack gap="1">
                    <Checkbox.Root
                      checked={includeMedia}
                      colorPalette="red"
                      size="sm"
                      onCheckedChange={handleIncludeMediaChange}
                    >
                      <Checkbox.HiddenInput aria-describedby={mediaHintId} />
                      <Checkbox.Control>
                        <Checkbox.Indicator />
                      </Checkbox.Control>
                      <Checkbox.Label fontSize="md">{includeMediaLabel}</Checkbox.Label>
                    </Checkbox.Root>
                    <Text color="fg.muted" fontSize="sm" id={mediaHintId} ps="5">
                      {t('widgets.gallery.deleteBoardIncludeMediaHint')}
                    </Text>
                  </Stack>
                ) : null}
                {itemCount > 0 && usageCheck ? (
                  <GalleryMediaUsagePanel
                    // Once the answer says none of the media is the caller's, nothing goes and there is no use to
                    // report: the panel only reads out the description that changed with it.
                    announcement={deletesNoMedia ? withMediaDescription : undefined}
                    check={usageCheck}
                    isRetryDisabled={!isOpen}
                    onRetry={handleRetryUsage}
                  />
                ) : null}
                {needsTypedName ? (
                  <Stack gap="1">
                    <chakra.label fontSize="md" htmlFor={nameInputId}>
                      {t('widgets.gallery.deleteBoardTypeName', { name: boardName })}
                    </chakra.label>
                    <Input
                      autoComplete="off"
                      id={nameInputId}
                      spellCheck={false}
                      value={typedName}
                      onChange={handleTypedNameChange}
                    />
                  </Stack>
                ) : null}
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button ref={cancelRef} variant="ghost" onClick={onCancel}>
                {t('common.cancel')}
              </Button>
              <Button colorPalette="red" disabled={!canDelete} onClick={handleConfirm}>
                {deleteLabel}
              </Button>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
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
 * Where a board can go: the Library, then under a Projects heading the open project and the account's other projects
 * by name — every tier but the one it is in. Nested so the board menu keeps its one-level shape for the common actions.
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
  const projects = useMemo(() => {
    const others = [...projectNames]
      .filter(([id]) => id !== projectId && id !== board.projectId)
      .map(([id, name]) => ({ id, label: name }))
      .sort((left, right) => left.label.localeCompare(right.label));

    return [...(board.projectId !== projectId ? [{ id: projectId, label: projectName }] : []), ...others];
  }, [board.projectId, projectId, projectName, projectNames]);
  const canMoveToLibrary = board.projectId !== null;

  if (!canMoveToLibrary && projects.length === 0) {
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
          <ScrollableMenuContent minW="12rem">
            {canMoveToLibrary ? (
              <BoardMoveMenuItem
                boardId={board.id}
                destinationId={null}
                label={t('widgets.gallery.boardGroups.library')}
                onMove={handleMove}
              />
            ) : null}
            {canMoveToLibrary && projects.length > 0 ? <Menu.Separator /> : null}
            {projects.length > 0 ? (
              <Menu.ItemGroup>
                <Menu.ItemGroupLabel>{t('widgets.gallery.boardGroups.projects')}</Menu.ItemGroupLabel>
                {projects.map((destination) => (
                  <BoardMoveMenuItem
                    key={destination.id}
                    boardId={board.id}
                    destinationId={destination.id}
                    label={destination.label}
                    onMove={handleMove}
                  />
                ))}
              </Menu.ItemGroup>
            ) : null}
          </ScrollableMenuContent>
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

/** Fixes where new results go, which stops following the selected board; null returns them to the default. */
const BoardAutoAddMenuItem = ({ boardId, label }: { boardId: string | null; label: string }) => {
  const { actions } = useGalleryWidget();
  const handleClick = useCallback(
    () => actions.updateSettings({ autoAddBoardId: boardId, followSelectedBoard: false }),
    [actions, boardId]
  );

  return <BoardMenuItem icon={SquareArrowDownIcon} label={label} value="auto-add-board" onClick={handleClick} />;
};

const BoardDownloadMenuItem = ({ board }: { board: GalleryBoard }) => {
  const { t } = useTranslation();
  const { actions } = useGalleryWidget();
  const handleClick = useCallback(() => void actions.downloadBoard(board.id), [actions, board.id]);

  return (
    <BoardMenuItem
      icon={DownloadIcon}
      label={
        board.videoCount > 0
          ? t('widgets.gallery.downloadBoardWithOmission', { count: board.videoCount })
          : t('widgets.gallery.downloadBoard')
      }
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

/**
 * Whose board someone else's board is and what the caller may do with it, above its actions: the menu offers only
 * what is allowed, so this says why the rest is missing and whom to ask. Nothing on the caller's own boards.
 */
const hasAccessHeader = (board: GalleryBoard) => board.access !== undefined && board.access.role !== 'owner';

const BoardAccessHeader = ({ board, id }: { board: GalleryBoard; id: string }) => {
  const { t } = useTranslation();
  const { access } = board;

  if (access === undefined || !hasAccessHeader(board)) {
    return null;
  }

  const owner = board.ownerName ?? t('widgets.gallery.boardAccess.deletedOwner');
  const level = access.adminOverride || access.role === null ? 'admin' : access.role;

  return (
    <Menu.ItemGroup>
      <Menu.ItemGroupLabel color="fg.muted" fontSize="sm" fontWeight="normal" maxW="18rem" textTransform="none">
        <Stack gap="0.5" id={id}>
          <Text>{t(`widgets.gallery.boardAccess.header.${level}`, { owner })}</Text>
          {level === 'viewer' ? <Text>{t('widgets.gallery.boardAccess.askToAdd', { owner })}</Text> : null}
        </Stack>
      </Menu.ItemGroupLabel>
      <Menu.Separator />
    </Menu.ItemGroup>
  );
};

const GENERAL_ACCESS_OPTIONS: readonly GalleryGeneralAccess[] = ['restricted', 'viewer', 'contributor'];

/** Who else may open the board, for its owner and admins: applied at once, and reported. */
const BoardGeneralAccessSubMenu = ({ board }: { board: GalleryBoard }) => {
  const { t } = useTranslation();
  const { actions } = useGalleryWidget();
  const current = board.access?.generalAccess ?? 'restricted';
  const handleValueChange = useCallback(
    ({ value }: { value: string }) => {
      if (value !== current && GENERAL_ACCESS_OPTIONS.includes(value as GalleryGeneralAccess)) {
        void actions.setBoardGeneralAccess(board.id, value as GalleryGeneralAccess);
      }
    },
    [actions, board.id, current]
  );
  const describe = (option: GalleryGeneralAccess) =>
    option !== 'restricted'
      ? t(`widgets.gallery.boardAccess.general.${option}Description`)
      : isOwnGalleryBoard(board)
        ? t('widgets.gallery.boardAccess.general.restrictedDescription')
        : t('widgets.gallery.boardAccess.general.restrictedDescriptionOther', {
            owner: board.ownerName ?? t('widgets.gallery.boardAccess.deletedOwner'),
          });

  return (
    <Menu.Root positioning={SUBMENU_POSITIONING}>
      <Menu.TriggerItem>
        <HStack gap="2" minW="0" w="full">
          <Icon as={UsersIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
          <Text flex="1" fontSize="md">
            {t('widgets.gallery.boardAccess.menu')}
          </Text>
          <Text color="fg.muted" fontSize="sm">
            {t(`widgets.gallery.boardAccess.general.${current}`)}
          </Text>
          <Icon as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
        </HStack>
      </Menu.TriggerItem>
      <Portal>
        <Menu.Positioner>
          <MenuContent maxW="20rem" minW="16rem">
            <Menu.RadioItemGroup value={current} onValueChange={handleValueChange}>
              <Menu.ItemGroupLabel color="fg.muted" fontSize="sm" fontWeight="normal" textTransform="none">
                {t('widgets.gallery.boardAccess.general.label')}
              </Menu.ItemGroupLabel>
              {GENERAL_ACCESS_OPTIONS.map((option) => (
                <GeneralAccessOption
                  key={option}
                  description={describe(option)}
                  label={t(`widgets.gallery.boardAccess.general.${option}`)}
                  value={option}
                />
              ))}
            </Menu.RadioItemGroup>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

/** One choice, named by its label alone and described by its consequence, so each is announced distinctly. */
const GeneralAccessOption = ({
  description,
  label,
  value,
}: {
  description: string;
  label: string;
  value: GalleryGeneralAccess;
}) => {
  const labelId = useId();
  const descriptionId = useId();

  return (
    <Menu.RadioItem alignItems="flex-start" aria-describedby={descriptionId} aria-labelledby={labelId} value={value}>
      <Menu.ItemIndicator mt="1" />
      <Stack gap="0.5" minW="0">
        <Menu.ItemText fontSize="md" id={labelId}>
          {label}
        </Menu.ItemText>
        <Text color="fg.muted" fontSize="sm" id={descriptionId} whiteSpace="normal">
          {description}
        </Text>
      </Stack>
    </Menu.RadioItem>
  );
};
