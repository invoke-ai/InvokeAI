import type { GalleryBoard, GalleryView } from '@features/gallery/core/types';

import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { toGalleryItemKey } from '@features/gallery/core/items';
import { GALLERY_AUTO_ADD_FOLLOW, getGallerySettings } from '@features/gallery/core/settings';
import {
  createGalleryBoard,
  deleteGalleryBoard,
  downloadGalleryArchive,
  updateGalleryBoard,
} from '@features/gallery/data/backend';
import { invalidateGallery, patchGalleryBoardCaches } from '@features/gallery/data/queryCache';
import { downloadBlob } from '@platform/browser/downloadBlob';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { useQueryClient } from '@tanstack/react-query';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { GalleryActions } from './GalleryWidgetContext';

import { useGalleryUi } from './GalleryUiContext';
import { useGalleryUploadAction } from './useGalleryUploadAction';

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export const useGalleryActions = ({
  boards,
  getCurrentGalleryLocation,
  loadMore,
  selectedBoardId,
}: {
  boards: GalleryBoard[];
  /** The latest rendered gallery, for settling an action the user may have moved on from while it ran. */
  getCurrentGalleryLocation: () => { autoAddBoardId: string; galleryView: GalleryView; selectedBoardId: string };
  loadMore: () => void;
  selectedBoardId: string;
}): GalleryActions => {
  const { ensureProjectOnServer, exportProject, gallery, galleryValues, notifications, projectId, widgets } =
    useGalleryUi();
  const { autoAddBoardId, showOtherProjectBoards } = getGallerySettings(galleryValues);
  const queryClient = useQueryClient();
  const { t } = useTranslation();
  const uploadFiles = useGalleryUploadAction({ boards, getCurrentGalleryLocation, selectedBoardId });

  return useMemo<GalleryActions>(() => {
    const recordError = (error: unknown) =>
      notifications.reportError({ area: 'gallery-actions', message: toErrorMessage(error), namespace: 'gallery' });
    const recordSuccess = (title: string, message?: string) => notifications.add({ kind: 'success', message, title });
    const refresh = () => void invalidateGallery(queryClient);
    const getBoard = (boardId: string) => boards.find((board) => board.id === boardId);
    const getBoardName = (boardId: string) => {
      const board = getBoard(boardId);

      return board ? getGalleryBoardLabel(board, t) : t('widgets.gallery.uncategorized');
    };
    const formatImageCount = (count: number) => t('widgets.gallery.imageCount', { count });
    const formatVideoCount = (count: number) => t('widgets.gallery.videoCount', { count });

    return {
      archiveBoard: async (boardId, archived) => {
        const owner = captureAccountScope();
        const rollback = patchGalleryBoardCaches(queryClient, boardId, { archived });
        const movedSelectionAway = archived && boardId === selectedBoardId;

        if (movedSelectionAway) {
          gallery.selectBoard('none');
        }

        try {
          await updateGalleryBoard(boardId, { archived }, owner.signal);

          assertAccountScopeCurrent(owner);
          recordSuccess(
            t(archived ? 'widgets.gallery.boardArchived' : 'widgets.gallery.boardUnarchived', {
              name: getBoardName(boardId),
            })
          );
          refresh();
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          rollback();

          if (movedSelectionAway && getCurrentGalleryLocation().selectedBoardId === 'none') {
            gallery.selectBoard(boardId);
          }

          recordError(error);
        }
      },
      createBoard: async (boardName, projectId) => {
        const owner = captureAccountScope();

        try {
          if (projectId !== null) {
            await ensureProjectOnServer?.();
            assertAccountScopeCurrent(owner);
          }

          const board = await createGalleryBoard(boardName, projectId, owner.signal);

          assertAccountScopeCurrent(owner);
          gallery.selectBoard(board.id);
          recordSuccess(t('widgets.gallery.boardCreated', { name: board.name }));
          refresh();

          return true;
        } catch (error: unknown) {
          if (isAccountScopeCurrent(owner)) {
            recordError(error);
          }

          return false;
        }
      },
      deleteBoard: async (boardId, includeImages) => {
        const owner = captureAccountScope();

        try {
          const boardName = getBoardName(boardId);

          const outcome = await deleteGalleryBoard(boardId, includeImages, owner.signal);

          assertAccountScopeCurrent(owner);
          const failedCount = outcome.failedImageNames.length + outcome.failedVideoNames.length;
          const title = t(
            failedCount > 0 ? 'widgets.gallery.deleteBoardPartialTitle' : 'widgets.gallery.deleteBoardSuccessTitle',
            { name: boardName }
          );
          const message = includeImages
            ? t('widgets.gallery.deleteBoardMediaOutcome', {
                failedImages: formatImageCount(outcome.failedImageNames.length),
                failedVideos: formatVideoCount(outcome.failedVideoNames.length),
                images: formatImageCount(outcome.deletedImageNames.length),
                videos: formatVideoCount(outcome.deletedVideoNames.length),
              })
            : t('widgets.gallery.deleteBoardMoveOutcome', {
                images: formatImageCount(outcome.deletedBoardImageNames.length),
                videos: formatVideoCount(outcome.deletedBoardVideoNames.length),
              });
          recordSuccess(title, message);

          gallery.reconcileDeletedBoardOutcome(outcome);
          refresh();
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
      downloadBoard: async (boardId) => {
        const owner = captureAccountScope();

        try {
          const board = getBoard(boardId);
          const videoCount = board?.videoCount ?? 0;

          notifications.add({
            kind: 'info',
            message: t('widgets.gallery.boardArchivePreparing', {
              count: videoCount,
              name: getBoardName(boardId),
            }),
            title: t('widgets.gallery.preparingDownload'),
          });

          const { blob, fileName } = await downloadGalleryArchive({ boardId, signal: owner.signal });

          assertAccountScopeCurrent(owner);
          downloadBlob(blob, fileName);
          recordSuccess(t('widgets.gallery.downloadReady'));
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
      exportProject,
      loadMore,
      moveBoard: async (boardId, destinationProjectId, destinationLabel) => {
        const owner = captureAccountScope();
        const rollback = patchGalleryBoardCaches(queryClient, boardId, { projectId: destinationProjectId });
        // Into another project while other projects are hidden, the board leaves the panel as an archived one does:
        // it stops being the selection and the auto-add target rather than take results nothing on screen shows.
        const leavesView =
          destinationProjectId !== null && destinationProjectId !== projectId && !showOtherProjectBoards;
        const movedSelectionAway = leavesView && boardId === selectedBoardId;
        const stoppedAutoAdding = leavesView && boardId === autoAddBoardId;

        if (movedSelectionAway) {
          gallery.selectBoard('none');
        }
        if (stoppedAutoAdding) {
          gallery.updateSettings({ autoAddBoardId: GALLERY_AUTO_ADD_FOLLOW });
        }

        try {
          if (destinationProjectId !== null) {
            await ensureProjectOnServer?.();
            assertAccountScopeCurrent(owner);
          }

          await updateGalleryBoard(boardId, { projectId: destinationProjectId }, owner.signal);

          assertAccountScopeCurrent(owner);
          recordSuccess(
            t('widgets.gallery.boardMoved', { destination: destinationLabel, name: getBoardName(boardId) })
          );
          refresh();
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          rollback();

          // Each only while the user has not chosen again since.
          const current = getCurrentGalleryLocation();
          if (movedSelectionAway && current.selectedBoardId === 'none') {
            gallery.selectBoard(boardId);
          }
          if (stoppedAutoAdding && current.autoAddBoardId === GALLERY_AUTO_ADD_FOLLOW) {
            gallery.updateSettings({ autoAddBoardId: boardId });
          }

          recordError(error);
        }
      },
      refresh,
      renameBoard: async (boardId, boardName) => {
        const owner = captureAccountScope();
        const rollback = patchGalleryBoardCaches(queryClient, boardId, { name: boardName });

        try {
          await updateGalleryBoard(boardId, { name: boardName }, owner.signal);

          assertAccountScopeCurrent(owner);
          recordSuccess(t('widgets.gallery.boardRenamed', { name: boardName }));
          refresh();
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          rollback();
          recordError(error);
        }
      },
      selectBoard: gallery.selectBoard,
      selectItem: gallery.selectItem,
      selectItemRange: (items, primaryItem, selectionPage) =>
        selectionPage === undefined
          ? gallery.setItemMultiSelection(items.map(toGalleryItemKey), primaryItem)
          : gallery.setItemMultiSelection(items.map(toGalleryItemKey), primaryItem, selectionPage),
      setCompareItem: gallery.setCompareItem,
      setSearchTerm: gallery.setSearchTerm,
      setStarredOnly: gallery.setStarredOnly,
      setSemanticSearchMode: gallery.setSemanticSearchMode,
      setSemanticSearchText: gallery.setSemanticSearchText,
      commitSemanticSearch: gallery.commitSemanticSearch,
      clearSearch: gallery.clearSearch,
      // Store references in widget values, reset pagination, and clear replaced text/mode so hidden filters cannot
      // affect listing or Preview navigation.
      setSemanticImageQuery: (reference) =>
        widgets.patchGalleryValues({
          galleryPage: 0,
          searchTerm: '',
          semanticImageQuery: reference,
          semanticSearchText: null,
        }),
      setView: gallery.setView,
      toggleItemInSelection: gallery.toggleItemSelection,
      updateSettings: gallery.updateSettings,
      uploadFiles,
    };
  }, [
    autoAddBoardId,
    boards,
    ensureProjectOnServer,
    exportProject,
    gallery,
    getCurrentGalleryLocation,
    loadMore,
    notifications,
    projectId,
    queryClient,
    selectedBoardId,
    showOtherProjectBoards,
    t,
    uploadFiles,
    widgets,
  ]);
};
