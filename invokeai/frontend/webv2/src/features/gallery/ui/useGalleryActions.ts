import type { GalleryBoard, GalleryView } from '@features/gallery/core/types';

import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { toGalleryItemKey } from '@features/gallery/core/items';
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
  getCurrentGalleryLocation: () => { galleryView: GalleryView; selectedBoardId: string };
  loadMore: () => void;
  selectedBoardId: string;
}): GalleryActions => {
  const { exportProject, gallery, notifications } = useGalleryUi();
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
        // Optimistic: the board moves between sections immediately, and
        // archiving the board being viewed steps off it right away.
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
      createBoard: async (boardName) => {
        const owner = captureAccountScope();

        try {
          const board = await createGalleryBoard(boardName, owner.signal);

          assertAccountScopeCurrent(owner);
          gallery.selectBoard(board.id);
          recordSuccess(t('widgets.gallery.boardCreated', { name: board.name }));
          refresh();
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
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
      refresh,
      renameBoard: async (boardId, boardName) => {
        const owner = captureAccountScope();
        // Optimistic: the new name paints everywhere at once; the refresh
        // re-sorts name-ordered lists once the backend confirms.
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
      selectItemRange: (items, primaryItem) => gallery.setItemMultiSelection(items.map(toGalleryItemKey), primaryItem),
      setCompareItem: gallery.setCompareItem,
      setSearchTerm: gallery.setSearchTerm,
      setView: gallery.setView,
      toggleItemInSelection: gallery.toggleItemSelection,
      updateSettings: gallery.updateSettings,
      uploadFiles,
    };
  }, [
    boards,
    exportProject,
    gallery,
    getCurrentGalleryLocation,
    loadMore,
    notifications,
    queryClient,
    selectedBoardId,
    t,
    uploadFiles,
  ]);
};
