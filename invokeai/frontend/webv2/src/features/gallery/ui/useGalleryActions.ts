import type { GalleryBoard } from '@features/gallery/core/types';

import { toGalleryItemKey } from '@features/gallery/core/items';
import {
  createGalleryBoard,
  deleteGalleryBoard,
  downloadGalleryArchive,
  updateGalleryBoard,
  uploadGalleryImage,
} from '@features/gallery/data/backend';
import { invalidateGallery } from '@features/gallery/data/queryCache';
import { downloadBlob } from '@platform/browser/downloadBlob';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { useQueryClient } from '@tanstack/react-query';
import { useMemo } from 'react';

import type { GalleryActions } from './GalleryWidgetContext';

import { useGalleryUi } from './GalleryUiContext';

const ACCEPTED_UPLOAD_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export const useGalleryActions = ({
  boards,
  loadMore,
  projectBoardId,
  projectName,
  selectedBoardId,
}: {
  boards: GalleryBoard[];
  loadMore: () => void;
  projectBoardId: string | null;
  projectName: string;
  selectedBoardId: string;
}): GalleryActions => {
  const { gallery, notifications } = useGalleryUi();
  const queryClient = useQueryClient();

  return useMemo<GalleryActions>(() => {
    const recordError = (error: unknown) =>
      notifications.reportError({ area: 'gallery-actions', message: toErrorMessage(error), namespace: 'gallery' });
    const recordSuccess = (title: string, message?: string) => notifications.add({ kind: 'success', message, title });
    const refresh = () => void invalidateGallery(queryClient);
    const getBoard = (boardId: string) => boards.find((board) => board.id === boardId);
    const getBoardName = (boardId: string) => getBoard(boardId)?.name ?? 'Uncategorized';

    return {
      archiveBoard: async (boardId, archived) => {
        const owner = captureAccountScope();

        try {
          await updateGalleryBoard(boardId, { archived }, owner.signal);

          assertAccountScopeCurrent(owner);
          recordSuccess(`${archived ? 'Archived' : 'Unarchived'} board "${getBoardName(boardId)}"`);

          if (archived && boardId === selectedBoardId) {
            gallery.selectBoard('none');
          }

          refresh();
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
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
          recordSuccess(`Created board "${board.name}"`);
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
          recordSuccess(
            `Deleted board "${boardName}"`,
            includeImages ? 'Its images were permanently deleted.' : 'Its images were moved to Uncategorized.'
          );

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
            message: `Preparing an image archive of "${getBoardName(boardId)}". ${videoCount} ${
              videoCount === 1 ? 'video' : 'videos'
            } will be omitted.`,
            title: 'Preparing download',
          });

          const { blob, fileName } = await downloadGalleryArchive({ boardId, signal: owner.signal });

          assertAccountScopeCurrent(owner);
          downloadBlob(blob, fileName);
          recordSuccess('Download ready');
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
      loadMore,
      refresh,
      renameBoard: async (boardId, boardName) => {
        const owner = captureAccountScope();

        try {
          await updateGalleryBoard(boardId, { name: boardName }, owner.signal);

          assertAccountScopeCurrent(owner);
          recordSuccess(`Renamed board to "${boardName}"`);
          refresh();
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
      selectBoard: gallery.selectBoard,
      selectItem: gallery.selectItem,
      selectItemRange: (items, primaryItem) => gallery.setItemMultiSelection(items.map(toGalleryItemKey), primaryItem),
      selectProjectBoard: async () => {
        const owner = captureAccountScope();

        if (projectBoardId && boards.some((board) => board.id === projectBoardId)) {
          gallery.selectBoard(projectBoardId);
          return;
        }

        if (boards.length === 0) {
          return;
        }

        try {
          const board = await createGalleryBoard(projectName, owner.signal);

          assertAccountScopeCurrent(owner);
          gallery.setProjectBoard(board.id);
          gallery.selectBoard(board.id);
          recordSuccess(`Created project board "${board.name}"`);
          refresh();
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
      setCompareItem: gallery.setCompareItem,
      setSearchTerm: gallery.setSearchTerm,
      setView: gallery.setView,
      toggleItemInSelection: gallery.toggleItemSelection,
      updateSettings: gallery.updateSettings,
      uploadFiles: async (files) => {
        const owner = captureAccountScope();
        const accepted = files.filter((file) => ACCEPTED_UPLOAD_TYPES.has(file.type));

        if (accepted.length === 0) {
          notifications.reportError({
            area: 'gallery-upload',
            message: 'No supported image files to upload (PNG, JPEG, or WebP).',
            namespace: 'gallery',
          });
          return;
        }

        try {
          await Promise.all(
            accepted.map((file) => uploadGalleryImage(file, selectedBoardId, { signal: owner.signal }))
          );

          assertAccountScopeCurrent(owner);
          recordSuccess(
            `Uploaded ${accepted.length} ${accepted.length === 1 ? 'image' : 'images'}`,
            `Added to ${getBoardName(selectedBoardId)} as assets.`
          );
          refresh();
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
    };
  }, [boards, gallery, loadMore, notifications, projectBoardId, projectName, queryClient, selectedBoardId]);
};
