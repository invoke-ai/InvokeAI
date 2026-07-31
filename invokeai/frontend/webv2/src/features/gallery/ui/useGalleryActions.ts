import type { GalleryItem } from '@features/gallery/core/items';
import type { GalleryBoard, GalleryView } from '@features/gallery/core/types';

import { compareGalleryItems, legacyGeneratedImageToGalleryItem, toGalleryItemKey } from '@features/gallery/core/items';
import {
  classifyGalleryUpload,
  createGalleryBoard,
  deleteGalleryBoard,
  downloadGalleryArchive,
  isDateBoardId,
  updateGalleryBoard,
  uploadGalleryImage,
  uploadGalleryVideo,
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
import { useTranslation } from 'react-i18next';

import type { GalleryActions } from './GalleryWidgetContext';

import { useGalleryUi } from './GalleryUiContext';

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export const useGalleryActions = ({
  boards,
  getCurrentGalleryLocation,
  loadMore,
  projectBoardId,
  projectName,
  selectedBoardId,
}: {
  boards: GalleryBoard[];
  getCurrentGalleryLocation: () => { galleryView: GalleryView; selectedBoardId: string };
  loadMore: () => void;
  projectBoardId: string | null;
  projectName: string;
  selectedBoardId: string;
}): GalleryActions => {
  const { gallery, notifications } = useGalleryUi();
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return useMemo<GalleryActions>(() => {
    const recordError = (error: unknown) =>
      notifications.reportError({ area: 'gallery-actions', message: toErrorMessage(error), namespace: 'gallery' });
    const recordSuccess = (title: string, message?: string) => notifications.add({ kind: 'success', message, title });
    const refresh = () => void invalidateGallery(queryClient);
    const getBoard = (boardId: string) => boards.find((board) => board.id === boardId);
    const getBoardName = (boardId: string) => getBoard(boardId)?.name ?? t('widgets.gallery.uncategorized');
    const formatImageCount = (count: number) => t('widgets.gallery.imageCount', { count });
    const formatVideoCount = (count: number) => t('widgets.gallery.videoCount', { count });

    return {
      archiveBoard: async (boardId, archived) => {
        const owner = captureAccountScope();

        try {
          await updateGalleryBoard(boardId, { archived }, owner.signal);

          assertAccountScopeCurrent(owner);
          recordSuccess(
            t(archived ? 'widgets.gallery.boardArchived' : 'widgets.gallery.boardUnarchived', {
              name: getBoardName(boardId),
            })
          );

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
      loadMore,
      refresh,
      renameBoard: async (boardId, boardName) => {
        const owner = captureAccountScope();

        try {
          await updateGalleryBoard(boardId, { name: boardName }, owner.signal);

          assertAccountScopeCurrent(owner);
          recordSuccess(t('widgets.gallery.boardRenamed', { name: boardName }));
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
          recordSuccess(t('widgets.gallery.projectBoardCreated', { name: board.name }));
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

        if (isDateBoardId(selectedBoardId)) {
          notifications.reportError({
            area: 'gallery-upload',
            message: t('widgets.gallery.uploadDateBoardUnavailable'),
            namespace: 'gallery',
          });
          return;
        }

        const accepted = files.flatMap((file) => {
          const classification = classifyGalleryUpload(file);

          return classification ? [{ file, kind: classification.kind }] : [];
        });

        if (accepted.length === 0) {
          notifications.reportError({
            area: 'gallery-upload',
            message: t('widgets.gallery.uploadUnsupported'),
            namespace: 'gallery',
          });
          return;
        }

        try {
          const targetBoardId = getBoard(selectedBoardId)?.kind === 'board' ? selectedBoardId : 'none';
          const targetBoardName = getBoardName(selectedBoardId);
          const imageUploads = accepted.filter((upload) => upload.kind === 'image');
          const videoUploads = accepted.filter((upload) => upload.kind === 'video');
          const imageResultsPromise = Promise.allSettled(
            imageUploads.map(({ file }) => uploadGalleryImage(file, targetBoardId, { signal: owner.signal }))
          );
          const videoResultsPromise = (async () => {
            const results: PromiseSettledResult<GalleryItem>[] = [];

            for (const { file } of videoUploads) {
              owner.signal.throwIfAborted();

              try {
                results.push({
                  status: 'fulfilled',
                  value: await uploadGalleryVideo(file, targetBoardId, { signal: owner.signal }),
                });
              } catch (reason: unknown) {
                if (owner.signal.aborted) {
                  throw reason;
                }

                results.push({ reason, status: 'rejected' });
              }
            }

            return results;
          })();
          const [imageResults, videoResults] = await Promise.all([imageResultsPromise, videoResultsPromise]);

          assertAccountScopeCurrent(owner);
          const uploadedImages = imageResults.flatMap((result) =>
            result.status === 'fulfilled' ? [legacyGeneratedImageToGalleryItem(result.value)] : []
          );
          const uploadedVideos = videoResults.flatMap((result) =>
            result.status === 'fulfilled' ? [result.value] : []
          );
          const uploadedItems = [...uploadedImages, ...uploadedVideos];
          const failedCount = files.length - uploadedItems.length;

          if (uploadedItems.length === 0) {
            notifications.reportError({
              area: 'gallery-upload',
              message: t('widgets.gallery.uploadFailed', { failed: failedCount }),
              namespace: 'gallery',
            });
            return;
          }

          const currentGalleryLocation = getCurrentGalleryLocation();
          const visibleUploads = uploadedItems.filter(
            (item) =>
              item.boardId === currentGalleryLocation.selectedBoardId &&
              (currentGalleryLocation.galleryView === 'images'
                ? item.category === 'general'
                : item.category === 'control' || item.category === 'mask' || item.category === 'user')
          );
          const newestVisibleUpload = visibleUploads.reduce<GalleryItem | undefined>(
            (newest, item) =>
              newest === undefined || compareGalleryItems(item, newest, { orderDir: 'DESC' }) < 0 ? item : newest,
            undefined
          );

          if (newestVisibleUpload) {
            gallery.selectItem(newestVisibleUpload);
          }

          refresh();

          const summary = t('widgets.gallery.uploadSummary', {
            board: targetBoardName,
            failed: failedCount,
            images: formatImageCount(uploadedImages.length),
            videos: formatVideoCount(uploadedVideos.length),
          });
          const split =
            imageUploads.length > 0 && videoUploads.length > 0 ? ` ${t('widgets.gallery.uploadSplit')}` : '';
          recordSuccess(
            t(
              failedCount > 0 ? 'widgets.gallery.uploadPartialTitle' : 'widgets.gallery.uploadSuccessTitle',
              failedCount > 0
                ? { succeeded: uploadedItems.length, total: files.length }
                : { count: uploadedItems.length }
            ),
            `${summary}${split}`
          );
        } catch (error: unknown) {
          if (!isAccountScopeCurrent(owner)) {
            return;
          }

          recordError(error);
        }
      },
    };
  }, [
    boards,
    gallery,
    getCurrentGalleryLocation,
    loadMore,
    notifications,
    projectBoardId,
    projectName,
    queryClient,
    selectedBoardId,
    t,
  ]);
};
