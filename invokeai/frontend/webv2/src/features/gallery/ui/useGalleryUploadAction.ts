import type { GalleryItem } from '@features/gallery/core/items';
import type { GalleryBoard, GalleryView } from '@features/gallery/core/types';

import { canOnBoard } from '@features/gallery/core/boardAccess';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import {
  classifyGalleryUpload,
  compareGalleryItems,
  legacyGeneratedImageToGalleryItem,
} from '@features/gallery/core/items';
import {
  isDateBoardId,
  isInvalidGalleryBoardDestination,
  uploadGalleryImage,
  uploadGalleryVideo,
} from '@features/gallery/data/backend';
import {
  canAddToCachedGalleryBoard,
  ensureGalleryBoardDestinations,
  getCachedGalleryBoard,
  invalidateGallery,
  invalidateGalleryBoardDestinations,
} from '@features/gallery/data/queryCache';
import { createLogger } from '@platform/logging/logger';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { GalleryActions } from './GalleryWidgetContext';

import { useGalleryHost, useOptionalGalleryUi } from './GalleryUiContext';

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));
/** A real board's id: not Uncategorized and not a virtual board, which hold no uploads. */
const isGalleryBoardId = (boardId: string): boolean => boardId !== 'none' && !isInvalidGalleryBoardDestination(boardId);
const uploadLogger = createLogger({ area: 'upload', namespace: 'gallery' });

export const useGalleryUploadAction = ({
  boards,
  getCurrentGalleryLocation,
  selectedBoardId: selectedBoardIdOrGetter,
}: {
  boards: GalleryBoard[];
  /**
   * Where the Gallery widget is showing on completion; the newest upload visible there gets selected. Omit to leave
   * the selection alone, as hosts without a workbench must.
   */
  getCurrentGalleryLocation?: () => { galleryView: GalleryView; selectedBoardId: string };
  /**
   * A getter is read when the upload starts and, when it names a real board, once more after the listing of every
   * readable board is ensured, the second reading deciding: for hosts that resolve the board from that listing or
   * must not re-render on board changes.
   */
  selectedBoardId: string | (() => string);
}): GalleryActions['uploadFiles'] => {
  const { notifications } = useGalleryHost();
  const gallery = useOptionalGalleryUi()?.gallery;
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return useCallback(
    async (files) => {
      const owner = captureAccountScope();
      const readSelectedBoardId = () =>
        typeof selectedBoardIdOrGetter === 'function' ? selectedBoardIdOrGetter() : selectedBoardIdOrGetter;
      let selectedBoardId = readSelectedBoardId();

      // A board is judged from the listing of every readable board, never trusted as given: an upload to a board
      // that was deleted, archived out of reach or shared read-only since the panel listed it would only fail.
      // Uncategorized and virtual boards need no listing.
      if (isGalleryBoardId(selectedBoardId)) {
        await ensureGalleryBoardDestinations(queryClient);
        if (!isAccountScopeCurrent(owner)) {
          return [];
        }
        selectedBoardId = readSelectedBoardId();
      }

      if (isDateBoardId(selectedBoardId)) {
        notifications.reportError({
          area: 'gallery-upload',
          message: t('widgets.gallery.uploadDateBoardUnavailable'),
          namespace: 'gallery',
        });
        return [];
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
        return [];
      }

      const getBoard = (boardId: string) => boards.find((board) => board.id === boardId);
      const selectedBoard = getBoard(selectedBoardId);
      const targetBoardName = selectedBoard
        ? getGalleryBoardLabel(selectedBoard, t)
        : t('widgets.gallery.uncategorized');

      // Refused before any upload: the server would refuse each file, leaving nothing but errors. The reason comes
      // from the listing that decided, not the panel's older view of the board. The listing is read again at once,
      // so the panel stops offering the board and the next upload is judged afresh.
      if (isGalleryBoardId(selectedBoardId) && !canAddToCachedGalleryBoard(selectedBoardId, queryClient)) {
        const cachedBoard = getCachedGalleryBoard(selectedBoardId, queryClient);
        const cachedBoardName = cachedBoard ? getGalleryBoardLabel(cachedBoard, t) : targetBoardName;

        notifications.reportError({
          area: 'gallery-upload',
          message:
            cachedBoard && !canOnBoard(cachedBoard, 'add_media')
              ? t('widgets.gallery.uploadsUnavailableForViewOnlyBoard', { name: cachedBoardName })
              : cachedBoard || selectedBoard
                ? t('widgets.gallery.uploadsUnavailableForBoard', { name: cachedBoardName })
                : t('widgets.gallery.uploadsUnavailableForUnlistedBoard'),
          namespace: 'gallery',
        });
        void invalidateGalleryBoardDestinations(queryClient, owner);
        return [];
      }
      const formatImageCount = (count: number) => t('widgets.gallery.imageCount', { count });
      const formatVideoCount = (count: number) => t('widgets.gallery.videoCount', { count });
      const recordError = (error: unknown) =>
        notifications.reportError({ area: 'gallery-actions', message: toErrorMessage(error), namespace: 'gallery' });

      try {
        // A virtual board holds no uploads; a real board was judged above.
        const targetBoardId = isGalleryBoardId(selectedBoardId) ? selectedBoardId : 'none';
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
        const uploadedVideos = videoResults.flatMap((result) => (result.status === 'fulfilled' ? [result.value] : []));
        const uploadedItems = [...uploadedImages, ...uploadedVideos];
        const failedCount = files.length - uploadedItems.length;
        const firstRejection = [...imageResults, ...videoResults].find((result) => result.status === 'rejected');

        // A total failure is reported once, by the notification path below.
        if (uploadedItems.length > 0) {
          uploadLogger[failedCount > 0 ? 'warn' : 'info']({
            context: {
              boardId: targetBoardId,
              failed: failedCount,
              images: uploadedImages.length,
              requested: files.length,
              videos: uploadedVideos.length,
            },
            error: firstRejection?.reason,
            message:
              failedCount > 0
                ? `Uploaded ${uploadedItems.length} of ${files.length} files`
                : `Uploaded ${uploadedItems.length} files`,
            name: failedCount > 0 ? 'gallery.upload-partial' : 'gallery.upload-completed',
          });
        }

        // What landed is listed; a board that refused a file leaves the listing's view of it stale. Either way the
        // gallery, boards included, reads again.
        void invalidateGallery(queryClient);

        if (uploadedItems.length === 0) {
          const fallback = t('widgets.gallery.uploadFailed', { failed: failedCount });

          notifications.reportError({
            area: 'gallery-upload',
            message: firstRejection ? getApiErrorMessage(firstRejection.reason, fallback) : fallback,
            namespace: 'gallery',
          });
          return [];
        }

        const currentGalleryLocation = getCurrentGalleryLocation?.();
        const visibleUploads = currentGalleryLocation
          ? uploadedItems.filter(
              (item) =>
                item.boardId === currentGalleryLocation.selectedBoardId &&
                (currentGalleryLocation.galleryView === 'images'
                  ? item.category === 'general'
                  : item.category === 'control' || item.category === 'mask' || item.category === 'user')
            )
          : [];
        const newestVisibleUpload = visibleUploads.reduce<GalleryItem | undefined>(
          (newest, item) =>
            newest === undefined || compareGalleryItems(item, newest, { orderDir: 'DESC' }) < 0 ? item : newest,
          undefined
        );

        if (newestVisibleUpload) {
          gallery?.selectItem(newestVisibleUpload);
        }

        const summary = t('widgets.gallery.uploadSummary', {
          board: targetBoardName,
          failed: failedCount,
          images: formatImageCount(uploadedImages.length),
          videos: formatVideoCount(uploadedVideos.length),
        });
        const split = imageUploads.length > 0 && videoUploads.length > 0 ? ` ${t('widgets.gallery.uploadSplit')}` : '';
        notifications.add({
          kind: 'success',
          message: `${summary}${split}`,
          title: t(
            failedCount > 0 ? 'widgets.gallery.uploadPartialTitle' : 'widgets.gallery.uploadSuccessTitle',
            failedCount > 0 ? { succeeded: uploadedItems.length, total: files.length } : { count: uploadedItems.length }
          ),
        });

        return uploadedItems;
      } catch (error: unknown) {
        if (isAccountScopeCurrent(owner)) {
          recordError(error);
        }

        return [];
      }
    },
    [boards, gallery, getCurrentGalleryLocation, notifications, queryClient, selectedBoardIdOrGetter, t]
  );
};
