import type { GalleryItem } from '@features/gallery/core/items';
import type { GalleryBoard, GalleryView } from '@features/gallery/core/types';

import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { compareGalleryItems, legacyGeneratedImageToGalleryItem } from '@features/gallery/core/items';
import {
  classifyGalleryUpload,
  isDateBoardId,
  uploadGalleryImage,
  uploadGalleryVideo,
} from '@features/gallery/data/backend';
import { invalidateGallery } from '@features/gallery/data/queryCache';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { GalleryActions } from './GalleryWidgetContext';

import { useGalleryUi } from './GalleryUiContext';

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

export const useGalleryUploadAction = ({
  boards,
  getCurrentGalleryLocation,
  selectedBoardId,
}: {
  boards: GalleryBoard[];
  getCurrentGalleryLocation: () => { galleryView: GalleryView; selectedBoardId: string };
  selectedBoardId: string;
}): GalleryActions['uploadFiles'] => {
  const { gallery, notifications } = useGalleryUi();
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return useCallback(
    async (files) => {
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

      const getBoard = (boardId: string) => boards.find((board) => board.id === boardId);
      const selectedBoard = getBoard(selectedBoardId);
      const targetBoardName = selectedBoard
        ? getGalleryBoardLabel(selectedBoard, t)
        : t('widgets.gallery.uncategorized');
      const formatImageCount = (count: number) => t('widgets.gallery.imageCount', { count });
      const formatVideoCount = (count: number) => t('widgets.gallery.videoCount', { count });
      const recordError = (error: unknown) =>
        notifications.reportError({ area: 'gallery-actions', message: toErrorMessage(error), namespace: 'gallery' });

      try {
        const targetBoardId = selectedBoard?.kind === 'board' ? selectedBoardId : 'none';
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

        void invalidateGallery(queryClient);

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
      } catch (error: unknown) {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        recordError(error);
      }
    },
    [boards, gallery, getCurrentGalleryLocation, notifications, queryClient, selectedBoardId, t]
  );
};
