import type { GalleryItem } from '@features/gallery/core/items';
import type { GalleryBoard, GalleryView } from '@features/gallery/core/types';
import type { QueryClient } from '@tanstack/react-query';
import type { TFunction } from 'i18next';

import { canOnBoard } from '@features/gallery/core/boardAccess';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import {
  ALL_READABLE_BOARDS_ID,
  classifyGalleryUpload,
  compareGalleryItems,
  getGalleryItemView,
  legacyGeneratedImageToGalleryItem,
} from '@features/gallery/core/items';
import { formatGalleryMediaCounts } from '@features/gallery/core/mediaCounts';
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

import type { GalleryCommandsPort, GalleryNotificationsPort } from './GalleryUiContext';
import type { GalleryActions } from './GalleryWidgetContext';

import { useGalleryHost, useOptionalGalleryUi } from './GalleryUiContext';

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));
/** Uncategorized and virtual boards hold no uploads. */
const isGalleryBoardId = (boardId: string): boolean => boardId !== 'none' && !isInvalidGalleryBoardDestination(boardId);
const uploadLogger = createLogger({ area: 'upload', namespace: 'gallery' });

export interface GalleryUploadOptions {
  boards: readonly GalleryBoard[];
  /** Absent hosts leave the selection alone. */
  gallery?: Pick<GalleryCommandsPort, 'selectItem' | 'setView'>;
  /** Omit to leave the selection alone, as hosts without a workbench must. */
  getCurrentGalleryLocation?: () => { galleryView: GalleryView; selectedBoardId: string };
  notifications: GalleryNotificationsPort;
  queryClient: QueryClient;
  /** Gallery uploads switch to the tab that received them; uploads made elsewhere (a Canvas paste) leave it alone. */
  revealInReceivingTab?: boolean;
  /** A getter naming a real board is read again after the all-boards listing is ensured, and that reading decides. */
  selectedBoardId: string | (() => string);
  t: TFunction;
}

/** Resolves to the uploaded items, empty when nothing landed or the account changed; for callers outside React. */
export const uploadGalleryFiles = async (
  files: File[],
  {
    boards,
    gallery,
    getCurrentGalleryLocation,
    notifications,
    queryClient,
    revealInReceivingTab = false,
    selectedBoardId: selectedBoardIdOrGetter,
    t,
  }: GalleryUploadOptions
): Promise<GalleryItem[]> => {
  const owner = captureAccountScope();
  // Only while the Gallery is still where the upload started may it switch the tab.
  const launchLocation = getCurrentGalleryLocation?.();
  const readSelectedBoardId = () =>
    typeof selectedBoardIdOrGetter === 'function' ? selectedBoardIdOrGetter() : selectedBoardIdOrGetter;
  let selectedBoardId = readSelectedBoardId();

  // A search of every board has no board of its own; Uncategorized would be a guess.
  if (selectedBoardId === ALL_READABLE_BOARDS_ID) {
    notifications.reportError({
      area: 'gallery-upload',
      message: t('widgets.gallery.uploadsUnavailableForAllBoards'),
      namespace: 'gallery',
    });
    return [];
  }

  // Judged from the all-boards listing, never trusted as given: the board may be gone or read-only since it was listed.
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

  const selectedBoard =
    boards.find((board) => board.id === selectedBoardId) ?? getCachedGalleryBoard(selectedBoardId, queryClient);
  const targetBoardName = selectedBoard ? getGalleryBoardLabel(selectedBoard, t) : t('widgets.gallery.uncategorized');

  // Refused before any upload, from the listing that decided; reread the listing so the panel stops offering the board.
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
  const recordError = (error: unknown) =>
    notifications.reportError({ area: 'gallery-actions', message: toErrorMessage(error), namespace: 'gallery' });

  try {
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

    // A total failure is reported once, below.
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

    // A board that refused a file leaves the listing stale, so the gallery reads again either way.
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
    const uploadsOnBoard = currentGalleryLocation
      ? uploadedItems.filter((item) => item.boardId === currentGalleryLocation.selectedBoardId)
      : [];
    const visibleUploads = uploadsOnBoard.filter(
      (item) => getGalleryItemView(item) === currentGalleryLocation?.galleryView
    );
    // Images and videos land on different tabs; a Gallery the user moved meanwhile keeps where they took it.
    const isGalleryWhereLaunched =
      revealInReceivingTab &&
      launchLocation !== undefined &&
      currentGalleryLocation?.galleryView === launchLocation.galleryView &&
      currentGalleryLocation.selectedBoardId === launchLocation.selectedBoardId;
    const receivingView =
      visibleUploads.length === 0 && isGalleryWhereLaunched && uploadsOnBoard[0]
        ? getGalleryItemView(uploadsOnBoard[0])
        : null;
    const shownUploads = receivingView
      ? uploadsOnBoard.filter((item) => getGalleryItemView(item) === receivingView)
      : visibleUploads;
    const newestShownUpload = shownUploads.reduce<GalleryItem | undefined>(
      (newest, item) =>
        newest === undefined || compareGalleryItems(item, newest, { orderDir: 'DESC' }) < 0 ? item : newest,
      undefined
    );

    if (gallery && receivingView) {
      gallery.setView(receivingView);
    }
    if (newestShownUpload) {
      gallery?.selectItem(newestShownUpload);
    }

    const media = formatGalleryMediaCounts(t, { images: uploadedImages.length, videos: uploadedVideos.length });
    const message = [
      t('widgets.gallery.uploadSummary', { board: targetBoardName, media }),
      failedCount > 0 ? t('widgets.gallery.uploadFailedCount', { count: failedCount }) : '',
      uploadedImages.length > 0 && uploadedVideos.length > 0 ? t('widgets.gallery.uploadSplit') : '',
    ]
      .filter(Boolean)
      .join(' ');
    notifications.add({
      kind: 'success',
      message,
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
};

export const useGalleryUploadAction = ({
  boards,
  getCurrentGalleryLocation,
  revealInReceivingTab = false,
  selectedBoardId,
}: Pick<
  GalleryUploadOptions,
  'boards' | 'getCurrentGalleryLocation' | 'revealInReceivingTab' | 'selectedBoardId'
>): GalleryActions['uploadFiles'] => {
  const { notifications } = useGalleryHost();
  const gallery = useOptionalGalleryUi()?.gallery;
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  return useCallback(
    (files) =>
      uploadGalleryFiles(files, {
        boards,
        gallery,
        getCurrentGalleryLocation,
        notifications,
        queryClient,
        revealInReceivingTab,
        selectedBoardId,
        t,
      }),
    [boards, gallery, getCurrentGalleryLocation, notifications, queryClient, revealInReceivingTab, selectedBoardId, t]
  );
};
