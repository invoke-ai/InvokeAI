import { logger } from 'app/logging/logger';
import type { AppStartListening, RootState } from 'app/store/store';
import { omit } from 'es-toolkit/compat';
import { selectListBoardsQueryArgs } from 'features/gallery/store/gallerySelectors';
import { boardIdSelected, galleryViewChanged } from 'features/gallery/store/gallerySlice';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { boardsApi } from 'services/api/endpoints/boards';
import { videosApi } from 'services/api/endpoints/videos';
import type { VideoDTO } from 'services/api/types';

import { getVideoUploadFailedDescription } from './videoUploadFailedDescription';

const log = logger('gallery');

/**
 * Mirrors getUploadedToastDescription in imageUploaded.ts: names the board the video
 * landed on so the user knows where to find it.
 */
const getUploadedToastDescription = (boardId: string, state: RootState) => {
  if (boardId === 'none') {
    return t('toast.addedToUncategorized');
  }
  const queryArgs = selectListBoardsQueryArgs(state);
  const { data } = boardsApi.endpoints.listAllBoards.select(queryArgs)(state);
  const board = data?.find((b) => b.board_id === boardId);

  return t('toast.addedToBoard', { name: board?.board_name ?? boardId });
};

let lastUploadedToastTimeout: number | null = null;

/**
 * Success and failure feedback for video uploads, mirroring the image upload listeners.
 * Because the batch helpers (uploadVideos) aggregate with Promise.allSettled, this
 * per-mutation listener is the layer where an individual rejected MP4 becomes visible
 * to the user — including which file failed.
 */
export const addVideoUploadedListeners = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: videosApi.endpoints.uploadVideo.matchFulfilled,
    effect: (action, { dispatch, getState }) => {
      const videoDTO: VideoDTO = action.payload;
      const silent = action.meta.arg.originalArgs.silent;
      const isFirstUploadOfBatch = action.meta.arg.originalArgs.isFirstUploadOfBatch ?? true;

      if (silent || videoDTO.is_intermediate) {
        // If the video is silent or intermediate, we don't want to show a toast
        return;
      }

      const state = getState();

      log.debug({ videoDTO }, 'Video uploaded');

      const boardId = videoDTO.board_id ?? 'none';

      if (lastUploadedToastTimeout !== null) {
        window.clearTimeout(lastUploadedToastTimeout);
      }
      const toastApi = toast({
        id: 'VIDEO_UPLOADED',
        title: t('toast.videoUploaded'),
        description: getUploadedToastDescription(boardId, state),
        status: 'success',
        duration: null, // we will close the toast manually
      });
      lastUploadedToastTimeout = window.setTimeout(() => {
        toastApi.close();
      }, 3000);

      // Only navigate the gallery on the first upload of a batch — see the matching
      // comment in imageUploaded.ts for the board-hijacking failure mode this avoids.
      if (isFirstUploadOfBatch) {
        dispatch(boardIdSelected({ boardId }));
        dispatch(galleryViewChanged('assets'));
      }
    },
  });

  startAppListening({
    matcher: videosApi.endpoints.uploadVideo.matchRejected,
    effect: (action) => {
      const fileName = action.meta.arg.originalArgs.file.name;
      const sanitizedData = {
        arg: {
          ...omit(action.meta.arg.originalArgs, ['file']),
          file: `<Blob ${fileName}>`,
        },
      };
      log.error({ ...sanitizedData }, 'Video upload failed');
      toast({
        title: t('toast.videoUploadFailed'),
        description: getVideoUploadFailedDescription(fileName, action.error.message),
        status: 'error',
      });
    },
  });
};
