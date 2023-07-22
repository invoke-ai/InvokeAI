import { logger } from 'app/logging/logger';
import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '..';

export const addImageRemovedFromBoardFulfilledListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.removeImageFromBoard.matchFulfilled,
    effect: (action) => {
      const log = logger('images');
      const imageDTO = action.meta.arg.originalArgs;

      log.debug({ imageDTO }, 'Image removed from board');
    },
  });
};

export const addImageRemovedFromBoardRejectedListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.removeImageFromBoard.matchRejected,
    effect: (action) => {
      const log = logger('images');
      const imageDTO = action.meta.arg.originalArgs;

      log.debug({ imageDTO }, 'Problem removing image from board');
    },
  });
};
