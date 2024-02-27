import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { imagesApi } from 'services/api/endpoints/images';

export const addImageRemovedFromBoardFulfilledListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: imagesApi.endpoints.removeImageFromBoard.matchFulfilled,
    effect: (action) => {
      const log = logger('images');
      const imageDTO = action.meta.arg.originalArgs;

      log.debug({ imageDTO }, 'Image removed from board');
    },
  });

  startAppListening({
    matcher: imagesApi.endpoints.removeImageFromBoard.matchRejected,
    effect: (action) => {
      const log = logger('images');
      const imageDTO = action.meta.arg.originalArgs;

      log.debug({ imageDTO }, 'Problem removing image from board');
    },
  });
};
