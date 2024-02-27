import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { imagesApi } from 'services/api/endpoints/images';

export const addImageAddedToBoardFulfilledListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: imagesApi.endpoints.addImageToBoard.matchFulfilled,
    effect: (action) => {
      const log = logger('images');
      const { board_id, imageDTO } = action.meta.arg.originalArgs;

      // TODO: update listImages cache for this board

      log.debug({ board_id, imageDTO }, 'Image added to board');
    },
  });

  startAppListening({
    matcher: imagesApi.endpoints.addImageToBoard.matchRejected,
    effect: (action) => {
      const log = logger('images');
      const { board_id, imageDTO } = action.meta.arg.originalArgs;

      log.debug({ board_id, imageDTO }, 'Problem adding image to board');
    },
  });
};
