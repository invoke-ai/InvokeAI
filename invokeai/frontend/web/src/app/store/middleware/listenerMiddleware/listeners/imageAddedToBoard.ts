import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/store';
import { imagesApi } from 'services/api/endpoints/images';

const log = logger('gallery');

export const addImageAddedToBoardFulfilledListener = (startAppListening: AppStartListening) => {
  startAppListening({
    matcher: imagesApi.endpoints.addImageToBoard.matchFulfilled,
    effect: (action) => {
      const { board_id, image_name } = action.meta.arg.originalArgs;
      log.debug({ board_id, image_name }, 'Image added to board');
    },
  });

  startAppListening({
    matcher: imagesApi.endpoints.addImageToBoard.matchRejected,
    effect: (action) => {
      const { board_id, image_name } = action.meta.arg.originalArgs;
      log.debug({ board_id, image_name }, 'Problem adding image to board');
    },
  });
};
