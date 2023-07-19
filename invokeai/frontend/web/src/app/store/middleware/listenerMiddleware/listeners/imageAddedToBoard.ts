import { log } from 'app/logging/useLogger';
import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'boards' });

export const addImageAddedToBoardFulfilledListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.addImageToBoard.matchFulfilled,
    effect: (action, { getState, dispatch }) => {
      const { board_id, imageDTO } = action.meta.arg.originalArgs;

      // TODO: update listImages cache for this board

      moduleLog.debug({ data: { board_id, imageDTO } }, 'Image added to board');
    },
  });
};

export const addImageAddedToBoardRejectedListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.addImageToBoard.matchRejected,
    effect: (action, { getState, dispatch }) => {
      const { board_id, imageDTO } = action.meta.arg.originalArgs;

      moduleLog.debug(
        { data: { board_id, imageDTO } },
        'Problem adding image to board'
      );
    },
  });
};
