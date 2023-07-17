import { log } from 'app/logging/useLogger';
import { boardImagesApi } from 'services/api/endpoints/boardImages';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'boards' });

export const addImageAddedToBoardFulfilledListener = () => {
  startAppListening({
    matcher: boardImagesApi.endpoints.addImageToBoard.matchFulfilled,
    effect: (action, { getState, dispatch }) => {
      const { board_id, image_name } = action.meta.arg.originalArgs;

      moduleLog.debug(
        { data: { board_id, image_name } },
        'Image added to board'
      );
    },
  });
};

export const addImageAddedToBoardRejectedListener = () => {
  startAppListening({
    matcher: boardImagesApi.endpoints.addImageToBoard.matchRejected,
    effect: (action, { getState, dispatch }) => {
      const { board_id, image_name } = action.meta.arg.originalArgs;

      moduleLog.debug(
        { data: { board_id, image_name } },
        'Problem adding image to board'
      );
    },
  });
};
