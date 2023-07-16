import { log } from 'app/logging/useLogger';
import { boardImagesApi } from 'services/api/endpoints/boardImages';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'boards' });

export const addImageRemovedFromBoardFulfilledListener = () => {
  startAppListening({
    matcher: boardImagesApi.endpoints.removeImageFromBoard.matchFulfilled,
    effect: (action, { getState, dispatch }) => {
      const { board_id, image_name } = action.meta.arg.originalArgs;

      // TODO: update listImages cache for this board

      moduleLog.debug(
        { data: { board_id, image_name } },
        'Image added to board'
      );
    },
  });
};

export const addImageRemovedFromBoardRejectedListener = () => {
  startAppListening({
    matcher: boardImagesApi.endpoints.removeImageFromBoard.matchRejected,
    effect: (action, { getState, dispatch }) => {
      const { board_id, image_name } = action.meta.arg.originalArgs;

      moduleLog.debug(
        { data: { board_id, image_name } },
        'Problem adding image to board'
      );
    },
  });
};
