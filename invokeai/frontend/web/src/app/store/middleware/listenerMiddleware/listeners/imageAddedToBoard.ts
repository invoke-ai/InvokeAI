import { log } from 'app/logging/useLogger';
import { startAppListening } from '..';
import { imageMetadataReceived } from 'services/thunks/image';
import { api } from 'services/apiSlice';

const moduleLog = log.child({ namespace: 'boards' });

export const addImageAddedToBoardFulfilledListener = () => {
  startAppListening({
    matcher: api.endpoints.addImageToBoard.matchFulfilled,
    effect: (action, { getState, dispatch }) => {
      const { board_id, image_name } = action.meta.arg.originalArgs;

      moduleLog.debug(
        { data: { board_id, image_name } },
        'Image added to board'
      );

      dispatch(
        imageMetadataReceived({
          imageName: image_name,
        })
      );
    },
  });
};

export const addImageAddedToBoardRejectedListener = () => {
  startAppListening({
    matcher: api.endpoints.addImageToBoard.matchRejected,
    effect: (action, { getState, dispatch }) => {
      const { board_id, image_name } = action.meta.arg.originalArgs;

      moduleLog.debug(
        { data: { board_id, image_name } },
        'Problem adding image to board'
      );
    },
  });
};
