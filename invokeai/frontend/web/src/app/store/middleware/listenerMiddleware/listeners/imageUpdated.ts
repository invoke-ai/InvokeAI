import { log } from 'app/logging/useLogger';
import { imagesApi } from 'services/api/endpoints/images';
import { startAppListening } from '..';

const moduleLog = log.child({ namespace: 'image' });

export const addImageUpdatedFulfilledListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.updateImage.matchFulfilled,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug(
        {
          data: {
            oldImage: action.meta.arg.originalArgs,
            updatedImage: action.payload,
          },
        },
        'Image updated'
      );
    },
  });
};

export const addImageUpdatedRejectedListener = () => {
  startAppListening({
    matcher: imagesApi.endpoints.updateImage.matchRejected,
    effect: (action, { dispatch }) => {
      moduleLog.debug(
        { data: action.meta.arg.originalArgs },
        'Image update failed'
      );
    },
  });
};
