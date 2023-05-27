import { startAppListening } from '..';
import { imageUpdated } from 'services/thunks/image';
import { log } from 'app/logging/useLogger';

const moduleLog = log.child({ namespace: 'image' });

export const addImageUpdatedFulfilledListener = () => {
  startAppListening({
    actionCreator: imageUpdated.fulfilled,
    effect: (action, { dispatch, getState }) => {
      moduleLog.debug(
        { oldImage: action.meta.arg, updatedImage: action.payload },
        'Image updated'
      );
    },
  });
};

export const addImageUpdatedRejectedListener = () => {
  startAppListening({
    actionCreator: imageUpdated.rejected,
    effect: (action, { dispatch }) => {
      moduleLog.debug({ oldImage: action.meta.arg }, 'Image update failed');
    },
  });
};
